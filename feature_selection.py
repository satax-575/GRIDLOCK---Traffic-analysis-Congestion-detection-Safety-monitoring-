"""
feature_selection.py
--------------------
Feature selection for Gridlock 2.0 demand prediction.

Changes from original:
  - LGBM importance screening uses sqrt(demand) as target [MANDATORY-2]
  - New features added to candidate list (demand_same_slot_prev_day,
    highway_x_afternoon, geo_slot_mean_demand, demand_cv, etc.)
  - Variance and correlation filters unchanged
  - Lag cols for test are now real-valued (from add_test_lag_lookup),
    so TRAIN_ONLY_FEATURES list is no longer needed for alignment —
    test already has all lag columns populated.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Columns that must never be used as model features
DROP_ALWAYS = [
    "Index", "geohash", "timestamp", "demand",
    "geohash_prefix5", "geohash_prefix4",   # raw string prefixes
    "demand_sqrt",                            # transformed target — not a feature
    "geohash_mean_temp",                      # intermediate; deviation is the feature
]

TARGET_COL       = "demand"
TARGET_SQRT_COL  = "demand_sqrt"    # sqrt-transformed target used for importance scoring


# ─────────────────────────────────────────────
# 1. Variance Threshold Filter
# ─────────────────────────────────────────────

def remove_low_variance(df: pd.DataFrame, threshold: float = 0.001) -> list:
    """
    Remove numeric columns with variance below threshold.
    Returns list of retained column names.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    selector   = VarianceThreshold(threshold=threshold)
    selector.fit(numeric_df)
    retained = numeric_df.columns[selector.get_support()].tolist()
    dropped  = [c for c in numeric_df.columns if c not in retained]
    if dropped:
        print(f"[variance] dropped {len(dropped)} low-variance cols: {dropped}")
    return retained


# ─────────────────────────────────────────────
# 2. Correlation Filter
# ─────────────────────────────────────────────

def remove_high_correlation(df: pd.DataFrame, feature_cols: list,
                            threshold: float = 0.95) -> list:
    """
    Remove one from each pair with |Pearson r| >= threshold.
    Retains the feature that appears first (alphabetically stable).
    """
    corr_matrix = df[feature_cols].corr().abs()
    upper_tri   = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] >= threshold)]
    retained = [c for c in feature_cols if c not in to_drop]
    if to_drop:
        print(f"[correlation] dropped {len(to_drop)} highly correlated cols: {to_drop}")
    return retained


# ─────────────────────────────────────────────
# 3. LightGBM Importance Ranking
# ─────────────────────────────────────────────

def get_lgbm_importance(train: pd.DataFrame, feature_cols: list,
                        top_n: int = None) -> pd.DataFrame:
    """
    Fit a quick LightGBM regressor on sqrt(demand) [MANDATORY-2] and
    return feature importances sorted descending.

    Using sqrt(demand) (skewness ~1.58) rather than raw demand (skewness ~3.73)
    gives LGBM a better-conditioned target during importance ranking.
    """
    # Use sqrt-transformed target if already computed, else compute inline
    if TARGET_SQRT_COL in train.columns:
        y = train[TARGET_SQRT_COL].values
    else:
        y = np.sqrt(train[TARGET_COL].values)

    X = train[feature_cols]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)

    importance_df = (
        pd.DataFrame({"feature": feature_cols,
                      "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    if top_n:
        importance_df = importance_df.head(top_n)

    print(f"[lgbm importance] top features:\n"
          f"{importance_df.head(15).to_string(index=False)}")
    return importance_df


# ─────────────────────────────────────────────
# 4. Master Feature Selection Pipeline
# ─────────────────────────────────────────────

def select_features(train: pd.DataFrame, test: pd.DataFrame,
                    corr_threshold: float = 0.95,
                    var_threshold:  float = 0.001,
                    top_n_lgbm:     int   = None):
    """
    Full feature selection pipeline.

    Since test now has all lag features populated via add_test_lag_lookup()
    and add_cross_day_lag(), train and test share the same feature set.

    Returns
    -------
    selected_features : list[str]
    importance_df     : pd.DataFrame   — LGBM importances
    X_train, X_test   : pd.DataFrame   — model-ready matrices
    """
    # Candidate columns: all numeric except always-dropped ones
    all_numeric    = train.select_dtypes(include=[np.number]).columns.tolist()
    candidate_cols = [c for c in all_numeric if c not in DROP_ALWAYS]

    # Step 1 — Variance filter
    retained_var = remove_low_variance(train[candidate_cols], threshold=var_threshold)

    # Step 2 — Correlation filter
    retained_corr = remove_high_correlation(train, retained_var, threshold=corr_threshold)

    # Step 3 — LGBM importance (on sqrt target)
    importance_df    = get_lgbm_importance(train, retained_corr, top_n=top_n_lgbm)
    selected_features = importance_df["feature"].tolist()

    print(f"\n[select_features] final feature count: {len(selected_features)}")
    print(f"[select_features] selected: {selected_features}\n")

    # Both train and test now have the same columns
    # (test lag was filled in feature_engineer.py)
    # If test is still missing any column, fill with 0
    for col in selected_features:
        if col not in test.columns:
            print(f"[select_features] WARNING: '{col}' missing in test — filling 0")
            test[col] = 0.0

    X_train = train[selected_features].copy()
    X_test  = test[selected_features].copy()

    return selected_features, importance_df, X_train, X_test


if __name__ == "__main__":
    from preprocessor    import preprocess
    from feature_engineer import build_features
    BASE = "../dataset"
    train, test, _ = preprocess(
        f"{BASE}/train.csv", f"{BASE}/test.csv", f"{BASE}/sample_submission.csv"
    )
    train, test = build_features(train, test)
    # Pre-compute sqrt target for importance scoring
    train["demand_sqrt"] = np.sqrt(train["demand"])
    selected, imp_df, X_train, X_test = select_features(train, test)
    print(X_train.shape, X_test.shape)
    print(imp_df.head(20))
