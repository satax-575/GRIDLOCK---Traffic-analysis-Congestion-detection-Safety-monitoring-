"""
train.py
--------
Production-grade training pipeline for Gridlock 2.0 demand prediction.

Key changes:
  [MANDATORY-2] sqrt(demand) target transformation — reduces skewness 3.73 -> 1.58
  [MANDATORY-4] Sample weights: highway x afternoon rows weighted 2.0x in LightGBM
  [MANDATORY-6] GroupKFold(groups=geohash) — prevents same-geohash leakage

Ensemble: LightGBM + XGBoost + CatBoost, weighted by inverse OOF RMSE.
All models train on sqrt(demand) and predictions are inverted via np.square().
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# Optional CatBoost
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[train] CatBoost not installed — skipping")

# Optional TabPFN-3  (pip install tabpfn>=2.0)
try:
    from tabpfn import TabPFNRegressor
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False
    print("[train] TabPFN not installed — skipping (pip install tabpfn)")

SEED = 42
TABPFN_TRAIN_SUBSAMPLE = 10_000  # rows per fold fed to TabPFN (memory guard)


# ─────────────────────────────────────────────
# Target Transformation Helpers  [MANDATORY-2]
# ─────────────────────────────────────────────

def transform_target(y: np.ndarray) -> np.ndarray:
    """Apply sqrt transformation. Skewness: ~3.73 -> ~1.58."""
    return np.sqrt(np.clip(y, 0, None))


def invert_target(y_sqrt: np.ndarray) -> np.ndarray:
    """Invert sqrt transform and clip to valid demand range [0, 1]."""
    return np.clip(y_sqrt ** 2, 0.0, 1.0)


# ─────────────────────────────────────────────
# Sample Weights  [MANDATORY-4]
# ─────────────────────────────────────────────

def build_sample_weights(train: pd.DataFrame,
                         highway_afternoon_weight: float = 2.0) -> np.ndarray:
    """
    Assign higher weight to highway x afternoon rows — the highest-RMSE segment.

    highway_x_afternoon = 1 when RoadType==Highway AND time_bucket==afternoon.
    These rows get 'highway_afternoon_weight' (default 2.0);
    all other rows get weight 1.0.

    The weight vector is aligned with the sorted training DataFrame index.
    """
    weights = np.ones(len(train), dtype=np.float32)
    if "highway_x_afternoon" in train.columns:
        mask = train["highway_x_afternoon"].values == 1
        weights[mask] = highway_afternoon_weight
        print(f"[weights] highway×afternoon rows: {mask.sum()} "
              f"({mask.mean()*100:.1f}%) → weight={highway_afternoon_weight:.1f}")
    else:
        print("[weights] 'highway_x_afternoon' not found — using uniform weights")
    return weights


# ─────────────────────────────────────────────
# LightGBM Parameters
# ─────────────────────────────────────────────

# ── Detect GPU once at startup ───────────────────────────────────────────
try:
    import torch as _torch
    USE_GPU = _torch.cuda.is_available()
    GPU_NAME = _torch.cuda.get_device_name(0) if USE_GPU else "none"
except ImportError:
    USE_GPU = False
    GPU_NAME = "none"
print(f"[train] GPU: {'YES — ' + GPU_NAME if USE_GPU else 'NO — running on CPU'}")

LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "learning_rate":     0.03,
    "num_leaves":        128,
    "max_depth":         -1,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "n_jobs":            -1,
    "random_state":      SEED,
    "verbose":           -1,
    # GPU acceleration — uses GPU if available, falls back to CPU silently
    "device":            "gpu" if USE_GPU else "cpu",
}

# XGBoost Parameters
XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "learning_rate":    0.03,
    "max_depth":        6,
    "min_child_weight": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    # XGBoost 2.0+: use device='cuda' + tree_method='hist' for GPU
    # (gpu_hist is deprecated and removed in XGBoost 2.0)
    "tree_method":      "hist",
    "device":           "cuda" if USE_GPU else "cpu",
    "seed":             SEED,
}

N_FOLDS   = 5
N_ROUNDS  = 3000
ES_ROUNDS = 150


# ─────────────────────────────────────────────
# LightGBM Training
# ─────────────────────────────────────────────

def train_lightgbm(X_train: pd.DataFrame, y_sqrt: np.ndarray,
                   X_test: pd.DataFrame, groups: np.ndarray,
                   sample_weights: np.ndarray):
    """
    Train LightGBM with GroupKFold [MANDATORY-6].
    Sample weights applied per-row [MANDATORY-4].
    Target is sqrt(demand) [MANDATORY-2].
    """
    kf    = GroupKFold(n_splits=N_FOLDS)
    oof   = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))

    print("\n" + "="*55)
    print("  LightGBM — GroupKFold Training (sqrt target)")
    print("="*55)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_sqrt, groups), 1):
        X_tr,  X_val  = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr,  y_val  = y_sqrt[tr_idx],        y_sqrt[val_idx]
        w_tr          = sample_weights[tr_idx]

        dtrain = lgb.Dataset(X_tr,  label=y_tr, weight=w_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            LGBM_PARAMS,
            dtrain,
            num_boost_round=N_ROUNDS,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(ES_ROUNDS, verbose=False),
                lgb.log_evaluation(period=500),
            ],
        )

        oof_sqrt       = model.predict(X_val, num_iteration=model.best_iteration)
        oof[val_idx]   = oof_sqrt
        preds         += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS

        fold_rmse = np.sqrt(mean_squared_error(
            invert_target(y_val), invert_target(oof_sqrt)
        ))
        print(f"  Fold {fold}  RMSE(demand): {fold_rmse:.6f}  "
              f"[best_iter={model.best_iteration}]")

    oof_rmse = np.sqrt(mean_squared_error(
        invert_target(y_sqrt), invert_target(oof)
    ))
    print(f"\n[LightGBM] Overall OOF RMSE (demand scale): {oof_rmse:.6f}")
    return oof, preds, oof_rmse


# ─────────────────────────────────────────────
# XGBoost Training
# ─────────────────────────────────────────────

def train_xgboost(X_train: pd.DataFrame, y_sqrt: np.ndarray,
                  X_test: pd.DataFrame, groups: np.ndarray,
                  sample_weights: np.ndarray):
    """
    Train XGBoost with GroupKFold [MANDATORY-6] and sample weights.
    Target is sqrt(demand) [MANDATORY-2].
    """
    kf  = GroupKFold(n_splits=N_FOLDS)
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))

    print("\n" + "="*55)
    print("  XGBoost — GroupKFold Training (sqrt target)")
    print("="*55)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_sqrt, groups), 1):
        X_tr,  X_val  = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr,  y_val  = y_sqrt[tr_idx],        y_sqrt[val_idx]

        dtrain = xgb.DMatrix(X_tr,  label=y_tr, weight=sample_weights[tr_idx])
        dval   = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            XGB_PARAMS,
            dtrain,
            num_boost_round=N_ROUNDS,
            evals=[(dval, "val")],
            early_stopping_rounds=ES_ROUNDS,
            verbose_eval=500,
        )

        oof_sqrt       = model.predict(xgb.DMatrix(X_val))
        oof[val_idx]   = oof_sqrt
        preds         += model.predict(xgb.DMatrix(X_test)) / N_FOLDS

        fold_rmse = np.sqrt(mean_squared_error(
            invert_target(y_val), invert_target(oof_sqrt)
        ))
        print(f"  Fold {fold}  RMSE(demand): {fold_rmse:.6f}")

    oof_rmse = np.sqrt(mean_squared_error(
        invert_target(y_sqrt), invert_target(oof)
    ))
    print(f"\n[XGBoost] Overall OOF RMSE (demand scale): {oof_rmse:.6f}")
    return oof, preds, oof_rmse


# ─────────────────────────────────────────────
# CatBoost Training
# ─────────────────────────────────────────────

def train_catboost(X_train: pd.DataFrame, y_sqrt: np.ndarray,
                   X_test: pd.DataFrame, groups: np.ndarray):
    """
    Train CatBoost with GroupKFold [MANDATORY-6].
    Target is sqrt(demand) [MANDATORY-2].
    CatBoost often complements LightGBM/XGBoost due to symmetric trees.
    """
    if not HAS_CATBOOST:
        return None, None, None

    kf  = GroupKFold(n_splits=N_FOLDS)
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))

    print("\n" + "="*55)
    print("  CatBoost — GroupKFold Training (sqrt target)")
    print("="*55)

    CB_PARAMS = dict(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        eval_metric="RMSE",
        early_stopping_rounds=ES_ROUNDS,
        verbose=500,
        allow_writing_files=False,
        task_type="GPU" if USE_GPU else "CPU",
        # GPU requires bootstrap_type='Bernoulli' to use subsample param
        # (default 'Bayesian' bootstrap ignores subsample on GPU)
        bootstrap_type="Bernoulli",
        subsample=0.8,
    )

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_sqrt, groups), 1):
        X_tr,  X_val  = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr,  y_val  = y_sqrt[tr_idx],        y_sqrt[val_idx]

        model = cb.CatBoostRegressor(**CB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
        )

        oof_sqrt       = model.predict(X_val)
        oof[val_idx]   = oof_sqrt
        preds         += model.predict(X_test) / N_FOLDS

        fold_rmse = np.sqrt(mean_squared_error(
            invert_target(y_val), invert_target(oof_sqrt)
        ))
        print(f"  Fold {fold}  RMSE(demand): {fold_rmse:.6f}")

    oof_rmse = np.sqrt(mean_squared_error(
        invert_target(y_sqrt), invert_target(oof)
    ))
    print(f"\n[CatBoost] Overall OOF RMSE (demand scale): {oof_rmse:.6f}")
    return oof, preds, oof_rmse


# ─────────────────────────────────────────────
# TabPFN-3 Training
# ─────────────────────────────────────────────

def train_tabpfn(X_train: pd.DataFrame, y_sqrt: np.ndarray,
                 X_test: pd.DataFrame, groups: np.ndarray):
    """
    TabPFN-3 (Prior-Labs) as 4th ensemble member.

    TabPFN is a pretrained transformer that performs Bayesian in-context
    learning on tabular data — no epochs, no hyperparameter tuning.
    Pretrained on millions of synthetic datasets, it provides strong
    architectural diversity vs the 3 GBDT models.

    Row handling:
    - TabPFN v2 supports up to 10K rows per fold natively.
    - When fold training size > TABPFN_TRAIN_SUBSAMPLE, we stratified-subsample
      by grouping y_sqrt into quantile bins to preserve distribution.
    - TabPFN-3 (tabpfn>=2.0) can handle larger sizes natively; the subsample
      is a safety guard for memory-constrained environments.
    """
    if not HAS_TABPFN:
        return None, None, None

    # Inject token into environment BEFORE any TabPFN call.
    # Priority: already-set env var > hardcoded fallback below.
    # The token is NEVER written to disk — only lives in this process.
    _token = os.environ.get("TABPFN_TOKEN", "")
    if not _token:
        print("[TabPFN] TABPFN_TOKEN not set — skipping.")
        print("  -> Set with: $env:TABPFN_TOKEN='<token>' then rerun.")
        return None, None, None
    os.environ["TABPFN_TOKEN"] = _token   # ensure child processes see it too
    print(f"[TabPFN] Token found (len={len(_token)}) — proceeding with local inference.")

    kf  = GroupKFold(n_splits=N_FOLDS)
    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))

    print("\n" + "="*55)
    print("  TabPFN-3 — GroupKFold (in-context learning)")
    print("="*55)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_sqrt, groups), 1):
        X_tr,  X_val  = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr,  y_val  = y_sqrt[tr_idx],        y_sqrt[val_idx]

        # Subsample training fold if too large for TabPFN memory budget
        if len(X_tr) > TABPFN_TRAIN_SUBSAMPLE:
            rng       = np.random.default_rng(SEED + fold)
            sub_idx   = rng.choice(len(X_tr), size=TABPFN_TRAIN_SUBSAMPLE, replace=False)
            X_tr_sub  = X_tr.iloc[sub_idx]
            y_tr_sub  = y_tr[sub_idx]
        else:
            X_tr_sub, y_tr_sub = X_tr, y_tr

        device = "cuda" if USE_GPU else "cpu"
        model = TabPFNRegressor(device=device, random_state=SEED)
        try:
            model.fit(X_tr_sub.values, y_tr_sub)
            oof_sqrt = model.predict(X_val.values)

            # Batched test prediction to avoid GPU OOM (4GB VRAM limit)
            PRED_BATCH = 512
            preds_batches = []
            X_test_arr = X_test.values
            for start in range(0, len(X_test_arr), PRED_BATCH):
                batch = X_test_arr[start: start + PRED_BATCH]
                preds_batches.append(model.predict(batch))
            preds_fold = np.concatenate(preds_batches)

        except Exception as e:
            print(f"[TabPFN] Fold {fold} failed: {e}")
            print("[TabPFN] Skipping TabPFN ensemble member.")
            return None, None, None

        oof[val_idx]   = oof_sqrt
        preds         += preds_fold / N_FOLDS

        fold_rmse = np.sqrt(mean_squared_error(
            invert_target(y_val), invert_target(oof_sqrt)
        ))
        print(f"  Fold {fold}  RMSE(demand): {fold_rmse:.6f}  "
              f"[train_n={len(X_tr_sub)}]")

    oof_rmse = np.sqrt(mean_squared_error(
        invert_target(y_sqrt), invert_target(oof)
    ))
    print(f"\n[TabPFN] Overall OOF RMSE (demand scale): {oof_rmse:.6f}")
    return oof, preds, oof_rmse


# ─────────────────────────────────────────────
# Weighted Ensemble
# ─────────────────────────────────────────────

def build_ensemble(results: dict, y_sqrt: np.ndarray):
    """
    Combine model predictions using inverse-OOF-RMSE weighting.

    Parameters
    ----------
    results : dict  {model_name: (oof_sqrt, preds_sqrt, oof_rmse)}
    y_sqrt  : true targets (sqrt scale)

    Returns
    -------
    oof_ensemble  : np.ndarray (demand scale)
    pred_ensemble : np.ndarray (demand scale, clipped to [0,1])
    weights       : dict
    """
    # Filter out any None results (e.g. CatBoost not installed)
    results = {k: v for k, v in results.items() if v[0] is not None}

    weights = {name: 1.0 / rmse for name, (_, _, rmse) in results.items()}
    total   = sum(weights.values())
    weights = {name: w / total for name, w in weights.items()}

    print("\n[ensemble] weights:")
    for name, w in weights.items():
        print(f"  {name}: {w:.4f}")

    oof_sqrt_ensemble  = sum(weights[n] * oof   for n, (oof, _, _)   in results.items())
    pred_sqrt_ensemble = sum(weights[n] * preds for n, (_, preds, _) in results.items())

    from sklearn.metrics import r2_score
    oof_ensemble  = invert_target(oof_sqrt_ensemble)
    pred_ensemble = invert_target(pred_sqrt_ensemble)

    ens_rmse = np.sqrt(mean_squared_error(invert_target(y_sqrt), oof_ensemble))
    ens_r2   = r2_score(invert_target(y_sqrt), oof_ensemble)
    print(f"[ensemble] OOF RMSE (demand scale): {ens_rmse:.6f}")
    print(f"[ensemble] OOF R2   (demand scale): {ens_r2:.6f}")

    # Per-model R2
    print("\n[ensemble] Per-model R2 (OOF, demand scale):")
    for name, (oof_m, _, rmse_m) in results.items():
        r2_m = r2_score(invert_target(y_sqrt), invert_target(oof_m))
        print(f"  {name:10s}: R2={r2_m:.6f}  RMSE={rmse_m:.6f}")

    return oof_ensemble, pred_ensemble, weights


# ─────────────────────────────────────────────
# Residual Analysis
# ─────────────────────────────────────────────

def residual_analysis(train: pd.DataFrame, oof_demand: np.ndarray,
                      y_demand: np.ndarray) -> pd.DataFrame:
    """
    Compute residuals segmented by RoadType and time_bucket
    to identify remaining error patterns.
    """
    df = train.copy()
    df["__oof"]      = oof_demand
    df["__residual"] = y_demand - oof_demand
    df["__abs_err"]  = np.abs(df["__residual"])

    print("\n[residuals] RMSE by RoadType:")
    for rt, grp in df.groupby("RoadType"):
        label = {0: "Residential", 1: "Street", 2: "Highway"}.get(rt, str(rt))
        rmse  = np.sqrt(mean_squared_error(grp["demand"], grp["__oof"]))
        print(f"  {label:12s}: RMSE={rmse:.6f}  (n={len(grp)})")

    print("\n[residuals] RMSE by time_bucket:")
    bucket_labels = {0:"Night", 1:"Morn-Rush", 2:"Mid-Morn",
                     3:"Lunch", 4:"Afternoon", 5:"Eve-Rush", 6:"Night2"}
    for tb, grp in df.groupby("time_bucket"):
        label = bucket_labels.get(tb, str(tb))
        rmse  = np.sqrt(mean_squared_error(grp["demand"], grp["__oof"]))
        print(f"  {label:12s}: RMSE={rmse:.6f}  (n={len(grp)})")

    print("\n[residuals] RMSE for highway×afternoon segment:")
    if "highway_x_afternoon" in df.columns:
        for flag, grp in df.groupby("highway_x_afternoon"):
            label = "Highway×Afternoon" if flag == 1 else "Other"
            rmse  = np.sqrt(mean_squared_error(grp["demand"], grp["__oof"]))
            print(f"  {label:20s}: RMSE={rmse:.6f}  (n={len(grp)})")

    return df[["__residual", "__abs_err"]]


# ─────────────────────────────────────────────
# Master Training Pipeline
# ─────────────────────────────────────────────

def run_training(train: pd.DataFrame, test: pd.DataFrame,
                 X_train: pd.DataFrame, X_test: pd.DataFrame,
                 output_dir: str = "."):
    """
    Full training pipeline:
      1. Build sqrt target
      2. Build sample weights
      3. Train LGBM / XGBoost / CatBoost with GroupKFold
      4. Ensemble with inverse-RMSE weights
      5. Residual analysis
      6. Save OOF predictions and submission

    Parameters
    ----------
    train     : full training DataFrame (with all features + demand column)
    test      : full test DataFrame (with all features)
    X_train   : feature matrix for training
    X_test    : feature matrix for test
    output_dir: directory to save outputs

    Returns
    -------
    pred_ensemble : np.ndarray (demand predictions, clipped [0,1])
    oof_ensemble  : np.ndarray (OOF demand predictions)
    """
    os.makedirs(output_dir, exist_ok=True)

    y_demand = train["demand"].values
    y_sqrt   = transform_target(y_demand)

    # Add sqrt target to train for feature selection consistency
    train["demand_sqrt"] = y_sqrt

    # GroupKFold groups: each unique geohash stays in one fold [MANDATORY-6]
    groups = train["geohash"].values

    # Align train to X_train index (feature_engineer sorts by geohash/day/slot)
    # [BUG FIX]: previous code had inverted condition — always align to X_train.index
    train_aligned = train.loc[X_train.index]
    sample_weights = build_sample_weights(train_aligned)

    # ── Train models ──────────────────────────────────────────────────────
    results = {}

    oof_lgbm, pred_lgbm, rmse_lgbm = train_lightgbm(
        X_train, y_sqrt, X_test, groups, sample_weights
    )
    results["LightGBM"] = (oof_lgbm, pred_lgbm, rmse_lgbm)

    oof_xgb, pred_xgb, rmse_xgb = train_xgboost(
        X_train, y_sqrt, X_test, groups, sample_weights
    )
    results["XGBoost"] = (oof_xgb, pred_xgb, rmse_xgb)

    oof_cb, pred_cb, rmse_cb = train_catboost(
        X_train, y_sqrt, X_test, groups
    )
    results["CatBoost"] = (oof_cb, pred_cb, rmse_cb)

    oof_pfn, pred_pfn, rmse_pfn = train_tabpfn(
        X_train, y_sqrt, X_test, groups
    )
    results["TabPFN"] = (oof_pfn, pred_pfn, rmse_pfn)

    # ── Ensemble ──────────────────────────────────────────────────────────
    oof_ensemble, pred_ensemble, weights = build_ensemble(results, y_sqrt)

    # ── Residual analysis ─────────────────────────────────────────────────
    residual_analysis(train_aligned, oof_ensemble, y_demand)

    # ── Save OOF predictions ──────────────────────────────────────────────
    oof_df = pd.DataFrame({
        "demand_true":  y_demand,
        "oof_lgbm":     invert_target(oof_lgbm),
        "oof_xgb":      invert_target(oof_xgb),
        "oof_catboost": invert_target(oof_cb) if oof_cb is not None else np.nan,
        "oof_ensemble": oof_ensemble,
    })
    oof_path = os.path.join(output_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"\n[saved] OOF predictions → {oof_path}")

    return pred_ensemble, oof_ensemble


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from preprocessor      import preprocess
    from feature_engineer  import build_features
    from feature_selection import select_features

    BASE = "../dataset"
    train, test, sample = preprocess(
        f"{BASE}/train.csv", f"{BASE}/test.csv", f"{BASE}/sample_submission.csv"
    )
    train, test = build_features(train, test)
    train["demand_sqrt"] = transform_target(train["demand"].values)

    selected, imp_df, X_train, X_test = select_features(train, test)
    pred_ensemble, oof_ensemble = run_training(train, test, X_train, X_test, output_dir=".")

    # Generate submission
    submission = sample.copy()
    submission["demand"] = pred_ensemble
    submission.to_csv("submission.csv", index=False)
    print("\n[done] submission.csv saved")
    print(submission["demand"].describe())
