"""
preprocessor.py
---------------
Data cleaning and preprocessing for Gridlock 2.0 demand prediction pipeline.
All imputation statistics are fitted on TRAIN only to prevent data leakage.

Fixes vs previous version:
  - geohash_enc: replaced slow np.where(le.classes_ == x) O(N_classes)
    with O(1) dict lookup
  - Vectorised merge-based imputation (no apply/axis=1)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

ROAD_TYPE_ORDER = ["Residential", "Street", "Highway"]
WEATHER_ORDER   = ["Sunny", "Foggy", "Rainy", "Snowy"]
TARGET_COL      = "demand"


# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────

def load_data(train_path: str, test_path: str, sample_path: str):
    """Load raw CSVs and return (train, test, sample) DataFrames."""
    train  = pd.read_csv(train_path)
    test   = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    print(f"[load]  train={train.shape}  test={test.shape}")
    return train, test, sample


# ─────────────────────────────────────────────
# 2. Basic Cleaning
# ─────────────────────────────────────────────

def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all string columns."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


# ─────────────────────────────────────────────
# 3. Vectorised Missing-Value Imputation
# ─────────────────────────────────────────────

def _fit_imputation_stats(train: pd.DataFrame) -> dict:
    """
    Compute all imputation lookup tables from train only.
    Note: Called BEFORE encoding, so Weather/RoadType are still raw strings.
    """
    return {
        "temp_median":         train.groupby(["geohash", "Weather"])["Temperature"].median(),
        "temp_global_median":  float(train["Temperature"].median()),
        "road_mode":           train.groupby("geohash")["RoadType"].agg(
                                   lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        "road_global_mode":    train["RoadType"].mode().iloc[0],
        "weather_mode":        train.groupby(["geohash", "day"])["Weather"].agg(
                                   lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        "weather_global_mode": train["Weather"].mode().iloc[0],
    }


def _apply_imputation(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply pre-fitted imputation stats via vectorised merge (no apply/axis=1)."""
    df = df.copy()

    # ── Temperature ──────────────────────────────────────────────────────
    if df["Temperature"].isnull().any():
        lut = stats["temp_median"].reset_index()
        lut.columns = ["geohash", "Weather", "_tf"]
        df = df.merge(lut, on=["geohash", "Weather"], how="left")
        m = df["Temperature"].isnull()
        df.loc[m, "Temperature"] = df.loc[m, "_tf"].fillna(stats["temp_global_median"])
        df.drop(columns=["_tf"], inplace=True)

    # ── RoadType ─────────────────────────────────────────────────────────
    if df["RoadType"].isnull().any():
        lut = stats["road_mode"].reset_index()
        lut.columns = ["geohash", "_rf"]
        df = df.merge(lut, on="geohash", how="left")
        m = df["RoadType"].isnull()
        df.loc[m, "RoadType"] = df.loc[m, "_rf"].fillna(stats["road_global_mode"])
        df.drop(columns=["_rf"], inplace=True)

    # ── Weather ───────────────────────────────────────────────────────────
    if df["Weather"].isnull().any():
        lut = stats["weather_mode"].reset_index()
        lut.columns = ["geohash", "day", "_wf"]
        df = df.merge(lut, on=["geohash", "day"], how="left")
        m = df["Weather"].isnull()
        df.loc[m, "Weather"] = df.loc[m, "_wf"].fillna(stats["weather_global_mode"])
        df.drop(columns=["_wf"], inplace=True)

    return df


def impute_missing(train: pd.DataFrame, test: pd.DataFrame):
    """Fit on train, apply to both. Returns (train, test)."""
    stats = _fit_imputation_stats(train)
    train = _apply_imputation(train, stats)
    test  = _apply_imputation(test,  stats)
    print(f"[impute] remaining nulls → train: {train.isnull().sum().sum()}  "
          f"test: {test.isnull().sum().sum()}")
    return train, test


# ─────────────────────────────────────────────
# 4. Encode Categoricals
# ─────────────────────────────────────────────

def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame):
    """
    Binary, ordinal, and label encoding — all fitted on train vocabulary only.
    Unseen test geohashes receive geohash_enc = -1.

    FIX: geohash_enc now uses O(1) dict lookup instead of O(N_classes)
    np.where scan per row.
    """
    for df in [train, test]:
        df["LargeVehicles"] = (df["LargeVehicles"] == "Allowed").astype(np.int8)
        df["Landmarks"]     = (df["Landmarks"]     == "Yes").astype(np.int8)

    road_map = {r: i for i, r in enumerate(ROAD_TYPE_ORDER)}
    train["RoadType"] = train["RoadType"].map(road_map).fillna(0).astype(np.int8)
    test["RoadType"]  = test["RoadType"].map(road_map).fillna(0).astype(np.int8)

    weather_map = {w: i for i, w in enumerate(WEATHER_ORDER)}
    train["Weather"] = train["Weather"].map(weather_map).fillna(0).astype(np.int8)
    test["Weather"]  = test["Weather"].map(weather_map).fillna(0).astype(np.int8)

    # ── Geohash label encoding — O(1) dict lookup [BUG FIX] ──────────────
    le = LabelEncoder()
    train["geohash_enc"] = le.fit_transform(train["geohash"]).astype(np.int32)
    # Build a dict for fast O(1) lookup; unseen → -1
    hash_to_int = {h: i for i, h in enumerate(le.classes_)}
    test["geohash_enc"] = test["geohash"].map(hash_to_int).fillna(-1).astype(np.int32)

    print("[encode] categorical encoding done")
    return train, test


# ─────────────────────────────────────────────
# 5. Master Preprocessor
# ─────────────────────────────────────────────

def preprocess(train_path: str, test_path: str, sample_path: str):
    """End-to-end preprocessing. Returns (train_df, test_df, sample_df)."""
    train, test, sample = load_data(train_path, test_path, sample_path)
    train = clean_basic(train)
    test  = clean_basic(test)
    train, test = impute_missing(train, test)
    train, test = encode_categoricals(train, test)
    print(f"[preprocess] complete — train={train.shape}  test={test.shape}")
    return train, test, sample


if __name__ == "__main__":
    BASE = "../dataset"
    train, test, sample = preprocess(
        f"{BASE}/train.csv", f"{BASE}/test.csv", f"{BASE}/sample_submission.csv"
    )
    print(train.head(3))
    print(f"Train nulls: {train.isnull().sum().sum()}")
    print(f"Test  nulls: {test.isnull().sum().sum()}")
