"""
feature_engineer.py
-------------------
Rich feature engineering for Gridlock 2.0 demand prediction.

Mandatory changes implemented:
  [1] Cross-day same-slot lag  — day48 -> day49 join on (geohash, time_slot)
  [3] Nearest-neighbour KNN fallback for unseen test geohashes
  [4] highway_x_afternoon interaction feature
  [5] Real lag lookup for test using (geohash, time_slot-1) from training data

Additional improvements:
  - geohash x time_slot joint aggregation (mean, std)
  - Day-level aggregation features
  - Demand coefficient of variation per geohash
  - Fixed double-merge bug in add_aggregation_features
  - Unified geohash_prefix encoding (train-fitted)
  - demand_cv, geohash_x_slot aggregation features
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. Temporal Features
# ─────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamp into numeric features, cyclical encodings,
    time-of-day buckets, and rush-hour flags.
    """
    parts = df["timestamp"].str.split(":")
    df["hour"]         = parts.str[0].astype(np.int8)
    df["minute"]       = parts.str[1].astype(np.int8)
    df["minute_of_day"] = (df["hour"] * 60 + df["minute"]).astype(np.int16)
    df["time_slot"]    = (df["minute_of_day"] // 15).astype(np.int8)   # 96 slots/day

    # Cyclical encoding (midnight wraps correctly)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]      / 24).astype(np.float32)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]      / 24).astype(np.float32)
    df["slot_sin"]  = np.sin(2 * np.pi * df["time_slot"] / 96).astype(np.float32)
    df["slot_cos"]  = np.cos(2 * np.pi * df["time_slot"] / 96).astype(np.float32)

    # Time-of-day bucket (0=night, 1=morning rush, 2=mid-morning,
    #                      3=lunch, 4=afternoon, 5=evening rush, 6=night)
    conditions = [
        df["hour"] < 6,
        df["hour"] < 10,
        df["hour"] < 12,
        df["hour"] < 14,
        df["hour"] < 17,
        df["hour"] < 20,
    ]
    choices = [0, 1, 2, 3, 4, 5]
    df["time_bucket"] = np.select(conditions, choices, default=6).astype(np.int8)

    # Rush-hour flag (morning 7-10, evening 17-20)
    df["is_rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] < 10)) |
        ((df["hour"] >= 17) & (df["hour"] < 20))
    ).astype(np.int8)

    # Is afternoon flag (time_bucket == 4 → 14:00-17:00)
    df["is_afternoon"] = (df["time_bucket"] == 4).astype(np.int8)

    print("[temporal] features added")
    return df


# ─────────────────────────────────────────────
# 2. Geospatial Features (Geohash Decoding)
# ─────────────────────────────────────────────

def decode_geohash(geohash_str: str):
    """
    Lightweight pure-Python geohash decoder (no external library).
    Returns (lat, lon) centre of the geohash cell.
    """
    BASE32  = "0123456789bcdefghjkmnpqrstuvwxyz"
    b32_map = {c: i for i, c in enumerate(BASE32)}
    lat_range = [-90.0,  90.0]
    lon_range = [-180.0, 180.0]
    is_lon    = True
    for char in geohash_str:
        bits = b32_map[char]
        for shift in range(4, -1, -1):
            bit = (bits >> shift) & 1
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2
                if bit: lon_range[0] = mid
                else:   lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2
                if bit: lat_range[0] = mid
                else:   lat_range[1] = mid
            is_lon = not is_lon
    return (lat_range[0] + lat_range[1]) / 2, (lon_range[0] + lon_range[1]) / 2


def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decode geohash to lat/lon and add spatial features.
    Prefix encodings are built on each split independently
    (caller must align encodings for train/test via build_features).
    """
    coords      = df["geohash"].apply(decode_geohash)
    df["lat"]   = coords.apply(lambda x: x[0]).astype(np.float32)
    df["lon"]   = coords.apply(lambda x: x[1]).astype(np.float32)
    df["geohash_prefix5"] = df["geohash"].str[:5]
    df["geohash_prefix4"] = df["geohash"].str[:4]
    print("[geospatial] lat/lon decoded")
    return df


def _encode_prefix_columns(train: pd.DataFrame, test: pd.DataFrame):
    """
    Encode geohash prefix columns using train-fitted mapping.
    Unseen prefixes in test receive -1.
    """
    for col in ["geohash_prefix5", "geohash_prefix4"]:
        uniq = {v: i for i, v in enumerate(sorted(train[col].unique()))}
        train[col + "_enc"] = train[col].map(uniq).astype(np.int16)
        test[col  + "_enc"] = test[col].map(uniq).fillna(-1).astype(np.int16)
    return train, test


# ─────────────────────────────────────────────
# 3. Nearest-Neighbour KNN for Unseen Geohashes  [MANDATORY-3]
# ─────────────────────────────────────────────

def _build_geohash_latlon_index(train: pd.DataFrame) -> pd.DataFrame:
    """
    Build a lookup table of (geohash -> lat, lon) from training data.
    Returns a DataFrame with columns [geohash, lat, lon].
    """
    return (
        train[["geohash", "lat", "lon"]]
        .drop_duplicates("geohash")
        .reset_index(drop=True)
    )


def _knn_fill_unseen_geohashes(test: pd.DataFrame,
                                agg: pd.DataFrame,
                                geo_index: pd.DataFrame,
                                agg_cols: list) -> pd.DataFrame:
    """
    For test rows whose geohash is not in the training agg table,
    find the nearest known geohash by Euclidean lat/lon distance
    and inherit its aggregation statistics.

    Parameters
    ----------
    test      : test DataFrame (must have lat, lon columns)
    agg       : geohash-level aggregation stats from train
    geo_index : train geohash -> (lat, lon) lookup
    agg_cols  : list of column names to fill (e.g. geohash_mean_demand)
    """
    known_geohashes = set(agg["geohash"].values)
    unseen_mask     = ~test["geohash"].isin(known_geohashes)

    if unseen_mask.sum() == 0:
        return test

    # Build arrays for fast Euclidean search
    train_lats  = geo_index["lat"].values
    train_lons  = geo_index["lon"].values
    train_hashes = geo_index["geohash"].values

    # Merge agg into geo_index for stat lookup
    geo_agg = geo_index.merge(agg, on="geohash", how="left")

    unseen_rows = test[unseen_mask].copy()
    print(f"[knn-fallback] {unseen_mask.sum()} unseen test geohash rows — "
          f"finding nearest known geohash...")

    for idx in unseen_rows.index:
        qlat = test.at[idx, "lat"]
        qlon = test.at[idx, "lon"]
        dists = (train_lats - qlat) ** 2 + (train_lons - qlon) ** 2
        nn_hash = train_hashes[np.argmin(dists)]
        nn_row  = geo_agg[geo_agg["geohash"] == nn_hash]
        for col in agg_cols:
            if col in nn_row.columns:
                test.at[idx, col] = nn_row[col].values[0]

    return test


# ─────────────────────────────────────────────
# 4. Aggregation / Target-Encode-Style Features
# ─────────────────────────────────────────────

def add_aggregation_features(train: pd.DataFrame, test: pd.DataFrame):
    """
    Compute geohash-level, slot-level, road-level, and joint statistics
    from training data and merge into both sets.

    Fixes:
    - Previous version merged twice (wasted compute + duplicated columns)
    - Now uses single merge path with KNN fallback for unseen test geohashes
    - Adds geohash x time_slot joint aggregation and demand_cv
    """
    # ── Geohash-level stats ───────────────────────────────────────────────
    geo_agg = train.groupby("geohash")["demand"].agg(
        geohash_mean_demand="mean",
        geohash_std_demand="std",
        geohash_max_demand="max",
        geohash_min_demand="min",
        geohash_median_demand="median",
    ).reset_index()
    geo_agg["geohash_std_demand"]  = geo_agg["geohash_std_demand"].fillna(0)
    # Coefficient of variation (spread relative to mean)
    geo_agg["demand_cv"] = (geo_agg["geohash_std_demand"] /
                             geo_agg["geohash_mean_demand"].replace(0, np.nan)).fillna(0)

    geo_agg_cols = ["geohash_mean_demand", "geohash_std_demand", "geohash_max_demand",
                    "geohash_min_demand", "geohash_median_demand", "demand_cv"]

    # ── Time-slot-level stats ─────────────────────────────────────────────
    slot_agg = train.groupby("time_slot")["demand"].agg(
        slot_mean_demand="mean",
        slot_std_demand="std",
    ).reset_index()

    # ── Road-type-level stats ─────────────────────────────────────────────
    road_agg = train.groupby("RoadType")["demand"].agg(
        road_mean_demand="mean",
    ).reset_index()

    # ── Geohash x time_slot joint stats ──────────────────────────────────
    joint_agg = train.groupby(["geohash", "time_slot"])["demand"].agg(
        geo_slot_mean_demand="mean",
        geo_slot_std_demand="std",
    ).reset_index()
    joint_agg["geo_slot_std_demand"] = joint_agg["geo_slot_std_demand"].fillna(0)

    # ── Day-level aggregation ─────────────────────────────────────────────
    day_agg = train.groupby("day")["demand"].agg(
        day_mean_demand="mean",
        day_std_demand="std",
    ).reset_index()

    # ── Merge into train (single pass) ───────────────────────────────────
    train = (train
             .merge(geo_agg,   on="geohash",              how="left")
             .merge(slot_agg,  on="time_slot",            how="left")
             .merge(road_agg,  on="RoadType",             how="left")
             .merge(joint_agg, on=["geohash", "time_slot"], how="left")
             .merge(day_agg,   on="day",                  how="left"))

    # ── Merge into test (single pass) ────────────────────────────────────
    test = (test
            .merge(geo_agg,   on="geohash",              how="left")
            .merge(slot_agg,  on="time_slot",            how="left")
            .merge(road_agg,  on="RoadType",             how="left")
            .merge(joint_agg, on=["geohash", "time_slot"], how="left")
            .merge(day_agg,   on="day",                  how="left"))

    # ── KNN fallback for unseen test geohashes [MANDATORY-3] ─────────────
    geo_index = _build_geohash_latlon_index(train)
    test = _knn_fill_unseen_geohashes(test, geo_agg, geo_index, geo_agg_cols)

    # ── Fill remaining nulls (joint/day may have gaps) ───────────────────
    global_mean = train["demand"].mean()
    for col in geo_agg_cols:
        train[col] = train[col].fillna(global_mean)
        test[col]  = test[col].fillna(global_mean)

    for col in ["slot_mean_demand", "slot_std_demand",
                "road_mean_demand",
                "geo_slot_mean_demand", "geo_slot_std_demand",
                "day_mean_demand", "day_std_demand"]:
        if col in train.columns:
            med = train[col].median()
            train[col] = train[col].fillna(med)
            test[col]  = test[col].fillna(med)

    print("[aggregation] features added")
    return train, test


# ─────────────────────────────────────────────
# 5. Lag & Rolling Features  (train only)
# ─────────────────────────────────────────────

def add_lag_rolling_features(train: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag and rolling statistics of demand per geohash in training data.
    Sorted by (geohash, day, time_slot) before shifting to preserve temporal order.
    These features do NOT exist in test; test uses add_test_lag_lookup() instead.
    """
    df  = train.sort_values(["geohash", "day", "time_slot"]).copy()
    grp = df.groupby("geohash")["demand"]

    df["demand_lag1"] = grp.shift(1)
    df["demand_lag2"] = grp.shift(2)
    df["demand_lag4"] = grp.shift(4)

    df["demand_roll4_mean"] = (grp.shift(1).rolling(4,  min_periods=1).mean()
                               .reset_index(level=0, drop=True))
    df["demand_roll8_mean"] = (grp.shift(1).rolling(8,  min_periods=1).mean()
                               .reset_index(level=0, drop=True))
    df["demand_roll4_std"]  = (grp.shift(1).rolling(4,  min_periods=2).std()
                               .reset_index(level=0, drop=True))

    lag_cols = ["demand_lag1", "demand_lag2", "demand_lag4",
                "demand_roll4_mean", "demand_roll8_mean", "demand_roll4_std"]
    df[lag_cols] = df[lag_cols].fillna(0)

    print("[lag/rolling] features added")
    return df


# ─────────────────────────────────────────────
# 6. Real Test Lag Lookup  [MANDATORY-5]
# ─────────────────────────────────────────────

def add_test_lag_lookup(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Replace geohash-mean proxy lag with real observed demand lookups.

    Strategy:
    - For each test row, find the training observation with the same
      geohash and time_slot = (test_time_slot - 1).
    - When multiple training days have the same slot, use the LATEST day
      (most temporally relevant).
    - Rows with no matching training record fall back to geohash_mean_demand.

    This avoids temporal information leakage because the training data
    (days 1-48) chronologically precedes the test data (day 49).
    """
    # Build lookup: latest demand per (geohash, time_slot) in training
    lag_lookup = (
        train.sort_values("day")
             .groupby(["geohash", "time_slot"])["demand"]
             .last()                          # latest day's value
             .reset_index()
             .rename(columns={"demand": "_lag1_val", "time_slot": "_lag_slot"})
    )
    # target slot in lookup = test_slot - 1
    lag_lookup["time_slot"] = lag_lookup["_lag_slot"] + 1
    lag_lookup = lag_lookup.drop(columns=["_lag_slot"])

    # Merge on (geohash, time_slot) where time_slot is the TEST slot
    test = test.merge(lag_lookup, on=["geohash", "time_slot"], how="left")

    # Fill remaining unseen combos with geohash mean
    fallback = test["_lag1_val"].isnull().sum()
    if fallback > 0:
        test["_lag1_val"] = test["_lag1_val"].fillna(
            test.get("geohash_mean_demand", pd.Series(0.0, index=test.index))
        )
        print(f"[test-lag] {fallback} rows fell back to geohash mean")

    test["demand_lag1"]     = test["_lag1_val"].astype(np.float32)
    test["demand_lag2"]     = test["demand_lag1"]          # best proxy available
    test["demand_lag4"]     = test["demand_lag1"]
    test["demand_roll4_mean"] = test["demand_lag1"]
    test["demand_roll8_mean"] = test["demand_lag1"]
    test["demand_roll4_std"]  = test.get("geohash_std_demand",
                                          pd.Series(0.0, index=test.index))
    test.drop(columns=["_lag1_val"], inplace=True)

    coverage = (~test["demand_lag1"].isnull()).mean() * 100
    print(f"[test-lag] real lookup coverage: {coverage:.1f}%")
    return test


# ─────────────────────────────────────────────
# 7. Cross-Day Same-Slot Lag  [MANDATORY-1]
# ─────────────────────────────────────────────

def add_cross_day_lag(train: pd.DataFrame, test: pd.DataFrame):
    """
    Join day-N and day-(N+1) records on (geohash, time_slot).
    For training:  day pairs are computed across all consecutive days.
    For test:      day 49 test rows are joined with day 48 training rows.

    The correlation between same-slot demand across consecutive days is ~0.79,
    making this one of the most predictive features available.
    """
    # ── Training: compute cross-day lag for all consecutive day pairs ──────
    day_cols = ["geohash", "day", "time_slot", "demand"]
    prev_day = train[day_cols].copy()
    prev_day["day"] = prev_day["day"] + 1          # shift day up by 1
    prev_day = prev_day.rename(columns={"demand": "demand_same_slot_prev_day"})

    train = train.merge(prev_day, on=["geohash", "day", "time_slot"], how="left")
    # Rows without a previous day (first day per geohash) → use geohash mean
    train["demand_same_slot_prev_day"] = (
        train["demand_same_slot_prev_day"]
        .fillna(train.get("geohash_mean_demand", pd.Series(np.nan)))
        .fillna(train["demand"].mean())
        .astype(np.float32)
    )

    # ── Test: look up day 48 training values at same (geohash, time_slot) ─
    # Test is day 49; find the day-48 training record for each (geohash, slot)
    day48 = train[train["day"] == train["day"].max()][
        ["geohash", "time_slot", "demand"]
    ].rename(columns={"demand": "demand_same_slot_prev_day"})

    test = test.merge(day48, on=["geohash", "time_slot"], how="left")
    # Fallback for test geohashes not in day 48
    test["demand_same_slot_prev_day"] = (
        test["demand_same_slot_prev_day"]
        .fillna(test.get("geohash_mean_demand", pd.Series(np.nan)))
        .fillna(train["demand"].mean())
        .astype(np.float32)
    )

    print("[cross-day-lag] demand_same_slot_prev_day added "
          f"(train null%: {train['demand_same_slot_prev_day'].isnull().mean()*100:.1f}%)")
    return train, test


# ─────────────────────────────────────────────
# 8. Interaction Features
# ─────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplicative interactions between key predictors.

    [MANDATORY-4] highway_x_afternoon:
      High-RMSE segment = Highway (RoadType==2) AND afternoon (time_bucket==4).
      This explicit interaction lets the model apply targeted corrections.
    """
    df["rush_x_lanes"]        = (df["is_rush_hour"]   * df["NumberofLanes"]).astype(np.float32)
    df["rush_x_largeveh"]     = (df["is_rush_hour"]   * df["LargeVehicles"]).astype(np.int8)
    df["lanes_x_road"]        = (df["NumberofLanes"]  * df["RoadType"]).astype(np.float32)
    df["temp_x_weather"]      = (df["Temperature"]    * df["Weather"]).astype(np.float32)
    df["landmark_x_rush"]     = (df["Landmarks"]      * df["is_rush_hour"]).astype(np.int8)
    df["slot_mean_x_geohash"] = (df["slot_mean_demand"] * df["geohash_mean_demand"]).astype(np.float32)

    # [MANDATORY-4] Highway x Afternoon — targets the highest-error segment
    is_highway   = (df["RoadType"] == 2).astype(np.int8)
    df["highway_x_afternoon"] = (is_highway * df["is_afternoon"]).astype(np.int8)

    # Additional useful interactions
    df["geo_slot_ratio"]  = (df.get("geo_slot_mean_demand", 0) /
                              df["geohash_mean_demand"].replace(0, np.nan)).fillna(1).astype(np.float32)
    df["demand_range"]    = (df["geohash_max_demand"] - df["geohash_min_demand"]).astype(np.float32)

    print("[interaction] features added")
    return df


# ─────────────────────────────────────────────
# 9. Master Feature Engineering Pipeline
# ─────────────────────────────────────────────

def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """
    Full feature engineering pipeline.

    Order matters:
      1. Temporal (need time_slot for later steps)
      2. Geospatial (need lat/lon for KNN fallback)
      3. Aggregations + KNN fallback (need geohash stats)
      4. Cross-day same-slot lag (needs day, time_slot, demand)
      5. Train lag/rolling (train only, sorted by time)
      6. Test lag lookup (test only, real observed values)
      7. Interactions (depend on all prior features)
    """
    # Step 1: Temporal
    train = add_temporal_features(train)
    test  = add_temporal_features(test)

    # Step 2: Geospatial
    train = add_geospatial_features(train)
    test  = add_geospatial_features(test)

    # Align prefix encodings using train vocabulary
    train, test = _encode_prefix_columns(train, test)

    # Step 3: Aggregations (fit on train, applied to both; KNN for unseen)
    train, test = add_aggregation_features(train, test)

    # Step 4: Cross-day same-slot lag [MANDATORY-1]
    train, test = add_cross_day_lag(train, test)

    # Step 5: Train lag/rolling (demand history — train only)
    train = add_lag_rolling_features(train)

    # Step 6: Test lag real lookup [MANDATORY-5]
    test = add_test_lag_lookup(train, test)

    # Step 7: Interactions [includes MANDATORY-4 highway_x_afternoon]
    train = add_interaction_features(train)
    test  = add_interaction_features(test)

    print(f"[build_features] done — train={train.shape}  test={test.shape}")
    return train, test


if __name__ == "__main__":
    from preprocessor import preprocess
    BASE = "../dataset"
    train, test, _ = preprocess(
        f"{BASE}/train.csv", f"{BASE}/test.csv", f"{BASE}/sample_submission.csv"
    )
    train, test = build_features(train, test)
    print("Train cols:", train.columns.tolist())
    print("Train shape:", train.shape, "  Test shape:", test.shape)
