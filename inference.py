"""
inference.py
------------
Standalone inference script for Gridlock 2.0 demand prediction.

Runs the full pipeline end-to-end:
  1. Preprocess
  2. Feature engineering
  3. Feature selection
  4. Train (with GroupKFold, sqrt target, sample weights)
  5. Generate submission.csv

Usage:
  python inference.py                          # uses default dataset paths
  python inference.py --train ../dataset/train.csv --test ../dataset/test.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from preprocessor      import preprocess
from feature_engineer  import build_features
from feature_selection import select_features
from train             import transform_target, invert_target, run_training

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DEFAULT_TRAIN_PATH  = "../dataset/train.csv"
DEFAULT_TEST_PATH   = "../dataset/test.csv"
DEFAULT_SAMPLE_PATH = "../dataset/sample_submission.csv"
DEFAULT_OUTPUT_DIR  = "."


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def run_inference(train_path: str  = DEFAULT_TRAIN_PATH,
                  test_path: str   = DEFAULT_TEST_PATH,
                  sample_path: str = DEFAULT_SAMPLE_PATH,
                  output_dir: str  = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    """
    Full end-to-end pipeline. Returns submission DataFrame.

    Steps
    -----
    1. Preprocess train + test
    2. Build features (all mandatory + additional improvements)
    3. Select features (using sqrt target for importance scoring)
    4. Train ensemble (LightGBM + XGBoost + CatBoost) with GroupKFold
    5. Invert sqrt transform on predictions
    6. Clip to [0, 1] and save submission.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  GRIDLOCK 2.0 — DEMAND PREDICTION INFERENCE PIPELINE")
    print("="*60 + "\n")

    # ── Step 1: Preprocess ────────────────────────────────────────────────
    print(">>> Step 1: Preprocessing")
    train, test, sample = preprocess(train_path, test_path, sample_path)

    # ── Step 2: Feature Engineering ───────────────────────────────────────
    print("\n>>> Step 2: Feature Engineering")
    train, test = build_features(train, test)

    # ── Step 3: Feature Selection ─────────────────────────────────────────
    print("\n>>> Step 3: Feature Selection")
    # Compute sqrt target so feature_selection can use it
    train["demand_sqrt"] = transform_target(train["demand"].values)
    selected, importance_df, X_train, X_test = select_features(train, test)

    print(f"\nFeature importance (top 20):\n"
          f"{importance_df.head(20).to_string(index=False)}\n")

    # ── Step 4: Training ──────────────────────────────────────────────────
    print("\n>>> Step 4: Training Ensemble")
    pred_ensemble, oof_ensemble = run_training(
        train, test, X_train, X_test, output_dir=output_dir
    )

    # ── Step 5: Build Submission ──────────────────────────────────────────
    # NOTE: Feature engineering merges (cross-day lag, spatial clusters, etc.)
    # can change test row count. We build submission directly from the
    # processed test Index column rather than the sample template.
    print("\n>>> Step 5: Building Submission")

    # Get Index from the processed test DataFrame (source of truth for row IDs)
    test_index_col = None
    for col in ["Index", "index", "id", "ID"]:
        if col in test.columns:
            test_index_col = col
            break

    if test_index_col is not None and len(pred_ensemble) == len(test):
        submission = pd.DataFrame({
            test_index_col: test[test_index_col].values,
            "demand":        pred_ensemble,
        })
    elif len(pred_ensemble) == len(sample):
        # Predictions match sample template exactly
        submission = sample.copy()
        submission["demand"] = pred_ensemble
    else:
        # Fallback: build from test length
        print(f"[warn] pred length={len(pred_ensemble)}, sample length={len(sample)}, "
              f"test length={len(test)}. Using test-length index.")
        submission = pd.DataFrame({"demand": pred_ensemble})

    # Sanity checks
    assert submission["demand"].isnull().sum() == 0, "NaN in submission!"
    assert (submission["demand"] >= 0).all(),  "Negative demand in submission!"
    assert (submission["demand"] <= 1).all(),  "Demand > 1 in submission!"

    sub_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(sub_path, index=False)

    print(f"\n[done] Submission saved -> {sub_path}  (rows={len(submission)})")
    print(f"Prediction stats:\n{submission['demand'].describe().round(6)}")
    print("\n" + "="*60)
    return submission


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gridlock 2.0 Inference Pipeline")
    parser.add_argument("--train",  default=DEFAULT_TRAIN_PATH,  help="Path to train.csv")
    parser.add_argument("--test",   default=DEFAULT_TEST_PATH,   help="Path to test.csv")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE_PATH, help="Path to sample_submission.csv")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR,  help="Output directory for submission.csv")
    args = parser.parse_args()

    run_inference(
        train_path  = args.train,
        test_path   = args.test,
        sample_path = args.sample,
        output_dir  = args.outdir,
    )
