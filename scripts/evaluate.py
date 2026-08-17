#!/usr/bin/env python3
"""Evaluate the trained model against the held-out test set and a naive
majority-class baseline.

The test set (data/processed/test.csv) is produced once by
scripts/prepare_data.py's stratified split and never touched by
scripts/train.py - these rows were not used for training or for the
early-stopping validation channel, so the numbers here are a real
out-of-sample estimate, not training-set accuracy repeated back.

Writes docs/evaluation-metrics.json (committed - it's the receipt for the
numbers quoted in the README) and prints a summary table.
"""

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from churn_features import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402

TEST_CSV = REPO_ROOT / "data" / "processed" / "test.csv"
TRAIN_CSV = REPO_ROOT / "data" / "processed" / "train.csv"
LAST_RUN_PATH = REPO_ROOT / "data" / "last_training_run.json"
METRICS_OUT = REPO_ROOT / "docs" / "evaluation-metrics.json"


def majority_class(train_csv_path):
    """The baseline: whatever class was most common in the training split.
    For this dataset that's "No" (0) - roughly 73.5% of customers don't churn."""
    labels = pd.read_csv(train_csv_path, header=None)[0]
    return int(labels.mode()[0])


def compute_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_true, y_score), 4),
    }


def download_and_load_model(model_data_url):
    s3 = boto3.client("s3")
    bucket, key = model_data_url.replace("s3://", "", 1).split("/", 1)

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "model.tar.gz"
        s3.download_file(bucket, key, str(tar_path))
        with tarfile.open(tar_path) as tar:
            tar.extractall(tmp)

        booster = xgb.Booster()
        errors = []
        for candidate in Path(tmp).rglob("*"):
            if candidate.is_dir():
                continue
            try:
                booster.load_model(str(candidate))
                return booster
            except Exception as exc:  # noqa: BLE001 - trying multiple candidate files
                errors.append(f"{candidate.name}: {exc}")
        raise RuntimeError(
            "Could not load an XGBoost booster from any file in the model artifact. "
            f"Tried: {errors}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data-url", default=None, help="s3://.../model.tar.gz (defaults to data/last_training_run.json)")
    args = parser.parse_args()

    if not TEST_CSV.exists():
        sys.exit(f"{TEST_CSV} not found - run scripts/prepare_data.py first.")

    model_data_url = args.model_data_url
    if model_data_url is None:
        if not LAST_RUN_PATH.exists():
            sys.exit("No --model-data-url given and data/last_training_run.json not found - run scripts/train.py first.")
        model_data_url = json.loads(LAST_RUN_PATH.read_text())["model_data_url"]

    test_df = pd.read_csv(TEST_CSV)
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMN].values

    print(f"Loading model from {model_data_url}...")
    booster = download_and_load_model(model_data_url)

    model_scores = booster.predict(xgb.DMatrix(X_test))
    model_preds = (model_scores >= 0.5).astype(int)
    model_metrics = compute_metrics(y_test, model_preds, model_scores)

    baseline_class = majority_class(TRAIN_CSV)
    baseline_preds = [baseline_class] * len(y_test)
    baseline_scores = [float(baseline_class)] * len(y_test)
    baseline_metrics = compute_metrics(y_test, baseline_preds, baseline_scores)

    results = {
        "test_set_size": len(y_test),
        "test_set_churn_rate": round(float(y_test.mean()), 4),
        "model": model_metrics,
        "baseline": {"predicts": "No" if baseline_class == 0 else "Yes", **baseline_metrics},
    }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(results, indent=2))

    print(f"\nHeld-out test set: n={results['test_set_size']}, churn_rate={results['test_set_churn_rate']:.2%}\n")
    print(f"{'metric':<10} {'model':>8} {'baseline':>10}")
    for metric in ("accuracy", "precision", "recall", "auc"):
        print(f"{metric:<10} {model_metrics[metric]:>8} {baseline_metrics[metric]:>10}")
    print(f"\nWrote {METRICS_OUT}")


if __name__ == "__main__":
    main()
