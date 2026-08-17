#!/usr/bin/env python3
"""Clean, encode, and split the Telco churn dataset for SageMaker training.

Run after scripts/download_data.sh. Writes:
  data/processed/train.csv       - 70%, no header, label first column
                                    (SageMaker built-in XGBoost's required format)
  data/processed/validation.csv  - 15%, same format
  data/processed/test.csv        - 15%, held out entirely from training/tuning;
                                    kept with a header for evaluate.py

The 70/15/15 stratified split uses a fixed random_state so re-running this
script reproduces the exact same split - "held-out test set" only means
something if it's the same set every time.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from churn_features import FEATURE_COLUMNS, TARGET_COLUMN, encode_record, encode_target

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "Telco-Customer-Churn.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RANDOM_STATE = 42


def load_and_encode(raw_path):
    df = pd.read_csv(raw_path)
    y = df[TARGET_COLUMN].map(encode_target)
    records = df.drop(columns=["customerID", TARGET_COLUMN]).to_dict("records")
    X = pd.DataFrame([encode_record(record) for record in records], columns=FEATURE_COLUMNS)
    return X, y


def split(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def write_algorithm_csv(path, X, y):
    """No header, label first column - the format SageMaker's built-in
    XGBoost algorithm requires for its train/validation channels."""
    out = pd.concat([y.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    out.to_csv(path, header=False, index=False)


def write_labeled_csv(path, X, y):
    """Same layout, with a header - test.csv is read back by evaluate.py,
    not uploaded to SageMaker, so a header is just easier to inspect."""
    out = pd.concat([y.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    out.columns = [TARGET_COLUMN] + FEATURE_COLUMNS
    out.to_csv(path, index=False)


def main():
    if not RAW_PATH.exists():
        sys.exit(f"{RAW_PATH} not found - run scripts/download_data.sh first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_and_encode(RAW_PATH)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split(X, y)

    write_algorithm_csv(OUT_DIR / "train.csv", X_train, y_train)
    write_algorithm_csv(OUT_DIR / "validation.csv", X_val, y_val)
    write_labeled_csv(OUT_DIR / "test.csv", X_test, y_test)

    for name, y_split in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        print(f"{name:<11} n={len(y_split):>5}  churn_rate={y_split.mean():.3%}")


if __name__ == "__main__":
    main()
