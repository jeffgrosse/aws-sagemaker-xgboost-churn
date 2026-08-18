#!/usr/bin/env python3
"""Kick off a SageMaker training job using the built-in XGBoost algorithm.

Not a notebook, not a standing resource: this uploads data/processed/{train,
validation}.csv to S3, starts a training job on a single ml.m5.large (billed
per second, terminates itself when done), blocks until it finishes, and
prints the resulting model artifact location plus the resolved container
image URI - both of which `make deploy` / `sam deploy --guided` need as
template.yaml parameters.

Requires scripts/bootstrap_training_role.sh to have been run once already.

Uses the SageMaker Python SDK's classic v2 API (sagemaker.estimator.Estimator,
sagemaker.image_uris.retrieve) - see docs/ARCHITECTURE.md#sdk-v3-gotcha for
why this is pinned rather than using whatever `pip install sagemaker` gives
you today.
"""

import argparse
import json
import sys
from pathlib import Path

import boto3
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
LAST_RUN_PATH = REPO_ROOT / "data" / "last_training_run.json"
S3_PREFIX = "aws-sagemaker-xgboost-churn"

DEFAULT_XGBOOST_VERSION = "1.7-1"
DEFAULT_INSTANCE_TYPE = "ml.m5.large"
DEFAULT_NUM_ROUND = 150
DEFAULT_EARLY_STOPPING_ROUNDS = 10


def resolve_scale_pos_weight(labels):
    """XGBoost's scale_pos_weight, computed from the training split's actual
    class balance (~26.5% churn here) rather than hardcoded. Weighting the
    minority (churn) class up trades some overall accuracy for better recall
    on churners - the class a churn model exists to catch. Pure function so
    it's testable without touching S3 or SageMaker."""
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0:
        raise ValueError("No positive (churn) examples in training data.")
    return round(negatives / positives, 4)


def build_hyperparameters(scale_pos_weight, num_round=DEFAULT_NUM_ROUND):
    """The exact hyperparameter dict passed to the built-in XGBoost
    algorithm. Pure function, no SDK/AWS objects, so tests can assert on it
    directly."""
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "num_round": num_round,
        "early_stopping_rounds": DEFAULT_EARLY_STOPPING_ROUNDS,
        "max_depth": 5,
        "eta": 0.2,
        "subsample": 0.8,
        "scale_pos_weight": scale_pos_weight,
    }


def read_training_labels(train_csv_path):
    # Algorithm-mode CSV: no header, label is column 0.
    return pd.read_csv(train_csv_path, header=None)[0].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN (see scripts/bootstrap_training_role.sh)")
    # Deliberately not left to default to boto3's ambient session region -
    # that reads your CLI profile's default, which has no reason to match
    # where you actually want to train and, on this project's first run,
    # silently pointed a real CreateTrainingJob call at the wrong region's
    # (also zero-quota) account limits. us-east-1 matches template.yaml /
    # samconfig.toml.example / scripts/bootstrap_training_role.sh's default -
    # override consistently across all three if you deploy elsewhere.
    parser.add_argument("--region", default="us-east-1", help="Must match the region scripts/bootstrap_training_role.sh's policy was scoped to")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--xgboost-version", default=DEFAULT_XGBOOST_VERSION)
    parser.add_argument("--num-round", type=int, default=DEFAULT_NUM_ROUND)
    args = parser.parse_args()

    train_csv = PROCESSED_DIR / "train.csv"
    val_csv = PROCESSED_DIR / "validation.csv"
    if not train_csv.exists() or not val_csv.exists():
        sys.exit(f"{train_csv} / {val_csv} not found - run scripts/prepare_data.py first.")

    # Imported here (not top of file) so `--help` and the pure functions
    # above work without the sagemaker SDK installed.
    from sagemaker import Session, image_uris
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput

    boto_session = boto3.Session(region_name=args.region)
    region = boto_session.region_name
    session = Session(boto_session=boto_session)
    bucket = session.default_bucket()

    train_s3 = session.upload_data(str(train_csv), bucket=bucket, key_prefix=f"{S3_PREFIX}/data/train")
    val_s3 = session.upload_data(str(val_csv), bucket=bucket, key_prefix=f"{S3_PREFIX}/data/validation")
    print(f"Uploaded training data to {train_s3}")
    print(f"Uploaded validation data to {val_s3}")

    scale_pos_weight = resolve_scale_pos_weight(read_training_labels(train_csv))
    hyperparameters = build_hyperparameters(scale_pos_weight, num_round=args.num_round)
    print(f"scale_pos_weight={scale_pos_weight} (from training split's actual class balance)")

    image_uri = image_uris.retrieve("xgboost", region, version=args.xgboost_version)
    print(f"Training image: {image_uri}")

    estimator = Estimator(
        image_uri=image_uri,
        role=args.role_arn,
        instance_count=1,
        instance_type=args.instance_type,
        output_path=f"s3://{bucket}/{S3_PREFIX}/output",
        sagemaker_session=session,
        base_job_name="churn-xgboost",
    )
    estimator.set_hyperparameters(**hyperparameters)

    estimator.fit(
        {
            "train": TrainingInput(train_s3, content_type="text/csv"),
            "validation": TrainingInput(val_s3, content_type="text/csv"),
        },
        wait=True,
        logs=True,
    )

    run_info = {
        "training_job_name": estimator.latest_training_job.name,
        "model_data_url": estimator.model_data,
        "image_uri": image_uri,
        "region": region,
        "xgboost_version": args.xgboost_version,
    }
    LAST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_PATH.write_text(json.dumps(run_info, indent=2))

    print("\nTraining job complete.")
    print(f"  Training job name : {run_info['training_job_name']}")
    print(f"  Model artifact    : {run_info['model_data_url']}")
    print(f"  Image URI         : {run_info['image_uri']}")
    print(f"\nSaved to {LAST_RUN_PATH} - `make deploy` reads this automatically.")


if __name__ == "__main__":
    main()
