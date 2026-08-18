#!/usr/bin/env python3
"""Smoke-test the deployed API: POST a sample customer, confirm a real
prediction comes back through the full chain (API Gateway -> Lambda ->
SageMaker Serverless endpoint -> response).

    python3 scripts/invoke_endpoint.py --stack-name aws-sagemaker-xgboost-churn --api-key YOUR_KEY
    python3 scripts/invoke_endpoint.py --api-url https://abc123.execute-api.us-east-1.amazonaws.com/predict --api-key YOUR_KEY

--api-key must match the ApiKeyValue the stack was deployed with (or set
PREDICT_API_KEY instead of passing it every time).
"""

import argparse
import json
import os
import sys
import urllib.request

import boto3

SAMPLE_CUSTOMER = {
    "SeniorCitizen": 0,
    "tenure": 2,
    "MonthlyCharges": 89.10,
    "TotalCharges": "178.20",
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


def resolve_api_url(stack_name, region):
    # Region passed explicitly rather than left to boto3's ambient default -
    # see the same fix in scripts/train.py and scripts/bootstrap_training_role.sh
    # for why relying on the CLI profile's default region is a real footgun
    # here (this exact command failed with "stack does not exist" against
    # the wrong region the first time, for that reason).
    cfn = boto3.client("cloudformation", region_name=region)
    outputs = cfn.describe_stacks(StackName=stack_name)["Stacks"][0]["Outputs"]
    for output in outputs:
        if output["OutputKey"] == "PredictApiUrl":
            return output["OutputValue"]
    sys.exit(f"Stack {stack_name} has no PredictApiUrl output.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--stack-name", default="aws-sagemaker-xgboost-churn")
    # Falls back to AWS_DEFAULT_REGION, then "us-east-1" - matching
    # scripts/train.py and scripts/bootstrap_training_role.sh's fallback
    # chain exactly (see docs/ARCHITECTURE.md#region-default-footgun).
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        help="Must match the region the stack was deployed to",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PREDICT_API_KEY"),
        help="Value of the ApiKeyValue template parameter the stack was deployed with; can also be set via PREDICT_API_KEY",
    )
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("Missing API key - pass --api-key or set PREDICT_API_KEY (the ApiKeyValue you deployed with).")

    api_url = args.api_url or resolve_api_url(args.stack_name, args.region)

    request = urllib.request.Request(
        api_url,
        data=json.dumps(SAMPLE_CUSTOMER).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Api-Key": args.api_key},
        method="POST",
    )

    print(f"POST {api_url}")
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))

    print(json.dumps(body, indent=2))

    if "churn_probability" not in body or "churn_label" not in body:
        sys.exit("Response missing churn_probability/churn_label - endpoint chain is broken.")

    print("\nEnd-to-end prediction confirmed.")


if __name__ == "__main__":
    main()
