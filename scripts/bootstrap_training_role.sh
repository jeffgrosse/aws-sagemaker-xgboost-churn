#!/usr/bin/env bash
# One-time setup: creates the IAM role SageMaker training jobs assume. Not a
# CFN resource - training is script-driven (see docs/ARCHITECTURE.md), and
# this role only needs to exist once per account, well before the first
# `make train`. Idempotent: safe to re-run, skips creation if the role
# already exists.
#
# Deliberately a separate, narrowly-scoped role from the one template.yaml
# creates for the *hosting* side (SageMaker::Model's execution role) -
# training and hosting have different permission needs (write access to the
# default bucket vs. read-only on one artifact), and training's role has to
# exist before any CFN stack does, so it can't be the same resource anyway.
set -euo pipefail

ROLE_NAME="aws-sagemaker-xgboost-churn-training-role"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION="${AWS_DEFAULT_REGION:-$(aws configure get region || true)}"
REGION="${REGION:-us-east-1}"
BUCKET_ARN="arn:aws:s3:::sagemaker-${REGION}-${ACCOUNT_ID}"

if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  echo "Role ${ROLE_NAME} already exists - skipping creation."
else
  echo "Creating ${ROLE_NAME}..."
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --description "SageMaker execution role for aws-sagemaker-xgboost-churn training jobs" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }' >/dev/null

  aws iam put-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-name "default-bucket-and-logs" \
    --policy-document "{
      \"Version\": \"2012-10-17\",
      \"Statement\": [
        {
          \"Effect\": \"Allow\",
          \"Action\": [\"s3:GetObject\", \"s3:PutObject\"],
          \"Resource\": \"${BUCKET_ARN}/*\"
        },
        {
          \"Effect\": \"Allow\",
          \"Action\": \"s3:ListBucket\",
          \"Resource\": \"${BUCKET_ARN}\"
        },
        {
          \"Effect\": \"Allow\",
          \"Action\": [\"logs:CreateLogGroup\", \"logs:CreateLogStream\", \"logs:PutLogEvents\"],
          \"Resource\": \"arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/sagemaker/*\"
        }
      ]
    }" >/dev/null

  echo "Waiting for IAM role propagation..."
  sleep 10
fi

ROLE_ARN=$(aws iam get-role --role-name "${ROLE_NAME}" --query Role.Arn --output text)
echo "TRAINING_ROLE_ARN=${ROLE_ARN}"
