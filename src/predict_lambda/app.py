"""API Gateway -> Lambda -> SageMaker Serverless Inference endpoint.

Takes a JSON customer record, encodes it with churn_features.encode_record
(the exact same function used at training time - see churn_features.py's
module docstring for why that matters), invokes the deployed endpoint with
a CSV row, and returns a churn probability.
"""

import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

from churn_features import REQUIRED_FIELDS, encode_record

logger = logging.getLogger()
logger.setLevel(logging.INFO)

runtime = boto3.client("sagemaker-runtime")

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
CHURN_THRESHOLD = float(os.environ.get("CHURN_THRESHOLD", "0.5"))

COLD_START_RETRY_DELAY_SECONDS = 2


def _response(status_code, body_dict):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body_dict),
    }


def _invoke_endpoint(csv_row):
    """SageMaker Serverless Inference can return a ModelError ("could not
    get a response from the endpoint") on the first request after a period
    of no traffic, while a fresh container is still starting - observed
    directly while building this (confirmed via the endpoint's own
    CloudWatch logs: the container hadn't even logged a request attempt for
    the failed call). One retry after a short delay is enough in practice -
    a second cold start back to back within the same request is unusual.
    Any other error (bad input already validated earlier, throttling,
    genuine service failure) is not retried here and propagates to the
    caller's generic 502 handling."""
    try:
        return runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType="text/csv", Body=csv_row)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ModelError":
            raise
        logger.warning("ModelError from %s (likely a Serverless Inference cold start) - retrying once", ENDPOINT_NAME)
        time.sleep(COLD_START_RETRY_DELAY_SECONDS)
        return runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType="text/csv", Body=csv_row)


def lambda_handler(event, context):
    try:
        raw_body = event.get("body") or "{}"
        customer = json.loads(raw_body)
    except (TypeError, json.JSONDecodeError):
        return _response(400, {"error": "Request body must be valid JSON."})

    if not isinstance(customer, dict):
        return _response(400, {"error": "Request body must be a JSON object."})

    try:
        feature_vector = encode_record(customer)
    except ValueError as exc:
        return _response(400, {"error": str(exc), "required_fields": REQUIRED_FIELDS})

    csv_row = ",".join(str(value) for value in feature_vector)

    try:
        endpoint_response = _invoke_endpoint(csv_row)
        prediction = endpoint_response["Body"].read().decode("utf-8").strip()
        churn_probability = float(prediction)
    except Exception:
        # Anything from here down is an infrastructure problem (cold-start
        # timeout, throttling, malformed endpoint response), not a caller
        # input problem - log the detail for CloudWatch, keep the response
        # generic so we're not leaking endpoint internals to a public API.
        logger.exception("Endpoint invocation failed for %s", ENDPOINT_NAME)
        return _response(502, {"error": "Prediction service is temporarily unavailable."})

    return _response(
        200,
        {
            "churn_probability": round(churn_probability, 4),
            "churn_label": "Yes" if churn_probability >= CHURN_THRESHOLD else "No",
        },
    )
