import json
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from churn_features import FEATURE_COLUMNS
from conftest import API_KEY_VALUE

VALID_CUSTOMER = {
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


def _event(body_dict, api_key=API_KEY_VALUE):
    event = {"body": json.dumps(body_dict)}
    if api_key is not None:
        event["headers"] = {"x-api-key": api_key}
    return event


def _mock_invoke_endpoint(predict_app, probability_string):
    fake_body = MagicMock()
    fake_body.read.return_value = probability_string.encode("utf-8")
    predict_app.runtime = MagicMock()
    predict_app.runtime.invoke_endpoint.return_value = {"Body": fake_body}
    return predict_app.runtime


def test_high_probability_returns_yes_label(predict_app):
    _mock_invoke_endpoint(predict_app, "0.7321")
    response = predict_app.lambda_handler(_event(VALID_CUSTOMER), None)
    body = json.loads(response["body"])

    assert response["statusCode"] == 200
    assert body["churn_probability"] == 0.7321
    assert body["churn_label"] == "Yes"


def test_low_probability_returns_no_label(predict_app):
    _mock_invoke_endpoint(predict_app, "0.1")
    response = predict_app.lambda_handler(_event(VALID_CUSTOMER), None)
    body = json.loads(response["body"])

    assert body["churn_label"] == "No"


def test_invokes_endpoint_with_csv_content_type_and_configured_endpoint_name(predict_app):
    runtime = _mock_invoke_endpoint(predict_app, "0.5")
    predict_app.lambda_handler(_event(VALID_CUSTOMER), None)

    call_kwargs = runtime.invoke_endpoint.call_args.kwargs
    assert call_kwargs["EndpointName"] == "aws-sagemaker-xgboost-churn-test-endpoint"
    assert call_kwargs["ContentType"] == "text/csv"

    csv_values = call_kwargs["Body"].split(",")
    assert len(csv_values) == len(FEATURE_COLUMNS)
    # Confirms the request Lambda sends the endpoint is encoded via the same
    # churn_features.encode_record path used at training time - this is the
    # test that would catch train/serve encoding drift.
    assert float(csv_values[FEATURE_COLUMNS.index("tenure")]) == 2.0


def test_missing_field_returns_400_without_calling_endpoint(predict_app):
    runtime = _mock_invoke_endpoint(predict_app, "0.5")
    incomplete = dict(VALID_CUSTOMER)
    del incomplete["Contract"]

    response = predict_app.lambda_handler(_event(incomplete), None)
    body = json.loads(response["body"])

    assert response["statusCode"] == 400
    assert "Contract" in body["error"]
    runtime.invoke_endpoint.assert_not_called()


def test_invalid_json_body_returns_400(predict_app):
    response = predict_app.lambda_handler({"body": "{not json", "headers": {"x-api-key": API_KEY_VALUE}}, None)
    assert response["statusCode"] == 400


def test_missing_api_key_returns_401_without_calling_endpoint(predict_app):
    runtime = _mock_invoke_endpoint(predict_app, "0.5")
    response = predict_app.lambda_handler(_event(VALID_CUSTOMER, api_key=None), None)

    assert response["statusCode"] == 401
    runtime.invoke_endpoint.assert_not_called()


def test_wrong_api_key_returns_401_without_calling_endpoint(predict_app):
    runtime = _mock_invoke_endpoint(predict_app, "0.5")
    response = predict_app.lambda_handler(_event(VALID_CUSTOMER, api_key="wrong-key"), None)

    assert response["statusCode"] == 401
    runtime.invoke_endpoint.assert_not_called()


def test_missing_headers_key_entirely_returns_401(predict_app):
    # No "headers" key at all in the event (vs. headers present but without
    # x-api-key) - a slightly different shape, same rejection.
    response = predict_app.lambda_handler({"body": json.dumps(VALID_CUSTOMER)}, None)
    assert response["statusCode"] == 401


def _model_error():
    return ClientError(
        {"Error": {"Code": "ModelError", "Message": "Received server error (0) from model"}},
        "InvokeEndpoint",
    )


def test_model_error_retries_once_and_succeeds(predict_app, monkeypatch):
    # First call raises the exact error observed from a real Serverless
    # Inference cold start; second call succeeds.
    fake_body = MagicMock()
    fake_body.read.return_value = b"0.6"
    predict_app.runtime = MagicMock()
    predict_app.runtime.invoke_endpoint.side_effect = [_model_error(), {"Body": fake_body}]
    monkeypatch.setattr(predict_app.time, "sleep", lambda seconds: None)

    response = predict_app.lambda_handler(_event(VALID_CUSTOMER), None)
    body = json.loads(response["body"])

    assert response["statusCode"] == 200
    assert body["churn_probability"] == 0.6
    assert predict_app.runtime.invoke_endpoint.call_count == 2


def test_model_error_twice_returns_502(predict_app, monkeypatch):
    predict_app.runtime = MagicMock()
    predict_app.runtime.invoke_endpoint.side_effect = [_model_error(), _model_error()]
    monkeypatch.setattr(predict_app.time, "sleep", lambda seconds: None)

    response = predict_app.lambda_handler(_event(VALID_CUSTOMER), None)

    assert response["statusCode"] == 502
    assert predict_app.runtime.invoke_endpoint.call_count == 2


def test_non_model_error_client_error_is_not_retried(predict_app):
    throttling_error = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "InvokeEndpoint",
    )
    predict_app.runtime = MagicMock()
    predict_app.runtime.invoke_endpoint.side_effect = throttling_error

    response = predict_app.lambda_handler(_event(VALID_CUSTOMER), None)

    assert response["statusCode"] == 502
    assert predict_app.runtime.invoke_endpoint.call_count == 1


def test_endpoint_invocation_failure_returns_502_without_leaking_detail(predict_app):
    predict_app.runtime = MagicMock()
    predict_app.runtime.invoke_endpoint.side_effect = RuntimeError("some internal boto3 detail")

    response = predict_app.lambda_handler(_event(VALID_CUSTOMER), None)
    body = json.loads(response["body"])

    assert response["statusCode"] == 502
    assert "some internal boto3 detail" not in body["error"]
