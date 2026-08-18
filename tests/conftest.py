import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

ENDPOINT_NAME = "aws-sagemaker-xgboost-churn-test-endpoint"
API_KEY_VALUE = "test-api-key-do-not-use-in-production"


@pytest.fixture
def predict_app(monkeypatch):
    monkeypatch.setenv("ENDPOINT_NAME", ENDPOINT_NAME)
    monkeypatch.setenv("CHURN_THRESHOLD", "0.5")
    monkeypatch.setenv("API_KEY_VALUE", API_KEY_VALUE)

    import predict_lambda.app as module

    importlib.reload(module)
    return module
