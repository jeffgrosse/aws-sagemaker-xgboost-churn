import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

ENDPOINT_NAME = "aws-sagemaker-xgboost-churn-test-endpoint"


@pytest.fixture
def predict_app(monkeypatch):
    monkeypatch.setenv("ENDPOINT_NAME", ENDPOINT_NAME)
    monkeypatch.setenv("CHURN_THRESHOLD", "0.5")

    import predict_lambda.app as module

    importlib.reload(module)
    return module
