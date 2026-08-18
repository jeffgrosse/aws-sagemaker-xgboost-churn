from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from evaluate import compute_metrics, majority_class, resolve_deployed_threshold


def test_compute_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    metrics = compute_metrics(y_true, y_pred, y_score)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["auc"] == 1.0


def test_compute_metrics_constant_baseline_predicting_majority_class():
    # A majority-class baseline: always predicts 0, never catches a churner.
    y_true = [0, 0, 0, 1]
    y_pred = [0, 0, 0, 0]
    y_score = [0.0, 0.0, 0.0, 0.0]
    metrics = compute_metrics(y_true, y_pred, y_score)
    assert metrics["accuracy"] == 0.75
    assert metrics["recall"] == 0.0  # zero_division=0, not an exception
    assert metrics["precision"] == 0.0


def test_majority_class_picks_more_common_label(tmp_path):
    csv_path = tmp_path / "train.csv"
    # No header, label first column - matches the algorithm-mode CSV format
    # majority_class() actually reads.
    csv_path.write_text("0,1.0,2.0\n0,1.0,2.0\n0,1.0,2.0\n1,1.0,2.0\n")
    assert majority_class(csv_path) == 0


def test_majority_class_picks_churn_when_it_is_the_majority(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("1,1.0,2.0\n1,1.0,2.0\n0,1.0,2.0\n")
    assert majority_class(csv_path) == 1


def test_resolve_deployed_threshold_reads_real_deployed_value():
    cfn = MagicMock()
    cfn.describe_stacks.return_value = {
        "Stacks": [{"Parameters": [{"ParameterKey": "ChurnThreshold", "ParameterValue": "0.3"}]}]
    }
    assert resolve_deployed_threshold(cfn, "aws-sagemaker-xgboost-churn") == 0.3


def test_resolve_deployed_threshold_falls_back_when_stack_not_found():
    cfn = MagicMock()
    cfn.describe_stacks.side_effect = ClientError(
        {"Error": {"Code": "ValidationError", "Message": "Stack does not exist"}}, "DescribeStacks"
    )
    assert resolve_deployed_threshold(cfn, "aws-sagemaker-xgboost-churn") == 0.5


def test_resolve_deployed_threshold_falls_back_when_parameter_missing():
    # Stack exists but ChurnThreshold isn't in its Parameters for some
    # reason (e.g. template drift) - same safe fallback, not a KeyError.
    cfn = MagicMock()
    cfn.describe_stacks.return_value = {"Stacks": [{"Parameters": [{"ParameterKey": "Other", "ParameterValue": "x"}]}]}
    assert resolve_deployed_threshold(cfn, "aws-sagemaker-xgboost-churn") == 0.5
