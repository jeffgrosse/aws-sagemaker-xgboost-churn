import pytest

from train import build_hyperparameters, resolve_scale_pos_weight


def test_resolve_scale_pos_weight_matches_class_ratio():
    # 3 negatives, 1 positive -> weight the positive class 3x.
    labels = [0, 0, 0, 1]
    assert resolve_scale_pos_weight(labels) == 3.0


def test_resolve_scale_pos_weight_balanced_classes_is_one():
    labels = [0, 1, 0, 1]
    assert resolve_scale_pos_weight(labels) == 1.0


def test_resolve_scale_pos_weight_no_positives_raises():
    with pytest.raises(ValueError):
        resolve_scale_pos_weight([0, 0, 0])


def test_build_hyperparameters_is_binary_logistic_with_auc_eval():
    params = build_hyperparameters(scale_pos_weight=2.5)
    assert params["objective"] == "binary:logistic"
    assert params["eval_metric"] == "auc"
    assert params["scale_pos_weight"] == 2.5


def test_build_hyperparameters_num_round_is_overridable():
    params = build_hyperparameters(scale_pos_weight=1.0, num_round=50)
    assert params["num_round"] == 50
