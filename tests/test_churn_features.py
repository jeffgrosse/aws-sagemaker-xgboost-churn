import pytest

from churn_features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    clean_total_charges,
    encode_record,
    encode_target,
)

BASE_RECORD = {
    "SeniorCitizen": 0,
    "tenure": 5,
    "MonthlyCharges": 53.85,
    "TotalCharges": "271.10",
    "gender": "Female",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}


def test_feature_columns_count_matches_encoding_scheme():
    expected = len(NUMERIC_FEATURES) + sum(len(levels) - 1 for levels in CATEGORICAL_FEATURES.values())
    assert len(FEATURE_COLUMNS) == expected


def test_clean_total_charges_blank_with_zero_tenure_is_zero():
    assert clean_total_charges("", tenure=0) == 0.0
    assert clean_total_charges("  ", tenure=0) == 0.0


def test_clean_total_charges_blank_with_nonzero_tenure_raises():
    with pytest.raises(ValueError):
        clean_total_charges("", tenure=5)


def test_clean_total_charges_parses_numeric_string():
    assert clean_total_charges("108.15", tenure=3) == 108.15


def test_encode_target():
    assert encode_target("Yes") == 1
    assert encode_target("No") == 0


def test_encode_target_rejects_unknown_value():
    with pytest.raises(ValueError):
        encode_target("Maybe")


def test_encode_record_returns_vector_matching_feature_columns_length():
    vector = encode_record(BASE_RECORD)
    assert len(vector) == len(FEATURE_COLUMNS)
    assert all(isinstance(v, float) for v in vector)


def test_encode_record_numeric_prefix_matches_input():
    vector = encode_record(BASE_RECORD)
    assert vector[FEATURE_COLUMNS.index("tenure")] == 5.0
    assert vector[FEATURE_COLUMNS.index("MonthlyCharges")] == 53.85
    assert vector[FEATURE_COLUMNS.index("TotalCharges")] == 271.10


def test_encode_record_one_hot_reference_level_is_all_zeros():
    # gender's reference level is "Female" (first in CATEGORICAL_FEATURES) -
    # a Female record should have gender_Male == 0.
    vector = encode_record(BASE_RECORD)
    assert vector[FEATURE_COLUMNS.index("gender_Male")] == 0.0


def test_encode_record_one_hot_non_reference_level_is_one():
    record = dict(BASE_RECORD, gender="Male")
    vector = encode_record(record)
    assert vector[FEATURE_COLUMNS.index("gender_Male")] == 1.0


def test_encode_record_missing_field_raises_with_field_name():
    record = dict(BASE_RECORD)
    del record["Contract"]
    with pytest.raises(ValueError, match="Contract"):
        encode_record(record)


def test_encode_record_unrecognized_category_value_raises():
    record = dict(BASE_RECORD, InternetService="Satellite")
    with pytest.raises(ValueError, match="InternetService"):
        encode_record(record)


def test_encode_record_zero_tenure_blank_total_charges():
    record = dict(BASE_RECORD, tenure=0, TotalCharges="")
    vector = encode_record(record)
    assert vector[FEATURE_COLUMNS.index("TotalCharges")] == 0.0
