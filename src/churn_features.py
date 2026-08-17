"""Feature encoding for the Telco churn model.

Deliberately dependency-free (stdlib only): this module is imported both by
the pandas-based local pipeline (scripts/prepare_data.py, scripts/evaluate.py)
and by the predict Lambda (src/predict_lambda/app.py), which has no pandas
layer and shouldn't need one. `encode_record` works on a plain dict so both
callers share one code path - the whole point is that training-time encoding
and serving-time encoding can never drift apart, because they're the same
function.

XGBoost's built-in algorithm (used via SageMaker's managed container, not a
custom script) takes CSV input with no header and the label in the first
column. It has no notion of categorical columns of its own, so one-hot
encoding happens here, not inside the algorithm.
"""

TARGET_COLUMN = "Churn"

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# Column -> ordered category levels. The first level in each list is the
# reference category (dropped, like pandas' get_dummies(drop_first=True)) -
# not encoding it avoids the exact collinearity a one-hot-per-level scheme
# would introduce. Order matters: it fixes FEATURE_COLUMNS below, which is
# the one true column order shared by training data, evaluation, and every
# single-row inference request.
CATEGORICAL_FEATURES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

FEATURE_COLUMNS = NUMERIC_FEATURES + [
    f"{column}_{level}"
    for column, levels in CATEGORICAL_FEATURES.items()
    for level in levels[1:]
]

REQUIRED_FIELDS = NUMERIC_FEATURES + list(CATEGORICAL_FEATURES.keys())


def clean_total_charges(raw, tenure):
    """Coerce the raw TotalCharges field to a float.

    In the source data, TotalCharges is stored as a string, and is blank
    (not "0", an actual empty string) for the 11 customers with tenure == 0 -
    brand-new accounts that haven't been billed yet. A blank string with
    tenure == 0 is treated as $0 total charges; any other unparseable value
    is a real data problem and raises rather than being silently zeroed.
    """
    if isinstance(raw, str):
        raw = raw.strip()
    if raw in ("", None):
        if tenure == 0:
            return 0.0
        raise ValueError(f"TotalCharges is blank but tenure is {tenure}, not 0 - unexpected data")
    return float(raw)


def encode_target(value):
    """Map the raw Churn column ("Yes"/"No") to 1/0."""
    if value not in ("Yes", "No"):
        raise ValueError(f"Unrecognized Churn value: {value!r}")
    return 1 if value == "Yes" else 0


def encode_record(record):
    """Encode one customer record (dict of raw field values) to a feature vector.

    Returns a list of floats in FEATURE_COLUMNS order. Raises ValueError with
    a specific field name on anything missing or unrecognized, rather than
    silently producing a misencoded row - a wrong category value here is a
    caller bug (or a stale category list) that should fail loudly, not
    quietly skew a prediction.
    """
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    tenure = float(record["tenure"])
    total_charges = clean_total_charges(record["TotalCharges"], tenure)

    numeric_values = [
        float(record["SeniorCitizen"]),
        tenure,
        float(record["MonthlyCharges"]),
        total_charges,
    ]

    categorical_values = []
    for column, levels in CATEGORICAL_FEATURES.items():
        value = record[column]
        if value not in levels:
            raise ValueError(f"Unrecognized value for {column}: {value!r} (expected one of {levels})")
        for level in levels[1:]:
            categorical_values.append(1.0 if value == level else 0.0)

    return numeric_values + categorical_values
