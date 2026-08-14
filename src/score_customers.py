import pandas as pd
import joblib

from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "churn_pipeline.pkl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "model_output"
    / "customer_predictions_final.csv"
)


# --------------------------------------------------
# Load data and model
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

model = joblib.load(MODEL_FILE)


# --------------------------------------------------
# Features
# --------------------------------------------------

numeric_features = [
    "tenure_months",
    "monthly_charges",
    "total_charges"
]

categorical_features = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method"
]

features = (
    numeric_features
    + categorical_features
)


# --------------------------------------------------
# Predict
# --------------------------------------------------

X = df[features]

df["churn_probability"] = (
    model.predict_proba(X)[:, 1]
)


# --------------------------------------------------
# Optimized classification threshold
# --------------------------------------------------

OPTIMAL_THRESHOLD = 0.35

df["predicted_churn"] = (
    df["churn_probability"]
    >= OPTIMAL_THRESHOLD
).astype(int)


# --------------------------------------------------
# Risk categories
# --------------------------------------------------

def assign_risk(probability):

    if probability >= 0.60:
        return "High"

    elif probability >= 0.35:
        return "Medium"

    return "Low"


df["risk_category"] = (
    df["churn_probability"]
    .apply(assign_risk)
)


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("Customer scoring completed.")

print("\nRisk distribution:")
print(
    df["risk_category"]
    .value_counts()
)

print("\nSaved to:")
print(OUTPUT_FILE)
