import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_data.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "churn_pipeline.pkl"


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

TARGET = "churn_value"

X = df.drop(columns=[TARGET])
y = df[TARGET]


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


# --------------------------------------------------
# Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# --------------------------------------------------
# Complete pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# --------------------------------------------------
# Train
# --------------------------------------------------

print("Training final churn model...")

pipeline.fit(X_train, y_train)


# --------------------------------------------------
# Evaluate using optimized threshold
# --------------------------------------------------

OPTIMAL_THRESHOLD = 0.35

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= OPTIMAL_THRESHOLD
).astype(int)


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print("\nFINAL MODEL PERFORMANCE")
print("=" * 50)

print(f"Threshold : {OPTIMAL_THRESHOLD}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# --------------------------------------------------
# Save trained pipeline
# --------------------------------------------------

joblib.dump(
    pipeline,
    MODEL_FILE
)

print("\nModel saved to:")
print(MODEL_FILE)
