import pandas as pd
import matplotlib.pyplot as plt

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
    confusion_matrix
)


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_data.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "model_output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

TARGET = "churn_value"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# --------------------------------------------------
# 3. Feature groups
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
# 4. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 5. Preprocessing
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
            OneHotEncoder(
                handle_unknown="ignore"
            )
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
# 6. Model
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
# 7. Train
# --------------------------------------------------

pipeline.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 8. Get probabilities
# --------------------------------------------------

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# --------------------------------------------------
# 9. Test thresholds
# --------------------------------------------------

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]


results = []


for threshold in thresholds:

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()


    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn
    })


# --------------------------------------------------
# 10. Results dataframe
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 100)
print("THRESHOLD COMPARISON")
print("=" * 100)

print(
    results_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# 11. Best threshold based on F1
# --------------------------------------------------

best_row = results_df.loc[
    results_df["f1"].idxmax()
]


best_threshold = best_row[
    "threshold"
]


print("\n" + "=" * 60)
print("BEST F1 THRESHOLD")
print("=" * 60)

print(
    f"Threshold : {best_threshold:.2f}"
)

print(
    f"Precision : {best_row['precision']:.4f}"
)

print(
    f"Recall    : {best_row['recall']:.4f}"
)

print(
    f"F1 Score  : {best_row['f1']:.4f}"
)


# --------------------------------------------------
# 12. Plot threshold metrics
# --------------------------------------------------

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    results_df["threshold"],
    results_df["precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["threshold"],
    results_df["recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["threshold"],
    results_df["f1"],
    marker="o",
    label="F1"
)


plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best F1 = {best_threshold:.2f}"
)


plt.xlabel(
    "Classification Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Precision, Recall and F1 by Classification Threshold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "threshold_optimization.png"
)


plt.show()


# --------------------------------------------------
# 13. Save results
# --------------------------------------------------

OUTPUT_FILE = (
    OUTPUT_DIR
    / "threshold_comparison.csv"
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved threshold comparison:")
print(OUTPUT_FILE)
