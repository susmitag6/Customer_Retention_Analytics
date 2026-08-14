import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
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


# --------------------------------------------------
# 3. Segmentation features
# --------------------------------------------------

numeric_features = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "cltv"
]

categorical_features = [
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
# 4. Prepare features
# --------------------------------------------------

X = df[
    numeric_features + categorical_features
].copy()


# --------------------------------------------------
# 5. One-hot encode
# --------------------------------------------------

X_encoded = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=False
)


# --------------------------------------------------
# 6. Convert boolean → integer
# --------------------------------------------------

bool_cols = X_encoded.select_dtypes(
    include=["bool"]
).columns

X_encoded[
    bool_cols
] = X_encoded[
    bool_cols
].astype(int)


# --------------------------------------------------
# 7. Handle missing values
# --------------------------------------------------

X_encoded = (
    X_encoded
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)

X_encoded = (
    X_encoded
    .fillna(
        X_encoded.median()
    )
)


# --------------------------------------------------
# 8. Scale
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_encoded
)


# --------------------------------------------------
# 9. Final K-Means model
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=20
)


df["cluster"] = kmeans.fit_predict(
    X_scaled
)


# --------------------------------------------------
# 10. Business names
# --------------------------------------------------

segment_names = {
    0: "Established High-Value",
    1: "Low-Spend Stable",
    2: "New High-Risk",
    3: "Mid-Tenure Moderate-Risk"
}


df["segment_name"] = (
    df["cluster"]
    .map(segment_names)
)


# --------------------------------------------------
# 11. Segment summary
# --------------------------------------------------

summary = (
    df
    .groupby(
        [
            "cluster",
            "segment_name"
        ]
    )
    .agg(
        customers=(
            "customerid",
            "count"
        ),
        avg_tenure=(
            "tenure_months",
            "mean"
        ),
        avg_monthly_charges=(
            "monthly_charges",
            "mean"
        ),
        avg_total_charges=(
            "total_charges",
            "mean"
        ),
        avg_cltv=(
            "cltv",
            "mean"
        ),
        churn_rate=(
            "churn_value",
            "mean"
        )
    )
    .reset_index()
)


summary["churn_rate"] *= 100

summary = summary.round(2)


# --------------------------------------------------
# 12. Print summary
# --------------------------------------------------

print("=" * 70)
print("FINAL CUSTOMER SEGMENTS")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)


# --------------------------------------------------
# 13. Save segmentation dataset
# --------------------------------------------------

OUTPUT_FILE = (
    OUTPUT_DIR
    / "customer_segments_final.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 14. Save summary
# --------------------------------------------------

SUMMARY_FILE = (
    OUTPUT_DIR
    / "final_segment_summary.csv"
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


print("\nSaved customer segments:")
print(OUTPUT_FILE)

print("\nSaved segment summary:")
print(SUMMARY_FILE)
