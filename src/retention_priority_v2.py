import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "model_output"
    / "customer_analytics.csv"
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

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("RETENTION PRIORITY ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# Calculate value at risk
# --------------------------------------------------

df["value_at_risk"] = (
    df["churn_probability"]
    * df["cltv"]
)


# --------------------------------------------------
# Actionable value at risk
# Only customers above the optimized churn threshold
# --------------------------------------------------

df["actionable_value_at_risk"] = 0.0

target_mask = (
    df["churn_probability"] >= 0.35
)

df.loc[
    target_mask,
    "actionable_value_at_risk"
] = df.loc[
    target_mask,
    "value_at_risk"
]

# --------------------------------------------------
# 4. Create percentile ranking
# --------------------------------------------------

df["priority_percentile"] = (
    df["value_at_risk"]
    .rank(
        pct=True,
        method="average"
    )
)


# --------------------------------------------------
# 5. Create priority categories
# --------------------------------------------------

def priority_category(row):

    # Only customers above the optimized
    # churn threshold should enter retention targeting

    if row["churn_probability"] < 0.35:
        return "Low"

    percentile = row["priority_percentile"]

    if percentile >= 0.90:
        return "Critical"

    elif percentile >= 0.75:
        return "High"

    else:
        return "Medium"


df["retention_priority"] = df.apply(
    priority_category,
    axis=1
)


# --------------------------------------------------
# 6. Rank customers
# --------------------------------------------------

df["retention_rank"] = (
    df["value_at_risk"]
    .rank(
        method="dense",
        ascending=False
    )
    .astype(int)
)


# --------------------------------------------------
# 7. Summary
# --------------------------------------------------

priority_summary = (
    df
    .groupby(
        "retention_priority"
    )
    .agg(
        customers=(
            "customerid",
            "count"
        ),
        avg_churn_probability=(
            "churn_probability",
            "mean"
        ),
        avg_cltv=(
            "cltv",
            "mean"
        ),
        total_value_at_risk=(
            "value_at_risk",
            "sum"
        )
    )
    .reset_index()
)


priority_summary[
    "avg_churn_probability"
] *= 100


priority_summary = priority_summary.round(
    {
        "avg_churn_probability": 2,
        "avg_cltv": 2,
        "total_value_at_risk": 2
    }
)


print("\nPriority distribution:")

print(
    priority_summary.to_string(
        index=False
    )
)


# --------------------------------------------------
# 8. Top retention customers
# --------------------------------------------------

top_customers = (
    df[
        df["retention_priority"]
        .isin(
            [
                "Critical",
                "High"
            ]
        )
    ]
    .sort_values(
        "value_at_risk",
        ascending=False
    )
)


print("\nTop 20 retention opportunities:")

print(
    top_customers[
        [
            "retention_rank",
            "customerid",
            "churn_probability",
            "cltv",
            "value_at_risk",
            "risk_category",
            "retention_priority",
            "segment_name",
            "contract"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# 9. Save complete final dataset
# --------------------------------------------------

OUTPUT_FILE = (
    OUTPUT_DIR
    / "customer_retention_analytics.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 10. Save priority customers
# --------------------------------------------------

PRIORITY_FILE = (
    OUTPUT_DIR
    / "priority_retention_customers.csv"
)

top_customers.to_csv(
    PRIORITY_FILE,
    index=False
)


print("\nSaved final retention analytics:")
print(OUTPUT_FILE)

print("\nSaved priority customers:")
print(PRIORITY_FILE)
