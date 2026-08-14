import pandas as pd

from pathlib import Path


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = (
    BASE_DIR
    / "data"
    / "model_output"
    / "customer_predictions.csv"
)

SEGMENTS_FILE = (
    BASE_DIR
    / "data"
    / "model_output"
    / "customer_segments_final.csv"
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
# 2. Load datasets
# --------------------------------------------------

predictions = pd.read_csv(
    PREDICTIONS_FILE
)

segments = pd.read_csv(
    SEGMENTS_FILE
)


print("Prediction dataset:")
print(predictions.shape)

print("\nSegmentation dataset:")
print(segments.shape)


# --------------------------------------------------
# 3. Select segmentation information
# --------------------------------------------------

segment_data = segments[
    [
        "customerid",
        "cluster",
        "segment_name"

    ]
].copy()


# --------------------------------------------------
# 4. Merge
# --------------------------------------------------

analytics = predictions.merge(
    segment_data,
    on="customerid",
    how="left"
)

# --------------------------------------------------
# 5. Create meaningful segment names
# --------------------------------------------------

#analytics["segment_name"] =  segment_data["segment_name"]


# --------------------------------------------------
# 6. Create strategic customer group
# --------------------------------------------------

def create_priority(row):

    if (
        row["risk_category"] == "High"
        and row["segment_name"]
        == "Established High-Value Customer"
    ):
        return "Priority Retention"

    elif (
        row["risk_category"] == "High"
    ):
        return "High Risk"

    elif (
        row["segment_name"]
        == "Established High-Value Customer"
    ):
        return "High Value - Stable"

    else:
        return "Standard"


analytics["strategic_group"] = analytics.apply(
    create_priority,
    axis=1
)


# --------------------------------------------------
# 7. Summary
# --------------------------------------------------

print("\nSegment distribution:")

print(
    analytics[
        "segment_name"
    ].value_counts()
)


print("\nStrategic group distribution:")

print(
    analytics[
        "strategic_group"
    ].value_counts()
)


# --------------------------------------------------
# 8. High-value customers at risk
# --------------------------------------------------

priority_customers = analytics[
    analytics["strategic_group"]
    == "Priority Retention"
].copy()


priority_customers = priority_customers.sort_values(
    "churn_probability",
    ascending=False
)


print("\nTop priority customers:")

print(
    priority_customers[
        [
            "customerid",
            "churn_probability",
            "risk_category",
            "cltv",
            "segment_name",
            "strategic_group"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# --------------------------------------------------
# 9. Save final analytics dataset
# --------------------------------------------------

OUTPUT_FILE = (
    OUTPUT_DIR
    / "customer_analytics.csv"
)

analytics.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nFinal analytics dataset saved to:")

print(OUTPUT_FILE)
