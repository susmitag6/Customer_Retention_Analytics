import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Retention Analytics",
    page_icon=":bar_chart:",
    layout="wide"
)


# --------------------------------------------------
# 2. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "model_output"
    / "customer_retention_analytics.csv"
)


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


df = load_data()


# --------------------------------------------------
# 4. Dashboard title
# --------------------------------------------------

st.title("Customer Retention Analytics Dashboard")

st.caption(
    "Churn prediction, customer segmentation, "
    "CLTV analysis and retention prioritization"
)


# --------------------------------------------------
# 5. Sidebar filters
# --------------------------------------------------

st.sidebar.header("Filters")


# Risk filter
risk_options = sorted(
    df["risk_category"]
    .dropna()
    .unique()
)

selected_risk = st.sidebar.multiselect(
    "Risk Category",
    options=risk_options,
    default=risk_options
)


# Segment filter
segment_options = sorted(
    df["segment_name"]
    .dropna()
    .unique()
)

selected_segments = st.sidebar.multiselect(
    "Customer Segment",
    options=segment_options,
    default=segment_options
)


# Retention priority filter
priority_options = [
    "Critical",
    "High",
    "Medium",
    "Low"
]

selected_priority = st.sidebar.multiselect(
    "Retention Priority",
    options=priority_options,
    default=priority_options
)


# Contract filter
contract_options = sorted(
    df["contract"]
    .dropna()
    .unique()
)

selected_contracts = st.sidebar.multiselect(
    "Contract Type",
    options=contract_options,
    default=contract_options
)


# --------------------------------------------------
# 6. Apply filters
# --------------------------------------------------

filtered_df = df[
    df["risk_category"].isin(selected_risk)
    &
    df["segment_name"].isin(selected_segments)
    &
    df["retention_priority"].isin(selected_priority)
    &
    df["contract"].isin(selected_contracts)
].copy()


# --------------------------------------------------
# 7. KPI calculations
# --------------------------------------------------

total_customers = len(filtered_df)


if total_customers > 0:

    actual_churn_rate = (
        filtered_df["churn_value"].mean()
        * 100
    )

else:

    actual_churn_rate = 0


high_risk_customers = (
    filtered_df["risk_category"]
    .eq("High")
    .sum()
)


critical_customers = (
    filtered_df["retention_priority"]
    .eq("Critical")
    .sum()
)


actionable_value_at_risk = (
    filtered_df[
        "actionable_value_at_risk"
    ]
    .sum()
)


average_churn_probability = (
    filtered_df[
        "churn_probability"
    ]
    .mean()
    * 100
    if total_customers > 0
    else 0
)


# --------------------------------------------------
# 8. KPI cards
# --------------------------------------------------

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)


kpi1.metric(
    "Total Customers",
    f"{total_customers:,}"
)


kpi2.metric(
    "Actual Churn Rate",
    f"{actual_churn_rate:.1f}%"
)


kpi3.metric(
    "High Risk Customers",
    f"{high_risk_customers:,}"
)


kpi4.metric(
    "Critical Retention",
    f"{critical_customers:,}"
)


kpi5.metric(
    "Actionable Value at Risk",
    f"{actionable_value_at_risk:,.0f}"
)


st.divider()


# --------------------------------------------------
# 9. Risk distribution
# --------------------------------------------------

chart1, chart2 = st.columns(2)


risk_counts = (
    filtered_df[
        "risk_category"
    ]
    .value_counts()
    .rename_axis("risk_category")
    .reset_index(name="customers")
)


risk_chart = px.bar(
    risk_counts,
    x="risk_category",
    y="customers",
    title="Customer Churn Risk Distribution",
    category_orders={
        "risk_category": [
            "Low",
            "Medium",
            "High"
        ]
    },
    labels={
        "risk_category": "Risk Category",
        "customers": "Customers"
    }
)


chart1.plotly_chart(
    risk_chart,
    use_container_width=True
)


# --------------------------------------------------
# 10. Segment distribution
# --------------------------------------------------

segment_counts = (
    filtered_df[
        "segment_name"
    ]
    .value_counts()
    .rename_axis("segment_name")
    .reset_index(name="customers")
)


segment_chart = px.bar(
    segment_counts,
    x="segment_name",
    y="customers",
    title="Customer Segment Distribution",
    labels={
        "segment_name": "Segment",
        "customers": "Customers"
    }
)


segment_chart.update_layout(
    xaxis_tickangle=-20
)


chart2.plotly_chart(
    segment_chart,
    use_container_width=True
)


# --------------------------------------------------
# 11. Retention priority distribution
# --------------------------------------------------

chart3, chart4 = st.columns(2)


priority_counts = (
    filtered_df[
        "retention_priority"
    ]
    .value_counts()
    .rename_axis("retention_priority")
    .reset_index(name="customers")
)


priority_chart = px.bar(
    priority_counts,
    x="retention_priority",
    y="customers",
    title="Retention Priority Distribution",
    category_orders={
        "retention_priority": [
            "Critical",
            "High",
            "Medium",
            "Low"
        ]
    },
    labels={
        "retention_priority":
            "Retention Priority",
        "customers":
            "Customers"
    }
)


chart3.plotly_chart(
    priority_chart,
    use_container_width=True
)


# --------------------------------------------------
# 12. Average churn probability by segment
# --------------------------------------------------

segment_risk = (
    filtered_df
    .groupby(
        "segment_name",
        as_index=False
    )
    .agg(
        average_churn_probability=(
            "churn_probability",
            "mean"
        )
    )
)


segment_risk[
    "average_churn_probability"
] *= 100


segment_risk_chart = px.bar(
    segment_risk,
    x="segment_name",
    y="average_churn_probability",
    title="Average Predicted Churn Risk by Segment",
    labels={
        "segment_name": "Segment",
        "average_churn_probability":
            "Average Churn Probability (%)"
    }
)


segment_risk_chart.update_layout(
    xaxis_tickangle=-20
)


chart4.plotly_chart(
    segment_risk_chart,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# 13. CLTV vs Churn Probability
# --------------------------------------------------

st.subheader("Customer Value vs Churn Risk")


scatter = px.scatter(
    filtered_df,
    x="cltv",
    y="churn_probability",
    color="retention_priority",
    size="monthly_charges",
    hover_data=[
        "customerid",
        "segment_name",
        "risk_category",
        "contract",
        "tenure_months",
        "value_at_risk"
    ],
    category_orders={
        "retention_priority": [
            "Critical",
            "High",
            "Medium",
            "Low"
        ]
    },
    title="CLTV vs Predicted Churn Probability",
    labels={
        "cltv":
            "Customer Lifetime Value",
        "churn_probability":
            "Predicted Churn Probability",
        "retention_priority":
            "Retention Priority"
    }
)


scatter.add_hline(
    y=0.35,
    line_dash="dash",
    annotation_text="Churn threshold = 35%"
)


scatter.add_hline(
    y=0.60,
    line_dash="dot",
    annotation_text="High-risk threshold = 60%"
)


st.plotly_chart(
    scatter,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# 14. Value at Risk by segment
# --------------------------------------------------

st.subheader("Value at Risk")


segment_value = (
    filtered_df
    .groupby(
        "segment_name",
        as_index=False
    )
    .agg(
        actionable_value_at_risk=(
            "actionable_value_at_risk",
            "sum"
        )
    )
    .sort_values(
        "actionable_value_at_risk",
        ascending=False
    )
)


value_chart = px.bar(
    segment_value,
    x="segment_name",
    y="actionable_value_at_risk",
    title="Actionable Value at Risk by Customer Segment",
    labels={
        "segment_name":
            "Customer Segment",
        "actionable_value_at_risk":
            "Actionable Value at Risk"
    }
)


value_chart.update_layout(
    xaxis_tickangle=-20
)


st.plotly_chart(
    value_chart,
    use_container_width=True
)


st.divider()


# --------------------------------------------------
# 15. Retention customer table
# --------------------------------------------------

st.subheader("Top Retention Opportunities")


retention_df = filtered_df[
    filtered_df[
        "retention_priority"
    ].isin(
        [
            "Critical",
            "High",
            "Medium"
        ]
    )
].copy()


retention_df = retention_df.sort_values(
    [
        "retention_rank",
        "value_at_risk"
    ],
    ascending=[
        True,
        False
    ]
)


table_columns = [
    "retention_rank",
    "customerid",
    "churn_probability",
    "risk_category",
    "cltv",
    "value_at_risk",
    "retention_priority",
    "segment_name",
    "contract",
    "tenure_months",
    "monthly_charges"
]


table_df = retention_df[
    table_columns
].copy()


# Convert probability to percentage
table_df[
    "churn_probability"
] = (
    table_df[
        "churn_probability"
    ]
    * 100
).round(1)


# Round financial fields
table_df[
    "value_at_risk"
] = table_df[
    "value_at_risk"
].round(2)


table_df[
    "monthly_charges"
] = table_df[
    "monthly_charges"
].round(2)


# Rename columns
table_df = table_df.rename(
    columns={
        "retention_rank":
            "Rank",

        "customerid":
            "Customer ID",

        "churn_probability":
            "Churn Probability (%)",

        "risk_category":
            "Risk",

        "cltv":
            "CLTV",

        "value_at_risk":
            "Value at Risk",

        "retention_priority":
            "Priority",

        "segment_name":
            "Segment",

        "contract":
            "Contract",

        "tenure_months":
            "Tenure",

        "monthly_charges":
            "Monthly Charges"
    }
)


st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    height=500
)


# --------------------------------------------------
# PCA Customer Segment Visualization
# --------------------------------------------------

st.divider()

st.subheader("Customer Segment Visualization")

st.caption(
    "PCA reduces the customer features to two dimensions "
    "so we can visually compare the customer segments."
)


pca_chart = px.scatter(
    filtered_df,
    x="pca_1",
    y="pca_2",
    color="segment_name",
    hover_data=[
        "customerid",
        "segment_name",
        "churn_probability",
        "cltv",
        "tenure_months",
        "monthly_charges"
    ],
    labels={
        "pca_1": "Principal Component 1",
        "pca_2": "Principal Component 2",
        "segment_name": "Customer Segment"
    },
    title="Customer Segments — PCA Visualization"
)


st.plotly_chart(
    pca_chart,
    use_container_width=True
)


# --------------------------------------------------
# 16. Summary information
# --------------------------------------------------

st.divider()

summary1, summary2, summary3 = st.columns(3)


# Average predicted churn probability
summary1.metric(
    "Average Predicted Churn Risk",
    f"{average_churn_probability:.1f}%"
)


# Critical + High priority customers
critical_high_customers = (
    filtered_df["retention_priority"]
    .isin(["Critical", "High"])
    .sum()
)

summary2.metric(
    "Total Critical + High Customers",
    f"{critical_high_customers:,}"
)


# Critical value at risk
critical_value = filtered_df.loc[
    filtered_df["retention_priority"] == "Critical",
    "actionable_value_at_risk"
].sum()

summary3.metric(
    "Critical Value at Risk",
    f"{critical_value:,.0f}"
)

# --------------------------------------------------
# 17. Download
# --------------------------------------------------

st.divider()


csv = filtered_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Filtered Customer Data",
    data=csv,
    file_name="customer_retention_dashboard_data.csv",
    mime="text/csv"
)
