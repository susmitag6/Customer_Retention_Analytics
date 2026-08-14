import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "eda"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("TELCO CUSTOMER CHURN - EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# --------------------------------------------------
# 3. Basic dataset information
# --------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nChurn_score:")
print(df["churn_score"])

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# 4. Statistical summary
# --------------------------------------------------

print("\nNumerical summary:")
print(df.describe())


# --------------------------------------------------
# 5. Churn distribution
# --------------------------------------------------

print("\nChurn distribution:")
print(df["churn_value"].value_counts())

print("\nChurn percentage:")
churn_percentage = (
    df["churn_value"]
    .value_counts(normalize=True)
    * 100
)

print(churn_percentage)


# --------------------------------------------------
# 6. Churn visualization
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="churn_value"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "churn_distribution.png"
)

plt.show()


# --------------------------------------------------
# 7. Churn by contract
# --------------------------------------------------

if "contract" in df.columns:

    contract_churn = pd.crosstab(
        df["contract"],
        df["churn_value"],
        normalize="index"
    ) * 100

    print("\nChurn percentage by contract:")
    print(contract_churn)

    contract_churn.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5)
    )

    plt.title("Churn by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Percentage")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_by_contract.png"
    )

    plt.show()


# --------------------------------------------------
# 8. Churn by tenure
# --------------------------------------------------

if "tenure_months" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="churn_value",
        y="tenure_months"
    )

    plt.title("Tenure vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Tenure (Months)")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "tenure_vs_churn.png"
    )

    plt.show()


# --------------------------------------------------
# 9. Monthly charges vs churn
# --------------------------------------------------

if "monthly_charges" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="churn_value",
        y="monthly_charges"
    )

    plt.title("Monthly Charges vs Churn")
    plt.xlabel("Churn_value")
    plt.ylabel("Monthly Charge")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "monthly_charges_vs_churn.png"
    )

    plt.show()


# --------------------------------------------------
# 10. Churn by internet service
# --------------------------------------------------

if "internet_service" in df.columns:

    internet_churn = pd.crosstab(
        df["internet_service"],
        df["churn_value"],
        normalize="index"
    ) * 100

    print("\nChurn percentage by Internet Service:")
    print(internet_churn)


# --------------------------------------------------
# 11. Churn by payment method
# --------------------------------------------------

if "payment_method" in df.columns:

    payment_churn = pd.crosstab(
        df["payment_method"],
        df["churn_value"],
        normalize="index"
    ) * 100

    print("\nChurn percentage by Payment Method:")
    print(payment_churn)


# --------------------------------------------------
# 12. Correlation analysis
# --------------------------------------------------

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

if not numeric_df.empty:

    plt.figure(figsize=(10, 7))

    correlation = numeric_df.corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Numerical Feature Correlation")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_matrix.png"
    )

    plt.show()


# --------------------------------------------------
# 13. Finish
# --------------------------------------------------

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print(f"\nCharts saved in:")
print(OUTPUT_DIR)
