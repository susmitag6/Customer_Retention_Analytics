import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# 1. Define file paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "raw" / "Telco_customer_churn.xlsx"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "telco_customer_churn_clean.csv"


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_excel(RAW_FILE)

print(f"Dataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# --------------------------------------------------
# 3. Inspect columns
# --------------------------------------------------

print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# 4. Check duplicate rows
# --------------------------------------------------

duplicate_rows = df.duplicated().sum()

print(f"\nDuplicate rows: {duplicate_rows}")


# --------------------------------------------------
# 5. Check missing values
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 6. Check data types
# --------------------------------------------------

print("\nData types:")
print(df.dtypes)


# --------------------------------------------------
# 7. Clean column names
# --------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)


print("\nCleaned column names:")
print(df.columns.tolist())


# --------------------------------------------------
# 8. Remove duplicate rows
# --------------------------------------------------

df = df.drop_duplicates()


# --------------------------------------------------
# 9. Convert total_charges to numeric
# --------------------------------------------------

if "total_charges" in df.columns:
    df["total_charges"] = pd.to_numeric(
        df["total_charges"],
        errors="coerce"
    )


# --------------------------------------------------
# 10. Handle missing values
# --------------------------------------------------

print("\nMissing values after type conversion:")
print(df.isnull().sum())


# For numeric columns, use median
numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())


# For categorical columns, use the most frequent value
categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# --------------------------------------------------
# 11. Remove unnecessary whitespace
# --------------------------------------------------

for column in categorical_columns:
    df[column] = df[column].str.strip()


# --------------------------------------------------
# 12. Final validation
# --------------------------------------------------

print("\nFinal dataset information:")
print(df.info())

print("\nRemaining missing values:")
print(df.isnull().sum())

print(f"\nFinal rows: {len(df)}")
print(f"Final columns: {len(df.columns)}")


# --------------------------------------------------
# 13. Save cleaned dataset
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nClean dataset saved to:")
print(OUTPUT_FILE)
