import pandas as pd

# -----------------------------
# 1. Load dataset
# -----------------------------
input_path = "data/raw/cm1.csv"

df = pd.read_csv(input_path)

print("Original shape:", df.shape)

# -----------------------------
# 2. Remove ID column
# -----------------------------
if "id" in df.columns:
    df = df.drop(columns=["id"])

# -----------------------------
# 3. Convert target to 0/1
# -----------------------------
df["defects"] = df["defects"].astype(str).str.lower().map({
    "false": 0,
    "true": 1
})

# -----------------------------
# 4. Check target conversion
# -----------------------------
print("\nDefect values after conversion:")
print(df["defects"].value_counts())

# -----------------------------
# 5. Check missing values
# -----------------------------
print("\nMissing values:")
print(df.isnull().sum().sum())

# -----------------------------
# 6. Check duplicate rows
# -----------------------------
print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicates if any
df = df.drop_duplicates()

# -----------------------------
# 7. Check data types
# -----------------------------
print("\nData types:")
print(df.dtypes)

# -----------------------------
# 8. Final shape
# -----------------------------
print("\nFinal shape:", df.shape)

# -----------------------------
# 9. Save cleaned dataset
# -----------------------------
output_path = "data/processed/cm1_clean.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:")
print(output_path)