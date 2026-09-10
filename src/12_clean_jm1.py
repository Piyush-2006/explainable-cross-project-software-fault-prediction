import os
import pandas as pd


# ============================================
# 1. File paths
# ============================================

INPUT_PATH = "data/raw/jm1.csv"
OUTPUT_PATH = "data/processed/jm1_clean.csv"


# ============================================
# 2. Load dataset
# ============================================

df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)


# ============================================
# 3. Remove ID column
# ============================================

if "id" in df.columns:
    df = df.drop(columns=["id"])


# ============================================
# 4. Convert defect labels to 0/1
# ============================================

df["defects"] = (
    df["defects"]
    .astype(str)
    .str.lower()
    .map({
        "false": 0,
        "true": 1
    })
)


# ============================================
# 5. Check missing values
# ============================================

print("\nMissing values:")
print(df.isnull().sum().sum())


# ============================================
# 6. Remove duplicate rows
# ============================================

duplicates = df.duplicated().sum()

print("\nDuplicate rows:", duplicates)

df = df.drop_duplicates().reset_index(drop=True)


# ============================================
# 7. Create Module_ID
# ============================================

df.insert(0, "Module_ID", range(1, len(df) + 1))


# ============================================
# 8. Save cleaned dataset
# ============================================

os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================
# 9. Display final information
# ============================================

print("\nCleaned shape:", df.shape)

print("\nDefect distribution:")
print(df["defects"].value_counts())

print("\nFinal columns:")
print(df.columns.tolist())

print("\nSaved to:")
print(OUTPUT_PATH)