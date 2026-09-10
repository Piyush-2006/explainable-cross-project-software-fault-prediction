import pandas as pd

# Load CM1 dataset
file_path = "data/raw/cm1.csv"

df = pd.read_csv(file_path)

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DEFECT DISTRIBUTION =====")
print(df["defects"].value_counts())

print("\n===== DEFECT PERCENTAGE =====")
print(df["defects"].value_counts(normalize=True) * 100)

print("\n===== BASIC STATISTICS =====")
print(df.describe())