import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("data/processed/cm1_clean.csv")

# Separate features and target
X = df.drop("defects", axis=1)
y = df["defects"]

# Remove ID because it is only an identifier
if "id" in X.columns:
    X = X.drop("id", axis=1)

# Calculate Mutual Information
mi_scores = mutual_info_classif(
    X,
    y,
    random_state=42
)

# Create result table
mi_df = pd.DataFrame({
    "Feature": X.columns,
    "Mutual_Information": mi_scores
})

# Sort from highest to lowest
mi_df = mi_df.sort_values(
    "Mutual_Information",
    ascending=False
)

print("\n===== MUTUAL INFORMATION FEATURE IMPORTANCE =====")
print(mi_df.to_string(index=False))

# Save results
mi_df.to_csv(
    "data/processed/feature_selection.csv",
    index=False
)

# Plot
plt.figure(figsize=(10, 6))

plt.bar(
    mi_df["Feature"],
    mi_df["Mutual_Information"]
)

plt.title("Feature Importance using Mutual Information")
plt.xlabel("Software Metrics")
plt.ylabel("Mutual Information")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

print("\nFeature selection results saved to:")
print("data/processed/feature_selection.csv")