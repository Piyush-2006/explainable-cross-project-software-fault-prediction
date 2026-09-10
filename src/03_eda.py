import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/processed/cm1_clean.csv")

# Calculate correlation with defects
correlations = df.corr(numeric_only=True)["defects"].drop("defects")

# Sort by absolute correlation
correlations = correlations.reindex(
    correlations.abs().sort_values(ascending=False).index
)

print("\n===== FEATURE CORRELATION WITH DEFECTS =====")
print(correlations)

# Plot correlations
plt.figure(figsize=(10, 6))
correlations.plot(kind="bar")

plt.title("Correlation of Software Metrics with Defects")
plt.xlabel("Software Metrics")
plt.ylabel("Correlation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()