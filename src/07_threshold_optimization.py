import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/processed/cm1_clean.csv")

X = df.drop("defects", axis=1)
y = df["defects"]

if "id" in X.columns:
    X = X.drop("id", axis=1)


# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. APPLY SMOTE ONLY TO TRAINING DATA
# ==========================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# ==========================================
# 4. TRAIN RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_smote, y_train_smote)


# ==========================================
# 5. GET DEFECT PROBABILITIES
# ==========================================

probabilities = model.predict_proba(X_test)[:, 1]


# ==========================================
# 6. TEST DIFFERENT THRESHOLDS
# ==========================================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


# ==========================================
# 7. RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n==========================================")
print("       THRESHOLD OPTIMIZATION")
print("==========================================")

print(
    results_df.to_string(index=False)
)


# ==========================================
# 8. FIND BEST F1 THRESHOLD
# ==========================================

best_row = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n==========================================")
print("             BEST THRESHOLD")
print("==========================================")

print(
    f"Threshold : {best_row['Threshold']:.2f}"
)

print(
    f"Precision : {best_row['Precision']:.3f}"
)

print(
    f"Recall    : {best_row['Recall']:.3f}"
)

print(
    f"F1 Score  : {best_row['F1']:.3f}"
)


# ==========================================
# 9. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "data/processed/threshold_results.csv",
    index=False
)

print("\nResults saved to:")
print("data/processed/threshold_results.csv")


# ==========================================
# 10. PLOT
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["F1"],
    marker="o",
    label="F1"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Threshold Optimization")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()