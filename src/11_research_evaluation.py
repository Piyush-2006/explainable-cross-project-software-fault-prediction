import os
import warnings

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/processed/cm1_clean.csv"
OUTPUT_PATH = "data/processed/final_evaluation_results.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded")
print("Shape:", df.shape)


# ============================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================

# Remove synthetic/module ID if present
drop_columns = []

if "Module_ID" in df.columns:
    drop_columns.append("Module_ID")

if "id" in df.columns:
    drop_columns.append("id")

X = df.drop(columns=drop_columns + ["defects"], errors="ignore")
y = df["defects"].astype(int)

print("\nFeatures:", X.shape[1])
print("Defect distribution:")
print(y.value_counts())


# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================
# 70% Train
# 15% Validation
# 15% Test

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42,
)

print("\nDataset split:")
print("Train:", X_train.shape[0])
print("Validation:", X_val.shape[0])
print("Test:", X_test.shape[0])


# ============================================================
# 5. APPLY SMOTE ONLY TO TRAINING DATA
# ============================================================

print("\nBefore SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# ============================================================
# 6. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight=None,
    n_jobs=-1,
)

model.fit(X_train_smote, y_train_smote)

print("\nRandom Forest training completed.")


# ============================================================
# 7. VALIDATION PREDICTIONS
# ============================================================

val_probabilities = model.predict_proba(X_val)[:, 1]


# ============================================================
# 8. FIND BEST THRESHOLD USING VALIDATION ONLY
# ============================================================

threshold_results = []

thresholds = np.arange(0.10, 0.71, 0.05)

for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    threshold_results.append(
        {
            "threshold": round(float(threshold), 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


threshold_df = pd.DataFrame(threshold_results)

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\nValidation threshold results:")
print(threshold_df)

print("\nBest threshold:")
print(best_threshold)

print("Best validation F1:")
print(round(float(best_row["f1"]), 4))


# ============================================================
# 9. FINAL TEST EVALUATION
# ============================================================
# IMPORTANT:
# Test data was never used for threshold selection.

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


# ============================================================
# 10. CALCULATE FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions,
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities,
)


# ============================================================
# 11. DISPLAY FINAL RESULTS
# ============================================================

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(f"Threshold : {best_threshold:.2f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results = pd.DataFrame(
    [
        {
            "dataset": "CM1",
            "model": "Random Forest + SMOTE",
            "best_threshold": best_threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
    ]
)

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

results.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\nResults saved to:")
print(OUTPUT_PATH)