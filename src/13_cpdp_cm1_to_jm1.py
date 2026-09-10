import os
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


# ============================================================
# FILE PATHS
# ============================================================

CM1_PATH = "data/processed/cm1_clean.csv"
JM1_PATH = "data/processed/jm1_clean.csv"

OUTPUT_PATH = "data/processed/cpdp_cm1_to_jm1_results.csv"


# ============================================================
# LOAD DATASETS
# ============================================================

cm1 = pd.read_csv(CM1_PATH)
jm1 = pd.read_csv(JM1_PATH)

print("\nDatasets loaded")
print("CM1 shape:", cm1.shape)
print("JM1 shape:", jm1.shape)


# ============================================================
# FEATURES
# ============================================================

features = [
    "loc",
    "v(g)",
    "ev(g)",
    "iv(g)",
    "n",
    "v",
    "l",
    "d",
    "i",
    "e",
    "b",
    "t",
    "lOCode",
    "lOComment",
    "lOBlank",
    "locCodeAndComment",
    "uniq_Op",
    "uniq_Opnd",
    "total_Op",
    "total_Opnd",
    "branchCount",
]


# ============================================================
# PREPARE CM1
# ============================================================

X_train = cm1[features].apply(pd.to_numeric, errors="coerce")
y_train = cm1["defects"].astype(int)


# ============================================================
# PREPARE JM1
# ============================================================

X_test = jm1[features].apply(pd.to_numeric, errors="coerce")
y_test = jm1["defects"].astype(int)


# ============================================================
# HANDLE INVALID VALUES
# ============================================================

print("\nInvalid/non-numeric values handled.")

print("CM1 missing values before cleaning:",
      X_train.isnull().sum().sum())

print("JM1 missing values before cleaning:",
      X_test.isnull().sum().sum())


# Use CM1 feature medians for both datasets
cm1_medians = X_train.median()

X_train = X_train.fillna(cm1_medians)
X_test = X_test.fillna(cm1_medians)


print("CM1 remaining missing values:",
      X_train.isnull().sum().sum())

print("JM1 remaining missing values:",
      X_test.isnull().sum().sum())


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nCM1 training distribution:")
print(y_train.value_counts())

print("\nJM1 testing distribution:")
print(y_test.value_counts())


# ============================================================
# SMOTE ON SOURCE PROJECT (CM1)
# ============================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


print("\nCM1 distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# TRAIN MODEL ON CM1
# ============================================================

model.fit(
    X_train_smote,
    y_train_smote
)

print("\nRandom Forest trained on CM1.")


# ============================================================
# PREDICT ON JM1
# ============================================================

test_probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# CLASSIFICATION THRESHOLD
# ============================================================

# Threshold selected previously using CM1 validation.
# It is kept fixed here for cross-project testing.
threshold = 0.30

test_predictions = (
    test_probabilities >= threshold
).astype(int)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n======================================")
print("CROSS-PROJECT DEFECT PREDICTION")
print("CM1 → JM1")
print("======================================")

print(f"Threshold : {threshold:.2f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "train_project": "CM1",
        "test_project": "JM1",
        "model": "Random Forest + SMOTE",
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
])


os.makedirs(
    "data/processed",
    exist_ok=True
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nResults saved to:")
print(OUTPUT_PATH)