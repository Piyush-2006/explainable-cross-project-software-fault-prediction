import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
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

print("Before SMOTE:")
print(y_train.value_counts())


# ==========================================
# 3. SMOTE
# ==========================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())


# ==========================================
# 4. RANDOM FOREST WITHOUT BALANCING
# ==========================================

normal_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

normal_model.fit(X_train, y_train)

normal_pred = normal_model.predict(X_test)
normal_prob = normal_model.predict_proba(X_test)[:, 1]


# ==========================================
# 5. RANDOM FOREST WITH CLASS WEIGHT
# ==========================================

weighted_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

weighted_model.fit(X_train, y_train)

weighted_pred = weighted_model.predict(X_test)
weighted_prob = weighted_model.predict_proba(X_test)[:, 1]


# ==========================================
# 6. RANDOM FOREST WITH SMOTE
# ==========================================

smote_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

smote_model.fit(X_train_smote, y_train_smote)

smote_pred = smote_model.predict(X_test)
smote_prob = smote_model.predict_proba(X_test)[:, 1]


# ==========================================
# 7. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, y_true, predictions, probabilities):

    return {
        "Method": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities
        )
    }


# ==========================================
# 8. COMPARE RESULTS
# ==========================================

results = [

    evaluate_model(
        "Normal Random Forest",
        y_test,
        normal_pred,
        normal_prob
    ),

    evaluate_model(
        "Class Weighted Random Forest",
        y_test,
        weighted_pred,
        weighted_prob
    ),

    evaluate_model(
        "SMOTE Random Forest",
        y_test,
        smote_pred,
        smote_prob
    )
]

results_df = pd.DataFrame(results)

print("\n==========================================")
print("      IMBALANCE EXPERIMENT")
print("==========================================")

print(results_df.to_string(index=False))


# ==========================================
# 9. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "data/processed/imbalance_results.csv",
    index=False
)

print("\nResults saved to:")
print("data/processed/imbalance_results.csv")