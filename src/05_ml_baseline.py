import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/processed/cm1_clean.csv")

print("Dataset shape:", df.shape)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

X = df.drop("defects", axis=1)
y = df["defects"]

# ID is not a useful prediction feature
if "id" in X.columns:
    X = X.drop("id", axis=1)


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. SCALE DATA FOR LOGISTIC REGRESSION
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
}


# ==========================================
# 6. TRAIN AND EVALUATE
# ==========================================

results = []

for name, model in models.items():

    print("\nTraining:", name)

    # Logistic Regression uses scaled data
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]

    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
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
    auc = roc_auc_score(
        y_test,
        probabilities
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n==========================================")
print("        ML MODEL COMPARISON")
print("==========================================")

print(
    results_df.to_string(index=False)
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "data/processed/ml_baseline_results.csv",
    index=False
)

print("\nResults saved to:")
print("data/processed/ml_baseline_results.csv")