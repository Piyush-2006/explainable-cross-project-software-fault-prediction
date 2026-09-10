import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


# ==========================================
# 1. LOAD CLEAN DATA
# ==========================================

df = pd.read_csv("data/processed/cm1_clean.csv")

print("Dataset shape:", df.shape)

X = df.drop("defects", axis=1)
y = df["defects"]


# ==========================================
# 2. CREATE MODULE IDs
# ==========================================

# Your cleaned dataset no longer has original IDs,
# so create simple module numbers.

module_ids = range(1, len(df) + 1)


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    module_ids,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. APPLY SMOTE
# ==========================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nTraining data before SMOTE:", len(X_train))
print("Training data after SMOTE:", len(X_train_smote))


# ==========================================
# 5. TRAIN RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train_smote,
    y_train_smote
)


# ==========================================
# 6. GET DEFECT PROBABILITY
# ==========================================

probabilities = model.predict_proba(X_test)[:, 1]


# ==========================================
# 7. RISK SCORE
# ==========================================

risk_score = probabilities * 100


# ==========================================
# 8. RISK LEVEL
# ==========================================

def get_risk_level(score):

    if score >= 60:
        return "HIGH"

    elif score >= 30:
        return "MEDIUM"

    else:
        return "LOW"


risk_level = [
    get_risk_level(score)
    for score in risk_score
]


# ==========================================
# 9. TESTING PRIORITY
# ==========================================

def get_priority(level):

    if level == "HIGH":
        return "TEST FIRST"

    elif level == "MEDIUM":
        return "TEST SOON"

    else:
        return "NORMAL"


testing_priority = [
    get_priority(level)
    for level in risk_level
]


# ==========================================
# 10. CREATE RISK REPORT
# ==========================================

risk_report = pd.DataFrame({
    "Module_ID": list(id_test),
    "Defect_Probability": probabilities,
    "Risk_Score": risk_score,
    "Risk_Level": risk_level,
    "Testing_Priority": testing_priority,
    "Actual_Defect": y_test.values
})


# ==========================================
# 11. SORT BY RISK
# ==========================================

risk_report = risk_report.sort_values(
    by="Risk_Score",
    ascending=False
)


# ==========================================
# 12. DISPLAY TOP 20
# ==========================================

print("\n==========================================")
print("       SOFTWARE FAULT RISK REPORT")
print("==========================================")

print(
    risk_report.head(20).to_string(index=False)
)


# ==========================================
# 13. RISK DISTRIBUTION
# ==========================================

print("\n==========================================")
print("             RISK DISTRIBUTION")
print("==========================================")

print(
    risk_report["Risk_Level"].value_counts()
)


# ==========================================
# 14. SAVE REPORT
# ==========================================

risk_report.to_csv(
    "data/processed/risk_report.csv",
    index=False
)

print("\nRisk report saved to:")
print("data/processed/risk_report.csv")