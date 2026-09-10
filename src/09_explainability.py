import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/processed/cm1_clean.csv")

print("Dataset shape:", df.shape)


# ==========================================
# 2. PREPARE DATA
# ==========================================

X = df.drop(columns=["defects"])
y = df["defects"]

# Original ID was removed during cleaning
module_ids = np.arange(1, len(df) + 1)


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

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. SMOTE
# ==========================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("Training samples after SMOTE:", len(X_train_smote))


# ==========================================
# 5. RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(
    X_train_smote,
    y_train_smote
)

print("Random Forest trained successfully.")


# ==========================================
# 6. SHAP
# ==========================================

print("\nCreating SHAP explanations...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)


# ==========================================
# 7. HANDLE SHAP OUTPUT
# ==========================================

if isinstance(shap_values, list):

    shap_class1 = shap_values[1]

else:

    shap_class1 = np.asarray(shap_values)

    if shap_class1.ndim == 3:
        shap_class1 = shap_class1[:, :, 1]

    elif shap_class1.ndim == 2:
        shap_class1 = shap_class1

    else:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_class1.shape}"
        )


print("SHAP shape:", shap_class1.shape)


# ==========================================
# 8. GLOBAL FEATURE IMPORTANCE
# ==========================================

importance = np.abs(shap_class1).mean(axis=0)

feature_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_SHAP_Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Mean_SHAP_Importance",
    ascending=False
)

print("\n==========================================")
print("       SHAP FEATURE IMPORTANCE")
print("==========================================")

print(
    feature_importance.to_string(index=False)
)


# ==========================================
# 9. SAVE FEATURE IMPORTANCE
# ==========================================

feature_importance.to_csv(
    "data/processed/shap_feature_importance.csv",
    index=False
)

print("\nFeature importance saved to:")
print("data/processed/shap_feature_importance.csv")


# ==========================================
# 10. SHAP SUMMARY PLOT
# ==========================================

print("\nGenerating SHAP summary plot...")

shap.summary_plot(
    shap_class1,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    "data/processed/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP summary saved to:")
print("data/processed/shap_summary.png")


# ==========================================
# 11. CREATE RISK DATA
# ==========================================

probabilities = model.predict_proba(X_test)[:, 1]

risk_report = pd.DataFrame({
    "Module_ID": np.array(id_test),
    "Defect_Probability": probabilities
})


# IMPORTANT:
# Reset index so positions stay 0,1,2,3...
risk_report = risk_report.sort_values(
    by="Defect_Probability",
    ascending=False
).reset_index(drop=True)


# ==========================================
# 12. TOP RISK MODULES
# ==========================================

print("\n==========================================")
print("          TOP RISK MODULES")
print("==========================================")

print(
    risk_report.head(10).to_string(index=False)
)


# ==========================================
# 13. INDIVIDUAL SHAP EXPLANATIONS
# ==========================================

print("\n==========================================")
print("       INDIVIDUAL SHAP EXPLANATIONS")
print("==========================================")


for position in range(min(10, len(risk_report))):

    module_id = int(
        risk_report.iloc[position]["Module_ID"]
    )

    probability = (
        risk_report.iloc[position]["Defect_Probability"]
    )

    # Because risk_report was sorted using the
    # same test rows, we need the original index.
    #
    # Find the test row belonging to this module.
    original_row = np.where(
        np.array(id_test) == module_id
    )[0][0]

    row_shap = shap_class1[original_row]

    explanation = pd.DataFrame({
        "Feature": X_test.columns,
        "SHAP_Value": row_shap
    })

    explanation["Absolute_SHAP"] = (
        explanation["SHAP_Value"].abs()
    )

    explanation = explanation.sort_values(
        by="Absolute_SHAP",
        ascending=False
    )

    print("\n------------------------------------------")
    print(f"Module ID: {module_id}")
    print(f"Defect Probability: {probability:.2%}")
    print("------------------------------------------")

    print(
        explanation[
            ["Feature", "SHAP_Value"]
        ].head(5).to_string(index=False)
    )


# ==========================================
# 14. SAVE RISK REPORT
# ==========================================

risk_report.to_csv(
    "data/processed/shap_risk_report.csv",
    index=False
)

print("\nRisk report saved to:")
print("data/processed/shap_risk_report.csv")


# ==========================================
# 15. FINISHED
# ==========================================

print("\n==========================================")
print("       SHAP ANALYSIS COMPLETED")
print("==========================================")

print("\nGenerated files:")

print(
    "1. data/processed/shap_feature_importance.csv"
)

print(
    "2. data/processed/shap_summary.png"
)

print(
    "3. data/processed/shap_risk_report.csv"
)