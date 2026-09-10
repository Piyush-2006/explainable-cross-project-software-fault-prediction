import os
import pandas as pd
import shap

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATHS
# ============================================================

CM1_PATH = "data/processed/cm1_clean.csv"
JM1_PATH = "data/processed/jm1_clean.csv"

OUTPUT_PATH = "data/processed/cpdp_shap_feature_importance.csv"


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
# LOAD DATA
# ============================================================

cm1 = pd.read_csv(CM1_PATH)
jm1 = pd.read_csv(JM1_PATH)

print("\nDatasets loaded")
print("CM1:", cm1.shape)
print("JM1:", jm1.shape)


# ============================================================
# CLEAN DATA
# ============================================================

X_cm1 = cm1[features].apply(
    pd.to_numeric,
    errors="coerce"
)

y_cm1 = cm1["defects"].astype(int)


X_jm1 = jm1[features].apply(
    pd.to_numeric,
    errors="coerce"
)

y_jm1 = jm1["defects"].astype(int)


# ============================================================
# CM1 MEDIAN VALUES
# ============================================================

cm1_medians = X_cm1.median()

X_cm1 = X_cm1.fillna(cm1_medians)
X_jm1 = X_jm1.fillna(cm1_medians)


# ============================================================
# FUNCTION FOR CPDP + SHAP
# ============================================================

def run_cpdp_shap(
    source_name,
    source_X,
    source_y,
    target_name,
    target_X
):

    print("\n======================================")
    print(f"{source_name} → {target_name}")
    print("CPDP + SHAP")
    print("======================================")


    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------

    smote = SMOTE(random_state=42)

    X_source_smote, y_source_smote = smote.fit_resample(
        source_X,
        source_y
    )

    print("\nAfter SMOTE:")
    print(pd.Series(y_source_smote).value_counts())


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_source_smote,
        y_source_smote
    )

    print("\nModel trained.")


    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(target_X)


    # --------------------------------------------------------
    # HANDLE SHAP OUTPUT
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        if len(shap_values) == 2:
            values = shap_values[1]
        else:
            values = shap_values[0]

    else:

        values = shap_values

        if len(values.shape) == 3:
            values = values[:, :, 1]


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    mean_abs_shap = abs(values).mean(axis=0)

    importance = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs_shap
    })

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # DISPLAY TOP FEATURES
    # --------------------------------------------------------

    print("\nTop SHAP features:")

    print(
        importance.head(10).to_string(index=False)
    )


    # --------------------------------------------------------
    # ADD PROJECT INFORMATION
    # --------------------------------------------------------

    importance.insert(
        0,
        "train_project",
        source_name
    )

    importance.insert(
        1,
        "test_project",
        target_name
    )


    return importance


# ============================================================
# CM1 → JM1
# ============================================================

result_cm1_jm1 = run_cpdp_shap(
    "CM1",
    X_cm1,
    y_cm1,
    "JM1",
    X_jm1
)


# ============================================================
# JM1 → CM1
# ============================================================

result_jm1_cm1 = run_cpdp_shap(
    "JM1",
    X_jm1,
    y_jm1,
    "CM1",
    X_cm1
)


# ============================================================
# COMBINE RESULTS
# ============================================================

final_results = pd.concat(
    [
        result_cm1_jm1,
        result_jm1_cm1
    ],
    ignore_index=True
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

final_results.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n======================================")
print("CPDP SHAP ANALYSIS COMPLETED")
print("======================================")

print("\nResults saved to:")
print(OUTPUT_PATH)