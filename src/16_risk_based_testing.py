import os
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATHS
# ============================================================

CM1_PATH = "data/processed/cm1_clean.csv"
JM1_PATH = "data/processed/jm1_clean.csv"

OUTPUT_PATH = "data/processed/risk_based_testing_results.csv"


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
# HANDLE INVALID VALUES
# ============================================================

cm1_medians = X_cm1.median()

X_cm1 = X_cm1.fillna(cm1_medians)
X_jm1 = X_jm1.fillna(cm1_medians)


# ============================================================
# FUNCTION: TRAIN + RANK
# ============================================================

def calculate_testing_priority(
    source_name,
    source_X,
    source_y,
    target_name,
    target_X,
    target_y
):

    print("\n======================================")
    print(f"{source_name} → {target_name}")
    print("RISK-BASED TESTING")
    print("======================================")


    # --------------------------------------------------------
    # SMOTE
    # --------------------------------------------------------

    smote = SMOTE(random_state=42)

    X_source_smote, y_source_smote = smote.fit_resample(
        source_X,
        source_y
    )


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

    print("Model trained.")


    # --------------------------------------------------------
    # PREDICT DEFECT PROBABILITY
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        target_X
    )[:, 1]


    # --------------------------------------------------------
    # CREATE RISK TABLE
    # --------------------------------------------------------

    risk_table = pd.DataFrame({
        "Module_ID": range(1, len(target_X) + 1),
        "actual_defect": target_y.values,
        "defect_probability": probabilities
    })


    # Highest risk first
    risk_table = risk_table.sort_values(
        "defect_probability",
        ascending=False
    ).reset_index(drop=True)


    # Ranking
    risk_table["risk_rank"] = (
        risk_table.index + 1
    )


    # Risk score
    risk_table["risk_score"] = (
        risk_table["defect_probability"] * 100
    ).round(2)


    # Risk level
    risk_table["risk_level"] = risk_table[
        "risk_score"
    ].apply(
        lambda x:
            "HIGH" if x >= 60
            else "MEDIUM" if x >= 30
            else "LOW"
    )


    # --------------------------------------------------------
    # TOP-K RECALL
    # --------------------------------------------------------

    total_defects = risk_table[
        "actual_defect"
    ].sum()

    percentages = [
        10,
        20,
        30,
        50
    ]

    results = []


    for percentage in percentages:

        count = max(
            1,
            int(len(risk_table) * percentage / 100)
        )

        top_modules = risk_table.head(count)

        defects_found = top_modules[
            "actual_defect"
        ].sum()

        recall = (
            defects_found / total_defects
            if total_defects > 0
            else 0
        )

        results.append({
            "train_project": source_name,
            "test_project": target_name,
            "top_percent": percentage,
            "modules_tested": count,
            "defects_found": int(defects_found),
            "total_defects": int(total_defects),
            "recall_at_top_k": round(
                recall,
                4
            )
        })


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nRisk-Based Testing Results")

    for result in results:

        print(
            f"Top {result['top_percent']:>2}% | "
            f"Modules: {result['modules_tested']:>4} | "
            f"Defects found: "
            f"{result['defects_found']:>4}/"
            f"{result['total_defects']} | "
            f"Recall: "
            f"{result['recall_at_top_k']:.4f}"
        )


    # --------------------------------------------------------
    # SHOW TOP 10 MODULES
    # --------------------------------------------------------

    print("\nTop 10 highest-risk modules:")

    print(
        risk_table[
            [
                "Module_ID",
                "actual_defect",
                "defect_probability",
                "risk_score",
                "risk_level",
                "risk_rank"
            ]
        ].head(10).to_string(index=False)
    )


    return pd.DataFrame(results)


# ============================================================
# CM1 → JM1
# ============================================================

result_cm1_jm1 = calculate_testing_priority(
    "CM1",
    X_cm1,
    y_cm1,
    "JM1",
    X_jm1,
    y_jm1
)


# ============================================================
# JM1 → CM1
# ============================================================

result_jm1_cm1 = calculate_testing_priority(
    "JM1",
    X_jm1,
    y_jm1,
    "CM1",
    X_cm1,
    y_cm1
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
print("RISK-BASED TESTING COMPLETED")
print("======================================")

print("\nResults saved to:")
print(OUTPUT_PATH)