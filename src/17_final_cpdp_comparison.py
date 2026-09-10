import os
import pandas as pd


# ============================================================
# INPUT FILES
# ============================================================

CPDP_CM1_JM1 = (
    "data/processed/cpdp_cm1_to_jm1_results.csv"
)

CPDP_JM1_CM1 = (
    "data/processed/cpdp_jm1_to_cm1_results.csv"
)

RISK_RESULTS = (
    "data/processed/risk_based_testing_results.csv"
)

OUTPUT_PATH = (
    "data/processed/final_cpdp_comparison.csv"
)


# ============================================================
# LOAD CPDP RESULTS
# ============================================================

cm1_jm1 = pd.read_csv(CPDP_CM1_JM1)

jm1_cm1 = pd.read_csv(CPDP_JM1_CM1)

risk_results = pd.read_csv(RISK_RESULTS)


print("\n======================================")
print("FINAL CPDP COMPARISON")
print("======================================")


# ============================================================
# DISPLAY BASIC RESULTS
# ============================================================

print("\nCPDP Results Loaded")

print("\nCM1 → JM1:")
print(cm1_jm1.to_string(index=False))

print("\nJM1 → CM1:")
print(jm1_cm1.to_string(index=False))


# ============================================================
# EXTRACT MODEL RESULTS
# ============================================================

comparison = pd.concat(
    [
        cm1_jm1,
        jm1_cm1
    ],
    ignore_index=True
)


# ============================================================
# EXTRACT RISK-BASED TESTING RESULTS
# ============================================================

risk_pivot = risk_results.pivot_table(
    index=[
        "train_project",
        "test_project"
    ],
    columns="top_percent",
    values="recall_at_top_k"
).reset_index()


# Rename columns

risk_pivot = risk_pivot.rename(
    columns={
        10: "recall_top_10",
        20: "recall_top_20",
        30: "recall_top_30",
        50: "recall_top_50"
    }
)


# ============================================================
# MERGE RESULTS
# ============================================================

final_comparison = comparison.merge(
    risk_pivot,
    on=[
        "train_project",
        "test_project"
    ],
    how="left"
)


# ============================================================
# CONVERT VALUES TO PERCENTAGES
# ============================================================

percentage_columns = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "recall_top_10",
    "recall_top_20",
    "recall_top_30",
    "recall_top_50"
]


for column in percentage_columns:

    if column in final_comparison.columns:

        final_comparison[column] = (
            final_comparison[column] * 100
        ).round(2)


# ============================================================
# SAVE FINAL TABLE
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

final_comparison.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY FINAL RESEARCH TABLE
# ============================================================

print("\n======================================")
print("FINAL RESEARCH COMPARISON TABLE")
print("======================================")

display_columns = [
    "train_project",
    "test_project",
    "model",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "recall_top_10",
    "recall_top_20",
    "recall_top_30",
    "recall_top_50"
]


print(
    final_comparison[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# BEST DIRECTION
# ============================================================

best_auc_index = final_comparison[
    "roc_auc"
].idxmax()

best_f1_index = final_comparison[
    "f1"
].idxmax()

best_top20_index = final_comparison[
    "recall_top_20"
].idxmax()


print("\n======================================")
print("KEY FINDINGS")
print("======================================")


print(
    "\nBest ROC-AUC:"
)

print(
    final_comparison.loc[
        best_auc_index,
        [
            "train_project",
            "test_project",
            "roc_auc"
        ]
    ].to_dict()
)


print(
    "\nBest F1 Score:"
)

print(
    final_comparison.loc[
        best_f1_index,
        [
            "train_project",
            "test_project",
            "f1"
        ]
    ].to_dict()
)


print(
    "\nBest Recall@Top-20%:"
)

print(
    final_comparison.loc[
        best_top20_index,
        [
            "train_project",
            "test_project",
            "recall_top_20"
        ]
    ].to_dict()
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n======================================")
print("COMPARISON COMPLETED")
print("======================================")

print("\nFinal file saved to:")
print(OUTPUT_PATH)