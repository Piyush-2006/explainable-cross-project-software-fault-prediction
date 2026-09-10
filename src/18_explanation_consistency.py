import os
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# INPUT
# ============================================================

INPUT_PATH = (
    "data/processed/cpdp_shap_feature_importance.csv"
)

OUTPUT_PATH = (
    "data/processed/explanation_consistency_results.csv"
)


# ============================================================
# LOAD SHAP RESULTS
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("\n======================================")
print("EXPLANATION CONSISTENCY ANALYSIS")
print("======================================")


# ============================================================
# SEPARATE THE TWO DIRECTIONS
# ============================================================

cm1_jm1 = df[
    (df["train_project"] == "CM1") &
    (df["test_project"] == "JM1")
].copy()

jm1_cm1 = df[
    (df["train_project"] == "JM1") &
    (df["test_project"] == "CM1")
].copy()


# ============================================================
# SORT BY SHAP IMPORTANCE
# ============================================================

cm1_jm1 = cm1_jm1.sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(drop=True)

jm1_cm1 = jm1_cm1.sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(drop=True)


# ============================================================
# FEATURE RANKINGS
# ============================================================

cm1_jm1["rank"] = range(
    1,
    len(cm1_jm1) + 1
)

jm1_cm1["rank"] = range(
    1,
    len(jm1_cm1) + 1
)


# ============================================================
# DISPLAY TOP FEATURES
# ============================================================

print("\nCM1 → JM1 Top 10 features:")

print(
    cm1_jm1[
        ["rank", "feature", "mean_abs_shap"]
    ].head(10).to_string(index=False)
)


print("\nJM1 → CM1 Top 10 features:")

print(
    jm1_cm1[
        ["rank", "feature", "mean_abs_shap"]
    ].head(10).to_string(index=False)
)


# ============================================================
# TOP-5 OVERLAP
# ============================================================

top5_cm1_jm1 = set(
    cm1_jm1.head(5)["feature"]
)

top5_jm1_cm1 = set(
    jm1_cm1.head(5)["feature"]
)

top5_common = (
    top5_cm1_jm1 &
    top5_jm1_cm1
)


top5_overlap = (
    len(top5_common) / 5
)


# ============================================================
# TOP-10 OVERLAP
# ============================================================

top10_cm1_jm1 = set(
    cm1_jm1.head(10)["feature"]
)

top10_jm1_cm1 = set(
    jm1_cm1.head(10)["feature"]
)

top10_common = (
    top10_cm1_jm1 &
    top10_jm1_cm1
)


top10_overlap = (
    len(top10_common) / 10
)


# ============================================================
# SPEARMAN RANK CORRELATION
# ============================================================

rank_cm1_jm1 = (
    cm1_jm1
    .set_index("feature")["rank"]
)

rank_jm1_cm1 = (
    jm1_cm1
    .set_index("feature")["rank"]
)


# Keep only common features
common_features = (
    rank_cm1_jm1.index
    .intersection(rank_jm1_cm1.index)
)


ranks_1 = rank_cm1_jm1.loc[
    common_features
]

ranks_2 = rank_jm1_cm1.loc[
    common_features
]


spearman_correlation, p_value = spearmanr(
    ranks_1,
    ranks_2
)


# ============================================================
# COMMON TOP FEATURES
# ============================================================

common_top_features = sorted(
    top10_common
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n======================================")
print("CONSISTENCY RESULTS")
print("======================================")

print(
    f"\nTop-5 common features: "
    f"{len(top5_common)}/5"
)

print(
    f"Top-5 overlap: "
    f"{top5_overlap * 100:.2f}%"
)


print(
    f"\nTop-10 common features: "
    f"{len(top10_common)}/10"
)

print(
    f"Top-10 overlap: "
    f"{top10_overlap * 100:.2f}%"
)


print(
    f"\nSpearman rank correlation: "
    f"{spearman_correlation:.4f}"
)

print(
    f"P-value: "
    f"{p_value:.4f}"
)


print(
    "\nCommon Top-10 features:"
)

for feature in common_top_features:
    print(
        f"- {feature}"
    )


# ============================================================
# INTERPRETATION
# ============================================================

if spearman_correlation >= 0.7:

    interpretation = (
        "High explanation consistency"
    )

elif spearman_correlation >= 0.4:

    interpretation = (
        "Moderate explanation consistency"
    )

else:

    interpretation = (
        "Low explanation consistency"
    )


print(
    f"\nInterpretation: {interpretation}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "comparison": "CM1 → JM1 vs JM1 → CM1",
        "top5_common_features": len(top5_common),
        "top5_overlap_percent": round(
            top5_overlap * 100,
            2
        ),
        "top10_common_features": len(top10_common),
        "top10_overlap_percent": round(
            top10_overlap * 100,
            2
        ),
        "spearman_correlation": round(
            spearman_correlation,
            4
        ),
        "p_value": round(
            p_value,
            4
        ),
        "interpretation": interpretation
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


# ============================================================
# COMPLETE
# ============================================================

print("\n======================================")
print("EXPLANATION CONSISTENCY COMPLETED")
print("======================================")

print("\nResults saved to:")
print(OUTPUT_PATH)