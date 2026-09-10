import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/processed/cm1_clean.csv"
)

X = df.drop(columns=["defects"])
y = df["defects"]


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


# ==========================================
# 3. SMOTE
# ==========================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# ==========================================
# 4. TRAIN MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(
    X_train_smote,
    y_train_smote
)


# ==========================================
# 5. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "data/processed/fault_prediction_model.pkl"
)


# ==========================================
# 6. SAVE FEATURE NAMES
# ==========================================

joblib.dump(
    list(X.columns),
    "data/processed/model_features.pkl"
)


print("==========================================")
print("       MODEL TRAINING COMPLETED")
print("==========================================")

print("Training samples:", len(X_train))
print("After SMOTE:", len(X_train_smote))

print("\nModel saved:")
print("data/processed/fault_prediction_model.pkl")

print("\nFeatures saved:")
print("data/processed/model_features.pkl")