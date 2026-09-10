import os
import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# Paths
# ============================================================

MODEL_PATH = "data/processed/fault_prediction_model.pkl"
FEATURES_PATH = "data/processed/model_features.pkl"


# ============================================================
# Load trained model
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(
        f"Feature file not found: {FEATURES_PATH}"
    )

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Software Fault Risk Prediction API",
    description="Explainable Software Fault Risk Prediction using Random Forest, SMOTE and SHAP",
    version="1.0"
)


# ============================================================
# CORS
# Allows dashboard/index.html to communicate with FastAPI
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Input data model
# ============================================================

class ModuleMetrics(BaseModel):

    loc: float
    vg: float
    evg: float
    ivg: float

    n: float
    v: float
    l: float
    d: float
    i: float
    e: float
    b: float
    t: float

    lOCode: float
    lOComment: float
    lOBlank: float
    locCodeAndComment: float

    uniq_Op: float
    uniq_Opnd: float
    total_Op: float
    total_Opnd: float

    branchCount: float


# ============================================================
# Health check
# ============================================================

@app.get("/")
def home():

    return {
        "project": "Explainable Software Fault Risk Prediction",
        "status": "API is running",
        "model": "Random Forest + SMOTE",
        "explainability": "SHAP"
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict_fault_risk(data: ModuleMetrics):

    # --------------------------------------------------------
    # Convert API input to dictionary
    # --------------------------------------------------------

    input_data = {
        "loc": data.loc,
        "v(g)": data.vg,
        "ev(g)": data.evg,
        "iv(g)": data.ivg,
        "n": data.n,
        "v": data.v,
        "l": data.l,
        "d": data.d,
        "i": data.i,
        "e": data.e,
        "b": data.b,
        "t": data.t,
        "lOCode": data.lOCode,
        "lOComment": data.lOComment,
        "lOBlank": data.lOBlank,
        "locCodeAndComment": data.locCodeAndComment,
        "uniq_Op": data.uniq_Op,
        "uniq_Opnd": data.uniq_Opnd,
        "total_Op": data.total_Op,
        "total_Opnd": data.total_Opnd,
        "branchCount": data.branchCount
    }

    # --------------------------------------------------------
    # Create DataFrame using exact model feature order
    # --------------------------------------------------------

    X = pd.DataFrame([input_data])

    X = X.reindex(columns=feature_names)

    # --------------------------------------------------------
    # Predict probability
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(X)[0][1]
    )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    risk_score = round(probability * 100, 2)

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if risk_score >= 60:
        risk_level = "HIGH"
        testing_priority = "TEST FIRST"

    elif risk_score >= 30:
        risk_level = "MEDIUM"
        testing_priority = "TEST SOON"

    else:
        risk_level = "LOW"
        testing_priority = "TEST LATER"

    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------

    explanation = []

    try:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):

            # Binary classification:
            # index 1 = defect class
            if len(shap_values) > 1:
                values = np.asarray(shap_values[1])[0]
            else:
                values = np.asarray(shap_values[0])[0]

        else:

            values = np.asarray(shap_values)

            # Possible shape:
            # (1, features, classes)
            if values.ndim == 3:

                if values.shape[2] > 1:
                    values = values[0, :, 1]
                else:
                    values = values[0, :, 0]

            # Possible shape:
            # (1, features)
            elif values.ndim == 2:

                values = values[0]

            # Possible shape:
            # (features,)
            elif values.ndim == 1:

                values = values

        # Make sure the number of SHAP values matches features
        values = np.asarray(values).flatten()

        if len(values) == len(feature_names):

            shap_data = []

            for feature, value in zip(feature_names, values):

                shap_data.append({
                    "feature": feature,
                    "shap_value": round(float(value), 4),
                    "effect": (
                        "increases risk"
                        if value > 0
                        else "decreases risk"
                    )
                })

            # Sort by absolute SHAP impact
            shap_data.sort(
                key=lambda x: abs(x["shap_value"]),
                reverse=True
            )

            # Return top 5 factors
            explanation = shap_data[:5]

    except Exception as error:

        explanation = [
            {
                "feature": "SHAP",
                "shap_value": 0,
                "effect": f"Explanation unavailable: {str(error)}"
            }
        ]

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
        "defect_probability": round(probability, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "testing_priority": testing_priority,
        "explanation": explanation
    }