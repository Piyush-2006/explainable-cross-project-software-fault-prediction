from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "data" / "processed" / "fault_prediction_model.pkl"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "model_features.pkl"
DASHBOARD_PATH = BASE_DIR / "dashboard" / "index.html"


# ============================================================
# Load model
# ============================================================

model = joblib.load(MODEL_PATH)
model_features = joblib.load(FEATURES_PATH)

explainer = shap.TreeExplainer(model)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Explainable Software Fault Risk Prediction API",
    description="Machine learning based software fault prediction with SHAP explanations.",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Input schema
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
# Root route - Dashboard
# ============================================================

@app.get("/")
def home():
    return FileResponse(DASHBOARD_PATH)


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Random Forest + SMOTE",
        "explainability": "SHAP"
    }


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict_fault(data: ModuleMetrics):

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

    # Create DataFrame in exact model feature order
    X = pd.DataFrame([input_data])

    X = X[model_features]

    # --------------------------------------------------------
    # Prediction probability
    # --------------------------------------------------------

    probability = float(model.predict_proba(X)[0][1])

    probability_percent = probability * 100

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    risk_score = probability_percent

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

    shap_values = explainer.shap_values(X)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):

        if len(shap_values) > 1:
            shap_array = np.asarray(shap_values[1])
        else:
            shap_array = np.asarray(shap_values[0])

    else:

        shap_array = np.asarray(shap_values)

        # Handle possible 3D output
        if shap_array.ndim == 3:

            if shap_array.shape[-1] == 2:
                shap_array = shap_array[:, :, 1]

            elif shap_array.shape[0] == 2:
                shap_array = shap_array[1]

    shap_array = np.asarray(shap_array).reshape(-1)

    # --------------------------------------------------------
    # Top 5 SHAP factors
    # --------------------------------------------------------

    feature_names = list(X.columns)

    shap_pairs = list(
        zip(feature_names, shap_array)
    )

    shap_pairs.sort(
        key=lambda x: abs(float(x[1])),
        reverse=True
    )

    top_factors = []

    for feature, contribution in shap_pairs[:5]:

        contribution = float(contribution)

        top_factors.append(
            {
                "feature": feature,
                "contribution": round(contribution, 4),
                "direction": (
                    "increases risk"
                    if contribution > 0
                    else "decreases risk"
                )
            }
        )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
        "defect_probability": round(
            probability_percent,
            2
        ),

        "risk_score": round(
            risk_score,
            2
        ),

        "risk_level": risk_level,

        "testing_priority": testing_priority,

        "shap_explanation": top_factors
    }