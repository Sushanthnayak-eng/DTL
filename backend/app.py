from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import json

# ------------------ App ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Load Artifacts ------------------
model = joblib.load("compost_model.pkl")
scaler = joblib.load("scaler.pkl")

with open("model_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_names"]

# ------------------ Input Schema ------------------
class CompostInput(BaseModel):
    Temperature: float
    MC_percent: float
    pH: float
    CN_Ratio: float
    Ammonia_mgkg: float
    Nitrate_mgkg: float
    TN_percent: float
    TOC_percent: float
    EC_mscm: float
    OM_percent: float
    T_Value: float
    GI_percent: float

# ------------------ Helper Functions ------------------
def classify_stage(score):
    if score < 35:
        return "Initial"
    elif score < 50:
        return "Active"
    elif score < 65:
        return "Stabilization"
    else:
        return "Mature"

def estimate_days(score):
    if score >= 70:
        return 0
    elif score >= 60:
        return int(5 + (70 - score) * 0.5)
    elif score >= 50:
        return int(10 + (60 - score) * 0.8)
    else:
        return int(15 + (50 - score) * 1.2)

# ------------------ Prediction API ------------------
@app.post("/predict")
def predict(data: CompostInput):

    input_dict = {
        'Temperature': data.Temperature,
        'MC(%)': data.MC_percent,
        'pH': data.pH,
        'C/N Ratio': data.CN_Ratio,
        'Ammonia(mg/kg)': data.Ammonia_mgkg,
        'Nitrate(mg/kg)': data.Nitrate_mgkg,
        'TN(%)': data.TN_percent,
        'TOC(%)': data.TOC_percent,
        'EC(ms/cm)': data.EC_mscm,
        'OM(%)': data.OM_percent,
        'T Value': data.T_Value,
        'GI(%)': data.GI_percent
    }

    # Order features EXACTLY like training
    X = np.array([input_dict[f] for f in FEATURES]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Prediction
    tree_preds = np.array([tree.predict(X_scaled)[0] for tree in model.estimators_])

    score = float(tree_preds.mean())
    std = float(tree_preds.std())

    return {
        "score": round(score, 2),
        "stage": classify_stage(score),
        "days_to_maturity": estimate_days(score),
        "confidence_interval": [
            round(score - 1.96 * std, 2),
            round(score + 1.96 * std, 2)
        ]
    }
