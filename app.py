"""
app.py
------
Service REST exposant la fonction predict() du modèle de churn prediction,
via FastAPI (Atelier 4).
"""

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "classifier.joblib"
ENCODER_PATH = "gender_encoder.joblib"
SCALER_PATH = "scaler.joblib"
FEATURES_PATH = "feature_columns.joblib"

app = FastAPI(title="Churn Prediction API")

model = joblib.load(MODEL_PATH)
gender_encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)

print(f"Modèle chargé depuis : {MODEL_PATH}")


class ChurnFeatures(BaseModel):
    CreditScore: int
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.post("/predict")
def predict(features: ChurnFeatures):
    """
    Effectue une prédiction de churn en fonction des caractéristiques
    fournies.

    Args:
        features (ChurnFeatures): les caractéristiques du client.

    Returns:
        dict: un dictionnaire avec la prédiction du modèle
              (0 = reste, 1 = churn).

    Raises:
        HTTPException: si une erreur survient pendant la prédiction.
    """
    try:
        data = features.dict()
        data["Gender"] = gender_encoder.transform([data["Gender"]])[0]

        row = [data[col] for col in feature_columns]
        row_scaled = scaler.transform(np.array(row).reshape(1, -1))

        prediction = model.predict(row_scaled)
        return {"prediction": int(prediction[0])}

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
