from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Crear router
router = APIRouter()

# Definir modelos de datos
class PredictionInput(BaseModel):
    region: str
    tenure: float
    age: int
    marital: str
    income: float
    gender: str
    ed: int
    employ: int
    retire: str
    reside: int

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    features_used: Dict[str, float]

# Cargar modelo
BASE_DIR = Path(__file__).resolve().parent.parent.parent
model = joblib.load(BASE_DIR / "models" / "best_model.pkl")

@router.post("/predict", response_model=PredictionOutput)
async def make_prediction(data: PredictionInput):
    try:
        # Convertir entrada a DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Realizar predicción
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data).max()
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability),
            "features_used": data.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/model-info")
async def get_model_info():
    return {
        "model_type": type(model).__name__,
        "features": list(model.feature_names_in_)
    }

@router.get("/health")
async def health_check():
    return {"status": "healthy"}