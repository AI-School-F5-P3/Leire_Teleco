from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
from pathlib import Path

# Obtener la ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

app = FastAPI(title="Telco Customer Predictor")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el mejor modelo
model_path = BASE_DIR / "models" / "best_model.pkl"
model = joblib.load(model_path)

@app.post("/predict")
async def predict(data: dict):
    try:
        # Convertir datos a DataFrame
        df = pd.DataFrame([data])
        
        # Realizar predicción
        prediction = model.predict(df)
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(model.predict_proba(df).max())
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    return {
        "model_type": type(model).__name__,
        "features": list(model.feature_names_in_)
    }