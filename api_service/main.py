import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from api_service.schemas import TransactionInput, PredictionOutput

app = FastAPI(
    title="API de Detección de Fraude",
    description="Servicio de inferencia para transacciones financieras",
    version="1.0.0"
)

# Cargar el pipeline completo (ColumnTransformer + Clasificador) al iniciar
try:
    model = joblib.load("models/fraud_pipeline.joblib")
except Exception as e:
    model = None
    print(f"Error cargando el modelo: {e}")


@app.post("/predict", response_model=PredictionOutput)
def predict_fraud(transaction: TransactionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible.")

    # 1. Convertir el objeto Pydantic a un DataFrame de 1 fila con los nombres exactos de columna
    input_data = pd.DataFrame([transaction.model_dump()])

    # 2. Obtener probabilidad de fraude (clase 1) y predicción binaria
    fraud_prob = float(model.predict_proba(input_data)[0][1])
    is_fraud = int(fraud_prob >= 0.5)

    # 3. Clasificar el nivel de riesgo
    if fraud_prob >= 0.75:
        risk_level = "HIGH"
    elif fraud_prob >= 0.35:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return PredictionOutput(
        is_fraud=is_fraud,
        fraud_probability=round(fraud_prob, 4),
        risk_level=risk_level
    )
