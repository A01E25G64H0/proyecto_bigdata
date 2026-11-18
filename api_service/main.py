from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from schemas import TransactionInput

app = FastAPI(
    title="API de Detección de Fraude Financiero",
    description="Servicio REST para inferencia de fraude sobre transacciones PaySim.",
    version="1.0.0"
)

model = joblib.load("fraud_model.pkl")
P99_CAP = 1623278.90  # Tope de Winsorización aplicado en Silver

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict_fraud(tx: TransactionInput):
    try:
        hour_of_day = tx.step % 24
        amount_capped = min(tx.amount, P99_CAP)
        log_amount = float(np.log1p(amount_capped))
        error_orig = tx.oldbalanceOrg - tx.amount - tx.newbalanceOrig
        error_dest = tx.oldbalanceDest + tx.amount - tx.newbalanceDest

        features_df = pd.DataFrame([{
            "type": tx.type,
            "hour_of_day": hour_of_day,
            "log_amount": log_amount,
            "oldbalanceOrg": tx.oldbalanceOrg,
            "newbalanceOrig": tx.newbalanceOrig,
            "oldbalanceDest": tx.oldbalanceDest,
            "newbalanceDest": tx.newbalanceDest,
            "is_merchant_dest": tx.is_merchant_dest,
            "errorBalanceOrig": error_orig,
            "errorBalanceDest": error_dest
        }])

        prob = float(model.predict_proba(features_df)[0][1])
        pred = int(prob >= 0.5)

        return {
            "is_fraud": pred,
            "fraud_probability": round(prob, 4),
            "decision": "BLOQUEAR" if pred == 1 else "APROBAR"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno de inferencia: {str(e)}")
