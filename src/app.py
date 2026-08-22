"""app.py — FastAPI service for real-time credit card fraud detection."""

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import pandas as pd
import joblib
from typing import List
# ----------------------------------------------------------------------
# Setup & Model Loading
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_pipeline.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"

# Load artifacts into memory ONE TIME when the server starts
try:
    pipeline = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold = json.load(f)["threshold"]
except FileNotFoundError:
    raise RuntimeError("Model artifacts not found. Please run src/evaluate.py first.")

# ----------------------------------------------------------------------
# Pydantic Input Validation (The Magic)
# ----------------------------------------------------------------------
# We dynamically create a Pydantic model that requires Time, Amount, and V1-V28
# All of them must be float values.
fields = {"Time": (float, ...), "Amount": (float, ...)}
for i in range(1, 29):
    fields[f"V{i}"] = (float, ...)

Transaction = create_model("Transaction", **fields)

# ----------------------------------------------------------------------
# FastAPI App Definition
# ----------------------------------------------------------------------
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="An API that uses a Random Forest pipeline to detect fraudulent transactions.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running! Go to /docs for the Swagger UI."}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    """
    Takes a single transaction payload, validates the 30 features, and returns a prediction.
    """
    try:
        # Convert the validated Pydantic model to a dictionary, then to a DataFrame
        data_dict = transaction.model_dump()
        df = pd.DataFrame([data_dict])
        
        # Predict
        fraud_probability = pipeline.predict_proba(df)[:, 1][0]
        is_fraud = int(fraud_probability >= threshold)
        
        return {
            "prediction": "Fraud" if is_fraud == 1 else "Legitimate",
            "class_id": is_fraud,
            "probability": round(float(fraud_probability), 4),
            "threshold": round(float(threshold), 4),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
def predict_fraud_batch(transactions: List[Transaction]):
    """
    Takes a list of transactions, validates all of them, and returns predictions in bulk.
    """
    try:
        # Convert the list of Pydantic models into a list of dictionaries, then to a single DataFrame
        data_dicts = [t.model_dump() for t in transactions]
        df = pd.DataFrame(data_dicts)
        
        # Predict the whole batch at once
        fraud_probabilities = pipeline.predict_proba(df)[:, 1]
        predictions = (fraud_probabilities >= threshold).astype(int)
        
        # Format the output list
        results = []
        for i in range(len(transactions)):
            results.append({
                "transaction_index": i,
                "prediction": "Fraud" if predictions[i] == 1 else "Legitimate",
                "class_id": int(predictions[i]),
                "probability": round(float(fraud_probabilities[i]), 4),
                "threshold": round(float(threshold), 4),
                "status": "success"
            })
            
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")