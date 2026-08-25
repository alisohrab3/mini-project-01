
import json
import logging
from contextlib import asynccontextmanager
from math import isfinite
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_pipeline.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"

EXPECTED_FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]



class Transaction(BaseModel):

    model_config = ConfigDict(extra="forbid")

    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    @field_validator("*")
    @classmethod
    def validate_finite_numbers(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("All features must be finite numeric values.")
        return value



def load_artifacts():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

    pipeline = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold = float(json.load(f)["threshold"])

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("The loaded pipeline does not support predict_proba().")

    if 1 not in pipeline.classes_:
        raise ValueError(
            f"Fraud class label 1 not found in model classes: {pipeline.classes_}"
        )

    return pipeline, threshold


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    try:
        app.state.pipeline, app.state.threshold = load_artifacts()
        app.state.fraud_class_index = list(app.state.pipeline.classes_).index(1)
        logger.info("Fraud model and calibrated threshold loaded successfully.")
    except Exception:
        logger.exception("Could not load model artifacts.")
        raise RuntimeError("Application could not start because model artifacts are invalid.")

    yield



app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for fraud scoring using a trained sklearn pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


def format_prediction(probability: float, threshold: float, index: int | None = None):

    class_id = int(probability >= threshold)

    response = {
        "prediction": "Fraud" if class_id == 1 else "Legitimate",
        "class_id": class_id,
        "probability": round(float(probability), 4),
        "threshold": round(float(threshold), 4),
        "status": "success",
    }

    if index is not None:
        response["transaction_index"] = index

    return response


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running.",
        "docs_url": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        data_dict = transaction.model_dump()
        df = pd.DataFrame([data_dict], columns=EXPECTED_FEATURES)

        probabilities = app.state.pipeline.predict_proba(df)
        fraud_probability = probabilities[0, app.state.fraud_class_index]

        return format_prediction(
            probability=float(fraud_probability),
            threshold=app.state.threshold,
        )

    except Exception:
        logger.exception("Single prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Internal prediction error.",
        )


@app.post("/predict/batch")
def predict_fraud_batch(
    transactions: Annotated[list[Transaction], Field(min_length=1)]
):

    try:
        data_dicts = [transaction.model_dump() for transaction in transactions]
        df = pd.DataFrame(data_dicts, columns=EXPECTED_FEATURES)

        probabilities = app.state.pipeline.predict_proba(df)
        fraud_probabilities = probabilities[:, app.state.fraud_class_index]

        return [
            format_prediction(
                probability=float(probability),
                threshold=app.state.threshold,
                index=index,
            )
            for index, probability in enumerate(fraud_probabilities)
        ]

    except Exception:
        logger.exception("Batch prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Internal batch prediction error.",
        )
