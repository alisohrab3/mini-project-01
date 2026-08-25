import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_pipeline.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"

EXPECTED_FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def load_artifacts():
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

    pipeline = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    threshold = float(threshold_data["threshold"])

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid threshold: {threshold}. It must be between 0 and 1.")

    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Loaded model pipeline does not support predict_proba().")

    if 1 not in pipeline.classes_:
        raise ValueError(
            f"Fraud class label 1 was not found in model classes: {pipeline.classes_}"
        )

    return pipeline, threshold


def validate_input(data):
    
    if not isinstance(data, dict):
        raise ValueError("Each transaction must be a JSON object.")

    missing = [feature for feature in EXPECTED_FEATURES if feature not in data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    cleaned = {}

    for feature in EXPECTED_FEATURES:
        try:
            value = float(data[feature])
        except (TypeError, ValueError):
            raise ValueError(
                f"Feature '{feature}' must be numeric; got {data[feature]!r}."
            )

        if not math.isfinite(value):
            raise ValueError(
                f"Feature '{feature}' must be finite, not NaN or infinity."
            )

        cleaned[feature] = value

    return cleaned


def predict_batch(input_list, pipeline, threshold):
    
    results = [None] * len(input_list)
    valid_data = []
    valid_indices = []

    for idx, record in enumerate(input_list):
        try:
            valid_data.append(validate_input(record))
            valid_indices.append(idx)
        except (ValueError, TypeError) as e:
            results[idx] = {
                "record_index": idx,
                "status": "error",
                "message": str(e),
            }

    if not valid_data:
        return results

    df = pd.DataFrame(valid_data, columns=EXPECTED_FEATURES)

    fraud_class_index = list(pipeline.classes_).index(1)
    fraud_probabilities = pipeline.predict_proba(df)[:, fraud_class_index]
    predictions = (fraud_probabilities >= threshold).astype(int)

    for i, original_idx in enumerate(valid_indices):
        probability = float(fraud_probabilities[i])
        class_id = int(predictions[i])

        results[original_idx] = {
            "record_index": original_idx,
            "prediction": "Fraud" if class_id == 1 else "Legitimate",
            "class_id": class_id,
            "probability": round(probability, 4),
            "threshold": round(float(threshold), 4),
            "status": "success",
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict credit-card fraud from a JSON file."
    )
    parser.add_argument("input_file", type=str, help="Path to input JSON file.")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    if isinstance(input_data, dict):
        input_data = [input_data]

    if not isinstance(input_data, list):
        raise ValueError(
            "Input JSON must be one transaction object or a list of transaction objects."
        )

    pipeline, threshold = load_artifacts()
    result = predict_batch(input_data, pipeline, threshold)

    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
