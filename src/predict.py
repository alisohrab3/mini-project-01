"""predict.py — batch prediction script with strict input validation."""

import json
import argparse
from pathlib import Path
import pandas as pd
import joblib

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_pipeline.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"

# The exact 30 features the pipeline expects
EXPECTED_FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

def load_artifacts():
    """Loads the trained pipeline and the calibrated threshold."""
    if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Please run evaluate.py first.")
    
    pipeline = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)
        
    return pipeline, threshold_data["threshold"]

def validate_input(data: dict):
    """Checks if all required columns are present in the dictionary."""
    missing = [f for f in EXPECTED_FEATURES if f not in data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    # Return a clean dictionary containing ONLY the expected features (drops unexpected extra columns)
    return {f: data[f] for f in EXPECTED_FEATURES}

def predict_batch(input_list: list, pipeline, threshold: float) -> list:
    """Takes a list of feature dictionaries, validates them, and processes predictions in a batch."""
    results = []
    valid_data = []
    valid_indices = []

    # 1. Validate all inputs
    for idx, record in enumerate(input_list):
        try:
            valid_record = validate_input(record)
            valid_data.append(valid_record)
            valid_indices.append(idx)
            # Placeholder for the result to maintain order
            results.append(None) 
        except ValueError as e:
            # If a record is invalid, log the error but don't crash the script
            results.append({
                "record_index": idx,
                "status": "error",
                "message": str(e)
            })

    # If absolutely no valid data was found, return early
    if not valid_data:
        return results

    # 2. Convert to DataFrame for batch processing
    df = pd.DataFrame(valid_data)
    
    # 3. Predict probabilities in one highly-efficient batch
    fraud_probabilities = pipeline.predict_proba(df)[:, 1]
    predictions = (fraud_probabilities >= threshold).astype(int)
    
    # 4. Format the final output
    for i, original_idx in enumerate(valid_indices):
        prob = fraud_probabilities[i]
        is_fraud = predictions[i]
        
        results[original_idx] = {
            "record_index": original_idx,
            "prediction": "Fraud" if is_fraud == 1 else "Legitimate",
            "class_id": int(is_fraud),
            "probability": round(float(prob), 4),
            "threshold": round(float(threshold), 4),
            "status": "success"
        }

    return results

def main():
    parser = argparse.ArgumentParser(description="Predict fraud from a batch JSON file.")
    parser.add_argument("input_file", type=str, help="Path to the input.json file")
    args = parser.parse_args()

    # Read the input JSON
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Force single dictionaries into a list so the batch logic always works
    if isinstance(input_data, dict):
        input_data = [input_data]

    # Load model and predict
    pipeline, threshold = load_artifacts()
    result = predict_batch(input_data, pipeline, threshold)

    # Print the output nicely formatted
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()