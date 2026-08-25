import json
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)
import utils

try:
    import train
    from train import TorchMLPClassifier, MLPModule
    sys.modules['__main__'].TorchMLPClassifier = TorchMLPClassifier
    sys.modules['__main__'].MLPModule = MLPModule
except (ImportError, AttributeError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
EVAL_FIGURES_DIR = REPORT_DIR / "figures" / "eval"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TARGET = "Class"
RANDOM_STATE = 42

def find_optimal_threshold(y_true, y_probabilities, min_recall=0.80):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probabilities)
    
    p_valid = precision[:-1]
    r_valid = recall[:-1]
    
    f1_scores = 2 * (p_valid * r_valid) / (p_valid + r_valid + 1e-12)
    
   
    valid_mask = (r_valid >= min_recall) & (thresholds > 0.01)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(f1_scores[valid_indices])]
    else:
        nonzero_mask = thresholds > 0.01
        nonzero_indices = np.where(nonzero_mask)[0]
        if len(nonzero_indices) > 0:
            best_idx = nonzero_indices[np.argmax(f1_scores[nonzero_indices])]
        else:
            best_idx = np.argmax(f1_scores)

    return float(thresholds[best_idx]), float(p_valid[best_idx]), float(r_valid[best_idx]), float(f1_scores[best_idx])

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a manually selected model.")
    parser.add_argument("--model", required=True, help="Filename of the model in the models/ folder.")
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = MODEL_DIR / args.model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model: {model_path}")

  
    utils.set_figure_output_dir(EVAL_FIGURES_DIR)

    print(f"Loading selected model: {args.model}")
    pipeline = joblib.load(model_path)

    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

   
    print("\nfinding best decision threshold with Out-Of-Fold predictions...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    
    threshold, threshold_precision, threshold_recall, threshold_f1 = find_optimal_threshold(y_train, oof_probs, min_recall=0.80)
    print(f"Chosen threshold = {threshold:.4f} (Recall={threshold_recall:.4f}, Precision={threshold_precision:.4f}, F1={threshold_f1:.4f})")

    precision, recall, threshold_values = precision_recall_curve(y_train, oof_probs)
    utils.plot_threshold_curve(threshold_values, precision, recall, chosen=threshold)

  
    print("\n--- Final evaluation on untouched Test Set ---")
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    test_pr_auc = average_precision_score(y_test, test_probabilities)
    test_roc_auc = roc_auc_score(y_test, test_probabilities)
    cm = confusion_matrix(y_test, test_predictions)

    print(f"Test PR-AUC  = {test_pr_auc:.4f} | Test ROC-AUC = {test_roc_auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, test_predictions, digits=4, zero_division=0))

    utils.plot_confusion_matrix(y_test, test_predictions)
    utils.plot_roc_curve(y_test, test_probabilities)
    utils.plot_pr_curve(y_test, test_probabilities)


    main_model_path = MODEL_DIR / "fraud_pipeline.joblib"
    joblib.dump(pipeline, main_model_path)

    with open(MODEL_DIR / "threshold.json", "w", encoding="utf-8") as file:
        json.dump({"threshold": threshold, "model_name": args.model}, file, indent=2)

    with open(REPORT_DIR / "final_test_metrics.json", "w", encoding="utf-8") as file:
        json.dump({
            "model_name": args.model,
            "test_pr_auc": test_pr_auc,
            "threshold": threshold,
            "confusion_matrix": cm.tolist(),
        }, file, indent=2)
        
    print("\nModel is successfully prepared for the final part which is predict.py")

if __name__ == "__main__":
    main()







































