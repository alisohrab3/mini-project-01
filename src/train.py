"""train.py — exhaustively train, evaluate, and export all fraud model combinations.

Usage (run from the project root):
    python src/train.py --cv-mode simple
    python src/train.py --cv-mode gridsearch
    python src/train.py --cv-mode randomized --n-iter 10
"""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, ParameterGrid, ParameterSampler, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import utils

# ----------------------------------------------------------------------
# PyTorch / MLP model
# ----------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    print("[warn] PyTorch not found — the MLP candidate will be skipped.")

if TORCH_AVAILABLE:
    class MLPModule(nn.Module):
        def __init__(self, input_dim, hidden_sizes, dropout):
            super().__init__()
            layers = []
            previous_size = input_dim
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(previous_size, hidden_size))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                previous_size = hidden_size
            layers.append(nn.Linear(previous_size, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
        def __init__(
            self, hidden_sizes=(64, 32), lr=1e-3, epochs=15, batch_size=512,
            dropout=0.0, weight_decay=0.0, random_state=42, verbose=False,
        ):
            self.hidden_sizes = hidden_sizes
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.dropout = dropout
            self.weight_decay = weight_decay
            self.random_state = random_state
            self.verbose = verbose

        def _build(self, input_dim):
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_ = MLPModule(
                input_dim=input_dim, hidden_sizes=list(self.hidden_sizes), dropout=self.dropout,
            ).to(self.device_)

        def fit(self, X, y):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            self.classes_ = np.array([0, 1])
            self._build(X.shape[1])

            dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            number_positive = int(y.sum())
            number_negative = len(y) - number_positive
            pos_weight = torch.tensor([number_negative / max(number_positive, 1)], dtype=torch.float32).to(self.device_)
            loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            self.model_.train()
            for epoch in range(self.epochs):
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(self.device_), y_batch.to(self.device_)
                    optimizer.zero_grad()
                    logits = self.model_(X_batch)
                    loss = loss_function(logits, y_batch)
                    loss.backward()
                    optimizer.step()
            return self

        def predict_proba(self, X):
            X = np.asarray(X, dtype=np.float32)
            self.model_.eval()
            probabilities = np.zeros((len(X), 2), dtype=np.float32)
            with torch.no_grad():
                for start_index in range(0, len(X), self.batch_size):
                    end_index = start_index + self.batch_size
                    X_batch = torch.from_numpy(X[start_index:end_index]).to(self.device_)
                    positive_probabilities = torch.sigmoid(self.model_(X_batch)).cpu().numpy().ravel()
                    probabilities[start_index:start_index + len(positive_probabilities), 1] = positive_probabilities
                    probabilities[start_index:start_index + len(positive_probabilities), 0] = 1 - positive_probabilities
            return probabilities

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# ----------------------------------------------------------------------
# Project paths and configuration
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
TRAIN_PATH = DATA_DIR / "train.csv"
TARGET = "Class"
RANDOM_STATE = 42

def build_preprocessor():
    return ColumnTransformer([
        ("robust_scale", RobustScaler(), ["Time", "Amount"]),
        ("pass", "passthrough", [f"V{i}" for i in range(1, 29)]),
    ], remainder="drop")

def get_candidate_models(preprocessor):
    candidates = {
        "LogisticRegression": {
            "pipeline": Pipeline([("prep", preprocessor), ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE))]),
            "default_params": {"model__C": 1.0, "model__solver": "lbfgs"},
            "grid": {"model__C": [0.01, 0.1, 1.0, 10.0], "model__solver": ["lbfgs", "liblinear"]},
            "random": {"model__C": np.logspace(-3, 1, 20).tolist()},
            "n_jobs": -1,
        },
        "KNN": {
            "pipeline": Pipeline([("prep", preprocessor), ("model", KNeighborsClassifier(n_jobs=-1))]),
            "default_params": {"model__n_neighbors": 5, "model__weights": "distance", "model__p": 2},
            "grid": {"model__n_neighbors": [3, 5, 7, 11], "model__weights": ["uniform", "distance"], "model__p": [1, 2]},
            "random": {"model__n_neighbors": list(range(3, 16)), "model__weights": ["uniform", "distance"], "model__p": [1, 2]},
            "n_jobs": -1,
        },
        "DecisionTree": {
            "pipeline": Pipeline([("prep", preprocessor), ("model", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE))]),
            "default_params": {"model__max_depth": 8, "model__min_samples_leaf": 5},
            "grid": {"model__max_depth": [4, 6, 8, 12], "model__min_samples_leaf": [2, 5, 10]},
            "random": {"model__max_depth": list(range(3, 16)), "model__min_samples_leaf": list(range(1, 20))},
            "n_jobs": -1,
        },
        "RandomForest": {
            "pipeline": Pipeline([("prep", preprocessor), ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
            "default_params": {"model__n_estimators": 150, "model__max_depth": 12, "model__min_samples_split": 5},
            "grid": {"model__n_estimators": [100, 200], "model__max_depth": [6, 10, 15], "model__min_samples_split": [2, 5]},
            "random": {"model__n_estimators": [100, 150, 200, 300], "model__max_depth": list(range(5, 20)), "model__min_samples_split": [2, 5, 10]},
            "n_jobs": -1,
        },
    }

    if TORCH_AVAILABLE:
        candidates["MLP"] = {
            "pipeline": Pipeline([("prep", preprocessor), ("model", TorchMLPClassifier(random_state=RANDOM_STATE))]),
            "default_params": {"model__hidden_sizes": (64, 32), "model__lr": 1e-3, "model__epochs": 15},
            "grid": {"model__hidden_sizes": [(64, 32), (128, 64)], "model__lr": [1e-3, 1e-4], "model__epochs": [15]},
            "random": {"model__hidden_sizes": [(32,), (64,), (64, 32), (128, 64)], "model__lr": np.logspace(-4, -2, 8).tolist(), "model__dropout": [0.0, 0.3], "model__epochs": [10, 20]},
            "n_jobs": 1,
        }
    
    return candidates

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-mode", choices=["simple", "gridsearch", "randomized"], default="gridsearch")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(RANDOM_STATE)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from: {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]

    preprocessor = build_preprocessor()
    all_candidates = get_candidate_models(preprocessor)
    selected_names = args.models if args.models else list(all_candidates.keys())
    candidates = {name: all_candidates[name] for name in selected_names if name in all_candidates}

    cv = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=RANDOM_STATE)
    summary_rows = []

    for model_name, config in candidates.items():
        print(f"\n=== Exhaustive Search: {model_name} ===")
        n_jobs = config.get("n_jobs", -1)
        
        if args.cv_mode == "simple":
            param_list = [config["default_params"]]
        elif args.cv_mode == "gridsearch":
            param_list = list(ParameterGrid(config["grid"]))
        else:
            param_list = list(ParameterSampler(config["random"], n_iter=args.n_iter, random_state=RANDOM_STATE))

        for idx, params in enumerate(param_list):
            print(f"  -> Combination {idx+1}/{len(param_list)}: {params}")
            
            current_pipeline = clone(config["pipeline"]).set_params(**params)
            
            # Generate OOF predictions
            oof_n_jobs = 1 if model_name == "MLP" else n_jobs
            oof_probs = cross_val_predict(current_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=oof_n_jobs)[:, 1]
            cv_pr_auc = average_precision_score(y_train, oof_probs)
            
            # Fit on entire training set
            current_pipeline.fit(X_train, y_train)
            train_probs = current_pipeline.predict_proba(X_train)[:, 1]
            train_pr_auc = average_precision_score(y_train, train_probs)
            
            overfitting_gap = train_pr_auc - cv_pr_auc
            param_str = "_".join([f"{k.split('__')[-1]}={v}" for k, v in params.items()])
            model_filename = f"{model_name}_{param_str}.joblib"
            
            joblib.dump(current_pipeline, MODEL_DIR / model_filename)

            # Baseline metrics for CSV
            oof_preds = (oof_probs >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_train, oof_preds).ravel()
            
            summary_rows.append({
                "model": f"{model_name} (Combo {idx+1})",
                "family": model_name,
                "saved_filename": model_filename,
                "best_cv_pr_auc": round(cv_pr_auc, 4),
                "train_pr_auc": round(train_pr_auc, 4),
                "overfitting_gap": round(overfitting_gap, 4),
                "Accuracy": round(accuracy_score(y_train, oof_preds), 4),
                "Precision (Fraud)": round(precision_score(y_train, oof_preds, zero_division=0), 4),
                "Recall (Fraud)": round(recall_score(y_train, oof_preds, zero_division=0), 4),
                "F1-Score": round(f1_score(y_train, oof_preds, zero_division=0), 4),
                "False Positives": fp,
                "False Negatives": fn,
                "params": str(params),
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="best_cv_pr_auc", ascending=False)
    utils.save_table(summary_df, "exhaustive_model_comparison")
    
    # ---------------------------------------------------------
    # GENERATE THE NEW PER-FAMILY PLOTS
    # ---------------------------------------------------------
    for family in summary_df["family"].unique():
        family_df = summary_df[summary_df["family"] == family]
        utils.plot_hyperparameter_sweep(family_df, family_name=family)

    print("\nTraining Complete! Check reports/figures/ for detailed hyperparameter sweep plots.")
    print("Next step: Manually select your best model and run src/evaluate.py")

if __name__ == "__main__":
    main()