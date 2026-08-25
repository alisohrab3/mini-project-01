import argparse
import time
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (
    StratifiedKFold,
    ParameterGrid,
    ParameterSampler,
    cross_val_predict,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import utils



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
            self,
            hidden_sizes=(64, 32),
            lr=1e-3,
            epochs=15,
            batch_size=512,
            dropout=0.0,
            weight_decay=0.0,
            random_state=42,
            verbose=False,
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
                input_dim=input_dim,
                hidden_sizes=list(self.hidden_sizes),
                dropout=self.dropout,
            ).to(self.device_)

        def fit(self, X, y):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

            self.classes_ = np.array([0, 1])

            self._build(X.shape[1])

            dataset = TensorDataset(
                torch.from_numpy(X),
                torch.from_numpy(y),
            )

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )

            loss_function = nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(
                self.model_.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

            self.model_.train()

            for _ in range(self.epochs):
                for X_batch, y_batch in loader:
                    X_batch = X_batch.to(self.device_)
                    y_batch = y_batch.to(self.device_)

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

                    X_batch = torch.from_numpy(
                        X[start_index:end_index]
                    ).to(self.device_)

                    positive_probabilities = torch.sigmoid(
                        self.model_(X_batch)
                    ).cpu().numpy().ravel()

                    actual_end_index = start_index + len(positive_probabilities)

                    probabilities[start_index:actual_end_index, 1] = (
                        positive_probabilities
                    )
                    probabilities[start_index:actual_end_index, 0] = (
                        1 - positive_probabilities
                    )

            return probabilities

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)



PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "smote"
CANDIDATES_DIR = MODEL_DIR / "candidates"
TRAIN_PATH = DATA_DIR / "train.csv"

TARGET = "Class"
RANDOM_STATE = 42


# Preprocessing
def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("robust_scale", RobustScaler(), ["Time", "Amount"]),
            ("pass", "passthrough", [f"V{i}" for i in range(1, 29)]),
        ],
        remainder="drop",
    )


# SMOTE model candidates
def get_candidate_models(preprocessor):


    def make_smote_pipeline(model):
        return Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "smote",
                    SMOTE(
                        sampling_strategy=0.10,
                        k_neighbors=5,
                        random_state=RANDOM_STATE,
                    ),
                ),
                ("model", model),
            ]
        )

    candidates = {
        "XGBoost": {
            "pipeline": make_smote_pipeline(
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1,
                    verbosity=0,
                )
            ),
            "default_params": {
                "smote__sampling_strategy": 0.10,
                "smote__k_neighbors": 5,
                "model__n_estimators": 200,
                "model__max_depth": 4,
                "model__learning_rate": 0.1,
                "model__subsample": 0.8,
                "model__colsample_bytree": 0.8,
            },
            "grid": {
                "smote__sampling_strategy": [0.05, 0.10, 0.25],
                "smote__k_neighbors": [3, 5],
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 4, 6],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
            "random": {
                "smote__sampling_strategy": [0.05, 0.10, 0.25],
                "smote__k_neighbors": [3, 5],
                "model__n_estimators": [100, 150, 200, 300],
                "model__max_depth": [3, 4, 5, 6, 8],
                "model__learning_rate": np.logspace(-2, -0.7, 8).tolist(),
                "model__subsample": [0.7, 0.8, 0.9, 1.0],
                "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            },
            "n_jobs": -1,
        },

        "LogisticRegression": {
            "pipeline": make_smote_pipeline(
                LogisticRegression(
                    class_weight=None,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                )
            ),
            "default_params": {
                "smote__sampling_strategy": 0.10,
                "smote__k_neighbors": 5,
                "model__C": 0.01,
                "model__solver": "lbfgs",
            },
            "grid": {
                "smote__sampling_strategy": [0.05, 0.10, 0.25, 0.5],
                "smote__k_neighbors": [3, 5],
                "model__C": [0.001, 0.01, 0.1, 1.0, 10.0],
                "model__solver": ["liblinear"],
            },
            "random": {
                "smote__sampling_strategy": [0.05, 0.10, 0.25],
                "smote__k_neighbors": [3, 5],
                "model__C": np.logspace(-3, 1, 20).tolist(),
                "model__solver": ["lbfgs", "liblinear"],
            },
            "n_jobs": -1,
        },

        "KNN": {
            "pipeline": make_smote_pipeline(
                KNeighborsClassifier(n_jobs=-1)
            ),
            "default_params": {
                "smote__sampling_strategy": 0.10,
                "smote__k_neighbors": 5,
                "model__n_neighbors": 5,
                "model__weights": "distance",
                "model__p": 2,
            },
            "grid": {
                "smote__sampling_strategy": [0.05, 0.10, 0.25],
                "smote__k_neighbors": [3, 5],
                "model__n_neighbors": [3, 5, 7, 11],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "random": {
                "smote__sampling_strategy": [0.05, 0.10, 0.25],
                "smote__k_neighbors": [3, 5],
                "model__n_neighbors": list(range(3, 16)),
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "n_jobs": -1,
        },

        "DecisionTree": {
            "pipeline": make_smote_pipeline(
                DecisionTreeClassifier(
                    class_weight=None,
                    random_state=RANDOM_STATE,
                )
            ),
            "default_params": {
                "smote__sampling_strategy": 0.10,
                "smote__k_neighbors": 5,
                "model__max_depth": 8,
                "model__min_samples_leaf": 5,
            },
            "grid": {
                "smote__sampling_strategy": [0.05, 0.10, 0.25],
                "smote__k_neighbors": [3, 5],
                "model__max_depth": [4, 6, 8, 12],
                "model__min_samples_leaf": [2, 5, 10],
            },
            "random": {
                "smote__sampling_strategy": [0.05, 0.10, 0.25],
                "smote__k_neighbors": [3, 5],
                "model__max_depth": list(range(3, 16)),
                "model__min_samples_leaf": list(range(1, 20)),
            },
            "n_jobs": -1,
        },

        "RandomForest": {
            "pipeline": make_smote_pipeline(
                RandomForestClassifier(
                    class_weight=None,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
            "default_params": {
                "smote__sampling_strategy": 0.10,
                "smote__k_neighbors": 5,
                "model__n_estimators": 150,
                "model__max_depth": 12,
                "model__min_samples_split": 5,
            },
            "grid": {
                "smote__sampling_strategy": [0.05, 0.10, 0.25],
                "smote__k_neighbors": [3, 5],
                "model__n_estimators": [100, 200],
                "model__max_depth": [6, 10, 15],
                "model__min_samples_split": [2, 5],
            },
            "random": {
                "smote__sampling_strategy": [0.05, 0.10, 0.25],
                "smote__k_neighbors": [3, 5],
                "model__n_estimators": [100, 150, 200, 300],
                "model__max_depth": list(range(5, 20)),
                "model__min_samples_split": [2, 5, 10],
            },
            "n_jobs": -1,
        },
    }

    if TORCH_AVAILABLE:
        candidates["MLP"] = {
            "pipeline": make_smote_pipeline(
                TorchMLPClassifier(random_state=RANDOM_STATE)
            ),
            "default_params": {
                "smote__sampling_strategy": 0.10,
                "smote__k_neighbors": 5,
                "model__hidden_sizes": (64, 32),
                "model__lr": 1e-3,
                "model__epochs": 15,
            },
            "grid": {
                "smote__sampling_strategy": [0.05, 0.10, 0.25],
                "smote__k_neighbors": [3, 5],
                "model__hidden_sizes": [(64, 32), (128, 64)],
                "model__lr": [1e-3, 1e-4],
                "model__epochs": [15],
            },
            "random": {
                "smote__sampling_strategy": [0.05, 0.10, 0.25],
                "smote__k_neighbors": [3, 5],
                "model__hidden_sizes": [
                    (32,),
                    (64,),
                    (64, 32),
                    (128, 64),
                ],
                "model__lr": np.logspace(-4, -2, 8).tolist(),
                "model__dropout": [0.0, 0.3],
                "model__epochs": [10, 20],
            },
            "n_jobs": 1,
        }

    return candidates



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cv-mode",
        choices=["simple", "gridsearch", "randomized"],
        default="gridsearch",
    )

    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
    )

    return parser.parse_args()



def calculate_probability_metrics(y_true, probabilities, threshold=0.5):


    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "accuracy": accuracy_score(y_true, predictions),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }



def main():
    args = parse_args()

    np.random.seed(RANDOM_STATE)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    SMOTE_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "smote"
    utils.set_figure_output_dir(SMOTE_FIGURES_DIR)

    print(f"Loading training data from: {TRAIN_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    print(f"Training rows: {len(train_df):,}")
    print(f"Fraud cases: {int(y_train.sum()):,}")
    print(f"Fraud rate: {y_train.mean():.6%}")

    preprocessor = build_preprocessor()
    all_candidates = get_candidate_models(preprocessor)

    selected_names = (
        args.models
        if args.models
        else list(all_candidates.keys())
    )

    candidates = {
        name: all_candidates[name]
        for name in selected_names
        if name in all_candidates
    }

    if not candidates:
        raise ValueError(
            f"No valid model names selected. Available models: "
            f"{list(all_candidates.keys())}"
        )

    cv = StratifiedKFold(
        n_splits=args.k_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    summary_rows = []

    diagnostics = {
        "_y_true": y_train.to_numpy(),
    }

    for model_name, config in candidates.items():
        print(f"\n=== SMOTE Search: {model_name} ===")

        n_jobs = config.get("n_jobs", -1)

        if args.cv_mode == "simple":
            param_list = [config["default_params"]]

        elif args.cv_mode == "gridsearch":
            param_list = list(ParameterGrid(config["grid"]))

        else:
            param_list = list(
                ParameterSampler(
                    config["random"],
                    n_iter=args.n_iter,
                    random_state=RANDOM_STATE,
                )
            )

        for idx, params in enumerate(param_list):
            print(
                f"  -> Combination {idx + 1}/{len(param_list)}: {params}"
            )

            start_time = time.perf_counter()

            current_pipeline = clone(
                config["pipeline"]
            ).set_params(**params)


            oof_n_jobs = 1 if model_name == "MLP" else n_jobs
            oof_probs = cross_val_predict(
                current_pipeline,
                X_train,
                y_train,
                cv=cv,
                method="predict_proba",
                n_jobs=oof_n_jobs,
            )[:, 1]


            current_pipeline.fit(X_train, y_train)

            train_probs = current_pipeline.predict_proba(X_train)[:, 1]

            elapsed_seconds = time.perf_counter() - start_time

            valid_metrics = calculate_probability_metrics(
                y_train,
                oof_probs,
                threshold=0.5,
            )

            train_metrics = calculate_probability_metrics(
                y_train,
                train_probs,
                threshold=0.5,
            )

            param_str = "_".join(
                [
                    f"{key.split('__')[-1]}={value}"
                    for key, value in params.items()
                ]
            )


            model_filename = (
                f"SMOTE_{model_name}_{param_str}.joblib"
            )

            model_path = MODEL_DIR / model_filename

            joblib.dump(current_pipeline, model_path)

            row = {
                "experiment": "SMOTE",
                "model": f"SMOTE {model_name} (Combo {idx + 1})",
                "family": model_name,
                "combination": idx + 1,
                "saved_filename": model_filename,
                "params": str(params),

 
                "valid_pr_auc": round(valid_metrics["pr_auc"], 6),
                "valid_roc_auc": round(valid_metrics["roc_auc"], 6),
                "valid_precision": round(valid_metrics["precision"], 6),
                "valid_recall": round(valid_metrics["recall"], 6),
                "valid_f1": round(valid_metrics["f1"], 6),
                "valid_accuracy": round(valid_metrics["accuracy"], 6),

                "train_pr_auc": round(train_metrics["pr_auc"], 6),
                "train_roc_auc": round(train_metrics["roc_auc"], 6),
                "train_precision": round(train_metrics["precision"], 6),
                "train_recall": round(train_metrics["recall"], 6),
                "train_f1": round(train_metrics["f1"], 6),
                "train_accuracy": round(train_metrics["accuracy"], 6),

                "pr_auc_gap": round(
                    train_metrics["pr_auc"] - valid_metrics["pr_auc"],
                    6,
                ),
                "roc_auc_gap": round(
                    train_metrics["roc_auc"] - valid_metrics["roc_auc"],
                    6,
                ),
                "precision_gap": round(
                    train_metrics["precision"] - valid_metrics["precision"],
                    6,
                ),
                "recall_gap": round(
                    train_metrics["recall"] - valid_metrics["recall"],
                    6,
                ),
                "f1_gap": round(
                    train_metrics["f1"] - valid_metrics["f1"],
                    6,
                ),


                "valid_false_positives": valid_metrics["false_positives"],
                "valid_false_negatives": valid_metrics["false_negatives"],
                "train_false_positives": train_metrics["false_positives"],
                "train_false_negatives": train_metrics["false_negatives"],

                # Timing
                "training_time_seconds": round(elapsed_seconds, 4),
                "training_time_minutes": round(
                    elapsed_seconds / 60.0,
                    4,
                ),
            }

            summary_rows.append(row)

            diagnostics[model_filename] = {
                "family": model_name,
                "model": row["model"],
                "oof_probs": oof_probs,
                "train_probs": train_probs,
                "valid_pr_auc": valid_metrics["pr_auc"],
            }

    if not summary_rows:
        raise RuntimeError("No models were trained.")

    summary_df = pd.DataFrame(summary_rows)


    summary_df = (
        summary_df
        .sort_values(by="valid_pr_auc", ascending=False)
        .reset_index(drop=True)
    )


    summary_df["best_cv_pr_auc"] = summary_df["valid_pr_auc"]
    summary_df["overfitting_gap"] = summary_df["pr_auc_gap"]
    summary_df["Precision (Fraud)"] = summary_df["valid_precision"]
    summary_df["Recall (Fraud)"] = summary_df["valid_recall"]
    summary_df["F1-Score"] = summary_df["valid_f1"]
    summary_df["False Positives"] = summary_df["valid_false_positives"]
    summary_df["False Negatives"] = summary_df["valid_false_negatives"]


    utils.save_table(
        summary_df,
        "smote_exhaustive_model_comparison",
    )

    top5_overall_df = summary_df.head(5).copy()

    utils.save_table(
        top5_overall_df,
        "smote_top5_overall_models",
    )


    top3_rows = []

    for family in summary_df["family"].unique():
        family_df = (
            summary_df[summary_df["family"] == family]
            .sort_values("valid_pr_auc", ascending=False)
            .head(3)
            .copy()
        )

        for rank, (_, row) in enumerate(
            family_df.iterrows(),
            start=1,
        ):
            source_path = MODEL_DIR / row["saved_filename"]

            candidate_filename = (
                f"SMOTE_{family}_rank{rank}_{source_path.name}"
            )

            destination_path = CANDIDATES_DIR / candidate_filename

            shutil.copy2(source_path, destination_path)

            top3_rows.append({
                "experiment": "SMOTE",
                "family": family,
                "rank_within_family": rank,
                "model": row["model"],
                "source_filename": row["saved_filename"],
                "candidate_filename": candidate_filename,
                "valid_pr_auc": row["valid_pr_auc"],
                "valid_roc_auc": row["valid_roc_auc"],
                "valid_precision": row["valid_precision"],
                "valid_recall": row["valid_recall"],
                "valid_f1": row["valid_f1"],
                "train_pr_auc": row["train_pr_auc"],
                "train_roc_auc": row["train_roc_auc"],
                "train_precision": row["train_precision"],
                "train_recall": row["train_recall"],
                "train_f1": row["train_f1"],
                "pr_auc_gap": row["pr_auc_gap"],
                "roc_auc_gap": row["roc_auc_gap"],
                "precision_gap": row["precision_gap"],
                "recall_gap": row["recall_gap"],
                "f1_gap": row["f1_gap"],
                "training_time_seconds": row[
                    "training_time_seconds"
                ],
            })

    top3_df = pd.DataFrame(top3_rows)

    utils.save_table(
        top3_df,
        "smote_top3_models_per_family",
    )


    ranking_metrics = {
        "pr_auc": "valid_pr_auc",
        "roc_auc": "valid_roc_auc",
        "precision": "valid_precision",
        "recall": "valid_recall",
        "f1": "valid_f1",
    }

    for family in summary_df["family"].unique():
        family_df = summary_df[
            summary_df["family"] == family
        ].copy()

        utils.plot_family_metric_rankings(
            family_df,
            family_name=family,
            metric_columns=ranking_metrics,
        )

        utils.plot_top3_family_comparison(
            family_df,
            family_name=family,
        )

        utils.plot_oof_pr_curves(
            family_df,
            diagnostics=diagnostics,
            family_name=family,
            top_n=3,
        )

        utils.plot_oof_probability_distributions(
            family_df,
            diagnostics=diagnostics,
            family_name=family,
            top_n=3,
        )

        utils.plot_hyperparameter_sweep(
            family_df,
            family_name=family,
        )


    utils.plot_top5_overall_models(summary_df)
    utils.plot_training_time_by_model(summary_df)
    utils.plot_training_time_by_family(
        summary_df,
        table_name="smote_training_time_by_family",
    )

    print("\nSMOTE Training Complete.")
    print("SMOTE models saved in:", MODEL_DIR)
    print("SMOTE top-3 candidates saved in:", CANDIDATES_DIR)
    print("SMOTE tables saved in: reports/tables/ (smote_*.csv / smote_*.md)")
    print(f"SMOTE figures saved in: {SMOTE_FIGURES_DIR.relative_to(PROJECT_ROOT)}/")
    print("Next step: compare baseline and SMOTE OOF metrics,")
    print("then select a candidate for src/evaluate.py.")


if __name__ == "__main__":
    main()
