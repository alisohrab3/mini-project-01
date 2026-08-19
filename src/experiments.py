"""experiments.py — standalone analysis experiments (independent of train.py).

Runs:
  1. Effect of scaling (KNN)                    [mandatory]
  2. Hyperparameter analysis (Decision Tree)    [mandatory]
  3. Impact of classification threshold         [mandatory]
  4. Class imbalance handling strategies        [bonus]
  5. Cross-model leaderboard                    [bonus]
  6. Learning curve (training size)             [bonus]

Educational Note:
This script compares model performance on both the internal validation split (Val)
and the held-out test split (Test) to observe generalization gaps.
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    precision_recall_curve, roc_auc_score,
)

# ----------------------------------------------------------------------
# Robust Path Configuration (Works from root or src/ directory)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import utils

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
TARGET = "Class"
RANDOM = 42


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------
def load_experiment_data():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Dataset files not found in: {BASE_DIR / 'data'}\n"
            f"Please run `python src/data_preparation.py` first to generate train.csv and test.csv."
        )
    
    print(f"Loading full training dataset from {TRAIN_PATH}...")
    df_train = pd.read_csv(TRAIN_PATH)
    print(f"Train dataset shape: {df_train.shape} (Fraud ratio: {df_train[TARGET].mean()*100:.4f}%)")

    print(f"Loading test dataset from {TEST_PATH}...")
    df_test = pd.read_csv(TEST_PATH)
    print(f"Test dataset shape:  {df_test.shape} (Fraud ratio: {df_test[TARGET].mean()*100:.4f}%)")

    return df_train, df_test


def get_split(data):
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    # Stratified split using the full training dataset
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM)


def metrics(y_true, y_pred, y_prob=None, prefix=""):
    p_fx = f"{prefix}_" if prefix else ""
    m = {
        f"{p_fx}Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        f"{p_fx}Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        f"{p_fx}F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        m[f"{p_fx}PR-AUC"] = round(average_precision_score(y_true, y_prob), 4)
        m[f"{p_fx}ROC-AUC"] = round(roc_auc_score(y_true, y_prob), 4)
    return m


def eval_val_and_test(model, X_val, y_val, X_test, y_test):
    """Helper to evaluate a model on both Validation and Test splits."""
    yp_val = model.predict(X_val)
    yp_test = model.predict(X_test)

    yprob_val = None
    yprob_test = None
    if hasattr(model, "predict_proba"):
        yprob_val = model.predict_proba(X_val)[:, 1]
        yprob_test = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        yprob_val = model.decision_function(X_val)
        yprob_test = model.decision_function(X_test)

    val_m = metrics(y_val, yp_val, yprob_val, prefix="Val")
    test_m = metrics(y_test, yp_test, yprob_test, prefix="Test")
    return {**val_m, **test_m}


# ----------------------------------------------------------------------
# 1) Effect of scaling
# ----------------------------------------------------------------------
def experiment_scaling(X_tr, X_val, y_tr, y_val, X_test, y_test, X_tr_s, X_val_s, X_test_s):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Effect of Scaling (KNN)")
    print("=" * 60)
    print("Fitting KNN on the full unscaled dataset...")
    knn_raw = KNeighborsClassifier(n_neighbors=5, leaf_size=50, n_jobs=-1).fit(X_tr, y_tr)
    res_raw = eval_val_and_test(knn_raw, X_val, y_val, X_test, y_test)

    print("Fitting KNN on the full scaled dataset...")
    knn_scaled = KNeighborsClassifier(n_neighbors=5, leaf_size=50, n_jobs=-1).fit(X_tr_s, y_tr)
    res_scaled = eval_val_and_test(knn_scaled, X_val_s, y_val, X_test_s, y_test)

    df = pd.DataFrame([
        {"Scaling": "Without Scaling", **res_raw},
        {"Scaling": "With Scaling", **res_scaled},
    ])
    print(df.to_string(index=False))
    utils.save_table(df, "exp1_scaling", sub="experiments/tables")
    utils.plot_grouped_bar_metrics(df, "Scaling", ["Val_Precision", "Val_Recall", "Val_F1", "Test_F1"],
                                   "KNN: Effect of Feature Scaling (Val vs Test)",
                                   sub="experiments/figures", filename="exp1_scaling.png")

    print("\nExplanation:")
    print("  -> KNN relies on Euclidean distance. Without scaling, features with high variance")
    print("     (such as 'Time' and 'Amount') dwarf the PCA-transformed 'V1' to 'V28' features,")
    print("     rendering the distance metric ineffective for identifying fraudulent neighbors.")
    print("  -> Decision Trees split nodes based on single-feature thresholds. Because each feature")
    print("     is split independently, monotonic scaling does not change the mathematical properties")
    print("     of the split, making decision trees scale-invariant.")
    return df


# ----------------------------------------------------------------------
# 2) Hyperparameter analysis (Decision Tree max_depth)
# ----------------------------------------------------------------------
def experiment_hyperparameter(X_tr_s, X_val_s, y_tr, y_val, X_test_s, y_test):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Hyperparameter Analysis (Decision Tree max_depth)")
    print("=" * 60)

    rows = []
    for md in [2, 5, 10, None]:
        print(f"Training Decision Tree with max_depth={md}...")
        dt = DecisionTreeClassifier(max_depth=md, class_weight="balanced", random_state=RANDOM).fit(X_tr_s, y_tr)
        res = eval_val_and_test(dt, X_val_s, y_val, X_test_s, y_test)
        rows.append({"max_depth": str(md), **res})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    utils.save_table(df, "exp2_hyperparameter", sub="experiments/tables")
    utils.plot_metric_vs_category(df, "max_depth", ["Val_F1", "Test_F1", "Val_PR-AUC", "Test_PR-AUC"],
                                  "Decision Tree: Effect of max_depth (Val vs Test)",
                                  sub="experiments/figures", filename="exp2_hyperparameter.png")

    print("\nAnalysis:")
    print("  -> Notice how at very high depth, the gap between training fit and test generalization can widen.")
    print("  -> Best Balance: A moderate max_depth prevents tree over-expansion while maintaining a strong balance.")
    return df


# ----------------------------------------------------------------------
# 3) Impact of classification threshold
# ----------------------------------------------------------------------
def experiment_threshold(X_tr_s, X_val_s, y_tr, y_val, X_test_s, y_test):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Impact of Classification Threshold")
    print("=" * 60)

    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM).fit(X_tr_s, y_tr)
    yprob_val = lr.predict_proba(X_val_s)[:, 1]
    yprob_test = lr.predict_proba(X_test_s)[:, 1]

    rows = []
    for t in [0.3, 0.5, 0.7]:
        yp_val = (yprob_val >= t).astype(int)
        yp_test = (yprob_test >= t).astype(int)
        val_m = metrics(y_val, yp_val, prefix="Val")
        test_m = metrics(y_test, yp_test, prefix="Test")
        rows.append({"Threshold": t, **val_m, **test_m})
        
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    utils.save_table(df, "exp3_threshold", sub="experiments/tables")

    prec, rec, th = precision_recall_curve(y_val, yprob_val)
    utils.plot_threshold_curve(th, prec, rec, chosen=None,
                               sub="experiments/figures", filename="exp3_threshold.png")

    print("\nExplanation:")
    print("  -> When the threshold decreases, Recall increases (we catch more true fraud cases) but")
    print("     Precision decreases (we generate more false alarms).")
    print("  -> Recommendation: A lower threshold (e.g., 0.3) is recommended for fraud detection because the")
    print("     financial and reputational cost of a missed fraud (false negative) is typically far higher")
    print("     than the operational cost of manually reviewing a false positive.")
    return df


# ----------------------------------------------------------------------
# 4) Class imbalance handling (bonus)
# ----------------------------------------------------------------------
def oversample_minority(X, y):
    X = np.asarray(X); y = np.asarray(y)
    pos, neg = X[y == 1], X[y == 0]
    n_neg, n_pos = len(neg), len(pos)
    idx = np.random.default_rng(RANDOM).integers(0, n_pos, size=max(0, n_neg - n_pos))
    pos_up = np.vstack([pos, pos[idx]])
    X_new = np.vstack([pos_up, neg])
    y_new = np.concatenate([np.ones(len(pos_up)), np.zeros(n_neg)])
    return X_new, y_new


def experiment_class_imbalance(X_tr_s, X_val_s, y_tr, y_val, X_test_s, y_test):
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Class Imbalance Handling (bonus)")
    print("=" * 60)

    rows = []
    # (a) baseline
    print("Training baseline Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, random_state=RANDOM).fit(X_tr_s, y_tr)
    rows.append({"Strategy": "Baseline (no handling)", **eval_val_and_test(lr, X_val_s, y_val, X_test_s, y_test)})

    # (b) class_weight
    print("Training class_weight='balanced' Logistic Regression...")
    lr_b = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM).fit(X_tr_s, y_tr)
    rows.append({"Strategy": "class_weight='balanced'", **eval_val_and_test(lr_b, X_val_s, y_val, X_test_s, y_test)})

    # (c) random oversampling of minority class
    print("Oversampling minority class and training...")
    X_up, y_up = oversample_minority(X_tr_s, y_tr)
    lr_o = LogisticRegression(max_iter=2000, random_state=RANDOM).fit(X_up, y_up)
    rows.append({"Strategy": "Random oversampling", **eval_val_and_test(lr_o, X_val_s, y_val, X_test_s, y_test)})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    utils.save_table(df, "exp4_imbalance", sub="experiments/tables")
    utils.plot_grouped_bar_metrics(df, "Strategy", ["Val_F1", "Test_F1", "Val_PR-AUC", "Test_PR-AUC"],
                                   "Effect of Class Imbalance Handling (Val vs Test)",
                                   sub="experiments/figures", filename="exp4_imbalance.png")
    return df


# ----------------------------------------------------------------------
# 5) Cross-model leaderboard (bonus)
# ----------------------------------------------------------------------
def experiment_model_comparison(X_tr_s, X_val_s, y_tr, y_val, X_test_s, y_test):
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Cross-Model Leaderboard (bonus)")
    print("=" * 60)

    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM),
        "DecisionTree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM),
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=12,
                                               class_weight="balanced", random_state=RANDOM, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5, leaf_size=50, n_jobs=-1),
    }

    try:
        from train import TorchMLPClassifier
        models["MLP"] = TorchMLPClassifier(hidden_sizes=(64, 32), lr=1e-3, epochs=12, random_state=RANDOM)
    except Exception:
        pass

    rows = []
    for name, model in models.items():
        print(f"Training {name} on full dataset...")
        model.fit(X_tr_s, y_tr)
        res = eval_val_and_test(model, X_val_s, y_val, X_test_s, y_test)
        rows.append({"Model": name, **res})

    df = pd.DataFrame(rows).sort_values("Val_PR-AUC", ascending=False)
    print(df.to_string(index=False))
    utils.save_table(df, "exp5_model_leaderboard", sub="experiments/tables")
    utils.plot_grouped_bar_metrics(df, "Model", ["Val_F1", "Test_F1", "Val_PR-AUC", "Test_PR-AUC"],
                                   "Model Family Comparison (Val vs Test)",
                                   sub="experiments/figures", filename="exp5_leaderboard.png")
    return df


# ----------------------------------------------------------------------
# 6) Learning curve (bonus)
# ----------------------------------------------------------------------
def experiment_learning_curve(X_full, y_full):
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Learning Curve (bonus)")
    print("=" * 60)
    print("Calculating learning curves using 5-fold CV across training subsets...")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_full, y_full, train_sizes=np.linspace(0.1, 1.0, 5),
        cv=cv, scoring="average_precision", n_jobs=-1, random_state=RANDOM
    )
    df = pd.DataFrame({
        "train_size": train_sizes.astype(int),
        "train_pr_auc_mean": train_scores.mean(axis=1).round(4),
        "cv_pr_auc_mean": val_scores.mean(axis=1).round(4),
        "cv_pr_auc_std": val_scores.std(axis=1).round(4),
    })
    print(df.to_string(index=False))
    utils.save_table(df, "exp6_learning_curve", sub="experiments/tables")
    utils.plot_learning_curve(train_sizes, train_scores, val_scores,
                              sub="experiments/figures", filename="exp6_learning_curve.png")
    return df


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run fraud-detection experiments.")
    p.add_argument("--only", type=int, nargs="*", default=None,
                   help="Run only selected experiments (e.g. --only 1 3 5).")
    return p.parse_args()


def main():
    args = parse_args()
    df_train, df_test = load_experiment_data()
    
    X_tr, X_val, y_tr, y_val = get_split(df_train)
    X_test = df_test.drop(columns=[TARGET])
    y_test = df_test[TARGET]

    print("Fitting Scaler on training split and transforming val & test...")
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    X_full = df_train.drop(columns=[TARGET])
    y_full = df_train[TARGET]

    exps = {
        1: lambda: experiment_scaling(X_tr, X_val, y_tr, y_val, X_test, y_test, X_tr_s, X_val_s, X_test_s),
        2: lambda: experiment_hyperparameter(X_tr_s, X_val_s, y_tr, y_val, X_test_s, y_test),
        3: lambda: experiment_threshold(X_tr_s, X_val_s, y_tr, y_val, X_test_s, y_test),
        4: lambda: experiment_class_imbalance(X_tr_s, X_val_s, y_tr, y_val, X_test_s, y_test),
        5: lambda: experiment_model_comparison(X_tr_s, X_val_s, y_tr, y_val, X_test_s, y_test),
        6: lambda: experiment_learning_curve(X_full, y_full),
    }
    
    only = args.only if args.only else sorted(exps)
    for num in only:
        exps[num]()

    print("\nAll requested experiments complete. Figures & tables are under reports/experiments/")


if __name__ == "__main__":
    main()
