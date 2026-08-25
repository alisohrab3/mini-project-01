import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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



BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

REPORTS_DIR = BASE_DIR / "reports"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"
TABLES_DIR = EXPERIMENTS_DIR / "tables"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
TARGET = "Class"
RANDOM = 42



def save_experiment_table(df: pd.DataFrame, filename: str):
    name = Path(filename).stem + ".csv"
    output_path = TABLES_DIR / name
    df.to_csv(output_path, index=False)
    print(f"[Saved Table]  -> {output_path}")
    return output_path


def save_experiment_figure(fig: plt.Figure, filename: str, dpi=300):
    name = Path(filename).stem + ".png"
    output_path = FIGURES_DIR / name
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[Saved Figure] -> {output_path}")
    return output_path


def _plot_grouped_bar_metrics(df, group_col, metrics_list, title, filename="metric_comparison.png"):
    x = np.arange(len(df))
    n_metrics = len(metrics_list)
    width = 0.8 / max(1, n_metrics)
    # colors = ["#1565c0", "#2e7d32", "#e53935", "#8e24aa", "#ef6c00"]
    colors = ["royalblue", "forestgreen", "crimson", "darkorchid", "darkorange"]
    
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 2), 6))
    for i, m_col in enumerate(metrics_list):
        if m_col not in df.columns:
            continue
        vals = df[m_col].to_numpy()
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=m_col, color=colors[i % len(colors)])
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df[group_col].astype(str), fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_experiment_figure(fig, filename)


def _plot_learning_curve(train_sizes, train_scores, val_scores, filename="exp6_learning_curve.png"):
    tr_mean = np.mean(train_scores, axis=1)
    tr_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, tr_mean, "o-", color="#e53935", label="Training PR-AUC")
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="#e53935")
    ax.plot(train_sizes, val_mean, "s-", color="#1565c0", label="CV PR-AUC")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#1565c0")

    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Average Precision (PR-AUC)")
    ax.set_title("Logistic Regression Learning Curve (5-Fold CV)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.35)
    save_experiment_figure(fig, filename)


def _plot_threshold_curve(y_true, y_prob, chosen_thresholds=None, filename="exp3_threshold.png"):
    prec, rec, th = precision_recall_curve(y_true, y_prob)
    p_curve = prec[:-1]
    r_curve = rec[:-1]
    f1_curve = (2 * p_curve * r_curve) / (p_curve + r_curve + 1e-12)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(th, p_curve, label="Precision", color="#1565c0", lw=2)
    ax.plot(th, r_curve, label="Recall", color="#2e7d32", lw=2)
    ax.plot(th, f1_curve, label="F1-Score", color="#e53935", lw=2)

    if chosen_thresholds:
        for t in chosen_thresholds:
            ax.axvline(t, linestyle="--", color="gray", alpha=0.7, label=f"Threshold = {t}")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Impact of Decision Threshold (Precision / Recall / F1)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    save_experiment_figure(fig, filename)



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



def experiment_scaling(X_tr, X_val, y_tr, y_val, X_test, y_test, X_tr_s, X_val_s, X_test_s):
    print("\n" + "=" * 65)
    print("MANDATORY EXPERIMENT 1: Effect of Scaling (KNN)")
    print("=" * 65)

    print("Fitting KNN on unscaled data (Raw)...")
    knn_raw = KNeighborsClassifier(n_neighbors=5, leaf_size=50, n_jobs=-1).fit(X_tr, y_tr)
    res_raw = eval_val_and_test(knn_raw, X_val, y_val, X_test, y_test)

    print("Fitting KNN on scaled data (StandardScaler)...")
    knn_scaled = KNeighborsClassifier(n_neighbors=5, leaf_size=50, n_jobs=-1).fit(X_tr_s, y_tr)
    res_scaled = eval_val_and_test(knn_scaled, X_val_s, y_val, X_test_s, y_test)

    df = pd.DataFrame([
        {"Model": "KNN", "Scaling": "Without Scaling", **res_raw},
        {"Model": "KNN", "Scaling": "With Scaling", **res_scaled},
    ])
    print(df.to_string(index=False))
    save_experiment_table(df, "exp1_scaling.csv")
    _plot_grouped_bar_metrics(
        df, "Scaling", ["Val_Precision", "Val_Recall", "Val_F1", "Test_F1"],
        "KNN: Effect of Feature Scaling (Val vs Test)",
        filename="exp1_scaling.png"
    )


    return df



def experiment_hyperparameter(X_tr_s, X_val_s, y_tr, y_val, X_test_s, y_test):
    print("\n" + "=" * 65)
    print("MANDATORY EXPERIMENT 2: Hyperparameter Analysis (Decision Tree max_depth)")
    print("=" * 65)

    rows = []
    for md in [2, 5, 10, None]:
        print(f"Training Decision Tree with max_depth={md}...")
        dt = DecisionTreeClassifier(max_depth=md, class_weight="balanced", random_state=RANDOM).fit(X_tr_s, y_tr)
        res = eval_val_and_test(dt, X_val_s, y_val, X_test_s, y_test)
        rows.append({"max_depth": str(md), **res})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    save_experiment_table(df, "exp2_hyperparameter.csv")
    _plot_grouped_bar_metrics(
        df, "max_depth", ["Val_F1", "Test_F1", "Val_PR-AUC", "Test_PR-AUC"],
        "Decision Tree: Effect of max_depth (Val vs Test)",
        filename="exp2_hyperparameter.png"
    )

    return df



def experiment_threshold(X_tr_s, X_val_s, y_tr, y_val, X_test_s, y_test):
    print("\n" + "=" * 65)
    print("MANDATORY EXPERIMENT 3: Impact of Classification Threshold")
    print("=" * 65)

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
    save_experiment_table(df, "exp3_threshold.csv")

    _plot_threshold_curve(
        y_val, yprob_val, chosen_thresholds=[0.3, 0.5, 0.7], filename="exp3_threshold.png"
    )


    return df



def oversample_minority(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    pos, neg = X[y == 1], X[y == 0]
    n_neg, n_pos = len(neg), len(pos)
    idx = np.random.default_rng(RANDOM).integers(0, n_pos, size=max(0, n_neg - n_pos))
    pos_up = np.vstack([pos, pos[idx]])
    X_new = np.vstack([pos_up, neg])
    y_new = np.concatenate([np.ones(len(pos_up)), np.zeros(n_neg)])
    return X_new, y_new


def experiment_class_imbalance(X_tr_s, X_val_s, y_tr, y_val, X_test_s, y_test):
    print("\n" + "=" * 65)
    print("EXPERIMENT 4: Class Imbalance Handling (Balanced Weights vs Oversampling)")
    print("=" * 65)

    rows = []
    print("Training Logistic Regression with class_weight='balanced'...")
    lr_b = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM).fit(X_tr_s, y_tr)
    rows.append({
        "Strategy": "class_weight='balanced'",
        **eval_val_and_test(lr_b, X_val_s, y_val, X_test_s, y_test)
    })

    print("Training Logistic Regression with Random Oversampling...")
    X_up, y_up = oversample_minority(X_tr_s, y_tr)
    lr_o = LogisticRegression(max_iter=2000, random_state=RANDOM).fit(X_up, y_up)
    rows.append({
        "Strategy": "Random Oversampling",
        **eval_val_and_test(lr_o, X_val_s, y_val, X_test_s, y_test)
    })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    save_experiment_table(df, "exp4_imbalance.csv")
    _plot_grouped_bar_metrics(
        df, "Strategy", ["Val_F1", "Test_F1", "Val_PR-AUC", "Test_PR-AUC"],
        "Comparison of Imbalance Handling Methods (Val vs Test)",
        filename="exp4_imbalance.png"
    )
    return df



def experiment_model_comparison(X_tr_s, X_val_s, y_tr, y_val, X_test_s, y_test):
    print("\n" + "=" * 65)
    print("EXPERIMENT 5: Cross-Model Leaderboard")
    print("=" * 65)

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
        print(f"Training {name}...")
        model.fit(X_tr_s, y_tr)
        res = eval_val_and_test(model, X_val_s, y_val, X_test_s, y_test)
        rows.append({"Model": name, **res})

    df = pd.DataFrame(rows).sort_values("Val_PR-AUC", ascending=False)
    print(df.to_string(index=False))
    save_experiment_table(df, "exp5_model_leaderboard.csv")
    _plot_grouped_bar_metrics(
        df, "Model", ["Val_F1", "Test_F1", "Val_PR-AUC", "Test_PR-AUC"],
        "Model Family Comparison (Val vs Test)",
        filename="exp5_leaderboard.png"
    )
    return df



def experiment_learning_curve(X_full, y_full):
    print("\n" + "=" * 65)
    print("EXPERIMENT 6: Learning Curve")
    print("=" * 65)

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
    save_experiment_table(df, "exp6_learning_curve.csv")
    _plot_learning_curve(
        train_sizes, train_scores, val_scores,
        filename="exp6_learning_curve.png"
    )
    return df



def parse_args():
    p = argparse.ArgumentParser(description="Run fraud-detection experiments.")
    p.add_argument("--only", type=int, nargs="*", default=None,
                   help="Run only selected experiments (e.g. --only 1 2 3).")
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

    print("\nAll requested experiments complete.")
    print(f"Tables saved in:  {TABLES_DIR}")
    print(f"Figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
