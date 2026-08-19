import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def get_report_dir(sub: str = "") -> Path:
    target_dir = REPORTS_DIR / sub if sub else REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir



# Tables and Figures 
def save_table(df: pd.DataFrame, name: str, sub: str = "tables"):
    out_dir = get_report_dir(sub)
    
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"

    df.to_csv(csv_path, index=False)
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))
    except ImportError:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))
            
    print(f" [Saved Table] -> {csv_path.relative_to(PROJECT_ROOT)}")


def save_fig(fig: plt.Figure, filename: str, sub: str = "figures", dpi: int = 300):
    out_dir = get_report_dir(sub)
    out_path = out_dir / filename

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f" [Saved Plot]  -> {out_path.relative_to(PROJECT_ROOT)}")


# Training figures
def plot_model_comparison(summary_df, sub="figures", filename="model_comparison.png"):
    """CV PR-AUC vs Train PR-AUC per model """
    df = summary_df.sort_values("best_cv_pr_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    w = 0.38
    ax.barh(x - w / 2, df["train_pr_auc"], height=w, label="Train PR-AUC", color="#90caf9")
    ax.barh(x + w / 2, df["best_cv_pr_auc"], height=w, label="CV PR-AUC", color="#1565c0")
    ax.set_yticks(x)
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("Average Precision (PR-AUC)")
    ax.set_title("Model Comparison — CV vs Train score")
    ax.legend(loc="lower right")
    for i, row in enumerate(df.itertuples()):
        ax.text(row.best_cv_pr_auc + 0.005, i, f"{row.best_cv_pr_auc:.3f}", va="center", fontsize=9)
    return save_fig(fig, filename, sub)

def plot_comprehensive_metrics(summary_df, sub="figures", filename="comprehensive_metrics.png"):
    """Grouped bar chart"""
    metrics = ["Precision (Fraud)", "Recall (Fraud)", "F1-Score"]
    df_melt = summary_df.melt(id_vars=["model"], value_vars=metrics, var_name="Metric", value_name="Score")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_melt, x="model", y="Score", hue="Metric", ax=ax, palette="tab10")
    ax.set_title("OOF Model Performance: Precision, Recall, and F1-Score", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.xticks(rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    return save_fig(fig, filename, sub)

def plot_error_breakdown(summary_df, sub="figures", filename="error_breakdown.png"):
    """False Positives and False Negatives."""
    df = summary_df[["model", "False Positives", "False Negatives"]].set_index("model")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind="bar", ax=ax, color=["#ff9800", "#f44336"])
    ax.set_title("OOF Error Breakdown: False Positives vs False Negatives", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count (Transactions)")
    ax.set_xlabel("Model")
    plt.xticks(rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    return save_fig(fig, filename, sub)


def plot_confusion_matrix(y_true, y_pred, sub="figures", filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Legit (0)", "Fraud (1)"],
        yticklabels=["Legit (0)", "Fraud (1)"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    return save_fig(fig, filename, sub)


def plot_roc_curve(y_true, y_probs, sub="figures", filename="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1565c0", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test Set)")
    ax.legend(loc="lower right")
    save_fig(fig, filename, sub)
    return roc_auc


def plot_pr_curve(y_true, y_probs, sub="figures", filename="pr_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#e53935", lw=2, label=f"PR curve (AP = {ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Test Set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    save_fig(fig, filename, sub)
    return ap


def plot_threshold_curve(thresholds, precision, recall, chosen=None, sub="figures", filename="threshold_curve.png"):
    p = np.asarray(precision)[:-1]
    r = np.asarray(recall)[:-1]
    f1 = 2 * p * r / (p + r + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, p, label="Precision", color="#1565c0")
    ax.plot(thresholds, r, label="Recall", color="#e53935")
    ax.plot(thresholds, f1, label="F1", color="#2e7d32")
    if chosen is not None:
        ax.axvline(chosen, color="grey", ls="--", lw=1, label=f"Chosen = {chosen:.3f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs. Decision Threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right")
    return save_fig(fig, filename, sub)

# Experiment figures
def plot_grouped_bar_metrics(df, hue_col, metric_cols, title, sub="figures", filename="grouped_bar.png"):
    melted = df.melt(id_vars=[hue_col], value_vars=metric_cols, var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=melted, x="Metric", y="Score", hue=hue_col, ax=ax, palette="Set2")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(title=hue_col)
    return save_fig(fig, filename, sub)


def plot_metric_vs_category(df, x_col, metric_cols, title, sub="figures", filename="metric_vs_param.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(df))
    colors = ["#1565c0", "#e53935", "#2e7d32", "#8e24aa"]
    for col, color in zip(metric_cols, colors):
        ax.plot(x, df[col], marker="o", label=col, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].astype(str))
    ax.set_xlabel(x_col)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend()
    return save_fig(fig, filename, sub)


def plot_learning_curve(train_sizes, train_scores, val_scores, sub="figures", filename="learning_curve.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#90caf9", label="Train PR-AUC")
    ax.fill_between(train_sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1),
                    color="#90caf9", alpha=0.3)
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color="#1565c0", label="CV PR-AUC")
    ax.fill_between(train_sizes,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1),
                    color="#1565c0", alpha=0.3)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Average Precision (PR-AUC)")
    ax.set_title("Learning Curve")
    ax.legend(loc="best")
    return save_fig(fig, filename, sub)


def plot_hyperparameter_sweep(family_df, family_name, sub="figures"):
    df = family_df.sort_values("best_cv_pr_auc", ascending=True).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))

    ax.plot(x, df["train_pr_auc"], label="Train PR-AUC (Overfitting Risk)", marker="o", color="#e53935", linestyle="--", alpha=0.7)
    ax.plot(x, df["best_cv_pr_auc"], label="CV PR-AUC (Generalization)", marker="s", color="#1565c0", linewidth=2)
    
    ax.fill_between(x, df["best_cv_pr_auc"], df["train_pr_auc"], color="#ffcdd2", alpha=0.3, label="Overfit Gap")
    
    labels = []
    for params in df["params"]:
        clean = params.replace("{", "").replace("}", "").replace("'model__", "").replace("'", "")
        labels.append(clean)
        
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    
    ax.set_title(f"Hyperparameter Sweep Analysis: {family_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Precision (PR-AUC)")
    ax.set_xlabel("Hyperparameter Combinations")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    filename = f"sweep_{family_name.lower()}.png"
    return save_fig(fig, filename, sub)