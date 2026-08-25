import os
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def get_report_dir(sub: str = "") -> Path:
    target_dir = REPORTS_DIR / sub if sub else REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def set_figure_output_dir(output_dir: str | Path) -> None:
    global FIGURES_DIR

    FIGURES_DIR = Path(output_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[utils] Figures will be saved to: {FIGURES_DIR}")



REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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


def save_fig(
    fig: plt.Figure,
    filename: str,
    sub: str | None = "figures",
    dpi: int = 300,
):

    if sub is None or sub == "figures":
        out_dir = FIGURES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = get_report_dir(sub)

    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f" [Saved Plot]  -> {out_path.relative_to(PROJECT_ROOT)}")



def plot_confusion_matrix(y_true, y_pred, sub="figures", filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
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


def plot_threshold_curve(
    thresholds, precision, recall, chosen=None, sub="figures", filename="threshold_curve.png"
):
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



def plot_hyperparameter_sweep(family_df, family_name, sub="figures"):
    """Plots Train vs CV PR-AUC for every combination of a specific model family."""
    df = family_df.sort_values("best_cv_pr_auc", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))

    ax.plot(
        x,
        df["train_pr_auc"],
        label="Train PR-AUC (Overfitting Risk)",
        marker="o",
        color="#e53935",
        linestyle="--",
        alpha=0.7,
    )
    ax.plot(
        x,
        df["best_cv_pr_auc"],
        label="CV PR-AUC (Generalization)",
        marker="s",
        color="#1565c0",
        linewidth=2,
    )
    ax.fill_between(
        x,
        df["best_cv_pr_auc"],
        df["train_pr_auc"],
        color="#ffcdd2",
        alpha=0.3,
        label="Overfit Gap",
    )

    labels = [
        params.replace("{", "").replace("}", "").replace("'model__", "").replace("'", "")
        for params in df["params"]
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Hyperparameter Sweep Analysis: {family_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Precision (PR-AUC)")
    ax.set_xlabel("Hyperparameter Combinations")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)

    filename = f"sweep_{family_name.lower()}.png"
    return save_fig(fig, filename, sub)


def plot_family_metric_rankings(family_df, family_name, metric_columns, sub="figures"):

    metric_info = {
        "pr_auc": {"valid": "valid_pr_auc", "train": "train_pr_auc", "gap": "pr_auc_gap", "title": "PR-AUC"},
        "roc_auc": {"valid": "valid_roc_auc", "train": "train_roc_auc", "gap": "roc_auc_gap", "title": "ROC-AUC"},
        "precision": {"valid": "valid_precision", "train": "train_precision", "gap": "precision_gap", "title": "Precision"},
        "recall": {"valid": "valid_recall", "train": "train_recall", "gap": "recall_gap", "title": "Recall"},
        "f1": {"valid": "valid_f1", "train": "train_f1", "gap": "f1_gap", "title": "F1"},
    }

    for metric_name in metric_columns:
        info = metric_info[metric_name]
        df = family_df.sort_values(by=info["valid"], ascending=False).reset_index(drop=True)
        labels = [f"#{i + 1}\nCombo {row['combination']}" for i, row in df.iterrows()]

        x = np.arange(len(df))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.75), 7))

        bars_valid = ax.bar(x - width, df[info["valid"]], width, label="Validation / OOF", color="#1565c0")
        bars_train = ax.bar(x, df[info["train"]], width, label="Train", color="#90caf9")
        bars_gap = ax.bar(x + width, df[info["gap"]], width, label="Gap (Train - Valid)", color="#ef6c00")

        def annotate_bars(bars, values):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                offset = 0.01 if value >= 0 else -0.035
                va = "bottom" if value >= 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + offset,
                    f"{value:.4f}",
                    ha="center",
                    va=va,
                    fontsize=8,
                    rotation=90,
                )

        annotate_bars(bars_valid, df[info["valid"]].to_numpy())
        annotate_bars(bars_train, df[info["train"]].to_numpy())
        annotate_bars(bars_gap, df[info["gap"]].to_numpy())

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.set_xlabel("Models ranked by validation / OOF score")
        ax.set_title(
            f"{family_name} — Ranked {info['title']}: Validation, Train and Gap",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_ylim(min(-0.1, df[info["gap"]].min() - 0.08), 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend()

        filename = f"ranking_{family_name.lower()}_{metric_name}.png"
        save_fig(fig, filename, sub)


def plot_top3_family_comparison(family_df, family_name, sub="figures"):

    df = family_df.sort_values("valid_pr_auc", ascending=False).head(3).reset_index(drop=True)
    if df.empty:
        return

    labels = [f"Rank {i + 1}\nCombo {row['combination']}" for i, row in df.iterrows()]
    x = np.arange(len(df))
    width = 0.16

    metric_specs = [
        ("valid_pr_auc", "Valid PR-AUC", "#1565c0"),
        ("train_pr_auc", "Train PR-AUC", "#90caf9"),
        ("pr_auc_gap", "PR-AUC Gap", "#ef6c00"),
        ("valid_recall", "Valid Recall", "#2e7d32"),
        ("valid_f1", "Valid F1", "#8e24aa"),
    ]

    fig, ax = plt.subplots(figsize=(11, 7))

    for offset, (column, label, color) in enumerate(metric_specs):
        values = df[column].to_numpy()
        bars = ax.bar(x + (offset - 2) * width, values, width, label=label, color=color)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            offset_y = 0.01 if value >= 0 else -0.035
            va = "bottom" if value >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset_y,
                f"{value:.4f}",
                ha="center",
                va=va,
                fontsize=8,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(min(-0.1, df["pr_auc_gap"].min() - 0.08), 1.15)
    ax.set_title(f"{family_name} — Top 3 Models by Validation PR-AUC", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=2)

    filename = f"top3_{family_name.lower()}_comparison.png"
    save_fig(fig, filename, sub)


def plot_oof_pr_curves(family_df, diagnostics, family_name, top_n=3, sub="figures"):
    df = family_df.sort_values("valid_pr_auc", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))

    y_true = diagnostics["_y_true"]
    baseline = np.mean(y_true)

    for _, row in df.iterrows():
        item = diagnostics[row["saved_filename"]]
        y_probs = item["oof_probs"]

        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        ap = average_precision_score(y_true, y_probs)

        ax.plot(recall, precision, linewidth=2, label=f"{row['model']} | AP={ap:.4f}")

    ax.axhline(baseline, color="gray", linestyle="--", label=f"Baseline (Prevalence) = {baseline:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        f"{family_name} — OOF Precision-Recall Curves (Top {min(top_n, len(df))})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)

    filename = f"oof_pr_curves_{family_name.lower()}.png"
    save_fig(fig, filename, sub)


def plot_oof_probability_distributions(family_df, diagnostics, family_name, top_n=3, sub="figures"):
    df = family_df.sort_values("valid_pr_auc", ascending=False).head(top_n)
    n_models = len(df)
    if n_models == 0:
        return

    fig, axes = plt.subplots(n_models, 1, figsize=(10, 4 * n_models), squeeze=False)
    y_true = np.asarray(diagnostics["_y_true"])

    for axis_index, (_, row) in enumerate(df.iterrows()):
        ax = axes[axis_index, 0]
        item = diagnostics[row["saved_filename"]]
        probs = item["oof_probs"]

        legitimate_probs = probs[y_true == 0]
        fraud_probs = probs[y_true == 1]

        sns.histplot(
            legitimate_probs,
            bins=40,
            stat="density",
            kde=True,
            color="#42a5f5",
            alpha=0.45,
            label=f"Legitimate (n={len(legitimate_probs):,}, med={np.median(legitimate_probs):.4f})",
            ax=ax,
        )

        sns.histplot(
            fraud_probs,
            bins=40,
            stat="density",
            kde=True,
            color="#e53935",
            alpha=0.45,
            label=f"Fraud (n={len(fraud_probs):,}, med={np.median(fraud_probs):.4f})",
            ax=ax,
        )

        ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Default Threshold (0.50)")
        ax.set_xlim(0, 1)
        ax.set_xlabel("OOF Predicted Probability of Fraud")
        ax.set_ylabel("Density")
        ax.set_title(f"{row['model']} | OOF PR-AUC = {row['valid_pr_auc']:.4f}", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"{family_name} — OOF Probability Distributions", y=1.01, fontsize=14, fontweight="bold")
    filename = f"oof_probability_distribution_{family_name.lower()}.png"
    save_fig(fig, filename, sub)


def plot_training_time_by_model(summary_df, sub="figures", filename="training_time_by_model.png"):
    df = summary_df.sort_values("training_time_seconds", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
    y = np.arange(len(df))

    bars = ax.barh(y, df["training_time_seconds"], color="#6a1b9a")
    labels = [f"{row['family']} - Combo {row['combination']}" for _, row in df.iterrows()]

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time by Model Configuration", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    max_val = df["training_time_seconds"].max()
    for bar, value in zip(bars, df["training_time_seconds"]):
        ax.text(
            value + max(max_val * 0.01, 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}s",
            va="center",
            fontsize=8,
        )

    save_fig(fig, filename, sub)


def plot_training_time_by_family(
    summary_df,
    sub="figures",
    filename="training_time_by_family.png",
    table_name="training_time_by_family",
):

    grouped = (
        summary_df
        .groupby("family")["training_time_seconds"]
        .agg(["sum", "mean", "count"])
        .reset_index()
        .sort_values("sum", ascending=False)
    )

    x = np.arange(len(grouped))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))

    total_bars = ax.bar(x - width / 2, grouped["sum"], width, label="Total Time (s)", color="#4a148c")
    mean_bars = ax.bar(x + width / 2, grouped["mean"], width, label="Avg Time / Combo (s)", color="#ab47bc")

    for bars in [total_bars, mean_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (grouped["sum"].max() * 0.01),
                f"{height:.2f}s",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(grouped["family"])
    ax.set_ylabel("Seconds")
    ax.set_title("Training Time Breakdown by Model Family", fontsize=13, fontweight="bold")
    ax.set_ylim(0, grouped["sum"].max() * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    save_fig(fig, filename, sub)

    save_table(
        grouped.rename(columns={
            "sum": "total_training_time_seconds",
            "mean": "mean_training_time_seconds",
            "count": "number_of_configurations",
        }),
        table_name,
    )


def plot_top5_overall_models(summary_df, sub="figures", filename="top5_overall_models_by_pr_auc.png"):

    required_columns = {"model", "params", "valid_pr_auc"}
    missing = required_columns - set(summary_df.columns)
    if missing:
        print(f"Skipping overall top-5 plot. Missing columns: {sorted(missing)}")
        return

    top5 = (
        summary_df
        .dropna(subset=["valid_pr_auc"])
        .sort_values("valid_pr_auc", ascending=False)
        .head(5)
        .copy()
        .reset_index(drop=True)
    )

    if top5.empty:
        print("Skipping overall top-5 plot: no valid rows found.")
        return

    
    top5["display_name"] = (
        "#" + (top5.index + 1).astype(str) + "  " + top5["model"].astype(str)
        + "\n" + top5["params"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(14, 8))


    plot_df = top5.iloc[::-1]

    bars = ax.barh(
        plot_df["display_name"],
        plot_df["valid_pr_auc"],
        color="#2563eb",
        edgecolor="black",
        linewidth=0.7,
    )

    ax.set_title(
        "Top 5 Overall Model Configurations by OOF PR-AUC",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Validation / OOF PR-AUC")
    ax.set_ylabel("Model Configuration")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    max_score = top5["valid_pr_auc"].max()
    ax.set_xlim(0, min(1.0, max_score * 1.20))

    for bar, val in zip(bars, plot_df["valid_pr_auc"]):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    save_fig(fig, filename, sub)
