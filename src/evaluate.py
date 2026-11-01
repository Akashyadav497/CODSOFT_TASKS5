"""
Model evaluation module.
Calculates and visualizes performance metrics (Precision, Recall, F1, PR-AUC, ROC-AUC),
generates metric comparison tables, confusion matrices, and ROC/PR curves.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, METRICS_CSV_PATH, METRICS_MD_PATH

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    technique: str
) -> Dict[str, Any]:
    """
    Computes key performance metrics for the fraud class (class 1).
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    metrics = {
        "Model": model_name,
        "Balancing_Technique": technique,
        "Precision": float(precision),
        "Recall": float(recall),
        "F1_Score": float(f1),
        "PR_AUC": float(pr_auc),
        "ROC_AUC": float(roc_auc),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    }
    return metrics


def save_metrics_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Saves performance comparison results to CSV and Markdown.
    """
    df_metrics = pd.DataFrame(results)
    # Sort by F1-Score and PR-AUC descending
    df_metrics = df_metrics.sort_values(by=["F1_Score", "PR_AUC"], ascending=False).reset_index(drop=True)

    METRICS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(METRICS_CSV_PATH, index=False)
    logger.info(f"Saved evaluation metrics to: {METRICS_CSV_PATH}")

    # Generate Markdown format
    md_content = "# Model Performance Comparison on Untouched Evaluation Data\n\n"
    md_content += df_metrics.to_markdown(index=False)
    md_content += "\n\n### Metric Interpretations:\n"
    md_content += "- **Precision**: Proportion of flagged transactions that were genuine fraud (TP / (TP + FP)). High precision prevents false alarms.\n"
    md_content += "- **Recall**: Proportion of actual fraudulent transactions detected (TP / (TP + FN)). High recall minimizes financial fraud leak.\n"
    md_content += "- **F1-Score**: Harmonic mean of Precision and Recall. Balances both costs.\n"
    md_content += "- **PR-AUC**: Area under the Precision-Recall curve. The single most honest metric for severely imbalanced fraud detection.\n"
    md_content += "- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.\n"

    with open(METRICS_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved evaluation markdown summary to: {METRICS_MD_PATH}")

    return df_metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: Path
) -> None:
    """
    Plots and saves an annotated confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Genuine (0)", "Fraud (1)"],
        yticklabels=["Genuine (0)", "Fraud (1)"],
        ax=ax
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, weight="bold", pad=10)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_and_roc(
    curves_data: List[Dict[str, Any]],
    output_dir: Path = FIGURES_DIR
) -> None:
    """
    Plots PR and ROC curves comparing trained algorithms.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. PR Curves
    plt.figure(figsize=(9, 6))
    for item in curves_data:
        p, r, _ = precision_recall_curve(item["y_true"], item["y_prob"])
        pr_auc = item["pr_auc"]
        plt.plot(r, p, label=f"{item['label']} (PR-AUC = {pr_auc:.3f})", lw=2)
    plt.title("Precision-Recall Curves (Untouched Test Set)", fontsize=13, weight="bold")
    plt.xlabel("Recall (Detection Rate)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. ROC Curves
    plt.figure(figsize=(9, 6))
    for item in curves_data:
        fpr, tpr, _ = roc_curve(item["y_true"], item["y_prob"])
        roc_auc = item["roc_auc"]
        plt.plot(fpr, tpr, label=f"{item['label']} (ROC-AUC = {roc_auc:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (0.50)")
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, weight="bold")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_comparison_bar(df_metrics: pd.DataFrame, output_path: Path = FIGURES_DIR / "model_comparison_bar.png") -> None:
    """
    Plots bar chart comparing Precision, Recall, F1 and PR-AUC across experiments.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    metric_cols = ["Precision", "Recall", "F1_Score", "PR_AUC"]
    labels = [f"{row['Model']} ({row['Balancing_Technique']})" for _, row in df_metrics.iterrows()]

    x = np.arange(len(labels))
    width = 0.18

    for i, col in enumerate(metric_cols):
        ax.bar(x + (i - 1.5) * width, df_metrics[col], width, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_title("Comprehensive Model & Balancing Comparison on Test Set", fontsize=13, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
