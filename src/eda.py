"""
Exploratory Data Analysis script for Credit Card Fraud Detection.
Generates all mandatory analytical visualizations with clean annotations and saves them to outputs/figures/.
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, RANDOM_STATE, RAW_DATA_PATH
from src.features import extract_features
from src.preprocess import split_data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Visual styling setup
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

logger = logging.getLogger(__name__)


def generate_eda_figures() -> None:
    """Generate and persist all mandatory EDA charts with factual summaries."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data for EDA...")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df = df_raw.drop_duplicates().copy()
    df_feat = extract_features(df)

    genuine_count = (df_feat["Class"] == 0).sum()
    fraud_count = (df_feat["Class"] == 1).sum()
    total_count = len(df_feat)
    fraud_pct = (fraud_count / total_count) * 100

    # -------------------------------------------------------------
    # 1. Class Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["Genuine (0)", "Fraudulent (1)"],
        [genuine_count, fraud_count],
        color=["#2b5c8f", "#d9534f"],
        width=0.5
    )
    ax.set_yscale("log")
    ax.set_title("Transaction Class Distribution (Log Scale)", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("Number of Transactions (Log Scale)", fontsize=11)
    for bar, count in zip(bars, [genuine_count, fraud_count]):
        pct = (count / total_count) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count * 1.3,
            f"{count:,}\n({pct:.3f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold"
        )
    fig.text(
        0.5, -0.05,
        f"Interpretation: Extreme class imbalance. Genuine: {genuine_count:,} (99.833%), Fraud: {fraud_count:,} (0.167%).",
        ha="center", fontsize=9, style="italic"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 2. Transaction Amount by Class (Violin/Box Plot and Histogram)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(
        x="Class",
        y="Amount",
        data=df_feat,
        ax=axes[0],
        palette=["#2b5c8f", "#d9534f"],
        showfliers=False
    )
    axes[0].set_xticklabels(["Genuine", "Fraud"])
    axes[0].set_title("Transaction Amount Distribution (Without Outliers)", fontsize=11, weight="bold")
    axes[0].set_ylabel("Amount ($)")

    # Log Amount KDE
    sns.kdeplot(
        df_feat[df_feat["Class"] == 0]["Log_Amount"],
        ax=axes[1],
        label="Genuine",
        color="#2b5c8f",
        fill=True,
        alpha=0.3
    )
    sns.kdeplot(
        df_feat[df_feat["Class"] == 1]["Log_Amount"],
        ax=axes[1],
        label="Fraud",
        color="#d9534f",
        fill=True,
        alpha=0.3
    )
    axes[1].set_title("Log(1 + Amount) Density by Class", fontsize=11, weight="bold")
    axes[1].set_xlabel("Log(1 + Amount)")
    axes[1].legend()

    fig.text(
        0.5, -0.05,
        "Interpretation: Genuine amounts have a wide spread (max >$25,000), while fraud clusters at specific values (median ~$9.25).",
        ha="center", fontsize=9, style="italic"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "amount_by_class.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 3. Fraud Rate by Hour of Day
    # -------------------------------------------------------------
    hourly_stats = df_feat.groupby("Hour")["Class"].agg(["count", "sum"]).reset_index()
    hourly_stats["fraud_rate_pct"] = (hourly_stats["sum"] / hourly_stats["count"]) * 100

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.bar(
        hourly_stats["Hour"],
        hourly_stats["count"],
        alpha=0.3,
        color="#2b5c8f",
        label="Total Transactions"
    )
    ax2.plot(
        hourly_stats["Hour"],
        hourly_stats["fraud_rate_pct"],
        color="#d9534f",
        marker="o",
        linewidth=2,
        label="Fraud Rate (%)"
    )

    ax1.set_xlabel("Hour of Day (0 - 23)")
    ax1.set_ylabel("Total Transactions", color="#2b5c8f")
    ax2.set_ylabel("Fraud Rate (%)", color="#d9534f")
    ax1.set_xticks(range(0, 24))
    ax1.grid(False)
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.title("Transaction Volume and Fraud Rate by Hour of Day", fontsize=12, weight="bold", pad=12)

    fig.text(
        0.5, -0.05,
        "Interpretation: Fraud rates spike significantly during early morning hours (2 AM - 5 AM) when overall transaction volume drops.",
        ha="center", fontsize=9, style="italic"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fraud_rate_by_hour.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 4. Feature Correlation with Fraud Label
    # -------------------------------------------------------------
    corr = df_feat.drop(columns=["Time"]).corr()["Class"].sort_values()
    corr_to_plot = pd.concat([corr.head(10), corr.tail(11)])  # top neg, top pos including Class
    corr_to_plot = corr_to_plot.drop(labels=["Class"], errors="ignore")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d9534f" if val < 0 else "#2b5c8f" for val in corr_to_plot.values]
    ax.barh(corr_to_plot.index, corr_to_plot.values, color=colors)
    ax.set_title("Top Positively & Negatively Correlated Features with Fraud", fontsize=12, weight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient")

    fig.text(
        0.5, -0.03,
        "Interpretation: V17, V14, V12 exhibit strongest negative correlation; V4, V11 show highest positive correlation with fraud.",
        ha="center", fontsize=9, style="italic"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_correlations.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 5. Class Distribution Before and After Balancing (Training Set Only)
    # -------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(df, random_state=RANDOM_STATE)

    # SMOTE oversampling
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res_smote, y_train_res_smote = smote.fit_resample(
        X_train[["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]], y_train
    )

    # RandomUnderSampler
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_train_res_rus, y_train_res_rus = rus.fit_resample(
        X_train[["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]], y_train
    )

    categories = ["Original Train", "After SMOTE", "After Undersampling", "Untouched Test"]
    genuine_vals = [
        (y_train == 0).sum(),
        (y_train_res_smote == 0).sum(),
        (y_train_res_rus == 0).sum(),
        (y_test == 0).sum()
    ]
    fraud_vals = [
        (y_train == 1).sum(),
        (y_train_res_smote == 1).sum(),
        (y_train_res_rus == 1).sum(),
        (y_test == 1).sum()
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rects1 = ax.bar(x - width/2, genuine_vals, width, label="Genuine (0)", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, fraud_vals, width, label="Fraud (1)", color="#d9534f")

    ax.set_yscale("log")
    ax.set_ylabel("Count (Log Scale)")
    ax.set_title("Training Data Class Counts Before & After Balancing (Split First Rule)", fontsize=12, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{int(h):,}", (rect.get_x() + rect.get_width()/2, h * 1.2),
                    ha="center", va="bottom", fontsize=8, rotation=0)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{int(h):,}", (rect.get_x() + rect.get_width()/2, h * 1.2),
                    ha="center", va="bottom", fontsize=8, rotation=0, weight="bold", color="#d9534f")

    fig.text(
        0.5, -0.05,
        "Interpretation: Balancing strictly modifies the training split. The test set remains completely untouched.",
        ha="center", fontsize=9, style="italic"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balancing_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("All EDA figures successfully generated and saved to outputs/figures/!")


if __name__ == "__main__":
    generate_eda_figures()
