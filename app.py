"""
Interactive Streamlit Web Application for Credit Card Fraud Detection.
Features:
  - Dataset Overview & Live Audit Metrics
  - Interactive Exploratory Data Analysis & Visualizations
  - Model Performance & Balancing Technique Comparison
  - Live Fraud Detection CSV Uploader with Probability Scoring
  - Interactive Decision Threshold Slider with Real-Time Precision / Recall Trade-Off
  - Export & Download for Flagged Transactions and Metrics
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.config import (
    BEST_MODEL_PATH,
    FIGURES_DIR,
    METRICS_CSV_PATH,
    PREPROCESSOR_PATH,
    RAW_DATA_PATH,
    THRESHOLD_CONFIG_PATH,
    DEFAULT_FRAUD_THRESHOLD,
)
from src.features import extract_features
from src.predict import FraudPredictor

# Page Configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2b5c8f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .fraud-badge {
        background-color: #d9534f;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .genuine-badge {
        background-color: #2b5c8f;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_cached_data():
    """Load sampled subset of transactions for responsive UI display."""
    if RAW_DATA_PATH.exists():
        df = pd.read_csv(RAW_DATA_PATH)
        df_clean = df.drop_duplicates()
        return df_clean
    return None


@st.cache_data
def load_metrics_table():
    """Load performance comparison metrics."""
    if METRICS_CSV_PATH.exists():
        return pd.read_csv(METRICS_CSV_PATH)
    return None


def main():
    st.sidebar.title("🛡️ Fraud Detection")
    st.sidebar.caption("CodSoft Data Science Task 5")

    menu = st.sidebar.radio(
        "Navigation Menu",
        [
            "📊 Executive Dashboard",
            "🔍 Exploratory Data Analysis",
            "🤖 Model Benchmark & Thresholds",
            "🚀 Live Fraud Detector & Batch Audit"
        ]
    )

    data = load_cached_data()
    metrics_df = load_metrics_table()

    # =========================================================================
    # Page 1: Executive Dashboard
    # =========================================================================
    if menu == "📊 Executive Dashboard":
        st.title("🛡️ Credit Card Fraud Detection: Executive Dashboard")
        st.markdown(
            "An end-to-end, reproducible Machine Learning system built to accurately detect fraudulent credit card transactions "
            "under severe class imbalance (~0.17% fraud prevalence)."
        )

        if data is not None:
            total_tx = len(data)
            total_fraud = (data["Class"] == 1).sum()
            total_genuine = total_tx - total_fraud
            fraud_rate = (total_fraud / total_tx) * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", f"{total_tx:,}")
            col2.metric("Genuine Transactions", f"{total_genuine:,}", "99.83%")
            col3.metric("Fraudulent Transactions", f"{total_fraud:,}", f"{fraud_rate:.3f}%")
            col4.metric("Imbalance Ratio", f"1 in {int(total_tx/total_fraud):,}")

            st.write("---")
            st.subheader("Key Architectural Safeguards")
            st.markdown("""
            - **Split-First Protocol:** Partitioning is performed *strictly* before any scaling or resampling, preventing data leakage.
            - **Honest Metric Evaluation:** Optimizing PR-AUC, F1-Score, and Recall rather than deceptive standard accuracy.
            - **Decision Threshold Optimization:** Calibrated on validation partition to balance false alarms and fraud leakage.
            - **Robust Preprocessing:** `RobustScaler` and `log1p` transformations to handle extreme outliers.
            """)

            st.subheader("Dataset Sample (First 5 Rows)")
            st.dataframe(data.head(5), use_container_width=True)

    # =========================================================================
    # Page 2: Exploratory Data Analysis
    # =========================================================================
    elif menu == "🔍 Exploratory Data Analysis":
        st.title("🔍 Exploratory Data Analysis (EDA)")
        st.markdown("In-depth analysis of financial amounts, diurnal patterns, and feature correlations.")

        tab1, tab2, tab3 = st.tabs(["Transaction Amounts & Classes", "Diurnal / Hourly Patterns", "Feature Correlations"])

        with tab1:
            st.subheader("Class Imbalance & Transaction Amounts")
            img1 = FIGURES_DIR / "class_distribution.png"
            img2 = FIGURES_DIR / "amount_by_class.png"
            c1, c2 = st.columns(2)
            if img1.exists():
                c1.image(str(img1), caption="Severe Class Imbalance (Log Scale)")
            if img2.exists():
                c2.image(str(img2), caption="Amount Distribution by Class")

        with tab2:
            st.subheader("Fraud Rate by Hour of Day")
            img3 = FIGURES_DIR / "fraud_rate_by_hour.png"
            if img3.exists():
                st.image(str(img3), use_container_width=True)
            st.info("💡 **Insight:** Fraud rates peak during early morning hours (02:00 - 05:00) when legitimate transaction traffic is lowest.")

        with tab3:
            st.subheader("Correlation & Feature Importance")
            img4 = FIGURES_DIR / "feature_correlations.png"
            if img4.exists():
                st.image(str(img4), use_container_width=True)

    # =========================================================================
    # Page 3: Model Benchmark & Thresholds
    # =========================================================================
    elif menu == "🤖 Model Benchmark & Thresholds":
        st.title("🤖 Model Performance Benchmark & Balancing Comparison")
        st.markdown("All models evaluated on the **untouched test set (20% holdout)**.")

        if metrics_df is not None:
            st.subheader("Comprehensive Metrics Comparison Table")
            st.dataframe(
                metrics_df.style.highlight_max(
                    subset=["Precision", "Recall", "F1_Score", "PR_AUC", "ROC_AUC"],
                    color="#d4edda"
                ),
                use_container_width=True
            )

            # Download metrics button
            csv_data = metrics_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Metrics CSV",
                data=csv_data,
                file_name="model_metrics_comparison.csv",
                mime="text/csv"
            )

        st.write("---")
        st.subheader("Evaluation Visualizations")
        col_a, col_b = st.columns(2)
        pr_img = FIGURES_DIR / "precision_recall_curves.png"
        roc_img = FIGURES_DIR / "roc_curves.png"
        if pr_img.exists():
            col_a.image(str(pr_img), caption="Precision-Recall Curves (Key Metric)")
        if roc_img.exists():
            col_b.image(str(roc_img), caption="ROC Curves")

        cm_img = FIGURES_DIR / "best_model_confusion_matrix.png"
        bar_img = FIGURES_DIR / "model_comparison_bar.png"
        col_c, col_d = st.columns(2)
        if cm_img.exists():
            col_c.image(str(cm_img), caption="Best Model Confusion Matrix")
        if bar_img.exists():
            col_d.image(str(bar_img), caption="Benchmark Comparison Bar Chart")

    # =========================================================================
    # Page 4: Live Fraud Detector & Batch Audit
    # =========================================================================
    elif menu == "🚀 Live Fraud Detector & Batch Audit":
        st.title("🚀 Live Fraud Detection & Batch Inference")

        # Threshold slider
        st.sidebar.subheader("⚙️ Decision Calibration")
        custom_threshold = st.sidebar.slider(
            "Fraud Decision Threshold",
            min_value=0.01,
            max_value=0.99,
            value=0.50,
            step=0.01,
            help="Lowering threshold catches more fraud (increases Recall) at the cost of more false alarms (decreases Precision)."
        )

        st.markdown("""
        Upload a transaction batch CSV containing required columns (`Time`, `Amount`, `V1` - `V28`).
        The calibrated model will predict the precise fraud probability and flag suspicious activity.
        """)

        uploaded_file = st.file_uploader("Upload Transactions CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded **{len(input_df):,}** transactions.")

                predictor = FraudPredictor()
                predictions = predictor.predict(input_df, custom_threshold=custom_threshold)

                fraud_flagged = (predictions["is_fraud"] == 1).sum()
                flagged_pct = (fraud_flagged / len(predictions)) * 100

                m1, m2, m3 = st.columns(3)
                m1.metric("Evaluated Transactions", f"{len(predictions):,}")
                m2.metric("Flagged as Suspicious", f"{fraud_flagged:,}", f"{flagged_pct:.2f}%")
                m3.metric("Decision Threshold", f"{custom_threshold:.2f}")

                st.subheader("Flagged High-Risk Transactions")
                flagged_df = predictions[predictions["is_fraud"] == 1]
                if not flagged_df.empty:
                    st.dataframe(flagged_df.head(100), use_container_width=True)

                    dl_csv = flagged_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Flagged Fraudulent Transactions CSV",
                        data=dl_csv,
                        file_name="flagged_fraud_transactions.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("No transactions exceeded the selected fraud threshold.")

            except Exception as e:
                st.error(f"Inference error: {e}")
        else:
            # Provide sample transaction demonstration
            st.info("Upload a CSV file or use the built-in test generator below.")
            if st.button("Generate & Test 5 Real Test Sample Transactions"):
                if data is not None:
                    sample_tx = data.sample(5, random_state=42).drop(columns=["Class"], errors="ignore")
                    predictor = FraudPredictor()
                    preds = predictor.predict(sample_tx, custom_threshold=custom_threshold)
                    st.dataframe(preds[["Time", "Amount", "fraud_probability", "decision_label"]], use_container_width=True)


if __name__ == "__main__":
    main()
