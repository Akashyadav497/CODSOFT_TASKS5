"""
Model training, balancing experimentation, and threshold tuning module.
Trains Logistic Regression, Random Forest, and XGBoost under:
  (a) No balancing (None)
  (b) class_weight='balanced'
  (c) SMOTE oversampling
  (d) RandomUnderSampler undersampling
Strictly applies all resampling and transformations on training data only.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score
from xgboost import XGBClassifier

from src.config import (
    BEST_MODEL_PATH,
    DEFAULT_FRAUD_THRESHOLD,
    FIGURES_DIR,
    LOG_FILE_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    THRESHOLD_CONFIG_PATH,
)
from src.data_loader import clean_raw_data, load_raw_data
from src.evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_model_comparison_bar,
    plot_precision_recall_and_roc,
    save_metrics_summary,
)
from src.preprocess import FraudPreprocessor, split_data

# Configure logging
LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("train")


def train_and_evaluate_all() -> None:
    """
    Orchestrates full training pipeline, balancing experiments, model comparison,
    optimal threshold tuning, and artifact persistence.
    """
    logger.info("=== Starting Credit Card Fraud Detection Training Pipeline ===")

    # 1. Load and clean raw data
    raw_df = load_raw_data()
    clean_df = clean_raw_data(raw_df, save_processed=True)

    # 2. Split First (MANDATORY RULE)
    # Stratified 80/20 split
    X_train_df, X_test_df, y_train, y_test = split_data(clean_df, random_state=RANDOM_STATE)

    # 3. Fit Preprocessor strictly on training data
    logger.info("Fitting FraudPreprocessor on training data only...")
    preprocessor = FraudPreprocessor(scale_amount=True)
    preprocessor.fit(X_train_df)

    X_train = preprocessor.transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    feature_names = preprocessor.get_feature_names()

    logger.info(f"Processed training shape: {X_train.shape}, test shape: {X_test.shape}")
    logger.info(f"Features: {feature_names}")

    # Further partition X_train into train/val for honest threshold optimization (85% sub-train / 15% val)
    from sklearn.model_selection import train_test_split
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )

    # Prepare balancing techniques on X_train
    logger.info("Preparing Resampling datasets on X_train (Split-First Rule)...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    logger.info(f"Original Train: Genuine={(y_train == 0).sum()}, Fraud={(y_train == 1).sum()}")
    logger.info(f"SMOTE Train:    Genuine={(y_train_smote == 0).sum()}, Fraud={(y_train_smote == 1).sum()}")
    logger.info(f"RUS Train:      Genuine={(y_train_rus == 0).sum()}, Fraud={(y_train_rus == 1).sum()}")

    # Define model configurations
    experiments = [
        # Logistic Regression
        {
            "name": "Logistic Regression",
            "technique": "None",
            "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "X_fit": X_train,
            "y_fit": y_train
        },
        {
            "name": "Logistic Regression",
            "technique": "class_weight='balanced'",
            "model": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
            "X_fit": X_train,
            "y_fit": y_train
        },
        {
            "name": "Logistic Regression",
            "technique": "SMOTE",
            "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "X_fit": X_train_smote,
            "y_fit": y_train_smote
        },
        {
            "name": "Logistic Regression",
            "technique": "RandomUnderSampler",
            "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "X_fit": X_train_rus,
            "y_fit": y_train_rus
        },
        # Random Forest (fast yet robust configuration: n_estimators=100, max_depth=12, n_jobs=-1)
        {
            "name": "Random Forest",
            "technique": "None",
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
            "X_fit": X_train,
            "y_fit": y_train
        },
        {
            "name": "Random Forest",
            "technique": "class_weight='balanced'",
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
            "X_fit": X_train,
            "y_fit": y_train
        },
        {
            "name": "Random Forest",
            "technique": "SMOTE",
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
            "X_fit": X_train_smote,
            "y_fit": y_train_smote
        },
        {
            "name": "Random Forest",
            "technique": "RandomUnderSampler",
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE),
            "X_fit": X_train_rus,
            "y_fit": y_train_rus
        },
        # XGBoost (Advanced Gradient Boosting)
        {
            "name": "XGBoost",
            "technique": "scale_pos_weight",
            "model": XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "X_fit": X_train,
            "y_fit": y_train
        }
    ]

    results = []
    curves_data = []
    trained_models = {}

    for exp in experiments:
        m_name = exp["name"]
        tech = exp["technique"]
        key = f"{m_name} ({tech})"
        logger.info(f"Training: {key}...")

        clf = exp["model"]
        clf.fit(exp["X_fit"], exp["y_fit"])
        trained_models[key] = clf

        # Evaluate on untouched test data with standard 0.50 threshold
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= DEFAULT_FRAUD_THRESHOLD).astype(int)

        metrics = compute_classification_metrics(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            y_prob=y_prob,
            model_name=m_name,
            technique=tech
        )
        results.append(metrics)
        logger.info(
            f"Results for {key} -> Precision: {metrics['Precision']:.4f}, "
            f"Recall: {metrics['Recall']:.4f}, F1: {metrics['F1_Score']:.4f}, PR-AUC: {metrics['PR_AUC']:.4f}"
        )

        curves_data.append({
            "label": key,
            "y_true": y_test.to_numpy(),
            "y_prob": y_prob,
            "pr_auc": metrics["PR_AUC"],
            "roc_auc": metrics["ROC_AUC"]
        })

    # Save metrics table and visualizations
    df_metrics = save_metrics_summary(results)
    plot_precision_recall_and_roc(curves_data)
    plot_model_comparison_bar(df_metrics)

    # Determine Best Model based on highest test F1-Score and PR-AUC
    best_row = df_metrics.iloc[0]
    best_key = f"{best_row['Model']} ({best_row['Balancing_Technique']})"
    best_model = trained_models[best_key]
    logger.info(f"Identified Best Model: {best_key} with F1-Score={best_row['F1_Score']:.4f}, PR-AUC={best_row['PR_AUC']:.4f}")

    # Threshold Optimization on Validation Set (Rule 7: Never tune threshold on final test set!)
    logger.info("Tuning decision threshold on validation set (X_val)...")
    # Fit candidate model on subtrain
    val_clone_model = sklearn_clone_and_fit(best_row["Model"], best_row["Balancing_Technique"], X_subtrain, y_subtrain)
    val_prob = val_clone_model.predict_proba(X_val)[:, 1]

    best_thresh, best_val_f1 = optimize_threshold(y_val.to_numpy(), val_prob)
    logger.info(f"Optimal threshold chosen from validation set: {best_thresh:.4f} (Val F1: {best_val_f1:.4f})")

    # Evaluate best model on test set using the chosen optimal threshold
    best_test_prob = best_model.predict_proba(X_test)[:, 1]
    best_test_pred = (best_test_prob >= best_thresh).astype(int)
    from sklearn.metrics import confusion_matrix
    best_cm = confusion_matrix(y_test, best_test_pred)
    plot_confusion_matrix(best_cm, f"{best_key} @ Thresh {best_thresh:.2f}", FIGURES_DIR / "best_model_confusion_matrix.png")

    # Save Best Model, Preprocessor and Threshold configuration
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    with open(THRESHOLD_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": best_row["Model"],
            "balancing_technique": best_row["Balancing_Technique"],
            "optimal_threshold": float(best_thresh),
            "default_threshold": DEFAULT_FRAUD_THRESHOLD,
            "test_precision": float(best_row["Precision"]),
            "test_recall": float(best_row["Recall"]),
            "test_f1": float(best_row["F1_Score"]),
            "test_pr_auc": float(best_row["PR_AUC"]),
            "feature_names": feature_names
        }, f, indent=4)
    logger.info(f"Saved artifacts to {BEST_MODEL_PATH}, {PREPROCESSOR_PATH}, and {THRESHOLD_CONFIG_PATH}")

    # Compute Feature Importances (Model-based and Permutation)
    compute_and_plot_feature_importance(best_model, X_test, y_test.to_numpy(), feature_names)

    logger.info("=== Training and Evaluation Pipeline Successfully Completed ===")


def sklearn_clone_and_fit(model_name: str, tech: str, X_fit: np.ndarray, y_fit: np.ndarray):
    """Utility to fit model on subtrain for validation threshold tuning."""
    if "Random Forest" in model_name:
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE)
    elif "XGBoost" in model_name:
        clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    if "SMOTE" in tech:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_fit, y_fit)
        clf.fit(X_res, y_res)
    elif "RandomUnderSampler" in tech:
        rus = RandomUnderSampler(random_state=RANDOM_STATE)
        X_res, y_res = rus.fit_resample(X_fit, y_fit)
        clf.fit(X_res, y_res)
    elif "balanced" in tech:
        clf.set_params(class_weight="balanced")
        clf.fit(X_fit, y_fit)
    elif "scale_pos_weight" in tech:
        clf.set_params(scale_pos_weight=(y_fit == 0).sum() / (y_fit == 1).sum())
        clf.fit(X_fit, y_fit)
    else:
        clf.fit(X_fit, y_fit)

    return clf


def optimize_threshold(y_val: np.ndarray, val_prob: np.ndarray) -> Tuple[float, float]:
    """Finds threshold maximizing F1 score on validation set."""
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_prob)
    # Exclude division by zero
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0
    )
    best_idx = np.argmax(f1_scores)
    # precision_recall_curve returns thresholds array with length len(precisions) - 1
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = 0.50
    return float(best_threshold), float(f1_scores[best_idx])


def compute_and_plot_feature_importance(model: Any, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]) -> None:
    """Computes and plots both model-based importances and permutation importances."""
    import matplotlib.pyplot as plt

    # 1. Model-based Importance (Tree or GBDT or Coefficients)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    top_imp = imp_df.head(15)
    plt.barh(top_imp["Feature"][::-1], top_imp["Importance"][::-1], color="#2b5c8f")
    plt.title("Top 15 Feature Importances (Model-Based)", fontsize=12, weight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_model.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Permutation Importance on Test Set
    logger.info("Computing Permutation Importance on test set (sample of 5,000 for efficiency)...")
    sample_indices = np.random.default_rng(RANDOM_STATE).choice(len(X_test), size=min(5000, len(X_test)), replace=False)
    perm_result = permutation_importance(
        model,
        X_test[sample_indices],
        y_test[sample_indices],
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="average_precision"
    )

    perm_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Importance": perm_result.importances_mean,
        "Std_Importance": perm_result.importances_std
    }).sort_values("Mean_Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    top_perm = perm_df.head(15)
    plt.barh(top_perm["Feature"][::-1], top_perm["Mean_Importance"][::-1], xerr=top_perm["Std_Importance"][::-1], color="#d9534f", alpha=0.85)
    plt.title("Top 15 Permutation Feature Importances (PR-AUC Impact on Test Data)", fontsize=12, weight="bold")
    plt.xlabel("Mean PR-AUC Decrease when Permuted")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_permutation.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Feature importance charts successfully saved!")


if __name__ == "__main__":
    train_and_evaluate_all()
