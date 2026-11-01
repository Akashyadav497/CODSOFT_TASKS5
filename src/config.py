"""
Central Configuration module for Credit Card Fraud Detection.
Defines all directory paths, hyperparameter defaults, feature lists, and random seeds.
"""

from pathlib import Path
from typing import List

# Base Directories
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_PATH: Path = DATA_DIR / "creditcard.csv"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
PROCESSED_DATA_PATH: Path = PROCESSED_DATA_DIR / "cleaned_creditcard.csv"

MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
METRICS_CSV_PATH: Path = OUTPUTS_DIR / "metrics.csv"
METRICS_MD_PATH: Path = OUTPUTS_DIR / "metrics.md"
PREDICTIONS_CSV_PATH: Path = OUTPUTS_DIR / "predictions.csv"
DATA_AUDIT_PATH: Path = OUTPUTS_DIR / "data_audit.md"
LOGS_DIR: Path = BASE_DIR / "logs"
LOG_FILE_PATH: Path = LOGS_DIR / "fraud_detection.log"

# Saved Model & Preprocessing Pipeline Artifacts
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.joblib"
THRESHOLD_CONFIG_PATH: Path = MODELS_DIR / "threshold_config.json"

# Modeling & Evaluation Constants
RANDOM_STATE: int = 42
TEST_SPLIT_RATIO: float = 0.20
DEFAULT_FRAUD_THRESHOLD: float = 0.50

# Feature Definitions
TARGET_COLUMN: str = "Class"
RAW_FEATURE_COLUMNS: List[str] = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount"
]

ENGINEERED_FEATURE_COLUMNS: List[str] = [
    "Hour",
    *[f"V{i}" for i in range(1, 29)],
    "Amount_Scaled"
]
