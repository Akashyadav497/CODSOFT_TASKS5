"""
Data loading and validation module.
Safely and efficiently loads raw and processed credit card transactions.
"""

import logging
from pathlib import Path
from typing import Tuple
import pandas as pd

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN

logger = logging.getLogger(__name__)


def load_raw_data(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw credit card transaction data from CSV.

    Args:
        data_path: Path to the raw CSV file.

    Returns:
        pd.DataFrame: Loaded transactions DataFrame.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if not data_path.exists():
        error_msg = f"Data file not found at: {data_path.resolve()}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Loading raw data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Raw data successfully loaded with shape: {df.shape}")
    return df


def clean_raw_data(df: pd.DataFrame, save_processed: bool = True) -> pd.DataFrame:
    """
    Deduplicates the dataset, reports missing values and saves cleaned data.

    Args:
        df: Raw DataFrame.
        save_processed: Whether to persist cleaned DataFrame to disk.

    Returns:
        pd.DataFrame: Deduplicated clean DataFrame.
    """
    initial_rows = len(df)
    duplicates = df.duplicated().sum()
    logger.info(f"Found {duplicates} duplicate rows out of {initial_rows} records.")

    df_cleaned = df.drop_duplicates().copy()
    logger.info(f"Cleaned dataset shape after deduplication: {df_cleaned.shape}")

    null_count = df_cleaned.isnull().sum().sum()
    if null_count > 0:
        logger.warning(f"Found {null_count} null values in cleaned dataset. Dropping nulls.")
        df_cleaned = df_cleaned.dropna()
    else:
        logger.info("Verified zero missing values in the dataset.")

    if save_processed:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info(f"Saved cleaned dataset to: {PROCESSED_DATA_PATH}")

    return df_cleaned


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Computes key summary statistics including class balance.

    Args:
        df: DataFrame containing the dataset.

    Returns:
        dict: Summary statistics dictionary.
    """
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    total = len(df)
    fraud_pct = (class_counts.get(1, 0) / total) * 100 if total > 0 else 0.0

    return {
        "total_transactions": total,
        "genuine_count": class_counts.get(0, 0),
        "fraud_count": class_counts.get(1, 0),
        "fraud_percentage": fraud_pct,
        "features_count": df.shape[1] - 1
    }
