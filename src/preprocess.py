"""
Preprocessing and data transformation module.
Implements scikit-learn compatible transformers and stratified data splitting
adhering strictly to the Split-First-Then-Scale principle.
"""

import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.config import RANDOM_STATE, TEST_SPLIT_RATIO, TARGET_COLUMN
from src.features import extract_features

logger = logging.getLogger(__name__)


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessor for Credit Card Fraud Detection.
    Fits feature scalers on training data only to prevent data leakage.
    """

    def __init__(self, scale_amount: bool = True):
        self.scale_amount = scale_amount
        # RobustScaler is chosen because financial transaction amounts contain extreme outliers.
        # It removes the median and scales the data according to the Interquartile Range (IQR).
        self.scaler = RobustScaler()
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FraudPreprocessor":
        """
        Fit scaler strictly on the training partition.
        """
        X_df = extract_features(X)

        # Select features: PCA features V1-V28, Hour, and Log_Amount
        pca_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in X_df.columns]
        other_cols = [col for col in ["Hour", "Log_Amount"] if col in X_df.columns]
        self.feature_names_ = pca_cols + other_cols

        if self.scale_amount and "Log_Amount" in self.feature_names_:
            self.scaler.fit(X_df[["Log_Amount"]])

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply learned transformations to arbitrary partition (Train, Val, Test, or Inference).
        """
        X_df = extract_features(X)

        if not self.feature_names_:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() before transform().")

        missing_cols = set(self.feature_names_) - set(X_df.columns)
        if missing_cols:
            raise ValueError(f"Input data is missing required transformed columns: {missing_cols}")

        X_transformed = X_df[self.feature_names_].copy()

        if self.scale_amount and "Log_Amount" in self.feature_names_:
            X_transformed["Log_Amount"] = self.scaler.transform(X_transformed[["Log_Amount"]])

        return X_transformed.to_numpy(dtype=np.float64)

    def get_feature_names(self) -> List[str]:
        return list(self.feature_names_)


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SPLIT_RATIO,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified split on the dataset.

    MANDATORY RULE:
    1. Stratified split so both train and test sets strictly maintain the real fraud ratio.
    2. Zero balancing or scaling is applied prior to this split to avoid data leakage.

    Args:
        df: Input cleaned DataFrame.
        test_size: Proportion of dataset to hold out as test set (default 0.20).
        random_state: Reproducibility seed (default 42).

    Returns:
        X_train, X_test, y_train, y_test
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    logger.info(
        f"Performing Stratified Split (test_size={test_size}, random_state={random_state})..."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    logger.info(
        f"Train shape: {X_train.shape}, Fraud cases: {y_train.sum()} ({y_train.mean():.4%})"
    )
    logger.info(
        f"Test shape:  {X_test.shape}, Fraud cases: {y_test.sum()} ({y_test.mean():.4%})"
    )

    return X_train, X_test, y_train, y_test
