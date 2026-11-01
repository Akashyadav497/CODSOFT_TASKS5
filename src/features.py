"""
Feature engineering module.
Extracts meaningful temporal and financial features from raw transactions.
"""

import numpy as np
import pandas as pd


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives domain-specific features from raw transactions.

    Features engineered:
      - Hour: Discretized hour of day (0-23) from elapsed seconds 'Time'
      - Log_Amount: Log1p transformed transaction amount to stabilize variance

    Args:
        df: Input DataFrame containing 'Time' and 'Amount'.

    Returns:
        pd.DataFrame: New DataFrame containing original features plus engineered features.
    """
    df_feat = df.copy()

    if "Time" in df_feat.columns:
        # Time represents seconds elapsed from the first transaction in the dataset.
        # Compute hour of day (0 to 23)
        df_feat["Hour"] = ((df_feat["Time"] // 3600) % 24).astype(int)

    if "Amount" in df_feat.columns:
        # np.log1p handles 0 amounts safely and dampens massive skewness
        df_feat["Log_Amount"] = np.log1p(np.maximum(0, df_feat["Amount"]))

    return df_feat
