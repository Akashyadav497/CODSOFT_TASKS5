"""
Prediction interface module.
Provides both Python API and command-line interface (CLI) to evaluate transactions,
strictly validating inputs and generating flagged predictions with fraud probabilities.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
import joblib
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    THRESHOLD_CONFIG_PATH,
    PREDICTIONS_CSV_PATH,
    DEFAULT_FRAUD_THRESHOLD,
    RAW_FEATURE_COLUMNS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("predict")


class FraudPredictor:
    """
    Production-ready Credit Card Fraud Predictor.
    Loads saved model, preprocessor and calibrated threshold.
    """

    def __init__(
        self,
        model_path: Path = BEST_MODEL_PATH,
        preprocessor_path: Path = PREPROCESSOR_PATH,
        threshold_config_path: Path = THRESHOLD_CONFIG_PATH
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{model_path.resolve()}'. Please run 'python -m src.train' first."
            )
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor file not found at '{preprocessor_path.resolve()}'. Please run 'python -m src.train' first."
            )

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        if threshold_config_path.exists():
            with open(threshold_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.threshold = float(config.get("optimal_threshold", DEFAULT_FRAUD_THRESHOLD))
        else:
            self.threshold = DEFAULT_FRAUD_THRESHOLD

        logger.info(f"Loaded predictor with active decision threshold = {self.threshold:.4f}")

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validates input transaction DataFrame.
        """
        if df.empty:
            raise ValueError("Input DataFrame is completely empty. Please provide at least one transaction row.")

        # Required columns: Time, Amount, and V1-V28
        missing_columns = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Validation Error: Input data is missing required feature column(s): {missing_columns}"
            )

        # Check for non-numeric data in critical columns
        for col in ["Time", "Amount"]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Validation Error: Column '{col}' must be numeric, but received dtype '{df[col].dtype}'.")

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], Path, str],
        custom_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Predict fraud status and probabilities for one or more transactions.

        Args:
            data: A pandas DataFrame, single transaction dict, or Path to CSV file.
            custom_threshold: Optional override for the classification probability threshold.

        Returns:
            pd.DataFrame: Contains original data plus 'fraud_probability', 'is_fraud', and 'decision_label'.
        """
        threshold = custom_threshold if custom_threshold is not None else self.threshold

        if isinstance(data, (str, Path)):
            file_path = Path(data)
            if not file_path.exists():
                raise FileNotFoundError(f"Transaction file not found: {file_path.resolve()}")
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                raise ValueError(f"Unable to parse CSV file: {e}")
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Unsupported data format. Provide a DataFrame, dictionary, or CSV file path.")

        self.validate_dataframe(df)

        # Transform features
        X_trans = self.preprocessor.transform(df)

        # Predict probabilities
        probabilities = self.model.predict_proba(X_trans)[:, 1]
        binary_predictions = (probabilities >= threshold).astype(int)

        results_df = df.copy()
        results_df["fraud_probability"] = probabilities
        results_df["is_fraud"] = binary_predictions
        results_df["decision_label"] = np.where(binary_predictions == 1, "FRAUDULENT", "GENUINE")

        return results_df


def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Inference CLI")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV transaction file")
    parser.add_argument("--output", "-o", type=str, default=str(PREDICTIONS_CSV_PATH), help="Path to save predictions CSV")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Custom decision threshold (0.0 to 1.0)")
    parser.add_argument("--only-fraud", action="store_true", help="Filter output to only flagged fraudulent transactions")

    args = parser.parse_args()

    try:
        predictor = FraudPredictor()
        preds = predictor.predict(args.input, custom_threshold=args.threshold)

        if args.only_fraud:
            preds = preds[preds["is_fraud"] == 1]

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out_path, index=False)

        flagged_count = (preds["is_fraud"] == 1).sum() if not args.only_fraud else len(preds)
        print(f"\n[SUCCESS] Processed {len(preds)} transactions.")
        print(f"[SUMMARY] Flagged Fraudulent: {flagged_count} | Output saved to: {out_path.resolve()}\n")

    except Exception as e:
        print(f"\n[ERROR] {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
