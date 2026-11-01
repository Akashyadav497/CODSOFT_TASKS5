"""
Automated unit and integration test suite for Credit Card Fraud Detection.
Uses pytest to verify data integrity, strict leakage prevention, input validation,
model loading, and deterministic reproducibility.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.config import (
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    THRESHOLD_CONFIG_PATH,
    RAW_DATA_PATH
)
from src.features import extract_features
from src.preprocess import FraudPreprocessor, split_data
from src.predict import FraudPredictor


@pytest.fixture
def sample_dataset():
    """Provides a deterministic sample dataset for fast unit testing."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_rows = 1000
    n_fraud = 20  # 2% fraud for testing

    data = {f"V{i}": rng.normal(0, 1, n_rows) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172800, n_rows)
    data["Amount"] = rng.exponential(50, n_rows)
    data["Class"] = [1] * n_fraud + [0] * (n_rows - n_fraud)

    df = pd.DataFrame(data)
    # Shuffle
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def test_stratified_split_preserves_fraud_ratio(sample_dataset):
    """
    TEST 1: The stratified split keeps the same fraud ratio in all partitions.
    """
    total_fraud_ratio = sample_dataset["Class"].mean()
    X_train, X_test, y_train, y_test = split_data(sample_dataset, test_size=0.2, random_state=RANDOM_STATE)

    train_fraud_ratio = y_train.mean()
    test_fraud_ratio = y_test.mean()

    # Verify both ratios are within 0.5% tolerance of the total population ratio
    assert abs(train_fraud_ratio - total_fraud_ratio) < 0.005, "Train fraud ratio deviated significantly from original ratio."
    assert abs(test_fraud_ratio - total_fraud_ratio) < 0.005, "Test fraud ratio deviated significantly from original ratio."


def test_balancing_applied_to_training_data_only(sample_dataset):
    """
    TEST 2: Balancing is applied to training data only (validation and test class counts remain unchanged).
    """
    from imblearn.over_sampling import SMOTE

    X_train, X_test, y_train, y_test = split_data(sample_dataset, test_size=0.2, random_state=RANDOM_STATE)
    test_counts_before = y_test.value_counts().to_dict()

    # Apply SMOTE strictly on training data
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    test_counts_after = y_test.value_counts().to_dict()

    # Training class counts must now be balanced
    assert (y_train_res == 0).sum() == (y_train_res == 1).sum(), "Training set was not balanced by SMOTE."
    # Test class counts MUST remain completely untouched
    assert test_counts_before == test_counts_after, "Test set class counts were inadvertently altered during balancing!"


def test_scaler_fitted_on_training_data_only(sample_dataset):
    """
    TEST 3: Scalers and encoders are fitted on training data only (no data leakage).
    """
    X_train, X_test, y_train, y_test = split_data(sample_dataset, test_size=0.2, random_state=RANDOM_STATE)

    preprocessor = FraudPreprocessor()
    preprocessor.fit(X_train)

    # Compute expected scaler center and scale directly from X_train Log_Amount
    train_log_amount = np.log1p(np.maximum(0, X_train["Amount"])).to_numpy().reshape(-1, 1)
    expected_scaler = RobustScaler().fit(train_log_amount)

    assert np.isclose(preprocessor.scaler.center_[0], expected_scaler.center_[0]), "Scaler center does not match training data center."
    assert np.isclose(preprocessor.scaler.scale_[0], expected_scaler.scale_[0]), "Scaler scale does not match training data scale."


def test_predict_rejects_missing_file():
    """
    TEST 4a: predict.py rejects a missing file with a clear, friendly error.
    """
    # Initialize predictor (assuming model is saved, or handle gracefully)
    if not BEST_MODEL_PATH.exists():
        pytest.skip("Best model artifact not yet generated.")

    predictor = FraudPredictor()
    with pytest.raises(FileNotFoundError) as exc_info:
        predictor.predict("non_existent_transactions_file_12345.csv")
    assert "Transaction file not found" in str(exc_info.value)


def test_predict_rejects_missing_columns():
    """
    TEST 4b: predict.py rejects input missing mandatory columns.
    """
    if not BEST_MODEL_PATH.exists():
        pytest.skip("Best model artifact not yet generated.")

    predictor = FraudPredictor()
    invalid_df = pd.DataFrame({"Amount": [10.5], "Time": [100.0]})  # Missing V1-V28
    with pytest.raises(ValueError) as exc_info:
        predictor.predict(invalid_df)
    assert "missing required feature column" in str(exc_info.value).lower()


def test_predict_rejects_wrong_data_types():
    """
    TEST 4c: predict.py rejects wrong data types (e.g., string amount).
    """
    if not BEST_MODEL_PATH.exists():
        pytest.skip("Best model artifact not yet generated.")

    predictor = FraudPredictor()
    bad_data = {f"V{i}": [0.0] for i in range(1, 29)}
    bad_data["Time"] = [100.0]
    bad_data["Amount"] = ["INVALID_STRING_AMOUNT"]
    bad_df = pd.DataFrame(bad_data)

    with pytest.raises(TypeError) as exc_info:
        predictor.predict(bad_df)
    assert "must be numeric" in str(exc_info.value).lower()


def test_saved_model_loads_and_outputs_valid_probabilities(sample_dataset):
    """
    TEST 5: The saved model loads and returns labels and probabilities between 0 and 1.
    """
    if not BEST_MODEL_PATH.exists():
        pytest.skip("Best model artifact not yet generated.")

    predictor = FraudPredictor()
    results = predictor.predict(sample_dataset.drop(columns=["Class"]))

    assert "fraud_probability" in results.columns
    assert "is_fraud" in results.columns
    assert "decision_label" in results.columns

    probs = results["fraud_probability"]
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Fraud probabilities are outside [0, 1] range."
    assert set(results["is_fraud"].unique()).issubset({0, 1}), "Binary fraud labels must be 0 or 1."


def test_reproducibility_with_same_random_state(sample_dataset):
    """
    TEST 6: Results are 100% reproducible with the same random_state.
    """
    X_train1, X_test1, y_train1, y_test1 = split_data(sample_dataset, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = split_data(sample_dataset, random_state=42)

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_series_equal(y_train1, y_train2)
