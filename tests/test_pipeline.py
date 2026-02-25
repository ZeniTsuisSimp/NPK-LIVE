"""
Unit Tests for the NPK Crop Recommendation MLOps Pipeline
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import load_data, preprocess, load_params


# ─── Test Data Loading ────────────────────────────────────────────────────────

class TestDataLoading:
    """Tests for data loading functionality."""

    def test_csv_exists(self):
        """Verify the dataset file exists."""
        assert os.path.exists("data/Crop_recommendation.csv"), \
            "Dataset file not found at data/Crop_recommendation.csv"

    def test_load_data_shape(self):
        """Verify loaded data has expected columns."""
        df = load_data("data/Crop_recommendation.csv")
        assert "N" in df.columns
        assert "P" in df.columns
        assert "K" in df.columns
        assert "Crop" in df.columns
        assert len(df) > 0

    def test_no_missing_values(self):
        """Verify no missing values in key columns."""
        df = load_data("data/Crop_recommendation.csv")
        assert df[["N", "P", "K", "Crop"]].isnull().sum().sum() == 0


# ─── Test Preprocessing ──────────────────────────────────────────────────────

class TestPreprocessing:
    """Tests for data preprocessing pipeline."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        n_samples = 200
        crops = ["Rice", "Wheat", "Corn", "Soybean", "Cotton"]
        return pd.DataFrame({
            "N": np.random.uniform(20, 200, n_samples),
            "P": np.random.uniform(10, 120, n_samples),
            "K": np.random.uniform(20, 200, n_samples),
            "Crop": np.random.choice(crops, n_samples),
        })

    def test_preprocess_output_keys(self, sample_df):
        """Verify preprocessing returns all expected keys."""
        result = preprocess(sample_df, test_size=0.2, random_state=42)
        expected_keys = [
            "X_train", "X_test", "y_train", "y_test",
            "scaler", "label_encoder", "feature_names", "target_names"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_preprocess_shapes(self, sample_df):
        """Verify train/test split ratios are correct."""
        result = preprocess(sample_df, test_size=0.2, random_state=42)
        total = len(sample_df)
        assert result["X_train"].shape[0] == int(total * 0.8)
        assert result["X_test"].shape[0] == int(total * 0.2)
        assert result["X_train"].shape[1] == 3  # N, P, K

    def test_scaling_mean_near_zero(self, sample_df):
        """Verify scaled training data has ~zero mean."""
        result = preprocess(sample_df, test_size=0.2, random_state=42)
        means = np.abs(result["X_train"].mean(axis=0))
        assert all(m < 0.1 for m in means), f"Scaled means not near zero: {means}"


# ─── Test Soil Health Score ────────────────────────────────────────────────────

class TestSoilHealth:
    """Tests for soil health computation (from app logic)."""

    def compute_soil_health(self, n, p, k):
        """Replicate the soil health scoring function."""
        opt_n, opt_p, opt_k = 105, 60, 100

        def bell(value, optimal, sigma=50):
            return math.exp(-0.5 * ((value - optimal) / sigma) ** 2) * 100

        score_n = bell(n, opt_n, 55)
        score_p = bell(p, opt_p, 35)
        score_k = bell(k, opt_k, 50)

        base = 0.35 * score_n + 0.30 * score_p + 0.35 * score_k

        values = [n, p, k]
        if min(values) > 0:
            ratio = max(values) / min(values)
            if ratio < 2.5:
                balance_bonus = max(0, 10 * (1 - (ratio - 1) / 1.5))
            else:
                balance_bonus = 0
        else:
            balance_bonus = 0

        return min(100, round(base + balance_bonus, 1))

    def test_optimal_values_high_score(self):
        """Optimal NPK values should give a high health score."""
        score = self.compute_soil_health(105, 60, 100)
        assert score >= 80, f"Optimal values gave low score: {score}"

    def test_zero_values_low_score(self):
        """Zero NPK values should give a low health score."""
        score = self.compute_soil_health(0, 0, 0)
        assert score < 50, f"Zero values gave high score: {score}"

    def test_score_range(self):
        """Score should always be between 0 and 100."""
        for _ in range(100):
            n = np.random.uniform(0, 300)
            p = np.random.uniform(0, 200)
            k = np.random.uniform(0, 250)
            score = self.compute_soil_health(n, p, k)
            assert 0 <= score <= 100, f"Score out of range: {score}"


# ─── Test Model Prediction ────────────────────────────────────────────────────

class TestModelPrediction:
    """Tests for model loading and prediction."""

    def test_model_file_exists(self):
        """Verify trained model file exists."""
        assert os.path.exists("models/npk_crop_model.pkl"), \
            "Model file not found at models/npk_crop_model.pkl"

    def test_model_loads(self):
        """Verify model bundle loads correctly."""
        import joblib
        bundle = joblib.load("models/npk_crop_model.pkl")
        assert "model" in bundle
        assert "scaler" in bundle
        assert "label_encoder" in bundle

    def test_prediction_output(self):
        """Verify model produces valid predictions."""
        import joblib
        bundle = joblib.load("models/npk_crop_model.pkl")
        input_data = np.array([[100, 50, 80]])
        scaled = bundle["scaler"].transform(input_data)
        prediction = bundle["model"].predict(scaled)
        assert len(prediction) == 1
        crop = bundle["label_encoder"].inverse_transform(prediction)
        assert isinstance(crop[0], str)


# ─── Test Params ──────────────────────────────────────────────────────────────

class TestParams:
    """Tests for parameter configuration."""

    def test_params_file_exists(self):
        """Verify params.yaml exists."""
        assert os.path.exists("params.yaml"), "params.yaml not found"

    def test_params_structure(self):
        """Verify params.yaml has expected sections."""
        params = load_params()
        assert "data" in params
        assert "model" in params
        assert "mlflow" in params
        assert "test_size" in params["data"]
        assert "n_estimators" in params["model"]
