"""
Prediction Pipeline.

Loads the artifacts produced by the training pipeline (StandardScaler,
KMeans model, outlier IQR bounds) and uses them to assign a customer
segment to new customers.

Two entry points are provided:
    * predict_from_rfm(df)          - df already has MonetaryValue,
                                       Frequency, Recency columns (one row
                                       per customer).
    * predict_from_transactions(df) - df is raw invoice-line data (same
                                       shape as the source Excel file); RFM
                                       features are computed first.

Usage:
    from src.pipeline.predict_pipeline import PredictPipeline
    import pandas as pd

    pipeline = PredictPipeline()
    result = pipeline.predict_from_rfm(pd.DataFrame({
        "MonetaryValue": [1200.50],
        "Frequency": [8],
        "Recency": [15],
    }))
"""
import os
import sys

import pandas as pd

from src.components.data_cleaning import DataCleaning
from src.components.data_transformation import RFM_FEATURES
from src.components.feature_engineering import FeatureEngineering
from src.components.model_trainer import DEFAULT_CLUSTER_LABELS
from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_object

logger = get_logger(__name__)

SCALER_PATH = os.path.join("artifacts", "models", "scaler.pkl")
MODEL_PATH = os.path.join("artifacts", "models", "kmeans_model.pkl")
OUTLIER_BOUNDS_PATH = os.path.join("artifacts", "models", "outlier_bounds.pkl")


class PredictPipeline:
    def __init__(
        self,
        scaler_path: str = SCALER_PATH,
        model_path: str = MODEL_PATH,
        outlier_bounds_path: str = OUTLIER_BOUNDS_PATH,
        cluster_labels: dict = None,
    ):
        self.scaler_path = scaler_path
        self.model_path = model_path
        self.outlier_bounds_path = outlier_bounds_path
        self.cluster_labels = cluster_labels or dict(DEFAULT_CLUSTER_LABELS)

        self._scaler = None
        self._model = None
        self._outlier_bounds = None

    def _load_artifacts(self):
        if self._scaler is None:
            self._scaler = load_object(self.scaler_path)
        if self._model is None:
            self._model = load_object(self.model_path)
        if self._outlier_bounds is None:
            self._outlier_bounds = load_object(self.outlier_bounds_path)

    def _classify_outlier(self, row: pd.Series):
        """Return the special negative cluster id for a customer, or None if not an outlier."""
        bounds = self._outlier_bounds

        is_monetary_outlier = (row["MonetaryValue"] > bounds["monetary_high"]) or (
            row["MonetaryValue"] < bounds["monetary_low"]
        )
        is_frequency_outlier = (row["Frequency"] > bounds["frequency_high"]) or (
            row["Frequency"] < bounds["frequency_low"]
        )

        if is_monetary_outlier and is_frequency_outlier:
            return -3  # DELIGHT
        if is_monetary_outlier:
            return -1  # PAMPER
        if is_frequency_outlier:
            return -2  # UPSELL
        return None

    def predict_from_rfm(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign a Cluster / ClusterLabel to each row of an RFM dataframe.

        Parameters
        ----------
        rfm_df : pd.DataFrame
            Must contain MonetaryValue, Frequency, Recency columns.

        Returns
        -------
        pd.DataFrame
            Copy of the input with Cluster and ClusterLabel columns added.
        """
        logger.info(f"Predicting cluster for {len(rfm_df)} customer(s)")
        try:
            self._load_artifacts()

            missing_cols = [c for c in RFM_FEATURES if c not in rfm_df.columns]
            if missing_cols:
                raise ValueError(f"Input is missing required columns: {missing_cols}")

            result_df = rfm_df.copy()
            result_df["Cluster"] = None

            outlier_mask = result_df.apply(
                lambda row: self._classify_outlier(row) is not None, axis=1
            )

            if outlier_mask.any():
                result_df.loc[outlier_mask, "Cluster"] = result_df.loc[outlier_mask].apply(
                    self._classify_outlier, axis=1
                )

            non_outlier_rows = result_df.loc[~outlier_mask]
            if len(non_outlier_rows) > 0:
                scaled = self._scaler.transform(non_outlier_rows[RFM_FEATURES])
                predicted_clusters = self._model.predict(scaled)
                result_df.loc[~outlier_mask, "Cluster"] = predicted_clusters

            result_df["Cluster"] = result_df["Cluster"].astype(int)
            result_df["ClusterLabel"] = result_df["Cluster"].map(self.cluster_labels)

            logger.info("Prediction completed")
            return result_df

        except Exception as e:
            logger.error("Error occurred during prediction")
            raise CustomException(e, sys)

    def predict_from_transactions(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        End-to-end prediction starting from raw invoice-line data (same shape
        as the original source Excel file). Cleans the data, computes RFM
        features, then assigns clusters.
        """
        logger.info(f"Predicting clusters from {len(transactions_df)} raw transaction rows")
        try:
            cleaned_df = DataCleaning().initiate_data_cleaning(transactions_df)
            rfm_df = FeatureEngineering().initiate_feature_engineering(cleaned_df)
            rfm_df = rfm_df.set_index("Customer ID")

            return self.predict_from_rfm(rfm_df)

        except Exception as e:
            logger.error("Error occurred during prediction from transactions")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example: predict segments for a couple of hand-crafted customers
    sample_customers = pd.DataFrame(
        {
            "MonetaryValue": [5000.0, 120.0],
            "Frequency": [15, 1],
            "Recency": [5, 200],
        },
        index=pd.Index(["demo_customer_1", "demo_customer_2"], name="Customer ID"),
    )

    pipeline = PredictPipeline()
    predictions = pipeline.predict_from_rfm(sample_customers)
    print(predictions)