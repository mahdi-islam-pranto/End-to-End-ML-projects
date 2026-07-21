"""
Data Transformation component.

Handles:
    * IQR-based outlier detection on MonetaryValue and Frequency.
    * Splitting the RFM dataframe into a "non outliers" set (used to fit the
      main KMeans model) and an "outliers" set (monetary-only, frequency-only,
      and both), which are handled as their own special segments later.
    * Standard scaling (mean 0, std 1) of the non-outlier RFM features, which
      is what KMeans is actually trained on.

The fitted StandardScaler is persisted so the exact same transformation can
be re-applied to new customers at prediction time.
"""
import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object

logger = get_logger(__name__)

RFM_FEATURES = ["MonetaryValue", "Frequency", "Recency"]


@dataclass
class DataTransformationConfig:
    scaler_path: str = os.path.join("artifacts", "models", "scaler.pkl")
    outlier_bounds_path: str = os.path.join("artifacts", "models", "outlier_bounds.pkl")
    iqr_multiplier: float = 1.5


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    @staticmethod
    def _iqr_bounds(series: pd.Series, multiplier: float):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return q1 - multiplier * iqr, q3 + multiplier * iqr

    def detect_outliers(self, rfm_df: pd.DataFrame):
        """
        Split the RFM dataframe using IQR outlier detection on MonetaryValue
        and Frequency.

        Returns
        -------
        non_outliers_df, monetary_outliers_df, frequency_outliers_df
        """
        m_low, m_high = self._iqr_bounds(rfm_df["MonetaryValue"], self.config.iqr_multiplier)
        monetary_outliers_df = rfm_df[
            (rfm_df["MonetaryValue"] > m_high) | (rfm_df["MonetaryValue"] < m_low)
        ].copy()

        f_low, f_high = self._iqr_bounds(rfm_df["Frequency"], self.config.iqr_multiplier)
        frequency_outliers_df = rfm_df[
            (rfm_df["Frequency"] > f_high) | (rfm_df["Frequency"] < f_low)
        ].copy()

        non_outliers_df = rfm_df[
            (~rfm_df.index.isin(monetary_outliers_df.index))
            & (~rfm_df.index.isin(frequency_outliers_df.index))
        ].copy()

        bounds = {
            "monetary_low": m_low,
            "monetary_high": m_high,
            "frequency_low": f_low,
            "frequency_high": f_high,
        }
        save_object(self.config.outlier_bounds_path, bounds)
        logger.info(f"Saved outlier IQR bounds to {self.config.outlier_bounds_path}: {bounds}")

        return non_outliers_df, monetary_outliers_df, frequency_outliers_df

    def build_outlier_segments(self, monetary_outliers_df, frequency_outliers_df):
        """
        Combine monetary/frequency outliers into three mutually exclusive
        segments with special (negative) cluster codes:
            -1 : monetary-only outliers  ("PAMPER")
            -2 : frequency-only outliers ("UPSELL")
            -3 : monetary AND frequency outliers ("DELIGHT")
        """
        overlap_indices = monetary_outliers_df.index.intersection(frequency_outliers_df.index)

        monetary_only = monetary_outliers_df.drop(overlap_indices).copy()
        frequency_only = frequency_outliers_df.drop(overlap_indices).copy()
        both = monetary_outliers_df.loc[overlap_indices].copy()

        monetary_only["Cluster"] = -1
        frequency_only["Cluster"] = -2
        both["Cluster"] = -3

        outlier_segments_df = pd.concat([monetary_only, frequency_only, both])
        return outlier_segments_df

    def scale_features(self, non_outliers_df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Standard-scale the RFM features, fitting (and persisting) or reusing the scaler."""
        if fit:
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(non_outliers_df[RFM_FEATURES])
            save_object(self.config.scaler_path, scaler)
            logger.info(f"Fitted StandardScaler and saved to {self.config.scaler_path}")
        else:
            from src.utils import load_object

            scaler = load_object(self.config.scaler_path)
            scaled_values = scaler.transform(non_outliers_df[RFM_FEATURES])

        scaled_df = pd.DataFrame(
            scaled_values, index=non_outliers_df.index, columns=RFM_FEATURES
        )
        return scaled_df

    def initiate_data_transformation(self, rfm_df: pd.DataFrame):
        """
        Run the full transformation pipeline.

        Returns
        -------
        scaled_data_df : scaled features for non-outlier customers (KMeans input)
        non_outliers_df : original-scale RFM rows for non-outlier customers
        outlier_segments_df : original-scale RFM rows for outlier customers,
                               already tagged with their special Cluster code
        """
        logger.info("Starting data transformation")
        try:
            non_outliers_df, monetary_outliers_df, frequency_outliers_df = self.detect_outliers(
                rfm_df
            )
            logger.info(
                f"Outlier split -> non-outliers: {len(non_outliers_df)}, "
                f"monetary outliers: {len(monetary_outliers_df)}, "
                f"frequency outliers: {len(frequency_outliers_df)}"
            )

            outlier_segments_df = self.build_outlier_segments(
                monetary_outliers_df, frequency_outliers_df
            )

            scaled_data_df = self.scale_features(non_outliers_df, fit=True)
            logger.info("Scaled non-outlier RFM features (mean=0, std=1)")

            return scaled_data_df, non_outliers_df, outlier_segments_df

        except Exception as e:
            logger.error("Error occurred during data transformation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    rfm_df = pd.read_csv("artifacts/data/rfm_features.csv", index_col=0)
    transformer = DataTransformation()
    scaled_df, non_outliers_df, outlier_segments_df = transformer.initiate_data_transformation(
        rfm_df
    )
    scaled_df.to_csv("artifacts/data/scaled_data.csv")
    non_outliers_df.to_csv("artifacts/data/non_outliers.csv")
    outlier_segments_df.to_csv("artifacts/data/outlier_segments.csv")