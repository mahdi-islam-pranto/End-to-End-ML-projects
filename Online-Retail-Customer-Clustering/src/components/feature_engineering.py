"""
Feature Engineering component.

Builds the classic RFM (Recency, Frequency, Monetary) feature set at the
customer level from cleaned invoice-line data.
"""
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureEngineeringConfig:
    customer_id_col: str = "Customer ID"
    invoice_col: str = "Invoice"
    invoice_date_col: str = "InvoiceDate"
    quantity_col: str = "Quantity"
    price_col: str = "Price"


class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig = FeatureEngineeringConfig()):
        self.config = config

    def add_sales_line_total(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["SalesLineTotal"] = df[self.config.quantity_col] * df[self.config.price_col]
        return df

    def build_rfm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate invoice-line data into one row per customer with RFM columns."""
        cfg = self.config

        df[cfg.invoice_date_col] = pd.to_datetime(df[cfg.invoice_date_col])

        aggregated_df = df.groupby(by=cfg.customer_id_col, as_index=False).agg(
            MonetaryValue=("SalesLineTotal", "sum"),
            Frequency=(cfg.invoice_col, "nunique"),
            LastInvoiceDate=(cfg.invoice_date_col, "max"),
        )

        max_invoice_date = aggregated_df["LastInvoiceDate"].max()
        aggregated_df["Recency"] = (max_invoice_date - aggregated_df["LastInvoiceDate"]).dt.days

        return aggregated_df

    def initiate_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature engineering pipeline and return the RFM dataframe."""
        logger.info("Starting feature engineering")
        try:
            df = self.add_sales_line_total(df)
            logger.info("Computed SalesLineTotal (Quantity * Price)")

            rfm_df = self.build_rfm_features(df)
            logger.info(
                f"Built RFM features for {len(rfm_df)} unique customers "
                f"(MonetaryValue, Frequency, Recency)"
            )

            return rfm_df

        except Exception as e:
            logger.error("Error occurred during feature engineering")
            raise CustomException(e, sys)


if __name__ == "__main__":
    sample_df = pd.read_csv("artifacts/data/cleaned_data.csv")
    fe = FeatureEngineering()
    rfm_df = fe.initiate_feature_engineering(sample_df)
    rfm_df.to_csv("artifacts/data/rfm_features.csv", index=False)