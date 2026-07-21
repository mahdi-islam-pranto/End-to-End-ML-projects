"""
Data Cleaning component.

Applies the cleaning rules discovered during EDA in the original notebook:
    * Invoice numbers must match the standard 6-digit pattern (drops
      cancellations/adjustments prefixed with "A" or "C").
    * StockCode must be a genuine product code (5 digits, 5 digits + letters,
      or the "PADS" special case) - drops discounts, postage, manual entries,
      bank charges, samples, test data, gift cards, etc.
    * Rows with a missing Customer ID are dropped (can't attribute to a
      customer for clustering).
    * Rows with Price <= 0 are dropped (returns/free items don't represent
      real monetary value).
"""
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataCleaningConfig:
    invoice_pattern: str = r"^\d{6}$"
    stock_code_numeric_pattern: str = r"^\d{5}$"
    stock_code_alpha_suffix_pattern: str = r"^\d{5}[a-zA-Z]+$"
    stock_code_pads_pattern: str = r"^PADS$"
    customer_id_col: str = "Customer ID"
    price_col: str = "Price"


class DataCleaning:
    def __init__(self, config: DataCleaningConfig = DataCleaningConfig()):
        self.config = config

    def clean_invoice(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Invoice"] = df["Invoice"].astype("str")
        mask = df["Invoice"].str.match(self.config.invoice_pattern) == True  # noqa: E712
        return df[mask]

    def clean_stock_code(self, df: pd.DataFrame) -> pd.DataFrame:
        df["StockCode"] = df["StockCode"].astype("str")
        mask = (
            (df["StockCode"].str.match(self.config.stock_code_numeric_pattern) == True)  # noqa: E712
            | (df["StockCode"].str.match(self.config.stock_code_alpha_suffix_pattern) == True)  # noqa: E712
            | (df["StockCode"].str.match(self.config.stock_code_pads_pattern) == True)  # noqa: E712
        )
        return df[mask]

    def drop_missing_customer_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=[self.config.customer_id_col])
        return df

    def drop_non_positive_price(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df[self.config.price_col] > 0.0]

    def initiate_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline and return the cleaned dataframe."""
        logger.info("Starting data cleaning")
        try:
            original_len = len(df)
            cleaned_df = df.copy()

            cleaned_df = self.clean_invoice(cleaned_df)
            logger.info(f"After invoice filtering: {len(cleaned_df)} rows")

            cleaned_df = self.clean_stock_code(cleaned_df)
            logger.info(f"After stock code filtering: {len(cleaned_df)} rows")

            cleaned_df = self.drop_missing_customer_id(cleaned_df)
            logger.info(f"After dropping missing customer IDs: {len(cleaned_df)} rows")

            cleaned_df = self.drop_non_positive_price(cleaned_df)
            logger.info(f"After dropping non-positive prices: {len(cleaned_df)} rows")

            retained_pct = (len(cleaned_df) / original_len) * 100 if original_len else 0
            logger.info(
                f"Data cleaning completed. Retained {len(cleaned_df)}/{original_len} "
                f"rows ({retained_pct:.2f}%)"
            )

            return cleaned_df

        except Exception as e:
            logger.error("Error occurred during data cleaning")
            raise CustomException(e, sys)


if __name__ == "__main__":
    sample_df = pd.read_csv("artifacts/data/raw_data.csv")
    cleaner = DataCleaning()
    cleaned = cleaner.initiate_data_cleaning(sample_df)
    cleaned.to_csv("artifacts/data/cleaned_data.csv", index=False)