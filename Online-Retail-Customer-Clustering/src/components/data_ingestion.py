"""
Data Ingestion component.

Responsible for reading the raw "Online Retail II" dataset (Excel file) and
saving a copy of the raw data as CSV inside the artifacts folder so that the
rest of the pipeline works off a stable, version-controlled artifact instead
of re-reading the (slow) Excel file every time.
"""
import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data", "raw_data.csv")
    source_file_path: str = os.path.join("data", "online_retail_II.xlsx")
    # Matches the original notebook, which read the first sheet (sheet_name=0).
    # The public "Online Retail II" dataset ships two sheets: "Year 2009-2010"
    # and "Year 2010-2011" - change this if you want the other year.
    sheet_name: int = 0


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def initiate_data_ingestion(self) -> str:
        """
        Read the source Excel file and persist it as a CSV artifact.

        Returns
        -------
        str
            Path to the saved raw data CSV artifact.
        """
        logger.info("Starting data ingestion")
        try:
            logger.info(f"Reading source data from {self.config.source_file_path}")

            df = pd.read_excel(self.config.source_file_path, sheet_name=self.config.sheet_name)

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)

            logger.info(
                f"Data ingestion completed. {len(df)} rows saved to {self.config.raw_data_path}"
            )

            return self.config.raw_data_path

        except Exception as e:
            logger.error("Error occurred during data ingestion")
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()