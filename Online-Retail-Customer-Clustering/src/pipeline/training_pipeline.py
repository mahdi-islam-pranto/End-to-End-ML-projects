"""
Training Pipeline.

Orchestrates the full workflow end to end:
    raw Excel data -> cleaning -> RFM feature engineering -> outlier removal
    & scaling -> KMeans training -> labeled customer segments.

Run directly with:
    python -m src.pipeline.training_pipeline
"""
import os
import sys

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.feature_engineering import FeatureEngineering
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

FINAL_OUTPUT_PATH = os.path.join("artifacts", "data", "full_clustering_result.csv")


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_cleaning = DataCleaning()
        self.feature_engineering = FeatureEngineering()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run(self) -> pd.DataFrame:
        logger.info("=== Starting training pipeline ===")
        try:
            # 1. Ingest raw data
            raw_data_path = self.data_ingestion.initiate_data_ingestion()
            raw_df = pd.read_csv(raw_data_path)

            # 2. Clean data
            cleaned_df = self.data_cleaning.initiate_data_cleaning(raw_df)

            # 3. Feature engineering (RFM)
            rfm_df = self.feature_engineering.initiate_feature_engineering(cleaned_df)
            rfm_df = rfm_df.set_index("Customer ID")

            # 4. Outlier removal + scaling
            scaled_data_df, non_outliers_df, outlier_segments_df = (
                self.data_transformation.initiate_data_transformation(rfm_df)
            )

            # 5. Train KMeans and assemble final labeled segments
            full_clustering_df = self.model_trainer.initiate_model_training(
                scaled_data_df, non_outliers_df, outlier_segments_df
            )

            os.makedirs(os.path.dirname(FINAL_OUTPUT_PATH), exist_ok=True)
            full_clustering_df.to_csv(FINAL_OUTPUT_PATH)

            logger.info(f"=== Training pipeline completed. Results saved to {FINAL_OUTPUT_PATH} ===")

            return full_clustering_df

        except Exception as e:
            logger.error("Training pipeline failed")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    result_df = pipeline.run()
    print(result_df["ClusterLabel"].value_counts())