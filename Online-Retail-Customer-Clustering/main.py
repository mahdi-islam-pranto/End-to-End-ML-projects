"""
Entry point for the Online Retail Customer Clustering project.

Run the full training pipeline:
    python main.py

Then generate plots (optional):
    python -m src.components.visualization

Then predict a segment for a new customer:
    python -m src.pipeline.predict_pipeline
"""
from src.pipeline.training_pipeline import TrainingPipeline


def main():
    pipeline = TrainingPipeline()
    result_df = pipeline.run()

    print("\nCustomer segment counts:")
    print(result_df["ClusterLabel"].value_counts())


if __name__ == "__main__":
    main()