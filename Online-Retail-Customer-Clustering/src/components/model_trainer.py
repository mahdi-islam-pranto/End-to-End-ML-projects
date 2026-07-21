"""
Model Trainer component.

    * Searches over a range of k values, recording inertia and silhouette
      score for each, to help choose the number of clusters.
    * Fits the final KMeans model (default k=4, matching the analysis in the
      original notebook) on the scaled, non-outlier RFM data.
    * Combines the main clusters with the pre-tagged outlier segments
      (monetary/frequency/both) and maps every cluster id to a
      business-friendly label.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object

logger = get_logger(__name__)

# Business-friendly names for each cluster id, matching the analysis done in
# the original notebook.
DEFAULT_CLUSTER_LABELS: Dict[int, str] = {
    0: "RETAIN",
    1: "RE-ENGAGE",
    2: "NURTURE",
    3: "REWARD",
    -1: "PAMPER",
    -2: "UPSELL",
    -3: "DELIGHT",
}


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "models", "kmeans_model.pkl")
    n_clusters: int = 4
    random_state: int = 42
    max_iter: int = 1000
    k_search_range: Tuple[int, int] = (2, 12)
    cluster_labels: Dict[int, str] = field(default_factory=lambda: dict(DEFAULT_CLUSTER_LABELS))


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config

    def find_optimal_k(self, scaled_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit KMeans for every k in the configured search range and record
        inertia + silhouette score, to help pick the best number of clusters.
        """
        k_min, k_max = self.config.k_search_range
        results: List[dict] = []

        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, max_iter=self.config.max_iter)
            labels = kmeans.fit_predict(scaled_data_df)
            sil_score = silhouette_score(scaled_data_df, labels)

            results.append({"k": k, "inertia": kmeans.inertia_, "silhouette_score": sil_score})
            logger.info(f"k={k} -> inertia={kmeans.inertia_:.2f}, silhouette={sil_score:.4f}")

        return pd.DataFrame(results)

    def train_final_model(self, scaled_data_df: pd.DataFrame) -> Tuple[KMeans, pd.Series]:
        """Fit the final KMeans model with the configured number of clusters."""
        kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            max_iter=self.config.max_iter,
        )
        cluster_labels = kmeans.fit_predict(scaled_data_df)

        save_object(self.config.model_path, kmeans)
        logger.info(
            f"Trained final KMeans (k={self.config.n_clusters}) and saved to "
            f"{self.config.model_path}"
        )

        return kmeans, pd.Series(cluster_labels, index=scaled_data_df.index, name="Cluster")

    def assemble_final_clusters(
        self,
        non_outliers_df: pd.DataFrame,
        cluster_labels: pd.Series,
        outlier_segments_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Attach cluster ids to the non-outlier customers, combine with the
        pre-tagged outlier segments, and map every cluster id to its
        business-friendly label.
        """
        non_outliers_df = non_outliers_df.copy()
        non_outliers_df["Cluster"] = cluster_labels

        full_df = pd.concat([non_outliers_df, outlier_segments_df])
        full_df["ClusterLabel"] = full_df["Cluster"].map(self.config.cluster_labels)

        return full_df

    def initiate_model_training(
        self,
        scaled_data_df: pd.DataFrame,
        non_outliers_df: pd.DataFrame,
        outlier_segments_df: pd.DataFrame,
        run_k_search: bool = True,
    ) -> pd.DataFrame:
        """Run the full model training pipeline and return the labeled customer dataframe."""
        logger.info("Starting model training")
        try:
            if run_k_search:
                k_search_results = self.find_optimal_k(scaled_data_df)
                os.makedirs("artifacts/data", exist_ok=True)
                k_search_results.to_csv("artifacts/data/k_search_results.csv", index=False)

            _, cluster_labels = self.train_final_model(scaled_data_df)

            full_clustering_df = self.assemble_final_clusters(
                non_outliers_df, cluster_labels, outlier_segments_df
            )
            logger.info(
                f"Model training completed. {len(full_clustering_df)} customers labeled "
                f"across {full_clustering_df['ClusterLabel'].nunique()} segments"
            )

            return full_clustering_df

        except Exception as e:
            logger.error("Error occurred during model training")
            raise CustomException(e, sys)


if __name__ == "__main__":
    scaled_data_df = pd.read_csv("artifacts/data/scaled_data.csv", index_col=0)
    non_outliers_df = pd.read_csv("artifacts/data/non_outliers.csv", index_col=0)
    outlier_segments_df = pd.read_csv("artifacts/data/outlier_segments.csv", index_col=0)

    trainer = ModelTrainer()
    full_clustering_df = trainer.initiate_model_training(
        scaled_data_df, non_outliers_df, outlier_segments_df
    )
    full_clustering_df.to_csv("artifacts/data/full_clustering_result.csv")