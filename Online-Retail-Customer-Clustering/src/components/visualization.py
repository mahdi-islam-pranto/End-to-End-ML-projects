"""
Visualization component.

Optional module that reproduces the plots from the original notebook (RFM
distributions, boxplots, elbow/silhouette curves, 3D scatter plots, cluster
violin plots, and the final cluster-distribution bar+line chart) and saves
them as PNG files instead of showing them inline, so this can run headlessly
on a server.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless backend, must be set before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

PLOTS_DIR = os.path.join("artifacts", "plots")

CLUSTER_COLORS = {
    0: "#1f77b4",  # Blue
    1: "#ff7f0e",  # Orange
    2: "#2ca02c",  # Green
    3: "#d62728",  # Red
    -1: "#9467bd",
    -2: "#8c564b",
    -3: "#e377c2",
}


def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_rfm_distributions(rfm_df: pd.DataFrame, filename: str = "rfm_distributions.png"):
    try:
        _ensure_plots_dir()
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.hist(rfm_df["MonetaryValue"], bins=10, color="skyblue", edgecolor="black")
        plt.title("Monetary Value Distribution")
        plt.xlabel("Monetary Value")
        plt.ylabel("Count")

        plt.subplot(1, 3, 2)
        plt.hist(rfm_df["Frequency"], bins=10, color="lightgreen", edgecolor="black")
        plt.title("Frequency Distribution")
        plt.xlabel("Frequency")
        plt.ylabel("Count")

        plt.subplot(1, 3, 3)
        plt.hist(rfm_df["Recency"], bins=20, color="salmon", edgecolor="black")
        plt.title("Recency Distribution")
        plt.xlabel("Recency")
        plt.ylabel("Count")

        plt.tight_layout()
        out_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved RFM distribution plot to {out_path}")

    except Exception as e:
        raise CustomException(e, sys)


def plot_k_search(k_search_results: pd.DataFrame, filename: str = "k_search.png"):
    try:
        _ensure_plots_dir()
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(k_search_results["k"], k_search_results["inertia"], marker="o")
        plt.title("KMeans Inertia for Different Values of k")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(
            k_search_results["k"],
            k_search_results["silhouette_score"],
            marker="o",
            color="orange",
        )
        plt.title("Silhouette Scores for Different Values of k")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.grid(True)

        plt.tight_layout()
        out_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved k-search plot to {out_path}")

    except Exception as e:
        raise CustomException(e, sys)


def plot_cluster_violin_plots(full_clustering_df: pd.DataFrame, filename: str = "cluster_violin_plots.png"):
    try:
        _ensure_plots_dir()
        plt.figure(figsize=(12, 18))

        for i, feature in enumerate(["MonetaryValue", "Frequency", "Recency"], start=1):
            plt.subplot(3, 1, i)
            sns.violinplot(
                x=full_clustering_df["Cluster"],
                y=full_clustering_df[feature],
                palette=CLUSTER_COLORS,
                hue=full_clustering_df["Cluster"],
                legend=False,
            )
            plt.title(f"{feature} by Cluster")
            plt.ylabel(feature)

        plt.tight_layout()
        out_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved cluster violin plots to {out_path}")

    except Exception as e:
        raise CustomException(e, sys)


def plot_cluster_summary(full_clustering_df: pd.DataFrame, filename: str = "cluster_summary.png"):
    """Bar chart of cluster sizes with average feature values overlaid as lines."""
    try:
        _ensure_plots_dir()
        df = full_clustering_df.copy()
        cluster_counts = df["ClusterLabel"].value_counts()
        df["MonetaryValue per 100 pounds"] = df["MonetaryValue"] / 100.00
        feature_means = df.groupby("ClusterLabel")[
            ["Recency", "Frequency", "MonetaryValue per 100 pounds"]
        ].mean()

        fig, ax1 = plt.subplots(figsize=(12, 8))

        sns.barplot(
            x=cluster_counts.index,
            y=cluster_counts.values,
            ax=ax1,
            palette="viridis",
            hue=cluster_counts.index,
            legend=False,
        )
        ax1.set_ylabel("Number of Customers", color="b")
        ax1.set_title("Cluster Distribution with Average Feature Values")

        ax2 = ax1.twinx()
        sns.lineplot(data=feature_means, ax=ax2, palette="Set2", marker="o")
        ax2.set_ylabel("Average Value", color="g")

        out_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved cluster summary plot to {out_path}")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    rfm_df = pd.read_csv("artifacts/data/rfm_features.csv")
    plot_rfm_distributions(rfm_df)

    k_search_results = pd.read_csv("artifacts/data/k_search_results.csv")
    plot_k_search(k_search_results)

    full_clustering_df = pd.read_csv("artifacts/data/full_clustering_result.csv")
    plot_cluster_violin_plots(full_clustering_df)
    plot_cluster_summary(full_clustering_df)