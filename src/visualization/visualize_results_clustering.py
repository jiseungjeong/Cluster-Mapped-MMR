#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import internal modules
from src.utils.embedding import EmbeddingProcessor
from src.data.dataset import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Visualize clustering results from experiment"
    )

    # Experiment results settings
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to experiment results JSON file",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Data directory path"
    )

    # Visualization settings
    parser.add_argument(
        "--method",
        type=str,
        choices=["tsne", "pca"],
        default="tsne",
        help="Method for dimensionality reduction",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--plot_3d", action="store_true", help="Plot in 3D (default is 2D)"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-mpnet-base-v2",
        help="Embedding model to use",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation of embeddings",
    )

    return parser.parse_args()


def analyze_clusters_from_results(results_data, example_pool):
    """Analyze clusters based on experiment results"""
    # Dictionary to store examples and their clusters
    example_clusters = defaultdict(list)
    test_query_embeddings = []
    test_questions = []

    # Extract selected examples for each test question
    for item in results_data:
        question_id = item.get("question_id", "")
        question = item.get("question", "")
        selected_examples = item.get("selected_examples", [])

        test_questions.append(
            {
                "id": question_id,
                "question": question,
                "selected_examples": selected_examples,
            }
        )

        # Group examples by test question
        for example_id in selected_examples:
            example_clusters[question_id].append(example_id)

    logger.info(f"Found {len(test_questions)} test questions with selected examples")

    # Find example objects from their IDs
    id_to_example = {example["id"]: example for example in example_pool}

    # Extract unique clusters
    clusters = []
    for question_id, example_ids in example_clusters.items():
        examples = [
            id_to_example.get(ex_id) for ex_id in example_ids if ex_id in id_to_example
        ]
        if examples:
            clusters.append({"question_id": question_id, "examples": examples})

    logger.info(f"Extracted {len(clusters)} clusters")
    return clusters, test_questions


def extract_all_examples_from_clusters(clusters):
    """Extract all examples from clusters with their cluster assignments"""
    all_examples = []
    cluster_labels = []
    example_ids = set()

    for cluster_idx, cluster in enumerate(clusters):
        for example in cluster["examples"]:
            if example["id"] not in example_ids:  # Avoid duplicates
                all_examples.append(example)
                cluster_labels.append(cluster_idx)
                example_ids.add(example["id"])

    logger.info(f"Extracted {len(all_examples)} unique examples across all clusters")
    return all_examples, np.array(cluster_labels)


def visualize_clusters_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_embeddings: Optional[np.ndarray] = None,
    output_file: str = "clustering_visualization_2d.png",
    title: str = "Clustering Visualization",
    method: str = "tsne",
):
    """Visualize clustering results in 2D using t-SNE or PCA"""
    # Dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)

        if test_embeddings is not None:
            test_reducer = TSNE(n_components=2, random_state=43)
            reduced_test = test_reducer.fit_transform(test_embeddings)
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)

        if test_embeddings is not None:
            reduced_test = reducer.transform(test_embeddings)

    # Plotting
    plt.figure(figsize=(12, 10))

    # Plot examples colored by cluster
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))

    for i, label in enumerate(unique_labels):
        cluster_points = reduced_embeddings[labels == label]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors[i]],
            label=f"Cluster {label}",
            alpha=0.7,
            s=50,
        )

    # Plot test questions if available
    if test_embeddings is not None and len(reduced_test) > 0:
        plt.scatter(
            reduced_test[:, 0],
            reduced_test[:, 1],
            c="black",
            marker="*",
            s=200,
            alpha=1,
            label="Test Questions",
        )

    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    logger.info(f"Plot saved to: {output_file}")
    plt.close()


def visualize_clusters_3d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_embeddings: Optional[np.ndarray] = None,
    output_file: str = "clustering_visualization_3d.png",
    title: str = "Clustering Visualization (3D)",
    method: str = "tsne",
):
    """Visualize clustering results in 3D using t-SNE or PCA"""
    # Dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=3, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)

        if test_embeddings is not None:
            test_reducer = TSNE(n_components=3, random_state=43)
            reduced_test = test_reducer.fit_transform(test_embeddings)
    else:  # PCA
        reducer = PCA(n_components=3, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)

        if test_embeddings is not None:
            reduced_test = reducer.transform(test_embeddings)

    # Plotting
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Plot examples colored by cluster
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))

    for i, label in enumerate(unique_labels):
        cluster_points = reduced_embeddings[labels == label]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            c=[colors[i]],
            label=f"Cluster {label}",
            alpha=0.7,
            s=50,
        )

    # Plot test questions if available
    if test_embeddings is not None and len(reduced_test) > 0:
        ax.scatter(
            reduced_test[:, 0],
            reduced_test[:, 1],
            reduced_test[:, 2],
            c="black",
            marker="*",
            s=200,
            alpha=1,
            label="Test Questions",
        )

    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    logger.info(f"Plot saved to: {output_file}")
    plt.close()


def main():
    """Main function"""
    args = parse_args()

    # Load experiment results
    logger.info(f"Loading experiment results from: {args.results_file}")
    with open(args.results_file, "r", encoding="utf-8") as f:
        results_data = json.load(f)

    logger.info(f"Loaded {len(results_data)} experiment results")

    # Extract dataset name from results
    dataset_name = (
        results_data[0].get("dataset", "unknown") if results_data else "unknown"
    )
    method_name = (
        results_data[0].get("method", "unknown") if results_data else "unknown"
    )

    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = Dataset(dataset_name, args.data_dir)

    # Get example pool
    example_pool = dataset.get_example_pool()
    logger.info(f"Example pool size: {len(example_pool)}")

    # Analyze clusters from results
    clusters, test_questions = analyze_clusters_from_results(results_data, example_pool)

    # Extract all examples with cluster labels
    all_examples, cluster_labels = extract_all_examples_from_clusters(clusters)

    # Initialize embedding processor
    embedding_processor = EmbeddingProcessor(args.embedding_model)

    # Calculate embeddings directly to avoid cache size mismatch
    embeddings = embedding_processor.model.encode([ex["question"] for ex in all_examples], show_progress_bar=True)
    # Old line: all_examples, dataset_name, args.force_recompute
    # Removed: )

    # Calculate embeddings for test questions
    test_embeddings = embedding_processor.embed_questions(
        test_questions, f"{dataset_name}_test", args.force_recompute
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Visualization title and output path
    experiment_name = os.path.splitext(os.path.basename(args.results_file))[0]
    if experiment_name == "results":
        experiment_name = os.path.basename(os.path.dirname(args.results_file))

    title = f"Clustering Results for {dataset_name.upper()} ({method_name} method)"
    output_file = os.path.join(
        args.output_dir,
        f"{experiment_name}_{args.method}_{'3d' if args.plot_3d else '2d'}.png",
    )

    # Visualize clustering results
    if args.plot_3d:
        visualize_clusters_3d(
            embeddings, cluster_labels, test_embeddings, output_file, title, args.method
        )
    else:
        visualize_clusters_2d(
            embeddings, cluster_labels, test_embeddings, output_file, title, args.method
        )

    # Log cluster statistics
    logger.info("Cluster statistics:")
    unique_labels = np.unique(cluster_labels)

    for label in unique_labels:
        count = np.sum(cluster_labels == label)
        percentage = count / len(cluster_labels) * 100
        logger.info(f"Cluster {label}: {count} examples ({percentage:.2f}%)")

    logger.info("Visualization completed.")


if __name__ == "__main__":
    main()
