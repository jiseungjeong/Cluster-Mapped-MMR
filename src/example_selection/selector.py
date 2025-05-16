import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod
import logging
from sklearn.cluster import KMeans
import hdbscan
from sklearn.metrics import silhouette_score
import random

logger = logging.getLogger(__name__)


class ExampleSelector(ABC):
    """Abstract base class for example selection techniques"""

    def __init__(self, example_pool: List[Dict], example_embeddings: np.ndarray):
        """
        Initialize example selector

        Args:
            example_pool: List of examples
            example_embeddings: Array of example embeddings
        """
        self.example_pool = example_pool
        self.example_embeddings = example_embeddings

    @abstractmethod
    def select_examples(
        self, query_embedding: np.ndarray, n_examples: int = 5
    ) -> List[Dict]:
        """
        Select examples for a query

        Args:
            query_embedding: Query embedding
            n_examples: Number of examples to select

        Returns:
            List of selected examples
        """
        pass


class RandomSelector(ExampleSelector):
    """Random example selection technique"""

    def select_examples(
        self, query_embedding: np.ndarray, n_examples: int = 5
    ) -> List[Dict]:
        """Select examples randomly"""
        random_indices = random.sample(
            range(len(self.example_pool)), min(n_examples, len(self.example_pool))
        )
        return [self.example_pool[i] for i in random_indices]


class MMRSelector(ExampleSelector):
    """MMR (Maximal Marginal Relevance) based example selection technique"""

    def __init__(
        self,
        example_pool: List[Dict],
        example_embeddings: np.ndarray,
        lambda_param: float = 0.7,
    ):
        """
        Initialize MMR selector

        Args:
            example_pool: List of examples
            example_embeddings: Array of example embeddings
            lambda_param: MMR diversity weight (0: diversity only, 1: similarity only)
        """
        super().__init__(example_pool, example_embeddings)
        self.lambda_param = lambda_param

    def select_examples(
        self, query_embedding: np.ndarray, n_examples: int = 5
    ) -> List[Dict]:
        """Select examples using MMR technique"""
        n_examples = min(n_examples, len(self.example_pool))

        # Normalized embedding vectors
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        normalized_examples = self.example_embeddings / np.linalg.norm(
            self.example_embeddings, axis=1, keepdims=True
        )

        # Calculate query-example similarity
        sim_query = np.dot(normalized_examples, normalized_query)

        # MMR-based example selection
        selected_indices = []
        remaining_indices = list(range(len(self.example_pool)))

        for _ in range(n_examples):
            if not remaining_indices:
                break

            # Calculate MMR
            if not selected_indices:  # First selection is the highest similarity
                mmr_idx = np.argmax(sim_query[remaining_indices])
                mmr_idx = remaining_indices[mmr_idx]
            else:
                # Calculate similarity matrix with selected examples
                selected_embeddings = normalized_examples[selected_indices]
                remaining_embeddings = normalized_examples[remaining_indices]

                # Calculate maximum similarity between remaining and selected examples
                sim_selected = np.max(
                    np.dot(remaining_embeddings, selected_embeddings.T), axis=1
                )

                # Calculate MMR scores
                mmr_scores = (
                    self.lambda_param * sim_query[remaining_indices]
                    - (1 - self.lambda_param) * sim_selected
                )

                # Select index with maximum MMR score
                mmr_idx = remaining_indices[np.argmax(mmr_scores)]

            # Update selected indices
            selected_indices.append(mmr_idx)
            remaining_indices.remove(mmr_idx)

        # Return selected examples
        return [self.example_pool[i] for i in selected_indices]


class ClusteringSelector(ExampleSelector):
    """Clustering-based example selection technique"""

    def __init__(
        self,
        example_pool: List[Dict],
        example_embeddings: np.ndarray,
        method: str = "kmeans",
        n_clusters: int = 5,
        silhouette_threshold: float = 0.1,
    ):
        """
        Initialize clustering selector

        Args:
            example_pool: List of examples
            example_embeddings: Array of example embeddings
            method: Clustering method ('kmeans' or 'hdbscan')
            n_clusters: Number of clusters (used only in kmeans)
            silhouette_threshold: Minimum silhouette score (for automatic k selection)
        """
        super().__init__(example_pool, example_embeddings)
        self.method = method.lower()
        self.n_clusters = n_clusters
        self.silhouette_threshold = silhouette_threshold
        self.labels = None
        self.centroids = None

        # Perform clustering
        self._perform_clustering()

    def _perform_clustering(self):
        """Perform clustering on embedding data"""
        if self.method == "kmeans":
            if self.n_clusters <= 0:  # Automatic k selection
                self._auto_select_k()
            else:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
                self.labels = kmeans.fit_predict(self.example_embeddings)
                self.centroids = kmeans.cluster_centers_

        elif self.method == "hdbscan":
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
            self.labels = clusterer.fit_predict(self.example_embeddings)

            # Calculate cluster centroids for HDBSCAN
            unique_labels = np.unique(self.labels)
            self.centroids = np.array(
                [
                    np.mean(self.example_embeddings[self.labels == label], axis=0)
                    for label in unique_labels
                    if label != -1  # -1 represents noise points
                ]
            )

        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")

        logger.info(
            f"Clustering completed (method: {self.method}, number of clusters: {len(np.unique(self.labels))})"
        )

    def _auto_select_k(self, max_k: int = 20, min_k: int = 2):
        """Select optimal k based on silhouette score"""
        best_k = min_k
        best_score = -1

        for k in range(min_k, min(max_k + 1, len(self.example_embeddings))):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.example_embeddings)

            # Calculate silhouette score
            score = silhouette_score(self.example_embeddings, labels)
            logger.debug(f"k={k}, silhouette score={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k

            # Stop if we reached a sufficiently good score
            if score > self.silhouette_threshold:
                break

        logger.info(
            f"Optimal k selected: {best_k} (silhouette score: {best_score:.4f})"
        )

        # Perform clustering again with selected k
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        self.labels = kmeans.fit_predict(self.example_embeddings)
        self.centroids = kmeans.cluster_centers_
        self.n_clusters = best_k

    def select_examples(
        self, query_embedding: np.ndarray, n_examples: int = 5
    ) -> List[Dict]:
        """Select examples based on clustering"""
        if self.labels is None or self.centroids is None:
            logger.warning("Clustering has not been performed.")
            return RandomSelector(
                self.example_pool, self.example_embeddings
            ).select_examples(query_embedding, n_examples)

        # If number of clusters is less than requested examples, select additional from some clusters
        num_clusters = len(np.unique(self.labels))
        if num_clusters < 0:  # All points are noise in HDBSCAN
            return RandomSelector(
                self.example_pool, self.example_embeddings
            ).select_examples(query_embedding, n_examples)

        # Calculate similarity between query embedding and cluster centroids
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        normalized_centroids = self.centroids / np.linalg.norm(
            self.centroids, axis=1, keepdims=True
        )
        centroid_similarities = np.dot(normalized_centroids, normalized_query)

        # Sort clusters by similarity
        sorted_cluster_indices = np.argsort(-centroid_similarities)

        selected_examples = []
        examples_needed = n_examples

        # Select examples from each cluster
        for cluster_idx in sorted_cluster_indices:
            if examples_needed <= 0:
                break

            # Example indices belonging to current cluster
            cluster_example_indices = np.where(self.labels == cluster_idx)[0]

            if len(cluster_example_indices) == 0:
                continue

            # Number of examples to select from this cluster
            n_from_cluster = min(
                len(cluster_example_indices),
                max(1, examples_needed // (num_clusters - len(selected_examples) + 1)),
            )

            # Calculate similarity between cluster examples and query
            cluster_embeddings = self.example_embeddings[cluster_example_indices]
            normalized_cluster = cluster_embeddings / np.linalg.norm(
                cluster_embeddings, axis=1, keepdims=True
            )
            similarities = np.dot(normalized_cluster, normalized_query)

            # Select top n_from_cluster examples by similarity
            top_indices = np.argsort(-similarities)[:n_from_cluster]
            selected_from_cluster = [
                self.example_pool[cluster_example_indices[i]] for i in top_indices
            ]

            selected_examples.extend(selected_from_cluster)
            examples_needed -= len(selected_from_cluster)

        # If we still don't have enough examples, randomly select from remaining
        if examples_needed > 0:
            all_selected_indices = [
                self.example_pool.index(ex) for ex in selected_examples
            ]
            remaining_indices = [
                i
                for i in range(len(self.example_pool))
                if i not in all_selected_indices
            ]

            if remaining_indices:
                random_indices = random.sample(
                    remaining_indices, min(examples_needed, len(remaining_indices))
                )
                selected_examples.extend([self.example_pool[i] for i in random_indices])

        return selected_examples


class CmMmrSelector(ExampleSelector):
    """CM-MMR (Clustering + MMR + Dynamic Mapping) example selection technique"""

    def __init__(
        self,
        example_pool: List[Dict],
        example_embeddings: np.ndarray,
        lambda_param: float = 0.7,
        n_clusters: int = 5,
    ):
        """
        Initialize CM-MMR selector

        Args:
            example_pool: List of examples
            example_embeddings: Array of example embeddings
            lambda_param: MMR diversity weight (0: diversity only, 1: similarity only)
            n_clusters: Number of clusters for K-means
        """
        super().__init__(example_pool, example_embeddings)
        self.lambda_param = lambda_param
        self.n_clusters = n_clusters
        self.labels = None
        self.centroids = None

        # Perform clustering (K-means only)
        self._perform_clustering()

    def _perform_clustering(self):
        """Perform K-means clustering on embedding data"""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels = kmeans.fit_predict(self.example_embeddings)
        self.centroids = kmeans.cluster_centers_
        logger.info(f"K-means clustering completed with {self.n_clusters} clusters")

    def select_examples(
        self, query_embedding: np.ndarray, n_examples: int = 5
    ) -> List[Dict]:
        """Select examples using CM-MMR technique"""
        # 1. Map query to closest cluster
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        normalized_centroids = self.centroids / np.linalg.norm(
            self.centroids, axis=1, keepdims=True
        )
        centroid_similarities = np.dot(normalized_centroids, normalized_query)
        closest_cluster = np.argmax(centroid_similarities)

        # 2. Get local candidate pool from the cluster
        cluster_example_indices = np.where(self.labels == closest_cluster)[0]
        cluster_examples = [self.example_pool[i] for i in cluster_example_indices]
        cluster_embeddings = self.example_embeddings[cluster_example_indices]

        # If cluster is too small, return all examples from it
        if len(cluster_examples) <= n_examples:
            return cluster_examples[:n_examples]

        # 3. Apply MMR to select diverse examples from the cluster
        normalized_cluster_embeddings = cluster_embeddings / np.linalg.norm(
            cluster_embeddings, axis=1, keepdims=True
        )

        # Calculate query-example similarity
        sim_query = np.dot(normalized_cluster_embeddings, normalized_query)

        # MMR-based example selection
        selected_indices = []
        remaining_indices = list(range(len(cluster_examples)))

        for _ in range(n_examples):
            if not remaining_indices:
                break

            # Calculate MMR
            if not selected_indices:  # First selection is the highest similarity
                mmr_idx = np.argmax(sim_query[remaining_indices])
                mmr_idx = remaining_indices[mmr_idx]
            else:
                # Calculate similarity matrix with selected examples
                selected_embeddings = normalized_cluster_embeddings[selected_indices]
                remaining_embeddings = normalized_cluster_embeddings[remaining_indices]

                # Calculate maximum similarity between remaining and selected examples
                sim_selected = np.max(
                    np.dot(remaining_embeddings, selected_embeddings.T), axis=1
                )

                # Calculate MMR scores
                mmr_scores = (
                    self.lambda_param * sim_query[remaining_indices]
                    - (1 - self.lambda_param) * sim_selected
                )

                # Select index with maximum MMR score
                mmr_idx = remaining_indices[np.argmax(mmr_scores)]

            # Update selected indices
            selected_indices.append(mmr_idx)
            remaining_indices.remove(mmr_idx)

        # Return selected examples
        return [cluster_examples[i] for i in selected_indices]


def get_selector(
    method: str, example_pool: List[Dict], example_embeddings: np.ndarray, **kwargs
) -> ExampleSelector:
    """
    Return example selector based on selected method

    Args:
        method: Selection method ('kmeans', 'hdbscan', 'mmr', 'random', 'cm-mmr')
        example_pool: Example pool
        example_embeddings: Example embeddings
        kwargs: Additional parameters

    Returns:
        ExampleSelector instance
    """
    method = method.lower()

    if method == "random":
        return RandomSelector(example_pool, example_embeddings)
    elif method == "mmr":
        lambda_param = kwargs.get("lambda_param", 0.7)
        return MMRSelector(example_pool, example_embeddings, lambda_param)
    elif method in ["kmeans", "hdbscan"]:
        return ClusteringSelector(
            example_pool,
            example_embeddings,
            method=method,
            n_clusters=kwargs.get("n_clusters", 5),
            silhouette_threshold=kwargs.get("silhouette_threshold", 0.1),
        )
    elif method == "cm-mmr":
        lambda_param = kwargs.get("lambda_param", 0.7)
        n_clusters = kwargs.get("n_clusters", 5)
        return CmMmrSelector(example_pool, example_embeddings, lambda_param, n_clusters)
    else:
        logger.warning(f"Unknown selection method: {method}, using random selector")
        return RandomSelector(example_pool, example_embeddings)
