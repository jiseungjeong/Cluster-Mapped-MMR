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


class SimilaritySelector(ExampleSelector):
    """Similarity-based example selection technique - baseline approach"""

    def select_examples(
        self, query_embedding: np.ndarray, n_examples: int = 5
    ) -> List[Dict]:
        """Select examples most similar to the query based on embedding similarity"""
        n_examples = min(n_examples, len(self.example_pool))

        # Normalize embeddings for cosine similarity
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        normalized_examples = self.example_embeddings / np.linalg.norm(
            self.example_embeddings, axis=1, keepdims=True
        )

        # Calculate similarity between query and all examples
        similarities = np.dot(normalized_examples, normalized_query)

        # Get indices of top-n most similar examples
        top_indices = np.argsort(-similarities)[:n_examples]

        # Return selected examples
        selected_examples = [self.example_pool[i] for i in top_indices]
        return selected_examples


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
        self.static_examples = None  # 새로 추가: 각 클러스터의 대표 예제 저장

        # Perform clustering
        self._perform_clustering()

        # 각 클러스터의 대표 예제 미리 선택
        self._preselect_cluster_representatives()

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

    def _preselect_cluster_representatives(self):
        """각 클러스터의 대표 예제를 미리 선택"""
        if self.labels is None or self.centroids is None:
            logger.warning("클러스터링이 수행되지 않았습니다.")
            return

        unique_clusters = np.unique(self.labels)
        if -1 in unique_clusters:  # HDBSCAN의 경우 노이즈 포인트 제외
            unique_clusters = unique_clusters[unique_clusters != -1]

        # 각 클러스터별 대표 예제 선택
        self.static_examples = {}
        for cluster_idx in unique_clusters:
            # 현재 클러스터에 속한 예제 인덱스 찾기
            cluster_example_indices = np.where(self.labels == cluster_idx)[0]

            if len(cluster_example_indices) == 0:
                continue

            # 클러스터 centroid 정규화
            centroid = self.centroids[cluster_idx]
            normalized_centroid = centroid / np.linalg.norm(centroid)

            # 클러스터 내 예제 임베딩 정규화
            cluster_embeddings = self.example_embeddings[cluster_example_indices]
            normalized_embeddings = cluster_embeddings / np.linalg.norm(
                cluster_embeddings, axis=1, keepdims=True
            )

            # centroid와의 유사도 계산
            similarities = np.dot(normalized_embeddings, normalized_centroid)

            # 유사도가 가장 높은 예제 선택
            most_similar_idx = np.argmax(similarities)
            representative_example = self.example_pool[
                cluster_example_indices[most_similar_idx]
            ]

            # 클러스터별 대표 예제 저장
            self.static_examples[cluster_idx] = representative_example

        logger.info(f"각 클러스터별 대표 예제 {len(self.static_examples)}개 선택 완료")

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
        """정적으로 미리 선택된 클러스터 대표 예제들 반환"""
        if self.static_examples is None or len(self.static_examples) == 0:
            logger.warning(
                "미리 선택된 예제가 없습니다. 무작위 선택 방법을 사용합니다."
            )
            return RandomSelector(
                self.example_pool, self.example_embeddings
            ).select_examples(query_embedding, n_examples)

        # 쿼리와 각 클러스터 centroid 간의 유사도 계산
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        normalized_centroids = self.centroids / np.linalg.norm(
            self.centroids, axis=1, keepdims=True
        )
        centroid_similarities = np.dot(normalized_centroids, normalized_query)

        # 유사도 기준으로 클러스터 인덱스 정렬
        sorted_cluster_indices = np.argsort(-centroid_similarities)

        # 정렬된 클러스터 순서대로 미리 선택된 대표 예제 반환
        selected_examples = []
        for cluster_idx in sorted_cluster_indices:
            if len(selected_examples) >= n_examples:
                break

            if cluster_idx in self.static_examples:
                selected_examples.append(self.static_examples[cluster_idx])

        # 충분한 예제를 못 찾았다면 무작위로 추가
        if len(selected_examples) < n_examples:
            remaining_needed = n_examples - len(selected_examples)
            already_selected_ids = {ex["id"] for ex in selected_examples}

            # 중복 없이 남은 예제 풀에서 선택
            remaining_examples = [
                ex for ex in self.example_pool if ex["id"] not in already_selected_ids
            ]

            if remaining_examples:
                random_indices = random.sample(
                    range(len(remaining_examples)),
                    min(remaining_needed, len(remaining_examples)),
                )
                selected_examples.extend(
                    [remaining_examples[i] for i in random_indices]
                )

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


class CmHdbscanMmrSelector(ExampleSelector):
    """CM-HDBSCAN-MMR (Clustering with HDBSCAN + MMR + Dynamic Mapping) example selection technique"""

    def __init__(
        self,
        example_pool: List[Dict],
        example_embeddings: np.ndarray,
        lambda_param: float = 0.7,
        min_cluster_size: int = 5,
        min_samples: int = None,
    ):
        """
        Initialize CM-HDBSCAN-MMR selector

        Args:
            example_pool: List of examples
            example_embeddings: Array of example embeddings
            lambda_param: MMR diversity weight (0: diversity only, 1: similarity only)
            min_cluster_size: Minimum size of clusters for HDBSCAN
            min_samples: Minimum number of samples in a neighborhood for HDBSCAN
        """
        super().__init__(example_pool, example_embeddings)
        self.lambda_param = lambda_param
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.labels = None
        self.centroids = None
        self.unique_labels = None

        # Perform clustering using HDBSCAN
        self._perform_clustering()

    def _perform_clustering(self):
        """Perform HDBSCAN clustering on embedding data"""
        # HDBSCAN 파라미터 설정
        hdbscan_params = {
            "min_cluster_size": self.min_cluster_size,
            "gen_min_span_tree": True,
            "prediction_data": True,  # 새로운 데이터 포인트 할당 가능하게
        }

        if self.min_samples:
            hdbscan_params["min_samples"] = self.min_samples

        # HDBSCAN 클러스터링 실행
        clusterer = hdbscan.HDBSCAN(**hdbscan_params)
        self.labels = clusterer.fit_predict(self.example_embeddings)
        self.clusterer = clusterer  # 모델 저장 (새로운 쿼리 포인트 할당 위해)

        # 고유 레이블 저장 (노이즈 포인트 제외)
        self.unique_labels = np.unique(self.labels)
        self.valid_labels = self.unique_labels[self.unique_labels != -1]

        # 각 클러스터의 중심점 계산
        self.centroids = []
        for label in self.valid_labels:
            cluster_points = self.example_embeddings[self.labels == label]
            centroid = np.mean(cluster_points, axis=0)
            self.centroids.append(centroid)

        self.centroids = np.array(self.centroids)

        # 클러스터 정보 로깅
        n_clusters = len(self.valid_labels)
        n_noise = np.sum(self.labels == -1)
        logger.info(
            f"HDBSCAN clustering completed with {n_clusters} clusters and {n_noise} noise points"
        )

    def select_examples(
        self, query_embedding: np.ndarray, n_examples: int = 5
    ) -> List[Dict]:
        """Select examples using CM-HDBSCAN-MMR technique"""
        # 노이즈 포인트를 처리할 대체 방법 준비
        fallback_selector = MMRSelector(
            self.example_pool, self.example_embeddings, self.lambda_param
        )

        # 유효한 클러스터가 없는 경우 대체 방법 사용
        if len(self.valid_labels) == 0:
            logger.warning(
                "HDBSCAN did not find any valid clusters. Using MMR selection as fallback."
            )
            return fallback_selector.select_examples(query_embedding, n_examples)

        # 1. 쿼리에 가장 가까운 클러스터 찾기
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        normalized_centroids = self.centroids / np.linalg.norm(
            self.centroids, axis=1, keepdims=True
        )
        centroid_similarities = np.dot(normalized_centroids, normalized_query)
        closest_cluster_idx = np.argmax(centroid_similarities)
        closest_cluster = self.valid_labels[closest_cluster_idx]

        # 2. 해당 클러스터에서 예제 후보 가져오기
        cluster_example_indices = np.where(self.labels == closest_cluster)[0]

        # 클러스터가 너무 작은 경우 대체 방법 사용
        if len(cluster_example_indices) < 2:
            logger.warning(
                f"Selected cluster has too few examples ({len(cluster_example_indices)}). Using MMR selection as fallback."
            )
            return fallback_selector.select_examples(query_embedding, n_examples)

        cluster_examples = [self.example_pool[i] for i in cluster_example_indices]
        cluster_embeddings = self.example_embeddings[cluster_example_indices]

        # 클러스터가 요청한 예제 수보다 작으면 모두 반환
        if len(cluster_examples) <= n_examples:
            return cluster_examples

        # 3. MMR을 사용하여 클러스터 내 다양한 예제 선택
        normalized_cluster_embeddings = cluster_embeddings / np.linalg.norm(
            cluster_embeddings, axis=1, keepdims=True
        )

        # 쿼리-예제 유사도 계산
        sim_query = np.dot(normalized_cluster_embeddings, normalized_query)

        # MMR 기반 예제 선택
        selected_indices = []
        remaining_indices = list(range(len(cluster_examples)))

        for _ in range(n_examples):
            if not remaining_indices:
                break

            # MMR 계산
            if not selected_indices:  # 첫 번째 선택은 가장 유사한 예제
                mmr_idx = np.argmax(sim_query[remaining_indices])
                mmr_idx = remaining_indices[mmr_idx]
            else:
                # 선택된 예제와의 유사도 행렬 계산
                selected_embeddings = normalized_cluster_embeddings[selected_indices]
                remaining_embeddings = normalized_cluster_embeddings[remaining_indices]

                # 남은 예제와 이미 선택된 예제 간의 최대 유사도 계산
                sim_selected = np.max(
                    np.dot(remaining_embeddings, selected_embeddings.T), axis=1
                )

                # MMR 점수 계산
                mmr_scores = (
                    self.lambda_param * sim_query[remaining_indices]
                    - (1 - self.lambda_param) * sim_selected
                )

                # 최대 MMR 점수를 가진 인덱스 선택
                mmr_idx = remaining_indices[np.argmax(mmr_scores)]

            # 선택된 인덱스 업데이트
            selected_indices.append(mmr_idx)
            remaining_indices.remove(mmr_idx)

        # 선택된 예제 반환
        return [cluster_examples[i] for i in selected_indices]


def get_selector(
    method: str, example_pool: List[Dict], example_embeddings: np.ndarray, **kwargs
) -> ExampleSelector:
    """
    Return example selector based on selected method

    Args:
        method: Selection method ('kmeans', 'hdbscan', 'mmr', 'random', 'cm-mmr', 'cm-hdbscan-mmr', 'similarity')
        example_pool: Example pool
        example_embeddings: Example embeddings
        kwargs: Additional parameters

    Returns:
        ExampleSelector instance
    """
    method = method.lower()

    if method == "random":
        return RandomSelector(example_pool, example_embeddings)
    elif method == "similarity":
        return SimilaritySelector(example_pool, example_embeddings)
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
    elif method == "cm-hdbscan-mmr":
        lambda_param = kwargs.get("lambda_param", 0.7)
        min_cluster_size = kwargs.get("min_cluster_size", 5)
        min_samples = kwargs.get("min_samples", None)
        return CmHdbscanMmrSelector(
            example_pool,
            example_embeddings,
            lambda_param,
            min_cluster_size,
            min_samples,
        )
    else:
        logger.warning(f"Unknown selection method: {method}, using random selector")
        return RandomSelector(example_pool, example_embeddings)
