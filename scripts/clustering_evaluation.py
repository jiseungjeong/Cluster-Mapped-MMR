#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import internal modules
from src.data.dataset import Dataset
from src.utils.embedding import EmbeddingProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering Evaluation")

    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="combined",
        help="Dataset to use (default: combined)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Data directory path"
    )

    # Embedding settings
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-mpnet-base-v2",
        help="Embedding model to use",
    )
    parser.add_argument(
        "--force_recompute", action="store_true", help="Whether to recompute embeddings"
    )

    # K-means settings
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of clusters for k-means",
    )

    # HDBSCAN settings
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=5,
        help="Minimum cluster size for HDBSCAN",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=None,
        help="Minimum samples for HDBSCAN",
    )

    return parser.parse_args()


def evaluate_clustering(embeddings, labels, method_name):
    """
    평가 지표로 클러스터링 품질 평가

    Args:
        embeddings: 데이터 임베딩
        labels: 클러스터 레이블
        method_name: 클러스터링 방법 이름

    Returns:
        평가 지표 결과 딕셔너리
    """
    # 단일 클러스터인 경우 평가 불가
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or -1 in unique_labels and len(unique_labels) < 3:
        logger.warning(
            f"{method_name}: 클러스터가 충분하지 않아 평가 불가능 (단일 클러스터)"
        )
        return {
            "method": method_name,
            "num_clusters": len(unique_labels) - (1 if -1 in unique_labels else 0),
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
        }

    # 노이즈 포인트(-1) 제외
    if -1 in unique_labels:
        mask = labels != -1
        filtered_embeddings = embeddings[mask]
        filtered_labels = labels[mask]
    else:
        filtered_embeddings = embeddings
        filtered_labels = labels

    # 평가 지표 계산
    try:
        s_score = silhouette_score(filtered_embeddings, filtered_labels)
        db_score = davies_bouldin_score(filtered_embeddings, filtered_labels)
        ch_score = calinski_harabasz_score(filtered_embeddings, filtered_labels)

        logger.info(f"\n{method_name} 클러스터링 평가 결과:")
        logger.info(
            f"  클러스터 수: {len(unique_labels) - (1 if -1 in unique_labels else 0)}"
        )
        logger.info(f"  Silhouette Score: {s_score:.4f} (높을수록 좋음, 범위: -1 ~ 1)")
        logger.info(f"  Davies-Bouldin Index: {db_score:.4f} (낮을수록 좋음)")
        logger.info(f"  Calinski-Harabasz Score: {ch_score:.4f} (높을수록 좋음)")

        return {
            "method": method_name,
            "num_clusters": len(unique_labels) - (1 if -1 in unique_labels else 0),
            "silhouette_score": s_score,
            "davies_bouldin_score": db_score,
            "calinski_harabasz_score": ch_score,
        }

    except Exception as e:
        logger.error(f"{method_name} 평가 중 오류 발생: {str(e)}")
        return {
            "method": method_name,
            "num_clusters": len(unique_labels) - (1 if -1 in unique_labels else 0),
            "error": str(e),
        }


def visualize_clusters_2d(embeddings, kmeans_labels, hdbscan_labels):
    """
    클러스터링 결과를 2D로 시각화

    Args:
        embeddings: 원본 임베딩
        kmeans_labels: K-means 클러스터 레이블
        hdbscan_labels: HDBSCAN 클러스터 레이블
    """
    try:
        # t-SNE로 고차원 임베딩을 2D로 축소
        from sklearn.manifold import TSNE

        logger.info("t-SNE로 임베딩 차원 축소 중...")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # 2x1 서브플롯 생성
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # K-means 클러스터 시각화
        scatter1 = axes[0].scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=kmeans_labels,
            cmap="viridis",
            alpha=0.7,
        )
        axes[0].set_title("K-means Clustering")
        axes[0].set_xlabel("t-SNE 1")
        axes[0].set_ylabel("t-SNE 2")

        # 클러스터 레이블 범례 추가
        legend1 = axes[0].legend(
            *scatter1.legend_elements(), title="Clusters", loc="upper right"
        )
        axes[0].add_artist(legend1)

        # HDBSCAN 클러스터 시각화
        scatter2 = axes[1].scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=hdbscan_labels,
            cmap="viridis",
            alpha=0.7,
        )
        axes[1].set_title("HDBSCAN Clustering")
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_ylabel("t-SNE 2")

        # 클러스터 레이블 범례 추가
        legend2 = axes[1].legend(
            *scatter2.legend_elements(), title="Clusters", loc="upper right"
        )
        axes[1].add_artist(legend2)

        plt.tight_layout()
        plt.savefig("clustering_visualization.png", dpi=300, bbox_inches="tight")
        logger.info("클러스터링 시각화 저장 완료: clustering_visualization.png")

    except Exception as e:
        logger.error(f"시각화 중 오류 발생: {str(e)}")


def main():
    args = parse_args()

    logger.info(f"데이터셋 로드 중: {args.dataset}")
    dataset = Dataset(args.dataset, args.data_dir)
    example_pool = dataset.get_example_pool()

    logger.info(f"예제 풀 크기: {len(example_pool)}")

    # 임베딩 계산
    embedding_processor = EmbeddingProcessor(args.embedding_model)
    embeddings = embedding_processor.embed_questions(
        example_pool, args.dataset, args.force_recompute
    )

    logger.info(f"임베딩 크기: {embeddings.shape}")

    # 임베딩 배열로 변환
    embeddings_array = np.array(embeddings)

    # K-means 클러스터링
    logger.info(f"K-means 클러스터링 수행 중 (n_clusters={args.n_clusters})...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(embeddings_array)

    # HDBSCAN 클러스터링
    logger.info(
        f"HDBSCAN 클러스터링 수행 중 (min_cluster_size={args.min_cluster_size})..."
    )
    hdbscan_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
    )
    hdbscan_labels = hdbscan_clusterer.fit_predict(embeddings_array)

    # 클러스터링 평가
    results = []
    results.append(evaluate_clustering(embeddings_array, kmeans_labels, "K-means"))
    results.append(evaluate_clustering(embeddings_array, hdbscan_labels, "HDBSCAN"))

    # 결과 표로 출력
    metrics_df = pd.DataFrame(results)
    logger.info("\n클러스터링 평가 지표 요약:")
    print(metrics_df.to_string(index=False))

    # 결과 저장
    metrics_df.to_csv("clustering_metrics.csv", index=False)
    logger.info("평가 지표 저장 완료: clustering_metrics.csv")

    # 클러스터링 시각화
    visualize_clusters_2d(embeddings_array, kmeans_labels, hdbscan_labels)

    logger.info("클러스터링 평가 완료")


if __name__ == "__main__":
    main()
