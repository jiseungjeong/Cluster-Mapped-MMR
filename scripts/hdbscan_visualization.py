#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.decomposition import PCA
import umap
import seaborn as sns
from tqdm import tqdm

# 데이터 파일 경로
data_file = "data/examples/combined_examples.json"

# 데이터 로드
print("Loading data...")
with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 질문 텍스트 추출 및 데이터셋 타입 저장
questions = [item["question"] for item in data]
dataset_types = [item["dataset"] for item in data]

# 문장 임베딩 모델 로드
print("Loading embedding model...")
model = SentenceTransformer("all-mpnet-base-v2")

# 질문 임베딩 생성
print("Creating sentence embeddings...")
embeddings = model.encode(questions, show_progress_bar=True)

# PCA로 임베딩 차원 축소 (50차원)
print("Reducing dimensions with PCA...")
pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(embeddings)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

# 분리된 데이터셋 처리
print("Processing each dataset separately for better clustering...")
cqa_mask = np.array(dataset_types) == "commonsenseqa"
gsm_mask = np.array(dataset_types) == "gsm8k"
arc_mask = np.array(dataset_types) == "arc"

# CommonsenseQA 데이터 클러스터링
print("Clustering CommonsenseQA data...")
cqa_clusterer = hdbscan.HDBSCAN(
    min_cluster_size=8,
    min_samples=2,
    cluster_selection_epsilon=0.2,
    metric="euclidean",
    cluster_selection_method="eom",
)
cqa_clusters = cqa_clusterer.fit_predict(reduced_embeddings[cqa_mask])

# GSM8K 데이터 클러스터링
print("Clustering GSM8K data...")
gsm_clusterer = hdbscan.HDBSCAN(
    min_cluster_size=8,
    min_samples=2,
    cluster_selection_epsilon=0.2,
    metric="euclidean",
    cluster_selection_method="eom",
)
gsm_clusters = gsm_clusterer.fit_predict(reduced_embeddings[gsm_mask])

# ARC 데이터 클러스터링
print("Clustering ARC data...")
arc_clusterer = hdbscan.HDBSCAN(
    min_cluster_size=8,
    min_samples=2,
    cluster_selection_epsilon=0.2,
    metric="euclidean",
    cluster_selection_method="eom",
)
arc_clusters = arc_clusterer.fit_predict(reduced_embeddings[arc_mask])

# 클러스터 ID 조정 (각 데이터셋별 오프셋 추가)
cqa_cluster_count = len(set(cqa_clusters)) - (1 if -1 in cqa_clusters else 0)
print(f"CommonsenseQA clusters: {cqa_cluster_count}")
print(f"CommonsenseQA noise points: {np.sum(cqa_clusters == -1)}")

# GSM8K 클러스터에 오프셋 추가 (노이즈 포인트 제외)
gsm_clusters_adj = np.array(gsm_clusters)
mask = gsm_clusters_adj != -1
gsm_clusters_adj[mask] += cqa_cluster_count
gsm_cluster_count = len(set(gsm_clusters)) - (1 if -1 in gsm_clusters else 0)
print(f"GSM8K clusters: {gsm_cluster_count}")
print(f"GSM8K noise points: {np.sum(gsm_clusters == -1)}")

# ARC 클러스터에 오프셋 추가 (노이즈 포인트 제외)
arc_clusters_adj = np.array(arc_clusters)
mask = arc_clusters_adj != -1
arc_clusters_adj[mask] += cqa_cluster_count + gsm_cluster_count
arc_cluster_count = len(set(arc_clusters)) - (1 if -1 in arc_clusters else 0)
print(f"ARC clusters: {arc_cluster_count}")
print(f"ARC noise points: {np.sum(arc_clusters == -1)}")

# 종합 클러스터 배열 생성
clusters = np.zeros(len(dataset_types), dtype=int)
clusters[cqa_mask] = cqa_clusters
clusters[gsm_mask] = gsm_clusters_adj
clusters[arc_mask] = arc_clusters_adj

# 클러스터 수 계산 (노이즈 포인트 제외)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f"Total clusters found: {n_clusters}")
print(f"Total noise points: {np.sum(clusters == -1)}")

# UMAP으로 2차원 시각화
print("Creating 2D UMAP visualization...")
umap_2d = umap.UMAP(
    n_components=2, random_state=42, min_dist=0.2, n_neighbors=30, metric="cosine"
)
vis_2d = umap_2d.fit_transform(reduced_embeddings)

# UMAP으로 3차원 시각화
print("Creating 3D UMAP visualization...")
umap_3d = umap.UMAP(
    n_components=3, random_state=42, min_dist=0.2, n_neighbors=30, metric="cosine"
)
vis_3d = umap_3d.fit_transform(reduced_embeddings)

# 데이터셋 유형에 따라 마커 설정
markers = {"commonsenseqa": "s", "gsm8k": "^", "arc": "o"}
dataset_markers = [markers[dt] for dt in dataset_types]

# 컬러맵 설정 (노이즈 포인트는 회색으로 표시)
unique_clusters = sorted(list(set(clusters)))
if -1 in unique_clusters:
    unique_clusters.remove(-1)

palette = sns.color_palette("tab20", n_colors=min(len(unique_clusters), 20))
cluster_colors = {}

for i, cluster_id in enumerate(unique_clusters):
    cluster_colors[cluster_id] = palette[i % 20]  # 20개 이상의 클러스터는 색상 반복

# 노이즈 포인트는 회색으로 표시
if -1 in set(clusters):
    cluster_colors[-1] = (0.7, 0.7, 0.7)  # 회색

# 각 포인트의 색상 지정
colors = [cluster_colors[c] for c in clusters]

# 2D 시각화 그래프 생성
plt.figure(figsize=(16, 14))

# 먼저 노이즈 포인트를 작게 표시
noise_mask = clusters == -1
for dt in np.unique(dataset_types):
    mask = (np.array(dataset_types) == dt) & noise_mask
    if np.any(mask):
        plt.scatter(
            vis_2d[mask, 0],
            vis_2d[mask, 1],
            c="gray",
            marker=markers[dt],
            s=50,
            alpha=0.5,
            edgecolors="w",
            linewidth=0.3,
            label=f"Noise points ({dt})" if dt == np.unique(dataset_types)[0] else None,
        )

# 클러스터 포인트 표시
for cluster_id in unique_clusters:
    cluster_mask = clusters == cluster_id
    for dt in np.unique(dataset_types):
        mask = (np.array(dataset_types) == dt) & cluster_mask
        if np.any(mask):
            plt.scatter(
                vis_2d[mask, 0],
                vis_2d[mask, 1],
                c=[cluster_colors[cluster_id]] * np.sum(mask),
                marker=markers[dt],
                s=80,
                alpha=0.7,
                edgecolors="w",
                linewidth=0.5,
                label=(
                    f"Cluster {cluster_id} ({dt})"
                    if dt == np.unique(dataset_types)[0]
                    else None
                ),
            )

# 클러스터 센터 표시
for cluster_id in unique_clusters:
    mask = clusters == cluster_id
    centroid_x = np.mean(vis_2d[mask, 0])
    centroid_y = np.mean(vis_2d[mask, 1])
    plt.scatter(
        centroid_x,
        centroid_y,
        s=200,
        marker="*",
        c="black",
        edgecolors="w",
        linewidth=1.5,
        label=f"Cluster center" if cluster_id == unique_clusters[0] else None,
    )
    plt.annotate(
        f"Cluster {cluster_id}",
        (centroid_x, centroid_y),
        fontsize=10,
        fontweight="bold",
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.title("HDBSCAN Clustering Visualized with UMAP (2D)", fontsize=16)
plt.xlabel("UMAP Dimension 1", fontsize=14)
plt.ylabel("UMAP Dimension 2", fontsize=14)
plt.grid(alpha=0.3)

# 범례 중복 제거 및 표시
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(),
    by_label.keys(),
    loc="upper right",
    fontsize=10,
    bbox_to_anchor=(1.15, 1),
)

plt.tight_layout()
plt.savefig("hdbscan_umap_2d.png", dpi=300, bbox_inches="tight")
print("2D visualization saved as 'hdbscan_umap_2d.png'")

# 3D 시각화 그래프 생성
fig = plt.figure(figsize=(16, 14))
ax = fig.add_subplot(111, projection="3d")

# 먼저 노이즈 포인트를 작게 표시
for dt in np.unique(dataset_types):
    mask = (np.array(dataset_types) == dt) & noise_mask
    if np.any(mask):
        ax.scatter(
            vis_3d[mask, 0],
            vis_3d[mask, 1],
            vis_3d[mask, 2],
            c="gray",
            marker=markers[dt],
            s=50,
            alpha=0.5,
            edgecolors="w",
            linewidth=0.3,
            label=f"Noise points ({dt})" if dt == np.unique(dataset_types)[0] else None,
        )

# 클러스터 포인트 표시
for cluster_id in unique_clusters:
    cluster_mask = clusters == cluster_id
    for dt in np.unique(dataset_types):
        mask = (np.array(dataset_types) == dt) & cluster_mask
        if np.any(mask):
            ax.scatter(
                vis_3d[mask, 0],
                vis_3d[mask, 1],
                vis_3d[mask, 2],
                c=[cluster_colors[cluster_id]] * np.sum(mask),
                marker=markers[dt],
                s=80,
                alpha=0.7,
                edgecolors="w",
                linewidth=0.5,
                label=(
                    f"Cluster {cluster_id} ({dt})"
                    if dt == np.unique(dataset_types)[0]
                    else None
                ),
            )

# 클러스터 센터 표시
for cluster_id in unique_clusters:
    mask = clusters == cluster_id
    centroid_x = np.mean(vis_3d[mask, 0])
    centroid_y = np.mean(vis_3d[mask, 1])
    centroid_z = np.mean(vis_3d[mask, 2])
    ax.scatter(
        centroid_x,
        centroid_y,
        centroid_z,
        s=200,
        marker="*",
        c="black",
        edgecolors="w",
        linewidth=1.5,
        label=f"Cluster center" if cluster_id == unique_clusters[0] else None,
    )
    ax.text(
        centroid_x,
        centroid_y,
        centroid_z,
        f"Cluster {cluster_id}",
        fontsize=10,
        fontweight="bold",
    )

ax.set_title("HDBSCAN Clustering Visualized with UMAP (3D)", fontsize=16)
ax.set_xlabel("UMAP Dimension 1", fontsize=14)
ax.set_ylabel("UMAP Dimension 2", fontsize=14)
ax.set_zlabel("UMAP Dimension 3", fontsize=14)

# 범례 중복 제거 및 표시
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig("hdbscan_umap_3d.png", dpi=300, bbox_inches="tight")
print("3D visualization saved as 'hdbscan_umap_3d.png'")

# 클러스터별 데이터 수 계산
cluster_counts = {i: np.sum(clusters == i) for i in sorted(set(clusters))}
dataset_cluster_counts = {}
for i in sorted(set(clusters)):
    cqa_count = np.sum(
        (np.array(clusters) == i) & (np.array(dataset_types) == "commonsenseqa")
    )
    gsm_count = np.sum((np.array(clusters) == i) & (np.array(dataset_types) == "gsm8k"))
    arc_count = np.sum((np.array(clusters) == i) & (np.array(dataset_types) == "arc"))
    dataset_cluster_counts[i] = {
        "commonsenseqa": cqa_count,
        "gsm8k": gsm_count,
        "arc": arc_count,
    }

# 결과 출력
print("\nClustering Summary:")
print(f"Total data points: {len(data)}")
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {np.sum(clusters == -1)}")

print("\nData points per cluster:")
for i in sorted(set(clusters)):
    print(f"  Cluster {i}: {cluster_counts[i]} items")
    print(f"    - CommonsenseQA: {dataset_cluster_counts[i]['commonsenseqa']} items")
    print(f"    - GSM8K: {dataset_cluster_counts[i]['gsm8k']} items")
    print(f"    - ARC: {dataset_cluster_counts[i]['arc']} items")
