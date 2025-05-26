#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import seaborn as sns
from tqdm import tqdm

# 데이터 파일 경로
data_file = "data/examples/combined_examples.json"

# 데이터 로드
print("데이터 로드 중...")
with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 질문 텍스트 추출 및 데이터셋 타입 저장
questions = [item["question"] for item in data]
dataset_types = [item["dataset"] for item in data]

# 문장 임베딩 모델 로드
print("모델 로딩 중...")
model = SentenceTransformer("all-mpnet-base-v2")

# 질문 임베딩 생성
print("문장 임베딩 생성 중...")
embeddings = model.encode(questions, show_progress_bar=True)

# K-means 클러스터링 수행 (k=3)
print("K-means 클러스터링 수행 중...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embeddings)

# PCA로 임베딩 차원 축소 (50차원)
print("PCA로 차원 축소 중...")
pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(embeddings)
print(f"PCA 분산 설명률: {pca.explained_variance_ratio_.sum():.4f}")

# UMAP으로 2차원 시각화
print("UMAP으로 2차원 시각화 중...")
umap_2d = umap.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=15)
vis_2d = umap_2d.fit_transform(reduced_embeddings)

# UMAP으로 3차원 시각화
print("UMAP으로 3차원 시각화 중...")
umap_3d = umap.UMAP(n_components=3, random_state=42, min_dist=0.1, n_neighbors=15)
vis_3d = umap_3d.fit_transform(reduced_embeddings)

# 데이터셋 유형에 따라 마커 설정
markers = {"commonsenseqa": "s", "gsm8k": "^", "arc": "o"}
dataset_markers = [markers[dt] for dt in dataset_types]

# 컬러맵 설정
colors = plt.cm.tab10(np.array(clusters) % 10)

# 2D 시각화 그래프 생성
plt.figure(figsize=(12, 10))
for i, dt in enumerate(np.unique(dataset_types)):
    mask = np.array(dataset_types) == dt
    scatter = plt.scatter(
        vis_2d[mask, 0],
        vis_2d[mask, 1],
        c=colors[mask],
        marker=markers[dt],
        s=70,
        alpha=0.7,
        edgecolors="w",
        linewidth=0.5,
        label=f"{dt} dataset",
    )

# 클러스터 센터 추가
for i in range(3):
    mask = clusters == i
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
        label=f"Cluster {i} center" if i == 0 else None,
    )
    plt.annotate(
        f"Cluster {i}",
        (centroid_x, centroid_y),
        fontsize=12,
        fontweight="bold",
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.title("K-means Clustering (k=3) Visualized with UMAP", fontsize=16)
plt.xlabel("UMAP Dimension 1", fontsize=14)
plt.ylabel("UMAP Dimension 2", fontsize=14)
plt.grid(alpha=0.3)

# 범례 중복 제거 및 표시
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=12)

plt.savefig("kmeans_umap_2d.png", dpi=300, bbox_inches="tight")
print("2D 시각화가 'kmeans_umap_2d.png'로 저장되었습니다.")

# 3D 시각화 그래프 생성
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection="3d")

for i, dt in enumerate(np.unique(dataset_types)):
    mask = np.array(dataset_types) == dt
    scatter = ax.scatter(
        vis_3d[mask, 0],
        vis_3d[mask, 1],
        vis_3d[mask, 2],
        c=colors[mask],
        marker=markers[dt],
        s=70,
        alpha=0.7,
        edgecolors="w",
        linewidth=0.5,
        label=f"{dt} dataset",
    )

# 클러스터 센터 추가
for i in range(3):
    mask = clusters == i
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
        label=f"Cluster {i} center" if i == 0 else None,
    )
    ax.text(
        centroid_x,
        centroid_y,
        centroid_z,
        f"Cluster {i}",
        fontsize=12,
        fontweight="bold",
    )

ax.set_title("K-means Clustering (k=3) Visualized with 3D UMAP", fontsize=16)
ax.set_xlabel("UMAP Dimension 1", fontsize=14)
ax.set_ylabel("UMAP Dimension 2", fontsize=14)
ax.set_zlabel("UMAP Dimension 3", fontsize=14)

# 범례 중복 제거 및 표시
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=12)

plt.savefig("kmeans_umap_3d.png", dpi=300, bbox_inches="tight")
print("3D 시각화가 'kmeans_umap_3d.png'로 저장되었습니다.")

# 클러스터별 데이터 수 계산
cluster_counts = {i: np.sum(clusters == i) for i in range(5)}
dataset_cluster_counts = {}
for i in range(3):
    cqa_count = np.sum(
        (np.array(clusters) == i) & (np.array(dataset_types) == "commonsenseqa")
    )
    gsm_count = np.sum((np.array(clusters) == i) & (np.array(dataset_types) == "gsm8k"))
    dataset_cluster_counts[i] = {"commonsenseqa": cqa_count, "gsm8k": gsm_count}

# 결과 출력
print("\n클러스터링 결과 요약:")
print(f"총 데이터 수: {len(data)}")
print(f"클러스터 수: 3")
print("\n각 클러스터별 데이터 수:")
for i in range(3):
    print(f"  클러스터 {i}: {cluster_counts[i]} 항목")
    print(f"    - CommonsenseQA: {dataset_cluster_counts[i]['commonsenseqa']} 항목")
    print(f"    - GSM8K: {dataset_cluster_counts[i]['gsm8k']} 항목")
