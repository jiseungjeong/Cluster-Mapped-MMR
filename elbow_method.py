#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm

# 데이터 파일 경로
data_file = "data/examples/total_examples_with_arc.json"

# 데이터 로드
with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 질문 텍스트 추출
questions = [item["question"] for item in data]

# 문장 임베딩 모델 로드
print("모델 로딩 중...")
model = SentenceTransformer("all-mpnet-base-v2")

# 질문 임베딩 생성
print("문장 임베딩 생성 중...")
embeddings = model.encode(questions, show_progress_bar=True)

# WCSS(Within-Cluster Sum of Squares) 계산
wcss = []
max_k = 20  # 최대 클러스터 수
print("다양한 k 값에 대해 클러스터링 수행 중...")

for k in tqdm(range(1, max_k + 1)):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    wcss.append(kmeans.inertia_)

# Elbow 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), wcss, "bo-")
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.grid(True)
plt.xticks(range(1, max_k + 1))

# 그래프에 주석 추가
plt.annotate(
    "Elbow Point",
    xy=(3, wcss[2]),
    xytext=(6, wcss[3] + 0.5 * (wcss[0] - wcss[-1])),
    arrowprops=dict(facecolor="red", shrink=0.05, width=2),
)

# 그래프 저장
plt.savefig("elbow_method_plot.png", dpi=300, bbox_inches="tight")
print("그래프가 'elbow_method_plot.png'로 저장되었습니다.")

# 그래프 표시
plt.show()

# 결과 요약 출력
print("\n분석 결과 요약:")
print(f"분석된 데이터 수: {len(data)}")
print("WCSS 값:")
for k, w in enumerate(wcss, 1):
    print(f"  클러스터 {k}: {w:.2f}")
