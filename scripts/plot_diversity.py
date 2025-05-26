import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# 데이터 설정
methods = [
    "Zero-shot",
    "random",
    "similarity",
    "MMR",
    "k-means",
    "HDBSCAN",
    "CM-MMR(k-means)",
    "CM-MMR(HDBSCAN)",
]

# 결과 데이터 (이미지 순서에 맞게 정렬)
scores = [
    0.0,  # Zero-shot (데이터 없음)
    0.793849,  # random
    0.478444,  # similarity
    0.542773,  # MMR
    0.744184,  # k-means
    0.808605,  # HDBSCAN
    0.536856,  # CM-MMR(k-means)
    0.526226,  # CM-MMR(HDBSCAN)
]

std_devs = [
    0.0,  # Zero-shot (데이터 없음)
    0.048213,  # random
    0.069615,  # similarity
    0.078229,  # MMR
    0.000000,  # k-means
    0.018309,  # HDBSCAN
    0.078159,  # CM-MMR(k-means)
    0.085666,  # CM-MMR(HDBSCAN)
]

# 폰트 설정 (한글 문제를 피하기 위해 영어 폰트 사용)
plt.rcParams["font.family"] = "Arial"
plt.figure(figsize=(14, 8))

# 형광 주황색 설정
fluorescent_orange = "#FF6200"

# 막대 그래프 생성
bars = plt.bar(methods, scores, color=fluorescent_orange, yerr=std_devs, capsize=5)

# 그래프 제목과 축 레이블 설정
plt.title("Average Diversity Score by Method", fontsize=16)
plt.xlabel("Method", fontsize=14)
plt.ylabel("Diversity Score (1 - Average Cosine Similarity)", fontsize=14)
plt.ylim(0, 1.0)  # 다양성 점수는 0-1 사이

# 막대 위에 값 표시
for i, bar in enumerate(bars):
    if i == 0:  # Zero-shot 데이터가 없는 경우
        continue
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

# X축 레이블 회전
plt.xticks(rotation=45, ha="right")

# 그리드 추가
plt.grid(axis="y", linestyle="--", alpha=0.7)

# 레이아웃 조정
plt.tight_layout()

# 그래프 저장
plt.savefig("diversity_score_by_method.png", dpi=300)
plt.savefig("diversity_score_by_method.pdf")

print(
    "그래프가 저장되었습니다: diversity_score_by_method.png, diversity_score_by_method.pdf"
)

# 그래프 표시
plt.show()
