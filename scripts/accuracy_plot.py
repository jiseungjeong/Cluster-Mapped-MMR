import matplotlib.pyplot as plt

# 예시 데이터: 방법론 이름과 정확도
methods = [
    "zero-shot",
    "random",
    "similiarity",
    "MMR",
    "k-means",
    "HDBSCAN",
    "CM-MMR(k-means)",
    "CM-MMR(HDBSCAN)",
]
accuracies = [
    83.33333333333334,
    81.33333333333334,
    83.33333333333334,
    84.33333333333334,
    82.66666666666667,
    85,
    84.33333333333334,
    86.33333333333333,
]

# 그래프 그리기
plt.figure(figsize=(10, 6))
bars = plt.bar(methods, accuracies)

# 정확도 수치를 막대 위에 표시
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 0.3,
        f"{yval:.1f}%",
        ha="center",
        va="bottom",
    )

# 그래프 설정
plt.ylim(80, 90)  # 여기서 y축 하한을 80으로 설정
plt.title("Accuracy Comparision between 8 example selection baselines")
plt.xlabel("Methods")
plt.ylabel("Accuracy (%)")
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
