import matplotlib.pyplot as plt
import numpy as np

# 데이터셋별 성능 데이터 (가정값 - 실제 실험 결과로 교체 필요)
datasets = ["ARC", "CommonsenseQA", "GSM8K", "Combined"]

# Zero-shot 성능 (예제 없이)
zero_shot_accuracies = [95, 72, 78, 84]  # 각 데이터셋별 zero-shot 정확도

# Few-shot random 성능 (5개 랜덤 예제 사용)
few_shot_random_accuracies = [96, 77, 89, 81]  # 각 데이터셋별 few-shot random 정확도

# 성능 향상/감소 계산
performance_changes = [
    few - zero for few, zero in zip(few_shot_random_accuracies, zero_shot_accuracies)
]

# 그래프 생성
fig, ax = plt.subplots(figsize=(12, 8))

# 바 위치 설정
x = np.arange(len(datasets))
width = 0.35

# Zero-shot과 Few-shot 바 그리기
bars1 = ax.bar(
    x - width / 2,
    zero_shot_accuracies,
    width,
    label="Zero-shot",
    color="lightblue",
    alpha=0.8,
)
bars2 = ax.bar(
    x + width / 2,
    few_shot_random_accuracies,
    width,
    label="Few-shot (Random)",
    color="lightgreen",
    alpha=0.8,
)

# Combined 데이터셋 바를 다른 색으로 강조
bars2[-1].set_color("orange")
bars2[-1].set_alpha(0.9)

# 정확도 값을 바 위에 표시
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    # Zero-shot 값
    ax.text(
        bar1.get_x() + bar1.get_width() / 2,
        bar1.get_height() + 1,
        f"{zero_shot_accuracies[i]}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12,
    )

    # Few-shot 값
    ax.text(
        bar2.get_x() + bar2.get_width() / 2,
        bar2.get_height() + 1,
        f"{few_shot_random_accuracies[i]}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12,
    )

    # 성능 변화 화살표 및 텍스트
    if performance_changes[i] > 0:
        # 성능 향상 (녹색 화살표) - 텍스트와 화살표 간격 조정
        ax.annotate(
            f"+{performance_changes[i]}%",
            xy=(i, max(zero_shot_accuracies[i], few_shot_random_accuracies[i]) + 6),
            ha="center",
            va="bottom",
            color="green",
            fontweight="bold",
            fontsize=15,
        )
        ax.arrow(
            i,
            max(zero_shot_accuracies[i], few_shot_random_accuracies[i]) + 2,
            0,
            2,
            head_width=0.1,
            head_length=0.5,
            fc="green",
            ec="green",
        )
    else:
        # 성능 감소 (빨간색 화살표)
        ax.annotate(
            f"{performance_changes[i]}%",
            xy=(i, max(zero_shot_accuracies[i], few_shot_random_accuracies[i]) + 6),
            ha="center",
            va="bottom",
            color="red",
            fontweight="bold",
            fontsize=15,
        )
        ax.arrow(
            i,
            max(zero_shot_accuracies[i], few_shot_random_accuracies[i]) + 2,
            0,
            -2,
            head_width=0.1,
            head_length=0.5,
            fc="red",
            ec="red",
        )

# 그래프 설정
ax.set_xlabel("Dataset Domain", fontsize=12, fontweight="bold")
ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
ax.set_title(
    "Accuracy Comparison Across Different Domains", fontsize=14, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_ylim(0, 105)

# 레이아웃 조정
plt.tight_layout()

# 저장 및 표시
plt.savefig("domain_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n=== 데이터셋별 성능 분석 ===")
print("Dataset\t\tZero-shot\tFew-shot\tChange")
print("-" * 50)
for i, dataset in enumerate(datasets):
    change_str = (
        f"+{performance_changes[i]}%"
        if performance_changes[i] > 0
        else f"{performance_changes[i]}%"
    )
    print(
        f"{dataset:12s}\t{zero_shot_accuracies[i]}%\t\t{few_shot_random_accuracies[i]}%\t\t{change_str}"
    )

print(f"\n주요 발견:")
print(f"- 단일 도메인 데이터셋(ARC, GSM8K)에서는 few-shot이 성능 향상을 보임")
print(f"- CommonsenseQA는 중간 정도의 향상")
print(f"- Combined 데이터셋에서는 오히려 성능이 감소함 (-3%)")
print(f"- 이는 도메인 혼합으로 인한 예제 선택의 어려움을 시사함")
