#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 데이터 수집
selection_times = []

# Zero 방법론 데이터
zero_data = {
    "shot": {  # 'random'에서 'shot'으로 변경
        "selection_time": {
            "mean": 1.4677047729492188e-05,
            "std": 1.0663186782023743e-05,
        },
    }
}

# Turbo 방법론 데이터 (Turbo 접두사 없이 정의)
other_methods_data = {
    "random": {
        "selection_time": {
            "mean": 2.1922588348388673e-05,
            "std": 7.5972600816746995e-06,
        },
    },
    "similarity": {
        "selection_time": {"mean": 0.0017971229553222657, "std": 0.0004122304841026219},
    },
    "MMR": {  # 대문자로 변경
        "selection_time": {"mean": 0.007757124106089274, "std": 0.004363183693208819},
    },
    "k-means": {  # 'kmeans'에서 'k-means'로 변경
        "selection_time": {
            "mean": 8.170763651529948e-05,
            "std": 1.7273439478668333e-05,
        },
    },
    "HDBSCAN": {  # 'hdbscan'에서 'HDBSCAN'으로 변경
        "selection_time": {
            "mean": 0.00020470301310221355,
            "std": 3.858030990059773e-05,
        },
    },
    "CM-MMR(k-means)": {  # 'cm-mmr'에서 'CM-MMR(k-means)'로 변경
        "selection_time": {"mean": 0.0016421898206075032, "std": 0.0013584709794871306},
    },
    "CM-MMR(HDBSCAN)": {  # 'cm-hdbscan-mmr'에서 'CM-MMR(HDBSCAN)'으로 변경
        "selection_time": {"mean": 0.0018919873237609863, "std": 0.0035900333565906098},
    },
}

# 이미지에 나온 순서대로 방법론 순서 지정
method_order = [
    "Zero-shot",
    "random",
    "similarity",
    "MMR",
    "k-means",
    "HDBSCAN",
    "CM-MMR(k-means)",
    "CM-MMR(HDBSCAN)",
]

# 데이터 준비
methods = []
selection_times_ordered = []

# 지정된 순서대로 데이터 추가
for method in method_order:
    if method == "Zero-shot":
        selection_times_ordered.append(zero_data["shot"]["selection_time"]["mean"])
    else:
        selection_times_ordered.append(
            other_methods_data[method]["selection_time"]["mean"]
        )

# 그래프 생성
fig, ax = plt.subplots(figsize=(12, 6))

# Selection Time 그래프
ax.bar(method_order, selection_times_ordered, color="lightgreen")
ax.set_title("Selection Time by Method", fontsize=14)
ax.set_xlabel("Method", fontsize=12)
ax.set_ylabel("Selection Time (seconds)", fontsize=12)
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("selection_time_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
