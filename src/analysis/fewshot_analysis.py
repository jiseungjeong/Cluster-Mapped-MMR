#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import subprocess
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="Few-Shot 예제 개수에 따른 정확도 분석"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "commonsenseqa", "combined"],
        default="combined",
        help="평가할 데이터셋",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["random"],
        default="random",
        help="예제 선택 방법",
    )
    parser.add_argument(
        "--shot_counts",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 8, 16, 32],
        help="평가할 few-shot 예제 개수 목록",
    )
    parser.add_argument(
        "--num_test_samples", type=int, default=300, help="테스트 샘플 수"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="데이터 디렉토리 경로"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results", help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="gpt-3.5-turbo",
        help="사용할 GPT 모델",
    )

    return parser.parse_args()


def run_single_experiment(
    dataset: str,
    method: str,
    num_examples: int,
    num_test_samples: int,
    data_dir: str,
    results_dir: str,
    gpt_model: str,
    is_in_analysis_dir: bool = False,
):
    """
    단일 실험 실행

    Args:
        dataset: 데이터셋 이름
        method: 예제 선택 방법
        num_examples: 선택할 예제 개수
        num_test_samples: 테스트 샘플 수
        data_dir: 데이터 디렉토리 경로
        results_dir: 결과 저장 디렉토리
        gpt_model: GPT 모델명
        is_in_analysis_dir: src/analysis 디렉토리에서 실행 중인지 여부

    Returns:
        (실험 성공 여부, 실험 결과 파일 경로)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{dataset}_{method}_{num_examples}shots_{timestamp}"
    experiment_dir = os.path.join(results_dir, experiment_name)

    logger.info(f"실험 실행: {dataset} / {method} / {num_examples} shots")

    # 데이터셋 파일 존재 여부 확인
    dataset_paths = {
        "gsm8k": os.path.join(data_dir, "processed", "gsm8k_test.json"),
        "commonsenseqa": os.path.join(data_dir, "processed", "commonsenseqa_test.json"),
        "combined": os.path.join(data_dir, "processed", "combined_test.json"),
    }

    if dataset in dataset_paths:
        dataset_path = dataset_paths[dataset]
        if not os.path.exists(dataset_path):
            logger.error(f"데이터셋 파일이 없습니다: {dataset_path}")
            return False, None
        else:
            logger.info(f"데이터셋 파일 확인됨: {dataset_path}")

    # 명령어 구성
    # 실행 위치에 따라 main.py 경로 결정
    if is_in_analysis_dir:
        main_py_path = "../main.py"  # src/analysis에서 실행 시
    else:
        main_py_path = "src/main.py"  # 프로젝트 루트에서 실행 시

    cmd = [
        "python",
        main_py_path,
        "--dataset",
        dataset,
        "--method",
        method,
        "--num_examples",
        str(num_examples),
        "--num_test_samples",
        str(num_test_samples),
    ]

    # 특정 방법에 따른 추가 인자
    if method == "mmr":
        cmd.extend(["--lambda_param", "0.7"])
    elif method == "kmeans":
        cmd.extend(["--n_clusters", "5"])

    # combined 데이터셋인 경우, 예제 파일 경로 명시적 지정
    if dataset == "combined":
        examples_path = os.path.join(data_dir, "examples", "combined_examples.json")
        if os.path.exists(examples_path):
            cmd.extend(["--examples_file", examples_path])
            logger.info(f"Combined 예제 파일 지정: {examples_path}")

    # 실험 실행
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"실험 완료: {experiment_name}")
        return True, os.path.join(experiment_dir, "results.json")
    except subprocess.CalledProcessError as e:
        logger.error(f"실험 실패: {experiment_name}, 오류: {str(e)}")
        return False, None


def calculate_accuracy(results_file: str) -> float:
    """
    결과 파일에서 정확도 계산

    Args:
        results_file: 결과 JSON 파일 경로

    Returns:
        정확도 (%)
    """
    if not os.path.exists(results_file):
        logger.error(f"결과 파일이 없습니다: {results_file}")
        return 0.0

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        # 정확도 계산
        correct_count = sum(1 for item in results if item.get("correct", False))
        total_count = len(results)

        if total_count == 0:
            return 0.0

        accuracy = (correct_count / total_count) * 100
        return accuracy

    except Exception as e:
        logger.error(f"결과 파일 처리 중 오류: {str(e)}")
        return 0.0


def plot_results(results: List[Dict], output_dir: str):
    """
    결과 시각화

    Args:
        results: 실험 결과 목록
        output_dir: 출력 디렉토리
    """
    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results)

    # 출력 디렉토리 생성
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 그래프 스타일 설정
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # 선 그래프 그리기
    plt.plot(
        df["num_examples"],
        df["accuracy"],
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=10,
        color="#1f77b4",
    )

    # 각 데이터 포인트에 값 표시
    for x, y in zip(df["num_examples"], df["accuracy"]):
        plt.text(x, y + 1, f"{y:.1f}%", ha="center", va="bottom", fontsize=9)

    # 그래프 제목과 라벨 설정 (영어로)
    plt.title("Impact of Few-Shot Example Count on Model Accuracy", fontsize=16, pad=20)
    plt.xlabel("Number of Few-Shot Examples", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)

    # X축 설정
    plt.xticks(df["num_examples"])

    # 그리드 및 레이아웃
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    # 이미지 저장
    plot_file = os.path.join(plots_dir, "fewshot_accuracy.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    logger.info(f"그래프 저장됨: {plot_file}")

    # CSV 저장
    csv_file = os.path.join(output_dir, "fewshot_results.csv")
    df.to_csv(csv_file, index=False)
    logger.info(f"결과 저장됨: {csv_file}")


def main():
    """메인 함수"""
    # 현재 작업 디렉토리 확인 (프로젝트 루트에서 실행되어야 함)
    cwd = os.getcwd()
    script_path = os.path.abspath(__file__)
    logger.info(f"현재 작업 디렉토리: {cwd}")
    logger.info(f"스크립트 경로: {script_path}")

    # 프로젝트 루트 디렉토리에서 실행 중인지 확인
    is_in_analysis_dir = cwd.endswith("/src/analysis")
    if is_in_analysis_dir:
        logger.info("src/analysis 디렉토리에서 실행 중입니다.")
    elif not (os.path.exists(os.path.join(cwd, "src", "main.py"))):
        logger.warning("이 스크립트는 프로젝트 루트 디렉토리에서 실행해야 합니다.")
        logger.warning("예: python src/analysis/fewshot_analysis.py")
        logger.warning(f"현재 위치: {cwd}")

    args = parse_args()

    # 실행 위치에 따라 결과 디렉토리 조정
    results_dir = args.results_dir
    if is_in_analysis_dir and results_dir == "./results":
        results_dir = "../../results"
        logger.info(f"실행 위치에 따라 결과 디렉토리를 조정합니다: {results_dir}")

    # 전체 결과 저장 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(results_dir, f"fewshot_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 각 shot 개수별로 실험 실행
    results = []

    for num_examples in args.shot_counts:
        success, results_file = run_single_experiment(
            dataset=args.dataset,
            method=args.method,
            num_examples=num_examples,
            num_test_samples=args.num_test_samples,
            data_dir=args.data_dir,
            results_dir=output_dir,
            gpt_model=args.gpt_model,
            is_in_analysis_dir=is_in_analysis_dir,
        )

        if success:
            accuracy = calculate_accuracy(results_file)
            logger.info(f"Shot count: {num_examples}, Accuracy: {accuracy:.2f}%")

            results.append(
                {
                    "num_examples": num_examples,
                    "accuracy": accuracy,
                    "results_file": results_file,
                }
            )

    # 결과 저장
    results_file = os.path.join(output_dir, "summary.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"종합 결과 저장됨: {results_file}")

    # 결과 시각화
    if results:
        plot_results(results, output_dir)
        logger.info("Few-shot 분석 완료!")
    else:
        logger.error("유효한 결과가 없습니다. 모든 실험이 실패했습니다.")


if __name__ == "__main__":
    main()
