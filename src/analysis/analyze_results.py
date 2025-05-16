#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="CoT 예제 선택 기법 실험 결과 분석")

    parser.add_argument(
        "--results_dir", type=str, required=True, help="분석할 결과 디렉토리"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/analysis",
        help="분석 결과 저장 디렉토리",
    )
    parser.add_argument("--plot", action="store_true", help="그래프 생성 여부")

    return parser.parse_args()


def load_experiment_results(results_dir: str) -> pd.DataFrame:
    """
    실험 결과 로드 및 통합

    Args:
        results_dir: 결과 디렉토리 경로

    Returns:
        통합된 결과 DataFrame
    """
    all_results = []

    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == "results.json":
                file_path = os.path.join(root, file)
                logger.info(f"결과 파일 로드: {file_path}")

                with open(file_path, "r", encoding="utf-8") as f:
                    results = json.load(f)

                    # 결과 디렉토리 정보 추가
                    experiment_name = os.path.basename(root)
                    for result in results:
                        result["experiment_name"] = experiment_name

                    all_results.extend(results)

    # 데이터 프레임 변환
    if not all_results:
        raise ValueError(f"결과 파일을 찾을 수 없습니다: {results_dir}")

    # 중첩 구조 평탄화
    flat_results = []
    for result in all_results:
        flat_result = {}
        for key, value in result.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_result[f"{key}_{sub_key}"] = sub_value
            else:
                flat_result[key] = value
        flat_results.append(flat_result)

    return pd.DataFrame(flat_results)


def analyze_accuracy(df: pd.DataFrame) -> Dict[str, Any]:
    """
    정확도 분석

    Args:
        df: 결과 DataFrame

    Returns:
        정확도 분석 결과
    """
    if "correct" not in df.columns:
        logger.warning("DataFrame에 'correct' 열이 없습니다. 정확도 분석을 건너뜁니다.")
        return {}

    accuracy_results = {}

    # 데이터셋별, 방법별 정확도
    for dataset in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset]
        dataset_results = {}

        for method in dataset_df["method"].unique():
            method_df = dataset_df[dataset_df["method"] == method]
            if "correct" in method_df.columns:
                accuracy = method_df["correct"].mean() * 100
                count = len(method_df)
                dataset_results[method] = {"accuracy": accuracy, "count": count}

        accuracy_results[dataset] = dataset_results

    return accuracy_results


def analyze_tokens(df: pd.DataFrame) -> Dict[str, Any]:
    """
    토큰 사용량 분석

    Args:
        df: 결과 DataFrame

    Returns:
        토큰 분석 결과
    """
    token_cols = [col for col in df.columns if col.startswith("tokens_")]
    if not token_cols:
        logger.warning("DataFrame에 토큰 관련 열이 없습니다. 토큰 분석을 건너뜁니다.")
        return {}

    token_results = {}

    # 데이터셋별, 방법별 토큰 통계
    for dataset in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset]
        dataset_results = {}

        for method in dataset_df["method"].unique():
            method_df = dataset_df[dataset_df["method"] == method]
            method_results = {}

            for col in token_cols:
                if col in method_df.columns:
                    method_results[col] = {
                        "mean": method_df[col].mean(),
                        "std": method_df[col].std(),
                        "median": method_df[col].median(),
                        "min": method_df[col].min(),
                        "max": method_df[col].max(),
                    }

            dataset_results[method] = method_results

        token_results[dataset] = dataset_results

    return token_results


def analyze_diversity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    예제 간 다양성 분석 (평균 코사인 유사도의 역수)

    Args:
        df: 결과 DataFrame

    Returns:
        다양성 분석 결과
    """
    if "selected_examples" not in df.columns:
        logger.warning(
            "DataFrame에 'selected_examples' 열이 없습니다. 다양성 분석을 건너뜁니다."
        )
        return {}

    from src.utils.embedding import EmbeddingProcessor
    from src.data.dataset import Dataset

    diversity_results = {}

    # 데이터셋 및 임베딩 모델 초기화
    embedding_processor = EmbeddingProcessor("paraphrase-multilingual-mpnet-base-v2")

    # 데이터셋별, 방법별 다양성 분석
    for dataset_name in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset_name]
        dataset_results = {}

        # 데이터셋 로드
        try:
            dataset = Dataset(dataset_name)
            example_pool = dataset.get_example_pool()

            # 예제 임베딩 계산
            example_embeddings = embedding_processor.embed_questions(
                example_pool, dataset_name, force_recompute=False
            )

            # 임베딩 정규화
            normalized_embeddings = example_embeddings / np.linalg.norm(
                example_embeddings, axis=1, keepdims=True
            )

            # 예제 ID를 인덱스로 변환하는 매핑 생성
            example_id_to_idx = {
                example["id"]: i for i, example in enumerate(example_pool)
            }

            # 방법별 다양성 계산
            for method in dataset_df["method"].unique():
                method_df = dataset_df[dataset_df["method"] == method]

                # 모든 선택된 예제 세트에 대한 다양성 평균 계산
                diversities = []

                for _, row in method_df.iterrows():
                    selected_ids = row["selected_examples"]

                    if not selected_ids or len(selected_ids) < 2:
                        continue

                    # 선택된 예제의 임베딩 인덱스 가져오기
                    try:
                        selected_indices = [
                            example_id_to_idx[example_id] for example_id in selected_ids
                        ]
                        selected_embeddings = normalized_embeddings[selected_indices]

                        # 코사인 유사도 행렬 계산
                        similarity_matrix = np.dot(
                            selected_embeddings, selected_embeddings.T
                        )

                        # 대각선 요소(자기 자신과의 유사도=1) 제외
                        n = similarity_matrix.shape[0]
                        similarity_sum = (similarity_matrix.sum() - n) / (n * (n - 1))

                        # 유사도의 역수 = 다양성
                        if similarity_sum > 0:
                            diversity = 1.0 / similarity_sum
                            diversities.append(diversity)
                    except Exception as e:
                        logger.debug(f"다양성 계산 중 오류: {str(e)}")
                        continue

                # 평균 다양성 계산
                if diversities:
                    dataset_results[method] = {
                        "mean": np.mean(diversities),
                        "std": np.std(diversities),
                        "median": np.median(diversities),
                        "min": np.min(diversities),
                        "max": np.max(diversities),
                    }
        except Exception as e:
            logger.error(f"데이터셋 {dataset_name} 처리 중 오류: {str(e)}")
            continue

        diversity_results[dataset_name] = dataset_results

    return diversity_results


def analyze_response_time(df: pd.DataFrame) -> Dict[str, Any]:
    """
    응답 시간 분석

    Args:
        df: 결과 DataFrame

    Returns:
        응답 시간 분석 결과
    """
    if "response_time" not in df.columns:
        logger.warning(
            "DataFrame에 'response_time' 열이 없습니다. 응답 시간 분석을 건너뜁니다."
        )
        return {}

    time_results = {}

    # 데이터셋별, 방법별 응답 시간 통계
    for dataset in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset]
        dataset_results = {}

        for method in dataset_df["method"].unique():
            method_df = dataset_df[dataset_df["method"] == method]
            if "response_time" in method_df.columns:
                dataset_results[method] = {
                    "mean": method_df["response_time"].mean(),
                    "std": method_df["response_time"].std(),
                    "median": method_df["response_time"].median(),
                    "min": method_df["response_time"].min(),
                    "max": method_df["response_time"].max(),
                }

        time_results[dataset] = dataset_results

    return time_results


def generate_plots(df: pd.DataFrame, analysis_results: Dict[str, Any], output_dir: str):
    """
    결과 분석을 위한 그래프 생성

    Args:
        df: 결과 DataFrame
        analysis_results: 분석 결과
        output_dir: 출력 디렉토리
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 폰트 설정
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    # 1. 정확도 비교 그래프
    if "accuracy" in analysis_results:
        plt.figure(figsize=(12, 6))

        datasets = list(analysis_results["accuracy"].keys())
        methods = []
        for dataset_results in analysis_results["accuracy"].values():
            methods.extend(list(dataset_results.keys()))
        methods = list(set(methods))

        for i, dataset in enumerate(datasets):
            accuracies = []
            for method in methods:
                if method in analysis_results["accuracy"][dataset]:
                    accuracies.append(
                        analysis_results["accuracy"][dataset][method]["accuracy"]
                    )
                else:
                    accuracies.append(0)

            x = np.arange(len(methods)) + i * 0.2
            plt.bar(x, accuracies, width=0.2, label=dataset)

        plt.xlabel("Examples Selection Methods")
        plt.ylabel("Accuracy (%)")
        plt.title("Compare accuracy by dataset and example selection method")
        plt.xticks(np.arange(len(methods)) + 0.2 * (len(datasets) - 1) / 2, methods)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "accuracy_comparison.png"), dpi=300)
        plt.close()

    # 2. 토큰 사용량 비교 그래프
    if "tokens" in analysis_results and df.shape[0] > 0:
        if "tokens_total_tokens" in df.columns:
            plt.figure(figsize=(12, 6))

            datasets = list(analysis_results["tokens"].keys())
            methods = []
            for dataset_results in analysis_results["tokens"].values():
                methods.extend(list(dataset_results.keys()))
            methods = list(set(methods))

            for i, dataset in enumerate(datasets):
                tokens = []
                for method in methods:
                    if (
                        method in analysis_results["tokens"][dataset]
                        and "tokens_total_tokens"
                        in analysis_results["tokens"][dataset][method]
                    ):
                        tokens.append(
                            analysis_results["tokens"][dataset][method][
                                "tokens_total_tokens"
                            ]["mean"]
                        )
                    else:
                        tokens.append(0)

                x = np.arange(len(methods)) + i * 0.2
                plt.bar(x, tokens, width=0.2, label=dataset)

            plt.xlabel("Examples Selection Methods")
            plt.ylabel("Average token usage")
            plt.title(
                "Average token usage between datasets and examples selection methods"
            )
            plt.xticks(np.arange(len(methods)) + 0.2 * (len(datasets) - 1) / 2, methods)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "token_usage_comparison.png"), dpi=300)
            plt.close()

    # 3. 응답 시간 비교 그래프
    if "response_time" in analysis_results and "response_time" in df.columns:
        plt.figure(figsize=(12, 6))

        datasets = list(analysis_results["response_time"].keys())
        methods = []
        for dataset_results in analysis_results["response_time"].values():
            methods.extend(list(dataset_results.keys()))
        methods = list(set(methods))

        for i, dataset in enumerate(datasets):
            times = []
            for method in methods:
                if method in analysis_results["response_time"][dataset]:
                    times.append(
                        analysis_results["response_time"][dataset][method]["mean"]
                    )
                else:
                    times.append(0)

            x = np.arange(len(methods)) + i * 0.2
            plt.bar(x, times, width=0.2, label=dataset)

        plt.xlabel("Examples Selection Methods")
        plt.ylabel("Average latency (sec))")
        plt.title("Average latency between datasets and examples selection methods")
        plt.xticks(np.arange(len(methods)) + 0.2 * (len(datasets) - 1) / 2, methods)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "response_time_comparison.png"), dpi=300)
        plt.close()

    # 4. 박스플롯: 토큰 사용량 분포
    if "tokens_total_tokens" in df.columns:
        plt.figure(figsize=(14, 7))
        sns.boxplot(x="method", y="tokens_total_tokens", hue="dataset", data=df)
        plt.xlabel("Examples Selection Methods")
        plt.ylabel("Total number of tokens")
        plt.title("Distribution of token usage by example selection method")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "token_usage_boxplot.png"), dpi=300)
        plt.close()

    # 5. 다양성 비교 그래프
    if "diversity" in analysis_results and df.shape[0] > 0:
        plt.figure(figsize=(12, 6))

        datasets = list(analysis_results["diversity"].keys())
        methods = []
        for dataset_results in analysis_results["diversity"].values():
            methods.extend(list(dataset_results.keys()))
        methods = list(set(methods))

        for i, dataset in enumerate(datasets):
            diversities = []
            for method in methods:
                if method in analysis_results["diversity"][dataset]:
                    diversities.append(
                        analysis_results["diversity"][dataset][method]["mean"]
                    )
                else:
                    diversities.append(0)

            x = np.arange(len(methods)) + i * 0.2
            plt.bar(x, diversities, width=0.2, label=dataset)

        plt.xlabel("Example Selection Method")
        plt.ylabel(
            "Diversity Indicators (inverse of mean cosine similarity between examples)"
        )
        plt.title("Compare Diversity by Dataset and Example Selection Method")
        plt.xticks(np.arange(len(methods)) + 0.2 * (len(datasets) - 1) / 2, methods)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "diversity_comparison.png"), dpi=300)
        plt.close()

    # 6. 종합 비교 그래프 (정확도, 다양성, 토큰 사용량, 지연시간)
    if df.shape[0] > 0:
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))

        dataset = list(df["dataset"].unique())[0]  # 첫 번째 데이터셋 선택
        methods = list(df["method"].unique())

        # 정확도
        accuracies = []
        for method in methods:
            if (
                "accuracy" in analysis_results
                and dataset in analysis_results["accuracy"]
                and method in analysis_results["accuracy"][dataset]
            ):
                accuracies.append(
                    analysis_results["accuracy"][dataset][method]["accuracy"]
                )
            else:
                accuracies.append(0)

        axs[0, 0].bar(methods, accuracies)
        axs[0, 0].set_title("Accuracy (%)")
        axs[0, 0].set_xlabel("Examples Selection Method")
        axs[0, 0].set_ylabel("Accuracy (%)")

        # 다양성
        diversities = []
        for method in methods:
            if (
                "diversity" in analysis_results
                and dataset in analysis_results["diversity"]
                and method in analysis_results["diversity"][dataset]
            ):
                diversities.append(
                    analysis_results["diversity"][dataset][method]["mean"]
                )
            else:
                diversities.append(0)

        axs[0, 1].bar(methods, diversities)
        axs[0, 1].set_title("Diversity Index")
        axs[0, 1].set_xlabel("Examples Selection Methods")
        axs[0, 1].set_ylabel("Diversity (inverse of cosine simiarity)")

        # 토큰 사용량
        tokens = []
        for method in methods:
            if (
                "tokens" in analysis_results
                and dataset in analysis_results["tokens"]
                and method in analysis_results["tokens"][dataset]
                and "tokens_total_tokens" in analysis_results["tokens"][dataset][method]
            ):
                tokens.append(
                    analysis_results["tokens"][dataset][method]["tokens_total_tokens"][
                        "mean"
                    ]
                )
            else:
                tokens.append(0)

        axs[1, 0].bar(methods, tokens)
        axs[1, 0].set_title("Average token usage")
        axs[1, 0].set_xlabel("Examples Selection Methods")
        axs[1, 0].set_ylabel("Average token usage")

        # 응답 시간
        times = []
        for method in methods:
            if (
                "response_time" in analysis_results
                and dataset in analysis_results["response_time"]
                and method in analysis_results["response_time"][dataset]
            ):
                times.append(analysis_results["response_time"][dataset][method]["mean"])
            else:
                times.append(0)

        axs[1, 1].bar(methods, times)
        axs[1, 1].set_title("Average Latency")
        axs[1, 1].set_xlabel("Examples Selection Methods")
        axs[1, 1].set_ylabel("Average latency (sec)")

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "comprehensive_comparison.png"), dpi=300)
        plt.close()

    logger.info(f"그래프가 {plots_dir} 디렉토리에 저장되었습니다.")


def main():
    """메인 함수"""
    args = parse_args()

    # 결과 디렉토리 확인
    if not os.path.exists(args.results_dir):
        logger.error(f"결과 디렉토리가 존재하지 않습니다: {args.results_dir}")
        return

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 결과 로드
    logger.info(f"결과 디렉토리에서 데이터 로드 중: {args.results_dir}")
    try:
        df = load_experiment_results(args.results_dir)
        logger.info(f"총 {len(df)}개 결과 로드됨")
    except Exception as e:
        logger.error(f"결과 로드 실패: {str(e)}")
        return

    # 결과가 없는 경우
    if df.empty:
        logger.error("분석할 결과가 없습니다.")
        return

    # 분석 수행
    analysis_results = {}

    logger.info("정확도 분석 중...")
    analysis_results["accuracy"] = analyze_accuracy(df)

    logger.info("토큰 사용량 분석 중...")
    analysis_results["tokens"] = analyze_tokens(df)

    logger.info("응답 시간 분석 중...")
    analysis_results["response_time"] = analyze_response_time(df)

    logger.info("다양성 분석 중...")
    analysis_results["diversity"] = analyze_diversity(df)

    # numpy 자료형을 파이썬 기본 자료형으로 변환 (JSON 직렬화를 위해)
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_numpy_types(obj.tolist())
        else:
            return obj

    # 결과 변환
    analysis_results = convert_numpy_types(analysis_results)

    # 분석 결과 저장
    analysis_path = os.path.join(args.output_dir, "analysis_results.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    logger.info(f"분석 결과 저장됨: {analysis_path}")

    # CSV 파일로 저장
    csv_path = os.path.join(args.output_dir, "all_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"모든 결과 저장됨: {csv_path}")

    # 결과 요약 출력
    print("\n===== 분석 결과 요약 =====")

    # 정확도 요약
    if analysis_results["accuracy"]:
        print("\n[정확도 분석]")
        for dataset, methods in analysis_results["accuracy"].items():
            print(f"\n데이터셋: {dataset}")
            print(f"{'방법':<15} {'정확도 (%)':<15} {'샘플 수':<10}")
            print("-" * 40)
            for method, stats in methods.items():
                print(
                    f"{method:<15} {stats['accuracy']:.2f}%{'':<8} {stats['count']:<10}"
                )

    # 토큰 사용량 요약
    if analysis_results["tokens"]:
        print("\n[토큰 사용량 분석]")
        for dataset, methods in analysis_results["tokens"].items():
            print(f"\n데이터셋: {dataset}")
            print(f"{'방법':<15} {'총 토큰 (평균)':<20} {'총 토큰 (중앙값)':<20}")
            print("-" * 60)
            for method, stats in methods.items():
                if "tokens_total_tokens" in stats:
                    total_mean = stats["tokens_total_tokens"]["mean"]
                    total_median = stats["tokens_total_tokens"]["median"]
                    print(
                        f"{method:<15} {total_mean:.2f}{'':<13} {total_median:.2f}{'':<13}"
                    )

    # 응답 시간 요약
    if analysis_results["response_time"]:
        print("\n[응답 시간 분석]")
        for dataset, methods in analysis_results["response_time"].items():
            print(f"\n데이터셋: {dataset}")
            print(f"{'방법':<15} {'평균 시간 (초)':<20} {'중앙값 시간 (초)':<20}")
            print("-" * 60)
            for method, stats in methods.items():
                mean_time = stats["mean"]
                median_time = stats["median"]
                print(f"{method:<15} {mean_time:.2f}{'':<13} {median_time:.2f}{'':<13}")

    # 다양성 요약
    if analysis_results["diversity"]:
        print("\n[다양성 분석]")
        for dataset, methods in analysis_results["diversity"].items():
            print(f"\n데이터셋: {dataset}")
            print(f"{'방법':<15} {'다양성 (평균)':<20} {'다양성 (중앙값)':<20}")
            print("-" * 60)
            for method, stats in methods.items():
                mean_diversity = stats["mean"]
                median_diversity = stats["median"]
                print(
                    f"{method:<15} {mean_diversity:.4f}{'':<13} {median_diversity:.4f}{'':<13}"
                )

    print("\n============================")

    # 그래프 생성
    if args.plot:
        logger.info("그래프 생성 중...")
        generate_plots(df, analysis_results, args.output_dir)

    logger.info("분석 완료!")


if __name__ == "__main__":
    main()
