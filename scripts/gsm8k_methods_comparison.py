#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import internal modules
from src.data.dataset import Dataset
from src.utils.embedding import EmbeddingProcessor
from src.example_selection.selector import get_selector
from src.llm.gpt_client import GPTClient
from src.main import evaluate_response

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_method_experiment(
    method_name,
    dataset_name="gsm8k",
    num_test_samples=100,
    num_examples=5,
    **method_kwargs,
):
    """
    특정 방법론에 대해 실험 실행

    Args:
        method_name: 방법론 이름
        dataset_name: 데이터셋 이름
        num_test_samples: 테스트 샘플 수
        num_examples: 선택할 예제 수 (zero-shot의 경우 0)
        **method_kwargs: 방법론별 추가 파라미터

    Returns:
        실험 결과 딕셔너리
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing method: {method_name}")
    logger.info(f"{'='*60}")

    # 데이터셋 로드
    dataset = Dataset(dataset_name, "./data")
    test_data = dataset.get_test_data(num_test_samples)
    example_pool = dataset.get_example_pool()

    logger.info(f"Test data count: {len(test_data)}")
    logger.info(f"Example pool size: {len(example_pool)}")

    # 임베딩 처리 (zero-shot이 아닌 경우에만)
    if method_name != "zero-shot":
        embedding_processor = EmbeddingProcessor("all-mpnet-base-v2")
        example_embeddings = embedding_processor.embed_questions(
            example_pool, dataset_name, force_recompute=False
        )

        # 셀렉터 초기화
        selector = get_selector(
            method_name, example_pool, example_embeddings, **method_kwargs
        )
    else:
        embedding_processor = EmbeddingProcessor("all-mpnet-base-v2")
        selector = None

    # GPT 클라이언트 초기화
    gpt_client = GPTClient()

    # 실험 결과 저장
    experiment_results = []
    total_accuracy = 0
    total_selection_time = 0
    total_response_time = 0
    total_tokens = 0
    successful_tests = 0

    for i, test_item in enumerate(test_data):
        try:
            logger.info(f"Test {i+1}/{len(test_data)} ({method_name})")

            # 예제 선택 (시간 측정)
            start_time = time.time()

            if method_name == "zero-shot":
                selected_examples = []
                selection_time = 0.0
            else:
                # 테스트 임베딩 계산
                test_embedding = embedding_processor.model.encode(test_item["question"])
                selected_examples = selector.select_examples(
                    test_embedding, n_examples=num_examples
                )
                selection_time = time.time() - start_time

            # 프롬프트 생성
            if method_name == "zero-shot":
                # Zero-shot 프롬프트 (예제 없이)
                prompt = f"""Solve the following math word problem step by step. Show your work and provide the final numerical answer.

Problem: {test_item["question"]}

Think through this step by step and then provide your final answer as a number.

Answer:"""
            else:
                prompt = gpt_client.create_cot_prompt(
                    test_item["question"], selected_examples
                )

            # GPT 호출
            gpt_result = gpt_client.call_gpt(prompt)
            response = gpt_result["response"]

            # 평가
            is_correct, extracted_answer = evaluate_response(
                test_item, response, dataset_name
            )

            # 결과 저장
            experiment_results.append(
                {
                    "question_id": test_item["id"],
                    "correct": is_correct,
                    "selection_time": selection_time,
                    "response_time": gpt_result["response_time"],
                    "tokens": gpt_result["tokens"]["total_tokens"],
                    "selected_examples": (
                        [ex["id"] for ex in selected_examples]
                        if selected_examples
                        else []
                    ),
                }
            )

            if is_correct:
                total_accuracy += 1
            total_selection_time += selection_time
            total_response_time += gpt_result["response_time"]
            total_tokens += gpt_result["tokens"]["total_tokens"]
            successful_tests += 1

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error in test {i+1}: {str(e)}")
            continue

    # 결과 계산
    if successful_tests > 0:
        accuracy = total_accuracy / successful_tests
        avg_selection_time = total_selection_time / successful_tests
        avg_response_time = total_response_time / successful_tests
        avg_tokens = total_tokens / successful_tests

        results = {
            "method": method_name,
            "accuracy": accuracy,
            "avg_selection_time": avg_selection_time,
            "avg_response_time": avg_response_time,
            "avg_tokens": avg_tokens,
            "successful_tests": successful_tests,
            "total_tests": len(test_data),
            "detailed_results": experiment_results,
        }

        logger.info(f"{method_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f} ({total_accuracy}/{successful_tests})")
        logger.info(f"  Avg Selection Time: {avg_selection_time:.6f} seconds")
        logger.info(f"  Avg Response Time: {avg_response_time:.4f} seconds")
        logger.info(f"  Avg Tokens: {avg_tokens:.1f}")
        logger.info(f"  Successful Tests: {successful_tests}/{len(test_data)}")

        return results
    else:
        logger.warning(f"No successful tests for {method_name}")
        return {
            "method": method_name,
            "accuracy": 0,
            "avg_selection_time": 0,
            "avg_response_time": 0,
            "avg_tokens": 0,
            "successful_tests": 0,
            "total_tests": len(test_data),
            "detailed_results": [],
        }


def run_all_methods_comparison(dataset_name="gsm8k", num_test_samples=100):
    """
    모든 방법론에 대해 비교 실험 실행

    Args:
        dataset_name: 데이터셋 이름
        num_test_samples: 테스트 샘플 수

    Returns:
        모든 방법론의 실험 결과
    """
    logger.info(f"Starting comprehensive methods comparison on {dataset_name}")

    # 테스트할 방법론들과 파라미터
    methods_config = [
        {"name": "zero-shot", "num_examples": 0, "kwargs": {}},
        {"name": "random", "num_examples": 5, "kwargs": {}},
        {"name": "similarity", "num_examples": 5, "kwargs": {}},
        {"name": "mmr", "num_examples": 5, "kwargs": {"lambda_param": 0.7}},
        {"name": "kmeans", "num_examples": 5, "kwargs": {"n_clusters": 5}},
        {"name": "hdbscan", "num_examples": 5, "kwargs": {"min_cluster_size": 5}},
        {
            "name": "cm-mmr",
            "num_examples": 5,
            "kwargs": {"lambda_param": 0.7, "n_clusters": 5},
        },
        {
            "name": "cm-hdbscan-mmr",
            "num_examples": 5,
            "kwargs": {"lambda_param": 0.7, "min_cluster_size": 5},
        },
    ]

    all_results = {}

    for method_config in methods_config:
        method_name = method_config["name"]
        num_examples = method_config["num_examples"]
        kwargs = method_config["kwargs"]

        try:
            result = run_method_experiment(
                method_name=method_name,
                dataset_name=dataset_name,
                num_test_samples=num_test_samples,
                num_examples=num_examples,
                **kwargs,
            )
            all_results[method_name] = result

        except Exception as e:
            logger.error(f"Failed to run experiment for {method_name}: {str(e)}")
            all_results[method_name] = {
                "method": method_name,
                "accuracy": 0,
                "avg_selection_time": 0,
                "avg_response_time": 0,
                "avg_tokens": 0,
                "successful_tests": 0,
                "total_tests": num_test_samples,
                "error": str(e),
            }

    return all_results


def analyze_and_visualize_results(all_results, dataset_name, save_dir="./analysis"):
    """
    모든 방법론의 결과를 분석하고 시각화

    Args:
        all_results: 모든 방법론의 실험 결과
        dataset_name: 데이터셋 이름
        save_dir: 결과 저장 디렉토리
    """
    logger.info("Analyzing and visualizing results...")

    # 결과 데이터 준비
    methods = []
    accuracies = []
    selection_times = []
    response_times = []
    tokens = []
    successful_tests = []

    # 방법론 이름 매핑 (시각화용)
    method_display_names = {
        "zero-shot": "Zero-shot",
        "random": "Random",
        "similarity": "Similarity",
        "mmr": "MMR",
        "kmeans": "K-means",
        "hdbscan": "HDBSCAN",
        "cm-mmr": "CM-MMR\n(K-means)",
        "cm-hdbscan-mmr": "CM-MMR\n(HDBSCAN)",
    }

    for method_name, result in all_results.items():
        if result["successful_tests"] > 0:
            methods.append(method_display_names.get(method_name, method_name))
            accuracies.append(result["accuracy"] * 100)  # 백분율로 변환
            selection_times.append(result["avg_selection_time"] * 1000)  # ms로 변환
            response_times.append(result["avg_response_time"])
            tokens.append(result["avg_tokens"])
            successful_tests.append(result["successful_tests"])

    # 최고 성능 찾기
    best_accuracy_idx = np.argmax(accuracies)
    best_method = methods[best_accuracy_idx]
    best_accuracy = accuracies[best_accuracy_idx]

    fastest_selection_idx = np.argmin([t for t in selection_times if t > 0])  # 0 제외
    fastest_method = (
        methods[fastest_selection_idx]
        if len([t for t in selection_times if t > 0]) > 0
        else "N/A"
    )

    # 결과 출력
    print("\n" + "=" * 80)
    print(f"{dataset_name.upper()} Dataset - Methods Comparison Results")
    print("=" * 80)
    print(f"Best Accuracy: {best_method} ({best_accuracy:.2f}%)")
    if fastest_method != "N/A":
        print(
            f"Fastest Selection: {fastest_method} ({selection_times[fastest_selection_idx]:.3f}ms)"
        )

    print("\nDetailed Results:")
    print("-" * 80)
    for i, method in enumerate(methods):
        print(
            f"{method:15s}: Acc={accuracies[i]:5.2f}%, "
            f"SelTime={selection_times[i]:6.3f}ms, "
            f"RespTime={response_times[i]:5.2f}s, "
            f"Tokens={tokens[i]:5.0f}, "
            f"Tests={successful_tests[i]}"
        )

    # 시각화
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 정확도 비교
    bars1 = ax1.bar(methods, accuracies, color="lightblue", alpha=0.7)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"{dataset_name.upper()}: Accuracy Comparison")
    ax1.set_ylim(0, max(accuracies) * 1.1)
    ax1.grid(True, alpha=0.3)

    # 정확도 값 표시
    for bar, acc in zip(bars1, accuracies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.setp(ax1.get_xticklabels(), rotation=15, fontsize=8)

    # 2. 선택 지연시간 비교 (로그 스케일)
    non_zero_times = [
        t if t > 0 else 0.001 for t in selection_times
    ]  # 0을 작은 값으로 대체
    bars2 = ax2.bar(methods, non_zero_times, color="lightcoral", alpha=0.7)
    ax2.set_ylabel("Selection Time (ms)")
    ax2.set_title(f"{dataset_name.upper()}: Selection Latency Comparison")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # 선택 시간 값 표시
    for bar, time_val in zip(bars2, selection_times):
        if time_val > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.2,
                f"{time_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        else:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                0.002,
                "0.000",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.setp(ax2.get_xticklabels(), rotation=15, fontsize=8)

    # 3. 정확도 vs 선택 시간 산점도
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    for i, (acc, time_val, method) in enumerate(
        zip(accuracies, selection_times, methods)
    ):
        ax3.scatter(
            time_val if time_val > 0 else 0.001,
            acc,
            c=[colors[i]],
            s=100,
            alpha=0.7,
            label=method,
        )

    ax3.set_xlabel("Selection Time (ms)")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title(f"{dataset_name.upper()}: Accuracy vs Selection Time")
    ax3.set_xscale("log")
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # 4. 응답 시간 비교
    bars4 = ax4.bar(methods, response_times, color="lightgreen", alpha=0.7)
    ax4.set_ylabel("Response Time (seconds)")
    ax4.set_title(f"{dataset_name.upper()}: GPT Response Time")
    ax4.grid(True, alpha=0.3)

    # 응답 시간 값 표시
    for bar, resp_time in zip(bars4, response_times):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{resp_time:.2f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.setp(ax4.get_xticklabels(), rotation=15, fontsize=8)

    plt.tight_layout()

    # 저장
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"{dataset_name}_methods_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {plot_path}")

    # 결과를 CSV로 저장
    df = pd.DataFrame(
        {
            "method": methods,
            "accuracy": accuracies,
            "selection_time_ms": selection_times,
            "response_time_s": response_times,
            "avg_tokens": tokens,
            "successful_tests": successful_tests,
        }
    )

    csv_path = os.path.join(save_dir, f"{dataset_name}_methods_comparison_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # JSON으로도 저장
    analysis_results = {
        "dataset": dataset_name,
        "best_accuracy_method": best_method,
        "best_accuracy": best_accuracy,
        "fastest_selection_method": fastest_method,
        "detailed_results": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    json_path = os.path.join(
        save_dir, f"{dataset_name}_methods_comparison_analysis.json"
    )
    with open(json_path, "w") as f:
        json.dump(analysis_results, f, indent=2)
    logger.info(f"Analysis results saved to {json_path}")

    plt.show()

    return analysis_results


def main():
    """메인 함수"""
    logger.info("Starting GSM8K Methods Comparison...")

    # 모든 방법론 실험 실행
    all_results = run_all_methods_comparison(
        dataset_name="gsm8k",
        num_test_samples=100,  # 전체 실험을 위해 100개로 설정
    )

    # 결과 분석 및 시각화
    analysis = analyze_and_visualize_results(all_results, "gsm8k", "./analysis")

    logger.info("\nGSM8K Methods Comparison completed!")
    logger.info("Key findings:")
    logger.info(
        f"- Best method: {analysis['best_accuracy_method']} ({analysis['best_accuracy']:.2f}%)"
    )
    logger.info(f"- Fastest selection: {analysis['fastest_selection_method']}")
    logger.info("Check ./analysis/ directory for detailed results and visualizations.")


if __name__ == "__main__":
    main()
