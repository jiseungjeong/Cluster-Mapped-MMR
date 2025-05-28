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
from src.example_selection.selector import CmHdbscanMmrSelector
from src.llm.gpt_client import GPTClient
from src.utils.experiment import ExperimentManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_response(test_item, response, dataset_name):
    """main.py에서 가져온 정확한 평가 함수"""
    # main.py의 evaluate_response 함수를 import해서 사용
    from src.main import evaluate_response as main_evaluate_response

    return main_evaluate_response(test_item, response, dataset_name)


def run_lambda_experiment(
    lambda_values, dataset_name="combined", num_test_samples=50, num_examples=5
):
    """
    다양한 lambda 값에 대해 CM-HDBSCAN-MMR 실험 실행

    Args:
        lambda_values: 테스트할 lambda 값들의 리스트
        dataset_name: 사용할 데이터셋
        num_test_samples: 테스트 샘플 수
        num_examples: 선택할 예제 수

    Returns:
        실험 결과 딕셔너리
    """
    logger.info(f"Starting lambda optimization experiment for CM-HDBSCAN-MMR")
    logger.info(f"Lambda values to test: {lambda_values}")

    # 데이터셋 로드
    dataset = Dataset(dataset_name, "./data")
    test_data = dataset.get_test_data(num_test_samples)
    example_pool = dataset.get_example_pool()

    logger.info(f"Test data count: {len(test_data)}")
    logger.info(f"Example pool size: {len(example_pool)}")

    # 임베딩 처리
    embedding_processor = EmbeddingProcessor("all-mpnet-base-v2")
    example_embeddings = embedding_processor.embed_questions(
        example_pool, dataset_name, force_recompute=False
    )

    # GPT 클라이언트 초기화
    gpt_client = GPTClient()

    # 결과 저장용
    results = {}

    for lambda_val in lambda_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing lambda = {lambda_val}")
        logger.info(f"{'='*60}")

        # CM-HDBSCAN-MMR 셀렉터 초기화
        selector = CmHdbscanMmrSelector(
            example_pool=example_pool,
            example_embeddings=example_embeddings,
            lambda_param=lambda_val,
            min_cluster_size=5,
            min_samples=None,
        )

        # 실험 결과 저장
        experiment_results = []
        total_accuracy = 0
        total_selection_time = 0
        successful_tests = 0

        for i, test_item in enumerate(test_data):
            try:
                logger.info(f"Test {i+1}/{len(test_data)} (lambda={lambda_val})")

                # 테스트 임베딩 계산
                test_embedding = embedding_processor.model.encode(test_item["question"])

                # 예제 선택 (시간 측정)
                start_time = time.time()
                selected_examples = selector.select_examples(
                    test_embedding, n_examples=num_examples
                )
                selection_time = time.time() - start_time

                # 프롬프트 생성
                prompt = gpt_client.create_cot_prompt(
                    test_item["question"], selected_examples
                )

                # GPT 호출
                gpt_result = gpt_client.call_gpt(prompt)
                response = gpt_result["response"]

                # 평가 (간단한 버전)
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
                        "tokens": gpt_result["tokens"],
                    }
                )

                if is_correct:
                    total_accuracy += 1
                total_selection_time += selection_time
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

            results[lambda_val] = {
                "accuracy": accuracy,
                "avg_selection_time": avg_selection_time,
                "successful_tests": successful_tests,
                "total_tests": len(test_data),
                "detailed_results": experiment_results,
            }

            logger.info(f"Lambda {lambda_val} Results:")
            logger.info(
                f"  Accuracy: {accuracy:.4f} ({total_accuracy}/{successful_tests})"
            )
            logger.info(f"  Avg Selection Time: {avg_selection_time:.6f} seconds")
            logger.info(f"  Successful Tests: {successful_tests}/{len(test_data)}")
        else:
            logger.warning(f"No successful tests for lambda {lambda_val}")
            results[lambda_val] = {
                "accuracy": 0,
                "avg_selection_time": 0,
                "successful_tests": 0,
                "total_tests": len(test_data),
                "detailed_results": [],
            }

    return results


def analyze_and_visualize_results(results, save_dir="./analysis"):
    """
    실험 결과 분석 및 시각화

    Args:
        results: 실험 결과 딕셔너리
        save_dir: 결과 저장 디렉토리
    """
    logger.info("Analyzing and visualizing results...")

    # 결과 데이터 준비
    lambda_values = list(results.keys())
    accuracies = [results[lam]["accuracy"] for lam in lambda_values]
    selection_times = [results[lam]["avg_selection_time"] for lam in lambda_values]

    # 최적 lambda 찾기
    best_lambda_accuracy = lambda_values[np.argmax(accuracies)]
    best_accuracy = max(accuracies)

    best_lambda_speed = lambda_values[np.argmin(selection_times)]
    best_speed = min(selection_times)

    # 결과 출력
    print("\n" + "=" * 80)
    print("CM-HDBSCAN-MMR Lambda Optimization Results")
    print("=" * 80)
    print(
        f"Best Lambda for Accuracy: {best_lambda_accuracy} (Accuracy: {best_accuracy:.4f})"
    )
    print(f"Best Lambda for Speed: {best_lambda_speed} (Time: {best_speed:.6f}s)")
    print("\nDetailed Results:")
    print("-" * 80)

    for lam in lambda_values:
        acc = results[lam]["accuracy"]
        time_val = results[lam]["avg_selection_time"]
        tests = results[lam]["successful_tests"]
        print(
            f"Lambda {lam:4.1f}: Accuracy={acc:.4f}, Time={time_val:.6f}s, Tests={tests}"
        )

    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 정확도 그래프
    ax1.plot(lambda_values, accuracies, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Lambda Value")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("CM-HDBSCAN-MMR: Accuracy vs Lambda")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(
        x=best_lambda_accuracy,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Best λ={best_lambda_accuracy}",
    )
    ax1.legend()

    # 선택 시간 그래프
    ax2.plot(lambda_values, selection_times, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Lambda Value")
    ax2.set_ylabel("Average Selection Time (seconds)")
    ax2.set_title("CM-HDBSCAN-MMR: Selection Time vs Lambda")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(
        x=best_lambda_speed,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"Fastest λ={best_lambda_speed}",
    )
    ax2.legend()

    plt.tight_layout()

    # 저장
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, "cm_hdbscan_lambda_optimization.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {plot_path}")

    # 결과를 CSV로 저장
    df = pd.DataFrame(
        {
            "lambda": lambda_values,
            "accuracy": accuracies,
            "avg_selection_time": selection_times,
            "successful_tests": [
                results[lam]["successful_tests"] for lam in lambda_values
            ],
        }
    )

    csv_path = os.path.join(save_dir, "cm_hdbscan_lambda_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # JSON으로도 저장
    json_path = os.path.join(save_dir, "cm_hdbscan_lambda_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to {json_path}")

    plt.show()

    return best_lambda_accuracy, best_accuracy


def main():
    """메인 함수"""
    # 테스트할 lambda 값들 (0.1부터 0.9까지)
    lambda_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # 실험 실행
    results = run_lambda_experiment(
        lambda_values=lambda_values,
        dataset_name="combined",
        num_test_samples=300,  # combined 데이터셋 전체 사용
        num_examples=5,
    )

    # 결과 분석 및 시각화
    best_lambda, best_accuracy = analyze_and_visualize_results(results)

    logger.info(f"\nOptimization completed!")
    logger.info(f"Recommended lambda value: {best_lambda}")
    logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
