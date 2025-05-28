#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimize_cm_hdbscan_lambda import (
    run_lambda_experiment,
    analyze_and_visualize_results,
)


def fine_tune_lambda(initial_best_lambda, range_width=0.2, num_points=11, **kwargs):
    """
    초기 최적 lambda 주변에서 세밀한 튜닝 수행

    Args:
        initial_best_lambda: 초기 실험에서 찾은 최적 lambda
        range_width: 탐색 범위 폭
        num_points: 테스트할 점의 개수
        **kwargs: run_lambda_experiment에 전달할 추가 인자들

    Returns:
        최적화된 lambda 값과 정확도
    """
    # 세밀한 lambda 값 범위 생성
    min_lambda = max(0.0, initial_best_lambda - range_width / 2)
    max_lambda = min(1.0, initial_best_lambda + range_width / 2)

    lambda_values = np.linspace(min_lambda, max_lambda, num_points)
    lambda_values = [round(lam, 3) for lam in lambda_values]  # 소수점 3자리로 반올림

    print(f"Fine-tuning around λ={initial_best_lambda}")
    print(f"Testing range: {min_lambda:.3f} to {max_lambda:.3f}")
    print(f"Lambda values: {lambda_values}")

    # 실험 실행
    results = run_lambda_experiment(lambda_values=lambda_values, **kwargs)

    # 결과 분석
    best_lambda, best_accuracy = analyze_and_visualize_results(
        results, save_dir="./analysis/fine_tuned"
    )

    return best_lambda, best_accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune CM-HDBSCAN-MMR lambda parameter"
    )
    parser.add_argument(
        "--initial_lambda",
        type=float,
        default=0.5,
        help="Initial best lambda value from coarse search",
    )
    parser.add_argument(
        "--range_width",
        type=float,
        default=0.2,
        help="Width of search range around initial lambda",
    )
    parser.add_argument(
        "--num_points", type=int, default=11, help="Number of lambda values to test"
    )
    parser.add_argument(
        "--num_test_samples", type=int, default=50, help="Number of test samples"
    )
    parser.add_argument(
        "--dataset", type=str, default="combined", help="Dataset to use"
    )

    args = parser.parse_args()

    # 세밀한 튜닝 실행
    best_lambda, best_accuracy = fine_tune_lambda(
        initial_best_lambda=args.initial_lambda,
        range_width=args.range_width,
        num_points=args.num_points,
        dataset_name=args.dataset,
        num_test_samples=args.num_test_samples,
        num_examples=5,
    )

    print(f"\nFine-tuning completed!")
    print(f"Optimized lambda value: {best_lambda}")
    print(f"Best accuracy achieved: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
