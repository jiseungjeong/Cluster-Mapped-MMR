#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import logging
import argparse
import sys
import itertools
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run all CoT example selection technique experiments"
    )

    # Dataset settings
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["gsm8k", "commonsenseqa", "combined"],
        default=["gsm8k", "commonsenseqa"],
        help="List of datasets to experiment with",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=["kmeans", "hdbscan", "mmr", "random", "cm-mmr"],
        default=["kmeans", "mmr", "random"],
        help="List of example selection methods to experiment with",
    )
    parser.add_argument(
        "--num_examples", type=int, default=5, help="Number of examples to select"
    )
    parser.add_argument(
        "--num_test_samples", type=int, default=100, help="Number of test samples"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Data directory path"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results", help="Results directory"
    )

    return parser.parse_args()


def run_experiment(
    dataset: str,
    method: str,
    num_examples: int,
    num_test_samples: int,
    data_dir: str,
    results_dir: str,
):
    """
    Run a single experiment

    Args:
        dataset: Dataset name
        method: Example selection method
        num_examples: Number of examples to select
        num_test_samples: Number of test samples
        data_dir: Data directory path
        results_dir: Results directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{dataset}_{method}_{timestamp}"

    logger.info(f"Running experiment: {dataset} / {method}")

    # Build command
    cmd = [
        "python",
        "src/main.py",
        "--dataset",
        dataset,
        "--method",
        method,
        "--num_examples",
        str(num_examples),
        "--num_test_samples",
        str(num_test_samples),
        "--data_dir",
        data_dir,
        "--results_dir",
        results_dir,
        "--experiment_name",
        experiment_name,
    ]

    # Add additional arguments for specific methods
    if method == "mmr":
        cmd.extend(["--lambda_param", "0.7"])
    elif method == "kmeans":
        cmd.extend(["--n_clusters", "5"])
    elif method == "cm-mmr":
        cmd.extend(["--lambda_param", "0.7", "--n_clusters", "5"])

    # Run experiment
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Experiment completed: {dataset} / {method}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment failed: {dataset} / {method}, error: {str(e)}")
        return False


def main():
    """Main function"""
    args = parse_args()

    # Create experiment combinations
    experiments = list(itertools.product(args.datasets, args.methods))

    logger.info(f"Planning to run {len(experiments)} experiments")

    # Create results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_results_dir = os.path.join(args.results_dir, f"batch_{timestamp}")
    os.makedirs(batch_results_dir, exist_ok=True)

    # Run each experiment
    for i, (dataset, method) in enumerate(experiments):
        logger.info(f"Experiment {i+1}/{len(experiments)}: {dataset} / {method}")

        success = run_experiment(
            dataset=dataset,
            method=method,
            num_examples=args.num_examples,
            num_test_samples=args.num_test_samples,
            data_dir=args.data_dir,
            results_dir=batch_results_dir,
        )

    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()
