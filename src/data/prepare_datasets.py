#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import argparse
import sys

# Add parent directory path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import internal modules
from src.data.dataset import Dataset

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function for dataset preparation"""
    parser = argparse.ArgumentParser(
        description="Dataset preparation for CoT experiments"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "commonsenseqa", "all"],
        default="all",
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Data directory path"
    )
    args = parser.parse_args()

    datasets = ["gsm8k", "commonsenseqa"] if args.dataset == "all" else [args.dataset]

    for dataset_name in datasets:
        logger.info(f"Preparing dataset: {dataset_name}")
        dataset = Dataset(dataset_name, args.data_dir)

        # Check test data
        test_data = dataset.get_test_data()
        logger.info(f"{dataset_name} test data count: {len(test_data)}")

        # Check example pool
        example_pool = dataset.get_example_pool()
        logger.info(f"{dataset_name} example pool size: {len(example_pool)}")

    logger.info("All datasets prepared successfully!")


if __name__ == "__main__":
    main()
