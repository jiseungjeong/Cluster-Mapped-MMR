#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import argparse
import sys
import time
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import internal modules
from src.data.dataset import Dataset
from src.utils.embedding import EmbeddingProcessor
from src.example_selection.selector import get_selector
from src.llm.gpt_client import GPTClient
from src.utils.experiment import ExperimentManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="CoT Example Selection Technique Experiment"
    )

    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "commonsenseqa", "combined"],
        default="gsm8k",
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Data directory path"
    )
    parser.add_argument(
        "--num_test_samples", type=int, default=100, help="Number of test samples"
    )

    # Example selection settings
    parser.add_argument(
        "--method",
        type=str,
        choices=["kmeans", "hdbscan", "mmr", "random", "cm-mmr"],
        default="kmeans",
        help="Example selection method",
    )
    parser.add_argument(
        "--num_examples", type=int, default=5, help="Number of examples to select"
    )
    parser.add_argument(
        "--lambda_param",
        type=float,
        default=0.7,
        help="MMR lambda parameter (only applies to MMR method)",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of clusters (only applies to kmeans method)",
    )

    # Embedding settings
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-mpnet-base-v2",
        help="Embedding model",
    )
    parser.add_argument(
        "--force_recompute", action="store_true", help="Whether to recompute embeddings"
    )

    # GPT settings
    parser.add_argument(
        "--gpt_model",
        type=str,
        default=None,
        help="GPT model name (None to use environment variable)",
    )
    parser.add_argument(
        "--gpt_temperature",
        type=float,
        default=None,
        help="GPT temperature setting (None to use environment variable)",
    )
    parser.add_argument(
        "--gpt_max_tokens",
        type=int,
        default=None,
        help="GPT maximum token count (None to use environment variable)",
    )

    # Experiment settings
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name (None for auto-generation)",
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results", help="Results directory"
    )

    return parser.parse_args()


def evaluate_gsm8k_response(test_item: Dict, response: str) -> bool:
    """
    Evaluate GSM8K response

    Args:
        test_item: Test item
        response: GPT response

    Returns:
        Correctness (True/False)
    """
    # Extract answer
    expected_answer = test_item.get("answer", "")
    if not expected_answer:
        return False

    # Extract only numbers
    expected_number = re.search(r"[-+]?\d*\.?\d+", expected_answer)
    if expected_number:
        expected_number = expected_number.group()
    else:
        return False

    # Try to find the answer in the response
    # 1) Find number after ####
    answer_pattern = r"####\s*([-+]?\d*\.?\d+)"
    matches = re.search(answer_pattern, response)

    # 2) Find number in the last line
    if not matches:
        last_line = response.strip().split("\n")[-1]
        matches = re.search(r"([-+]?\d*\.?\d+)", last_line)

    # 3) Find any number in the response
    if not matches:
        matches = re.findall(r"([-+]?\d*\.?\d+)", response)
        if matches:
            # Use the last number
            predicted_number = matches[-1]
        else:
            return False
    else:
        predicted_number = matches.group(1) if hasattr(matches, "group") else matches[0]

    # Compare numbers
    try:
        expected_value = float(expected_number)
        predicted_value = float(predicted_number)
        return abs(expected_value - predicted_value) < 1e-6
    except:
        return False


def evaluate_commonsenseqa_response(test_item: Dict, response: str) -> bool:
    """
    Evaluate CommonsenseQA response

    Args:
        test_item: Test item
        response: GPT response

    Returns:
        Correctness (True/False)
    """
    # Extract answer label
    expected_key = test_item.get("answerKey", "")
    if not expected_key:
        return False

    # Try to extract label from response
    # 1) Direct label matching
    label_pattern = r"([A-E])[.:]"
    matches = re.search(label_pattern, response)

    # 2) Search for "the answer is A" format
    if not matches:
        answer_pattern = r"the answer is\s*([A-E])"
        matches = re.search(answer_pattern, response)

    # 3) Search for "The answer is A" format (case insensitive)
    if not matches:
        english_pattern = r"[Tt]he answer is\s*([A-E])"
        matches = re.search(english_pattern, response)

    # 4) Last method: find any A-E label in the response
    if not matches:
        all_labels = re.findall(r"\b([A-E])\b", response)
        if all_labels:
            # Use the first label
            predicted_key = all_labels[0]
        else:
            return False
    else:
        predicted_key = matches.group(1)

    # Compare labels
    return predicted_key == expected_key


def evaluate_response(test_item: Dict, response: str, dataset_name: str) -> bool:
    """
    Evaluate GPT response

    Args:
        test_item: Test item
        response: GPT response
        dataset_name: Dataset name

    Returns:
        Correctness (True/False)
    """
    # For combined dataset, determine evaluation method based on the dataset field in the test item
    if dataset_name == "combined":
        item_dataset = test_item.get("dataset", "").lower()
        if item_dataset == "arc":
            # ARC multiple choice format evaluation
            return evaluate_arc_response(test_item, response)
        elif item_dataset == "commonsenseqa":
            return evaluate_commonsenseqa_response(test_item, response)
        else:
            logger.warning(f"Unknown dataset in combined data: {item_dataset}")
            return False

    # Original dataset evaluation
    elif dataset_name == "gsm8k":
        return evaluate_gsm8k_response(test_item, response)
    elif dataset_name == "commonsenseqa":
        return evaluate_commonsenseqa_response(test_item, response)
    else:
        logger.warning(f"No evaluation logic for dataset: {dataset_name}")
        return False


def evaluate_arc_response(test_item: Dict, response: str) -> bool:
    """
    Evaluate ARC multiple choice response

    Args:
        test_item: Test item
        response: GPT response

    Returns:
        Correctness (True/False)
    """
    # Get expected answer
    expected_answer = test_item.get("answer", "")
    if not expected_answer:
        return False

    # Try to extract answer from response
    # 1) Direct letter matching (A-D)
    answer_pattern = r"([A-D])[.:]"
    matches = re.search(answer_pattern, response)

    # 2) Search for "the answer is X" format
    if not matches:
        answer_pattern = r"the answer is\s*([A-D])"
        matches = re.search(answer_pattern, response.lower())

    # 3) Look for numbers (1-4) as possible answer choices
    if not matches:
        num_pattern = r"answer\s*(?:is|:)?\s*([1-4])"
        matches = re.search(num_pattern, response.lower())

        if matches:
            # Convert number to letter
            num_answer = matches.group(1)
            letter_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
            predicted_answer = letter_map.get(num_answer)
        else:
            # Try to find any standalone A, B, C, D in the response
            all_letters = re.findall(r"\b([A-D])\b", response)
            predicted_answer = all_letters[0] if all_letters else None
    else:
        predicted_answer = matches.group(1)

    return predicted_answer == expected_answer


def run_experiment(args):
    """
    Run experiment

    Args:
        args: Experiment arguments
    """
    # Create experiment name with timestamp
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.dataset}_{args.method}_{timestamp}"

    # Initialize experiment manager
    experiment = ExperimentManager(args.results_dir, args.experiment_name)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = Dataset(args.dataset, args.data_dir)

    # Get test data and example pool
    test_data = dataset.get_test_data(args.num_test_samples)
    example_pool = dataset.get_example_pool()

    logger.info(f"Test data count: {len(test_data)}")
    logger.info(f"Example pool size: {len(example_pool)}")

    # Initialize embedding processor
    embedding_processor = EmbeddingProcessor(args.embedding_model)

    # Calculate example pool embeddings
    example_embeddings = embedding_processor.embed_questions(
        example_pool, args.dataset, args.force_recompute
    )

    # Initialize GPT client
    gpt_client = GPTClient(
        model_name=args.gpt_model,
        temperature=args.gpt_temperature,
        max_tokens=args.gpt_max_tokens,
    )

    # Initialize example selector
    selector_kwargs = {"lambda_param": args.lambda_param, "n_clusters": args.n_clusters}

    selector = get_selector(
        args.method, example_pool, example_embeddings, **selector_kwargs
    )

    # Run experiment for each test item
    results = []
    total_latency = 0.0
    successful_tests = 0

    for i, test_item in enumerate(test_data):
        logger.info(f"Test ({i+1}/{len(test_data)}): {test_item['id']}")

        # Calculate test item embedding
        test_embedding = embedding_processor.model.encode(test_item["question"])

        # Select examples
        start_time = time.time()
        selected_examples = selector.select_examples(
            test_embedding, n_examples=args.num_examples
        )
        selection_time = time.time() - start_time

        # Extract selected example IDs
        selected_ids = [example["id"] for example in selected_examples]
        logger.info(f"Selected examples: {selected_ids}")

        # Create prompt
        prompt = gpt_client.create_cot_prompt(test_item["question"], selected_examples)

        # Call GPT
        try:
            gpt_result = gpt_client.call_gpt(prompt)
            response = gpt_result["response"]
            response_time = gpt_result["response_time"]

            # Total latency (selection + API call)
            total_test_latency = selection_time + response_time
            total_latency += total_test_latency
            successful_tests += 1

            # Evaluate correctness
            is_correct = evaluate_response(test_item, response, args.dataset)

            # Log result
            result = {
                "dataset": args.dataset,
                "method": args.method,
                "question_id": test_item["id"],
                "question": test_item["question"],
                "expected_answer": test_item.get("answer", ""),
                "response": response,
                "selected_examples": selected_ids,
                "correct": is_correct,
                "tokens": gpt_result["tokens"],
                "response_time": response_time,
                "selection_time": selection_time,
                "total_latency": total_test_latency,
            }

            results.append(result)
            experiment.log_result(result)

        except Exception as e:
            logger.error(f"GPT call failed: {str(e)}")

            # Log error result
            result = {
                "dataset": args.dataset,
                "method": args.method,
                "question_id": test_item["id"],
                "question": test_item["question"],
                "error": str(e),
            }

            experiment.log_result(result)

        # Rate limiting
        time.sleep(0.5)

    # Save results
    experiment.save_results()

    # Output summary
    summary = experiment.summarize_results()
    experiment.print_summary(summary)

    # Print latency metrics
    if successful_tests > 0:
        avg_latency = total_latency / successful_tests
        logger.info(f"Average total latency per test sample: {avg_latency:.2f} seconds")

        # Detailed latency analysis if results are available
        if results:
            selection_times = [
                r.get("selection_time", 0) for r in results if "selection_time" in r
            ]
            response_times = [
                r.get("response_time", 0) for r in results if "response_time" in r
            ]

            if selection_times:
                avg_selection = sum(selection_times) / len(selection_times)
                logger.info(
                    f"Average example selection time: {avg_selection:.2f} seconds"
                )

            if response_times:
                avg_response = sum(response_times) / len(response_times)
                logger.info(f"Average LLM response time: {avg_response:.2f} seconds")
    else:
        logger.warning("No successful tests to calculate average latency")

    return experiment


def main():
    """Main function"""
    args = parse_args()

    logger.info(f"Starting experiment: dataset={args.dataset}, method={args.method}")

    experiment = run_experiment(args)

    logger.info("Experiment completed!")


if __name__ == "__main__":
    main()
