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
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    # 질문 ID 로깅 (디버깅용)
    question_id = test_item.get("id", "unknown")

    # 1. answerKey 필드에서 정답 레이블 추출
    expected_key = test_item.get("answerKey", "")

    # 2. answerKey가 없다면 answer 필드에서 추출 시도
    if not expected_key and "answer" in test_item:
        answer_text = test_item.get("answer", "")
        # "B. concentration" 형식에서 "B"만 추출
        if answer_text and len(answer_text) >= 1:
            expected_key = answer_text[0]

    # 3. 여전히 정답이 없으면 False 반환
    if not expected_key:
        logger.warning(f"No answer key found for question: {question_id}")
        return False

    # 정답 키를 대문자로 변환하고 필요 없는 문자 제거
    expected_key = expected_key.strip().upper()
    if expected_key not in "ABCDE":
        logger.warning(f"Invalid answer key: {expected_key}")
        return False

    # 디버깅을 위한 로깅
    logger.debug(f"Evaluating {question_id}: Expected key = {expected_key}")

    # 추출된 패턴들을 저장할 변수
    extracted_patterns = []
    predicted_key = None

    # 가장 높은 우선순위: #### 기호 이후의 답변
    if "####" in response:
        after_hash = response.split("####")[-1].strip()
        # 직접적인 레이블 찾기 (A-E)
        hash_labels = re.findall(r"\b([A-E])\b", after_hash)
        if hash_labels:
            predicted_key = hash_labels[0].upper()
            extracted_patterns.append(f"After hash direct label: {predicted_key}")
        # 레이블이 없으면 "final answer: X" 패턴 찾기
        elif not predicted_key and re.search(
            r"final answer:?\s*([A-E])", after_hash, re.IGNORECASE
        ):
            matches = re.search(r"final answer:?\s*([A-E])", after_hash, re.IGNORECASE)
            predicted_key = matches.group(1).upper()
            extracted_patterns.append(f"After hash final answer: {predicted_key}")

    # 다음 우선순위: "Final answer:" 패턴 변형들
    if not predicted_key:
        final_patterns = [
            r"final answer:?\s*([A-E])[\.\s]",
            r"final answer:?\s*([A-E])\.",
            r"final answer:?\s*([A-E])$",
            r"final answer:?\s*\*\*([A-E])[.\s]",
            r"final answer:?\s*\*\*([A-E])\*\*",
            r"final answer:?\s*([A-E])[,;]",
            r"final answer:?\s*option\s+([A-E])",
            r"final answer:?\s*is\s+([A-E])",
        ]

        for pattern in final_patterns:
            matches = re.search(pattern, response, re.IGNORECASE)
            if matches:
                predicted_key = matches.group(1).upper()
                extracted_patterns.append(f"Final pattern: {predicted_key}")
                break

    # 다음 우선순위: "The answer is X" 패턴 변형들
    if not predicted_key:
        answer_patterns = [
            r"the answer is:?\s*([A-E])[\.\s]",
            r"the answer is:?\s*([A-E])\.",
            r"the answer is:?\s*([A-E])$",
            r"the answer is:?\s*\*\*([A-E])[.\s]",
            r"the answer is:?\s*\*\*([A-E])\*\*",
            r"the answer is:?\s*([A-E])[,;]",
            r"the answer is:?\s*option\s+([A-E])",
            r"my answer is:?\s*([A-E])",
        ]

        for pattern in answer_patterns:
            matches = re.search(pattern, response, re.IGNORECASE)
            if matches:
                predicted_key = matches.group(1).upper()
                extracted_patterns.append(f"Answer pattern: {predicted_key}")
                break

    # 다음 우선순위: 마크다운 포맷된 응답(**X**)
    if not predicted_key:
        markdown_patterns = [
            r"\*\*([A-E])\*\*",
            r"\*\*([A-E])[\.\s]",
            r"\*\*\s*([A-E])\s*\*\*",
        ]

        for pattern in markdown_patterns:
            matches = re.search(pattern, response)
            if matches:
                predicted_key = matches.group(1).upper()
                extracted_patterns.append(f"Markdown pattern: {predicted_key}")
                break

    # 마지막 부분에서 "X. 텍스트" 패턴 찾기
    if not predicted_key:
        # 마지막 5줄로 제한
        last_lines = "\n".join(response.strip().split("\n")[-5:])

        choice_patterns = [r"([A-E])\.\s*\w+", r"option\s+([A-E])"]

        for pattern in choice_patterns:
            matches = re.findall(pattern, last_lines, re.IGNORECASE)
            if matches:
                predicted_key = matches[-1].upper()
                extracted_patterns.append(f"Last lines choice pattern: {predicted_key}")
                break

    # 마지막 시도: 응답에서 A-E 레이블 찾기 (뒤에서 앞으로)
    if not predicted_key:
        all_labels = re.findall(r"\b([A-E])\b", response)
        if all_labels:
            # 마지막 레이블 사용
            predicted_key = all_labels[-1].upper()
            extracted_patterns.append(f"Last resort: {predicted_key}")

    # 패턴 중 하나라도 찾았는지 확인
    if not predicted_key:
        logger.warning(f"Could not extract answer from response for {question_id}")
        # 전체 응답 내용 로깅
        logger.debug(f"Full response for {question_id}: {response}")
        return False

    # 디버깅 로깅
    logger.debug(f"Extracted patterns for {question_id}: {extracted_patterns}")
    logger.debug(f"Predicted key: {predicted_key}, Expected key: {expected_key}")

    # 답변 비교
    is_correct = predicted_key == expected_key
    if not is_correct:
        logger.warning(
            f"Incorrect answer for {question_id}: extracted {predicted_key} != expected {expected_key}"
        )

    return is_correct


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

    # 정답 문자열에서 숫자와 문자만 추출
    expected_answer = re.sub(r"[^A-D0-9]", "", expected_answer).upper()

    # 숫자로 된 정답을 알파벳으로 변환 (필요한 경우)
    if expected_answer in ["1", "2", "3", "4"]:
        expected_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}[expected_answer]
    else:
        expected_letter = (
            expected_answer if expected_answer in ["A", "B", "C", "D"] else None
        )

    # 알파벳으로 된 정답을 숫자로 변환 (필요한 경우)
    if expected_answer in ["A", "B", "C", "D"]:
        expected_number = {"A": "1", "B": "2", "C": "3", "D": "4"}[expected_answer]
    else:
        expected_number = (
            expected_answer if expected_answer in ["1", "2", "3", "4"] else None
        )

    if not expected_letter and not expected_number:
        logger.warning(f"Invalid expected answer format: {expected_answer}")
        return False

    # 다양한 패턴으로 응답에서 답변 추출
    predicted_answer = None

    # 1) Final answer 패턴 (####으로 표시)
    final_answer_pattern = r"(?:final answer|answer)[\s:]*(?:is)?[\s:#]*\s*([1-4A-D])"
    matches = re.search(final_answer_pattern, response.lower())
    if matches:
        predicted_answer = matches.group(1).upper()

    # 2) 마지막 줄에서 숫자나 문자 찾기
    if not predicted_answer:
        last_line = response.strip().split("\n")[-1]
        last_line_pattern = r"([1-4A-D])(?:\.|:|$|\s|,)"
        matches = re.search(last_line_pattern, last_line)
        if matches:
            predicted_answer = matches.group(1).upper()

    # 3) "The answer is X" 패턴
    if not predicted_answer:
        answer_pattern = r"the answer is\s*([1-4A-D])"
        matches = re.search(answer_pattern, response.lower())
        if matches:
            predicted_answer = matches.group(1).upper()

    # 4) "Option X" 패턴
    if not predicted_answer:
        option_pattern = r"(?:option|choice)\s*([1-4A-D])"
        matches = re.search(option_pattern, response.lower())
        if matches:
            predicted_answer = matches.group(1).upper()

    # 5) 응답 전체에서 [A-D] or [1-4] 형태의 문자열 찾기
    if not predicted_answer:
        all_letters = re.findall(r"\b([A-D])\b", response.upper())
        if all_letters:
            predicted_answer = all_letters[-1]  # 마지막 문자 사용

    # 6) 응답 전체에서 숫자 1-4 찾기
    if not predicted_answer:
        all_numbers = re.findall(r"\b([1-4])\b", response)
        if all_numbers:
            predicted_answer = all_numbers[-1]  # 마지막 숫자 사용

    if not predicted_answer:
        return False

    # 추출된 답변이 숫자면 해당하는 알파벳으로 변환
    if predicted_answer in ["1", "2", "3", "4"]:
        predicted_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}[predicted_answer]
        predicted_number = predicted_answer
    # 추출된 답변이 알파벳이면 해당하는 숫자로 변환
    elif predicted_answer in ["A", "B", "C", "D"]:
        predicted_letter = predicted_answer
        predicted_number = {"A": "1", "B": "2", "C": "3", "D": "4"}[predicted_answer]
    else:
        return False

    # 예상 답변과 실제 답변 비교 (숫자 또는 알파벳 형식 모두 고려)
    return (expected_letter and expected_letter == predicted_letter) or (
        expected_number and expected_number == predicted_number
    )


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
