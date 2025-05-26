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
        choices=["gsm8k", "commonsenseqa", "arc", "combined"],
        default="gsm8k",
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Data directory path"
    )
    parser.add_argument(
        "--num_test_samples", type=int, default=100, help="Number of test samples"
    )

    # 반복 실험 설정
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="Number of times to repeat the experiment",
    )

    # Example selection settings
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "kmeans",
            "hdbscan",
            "mmr",
            "random",
            "cm-mmr",
            "cm-hdbscan-mmr",
            "similarity",
        ],
        default="kmeans",
        help="Example selection method",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        nargs="+",
        default=[5],
        help="Number of examples to select (can specify multiple values)",
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

    # HDBSCAN 관련 인자 추가
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=5,
        help="Minimum cluster size (only applies to HDBSCAN-based methods)",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=None,
        help="Minimum samples in neighborhood (only applies to HDBSCAN-based methods)",
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


def evaluate_gsm8k_response(test_item: Dict, response: str) -> tuple:
    """
    Evaluate GSM8K response

    Args:
        test_item: Test item
        response: GPT response

    Returns:
        Tuple of (correctness (True/False), extracted_answer)
    """
    # 디버그를 위한 질문 ID 로깅
    question_id = test_item.get("id", "unknown")

    # Extract answer
    expected_answer = test_item.get("answer", "")
    if not expected_answer:
        return False, None

    # Extract only numbers from expected answer, removing commas
    expected_answer = expected_answer.replace(",", "")
    expected_number = re.search(r"[-+]?\d*\.?\d+", expected_answer)
    if expected_number:
        expected_number = expected_number.group()
    else:
        return False, None

    # 추출된 패턴들을 저장할 변수
    extracted_patterns = []
    predicted_number = None

    # 응답이 비어 있으면 False 반환
    if not response or len(response.strip()) == 0:
        logger.warning(f"Empty response for {question_id}")
        return False, None

    # 최종 결론 또는 "Final answer", "Final step" 부분 식별
    final_sections = []

    # 1. #### 태그 이후 부분 (최우선)
    if "####" in response:
        after_hash = response.split("####")[-1].strip()
        final_sections.append(("#### 태그 이후", after_hash))

    # 2. "Final answer:" 또는 "Final step:" 이후 부분
    final_answer_match = re.search(
        r"(?:Final answer:|Final step:|Answer:|Therefore,)[^.!?]*",
        response,
        re.IGNORECASE,
    )
    if final_answer_match:
        final_answer_text = final_answer_match.group(0)
        final_sections.append(("Final answer/step", final_answer_text))

    # 3. 마지막 문단
    last_paragraph = response.strip().split("\n\n")[-1]
    if "**" in last_paragraph:  # 굵은 글씨가 있는 경우 우선 고려
        final_sections.append(("마지막 문단(굵은 글씨)", last_paragraph))
    else:
        final_sections.append(("마지막 문단", last_paragraph))

    # 응답 전체를 정규화 (달러 기호 제거 및 콤마 제거)
    clean_response = re.sub(r"\$\s*", "", response)
    clean_response = re.sub(r",", "", clean_response)

    # Try to find the answer in the response - multiple approaches in order of priority

    # 추출 우선순위 영역에서 정답 찾기
    for section_name, section_text in final_sections:
        # 달러 기호 및 콤마 제거
        section_text = re.sub(r"\$\s*", "", section_text)
        section_text = re.sub(r",", "", section_text)

        # 1) 굵은 글씨 패턴 (Bold pattern) - 최우선
        bold_matches = re.findall(r"\*\*([-+]?\d*\.?\d+)\*\*", section_text)
        if bold_matches:
            predicted_number = bold_matches[-1]  # 가장 마지막 굵은 글씨 숫자 사용
            extracted_patterns.append(f"{section_name} 굵은 글씨: {predicted_number}")
            break

        # 2) "Final answer: X" 또는 "answer: X" 패턴 찾기
        final_pattern = r"(?:final answer|final result|final step|answer|therefore)(?:[:\s])+\s*([-+]?\d*\.?\d+)"
        final_matches = re.search(final_pattern, section_text.lower())
        if final_matches:
            predicted_number = final_matches.group(1)
            extracted_patterns.append(f"{section_name} 결론 문구: {predicted_number}")
            break

        # 3) 숫자 뒤에 단위나 특수 표시가 있는 경우 (예: $42.00, 8%)
        unit_patterns = [
            r"([-+]?\d*\.?\d+)(?:\s*dollars|\s*\$|\s*%)",  # 42 dollars, $42, 8%
            r"([-+]?\d*\.?\d+)(?:\s*is|\s*=)",  # 42 is the answer, =42
        ]
        for pattern in unit_patterns:
            unit_matches = re.findall(pattern, section_text.lower())
            if unit_matches:
                predicted_number = unit_matches[-1]  # 마지막 매칭 사용
                extracted_patterns.append(
                    f"{section_name} 단위 포함: {predicted_number}"
                )
                break

        # 4) 일반 숫자 찾기
        if not predicted_number:
            number_matches = re.findall(r"([-+]?\d*\.?\d+)", section_text)
            if number_matches:
                predicted_number = number_matches[-1]  # 마지막 숫자 사용
                extracted_patterns.append(
                    f"{section_name} 일반 숫자: {predicted_number}"
                )
                break

    # 위 방법으로 찾지 못한 경우 기존 로직 사용

    # 1) #### 패턴 - #### 태그 이후의 텍스트 처리 개선
    if not predicted_number and "####" in response:
        # 마지막 #### 태그 이후의 텍스트만 고려
        after_hash = response.split("####")[-1].strip()

        # 직접 숫자 찾기 (달러 기호 제거)
        after_hash = after_hash.replace("$", "").replace(",", "").strip()
        hash_direct_pattern = r"([-+]?\d*\.?\d+)"
        hash_direct_matches = re.findall(hash_direct_pattern, after_hash)

        if hash_direct_matches:
            predicted_number = hash_direct_matches[0]
            extracted_patterns.append(f"Hash direct: {predicted_number}")

    # 2) 마크다운 굵은 글씨 패턴 및 달러 기호 처리 개선
    if not predicted_number:
        # 달러 기호 처리 및 콤마 제거
        dollar_clean_response = re.sub(r"\$\s*", "", response)
        dollar_clean_response = re.sub(r",", "", dollar_clean_response)

        bold_patterns = [
            r"\*\*([-+]?\d*\.?\d+)\*\*",  # **100**
            r"\*\*([-+]?\d*\.?\d+)(?:\D|$)",  # **100 followed by non-digit
        ]

        for pattern in bold_patterns:
            matches = re.findall(pattern, dollar_clean_response)
            if matches:
                # 마지막 굵은 글씨 숫자 사용
                predicted_number = matches[-1]
                extracted_patterns.append(f"Bold pattern: {predicted_number}")
                break

    # 3) 결론 문구 이후 숫자 찾기 (끝에서 시작까지 검색) 개선
    if not predicted_number:
        # 달러 기호 제거한 응답 사용
        clean_response = re.sub(r"\$\s*", "", response.lower())
        clean_response = re.sub(r",", "", clean_response)

        conclusion_patterns = [
            r"(?:final answer|final result|final total|final step|answer|result)(?:[:\s])+\s*([-+]?\d*\.?\d+)",
            r"(?:final|total|answer is|result is|sum is|equals)(?:\s*:)?\s*([-+]?\d*\.?\d+)",
            r"(?:answer|result|total)(?:\s*:)?\s*([-+]?\d*\.?\d+)",
            r"(?:is|=)\s*([-+]?\d*\.?\d+)(?:\s*classes|\s*dollars|\s*people|\s*students|\s*units|\s*kg|\s*miles|\s*meters)?",
            r"therefore[^.!?]*?(?:\s|is|:|=)\s*([-+]?\d*\.?\d+)",  # Therefore... 100
        ]

        # 모든 패턴에 대해 모든 매칭을 찾고 마지막 매칭을 선택
        all_matches = []

        # 원본 응답에서 찾기
        for pattern in conclusion_patterns:
            pattern_matches = list(re.finditer(pattern, response.lower()))
            if pattern_matches:
                all_matches.extend(pattern_matches)

        # 달러 기호 제거한 응답에서 찾기
        for pattern in conclusion_patterns:
            pattern_matches = list(re.finditer(pattern, clean_response))
            if pattern_matches:
                all_matches.extend(pattern_matches)

        # 매칭이 있으면 위치가 가장 뒤에 있는 매칭 선택
        if all_matches:
            # 매칭의 시작 위치를 기준으로 정렬
            all_matches.sort(key=lambda m: m.start())
            # 가장 마지막에 있는 매칭 선택
            last_match = all_matches[-1]
            predicted_number = last_match.group(1)
            extracted_patterns.append(f"Conclusion phrase: {predicted_number}")

    # 4) 마지막 3줄에서 숫자 찾기 개선
    if not predicted_number:
        last_lines = "\n".join(response.strip().split("\n")[-3:])
        # 달러 기호 제거 및 콤마 제거
        last_lines = re.sub(r"\$\s*", "", last_lines)
        last_lines = re.sub(r",", "", last_lines)
        number_matches = re.findall(r"([-+]?\d*\.?\d+)", last_lines)
        if number_matches:
            # Use the last number in the last 3 lines
            predicted_number = number_matches[-1]
            extracted_patterns.append(f"Last 3 lines: {predicted_number}")

    # 5) 마지막 수단: 응답에서 아무 숫자나 찾기
    if not predicted_number:
        # 달러 기호 제거 및 콤마 제거
        clean_response = re.sub(r"\$\s*", "", response)
        clean_response = re.sub(r",", "", clean_response)
        number_matches = re.findall(r"([-+]?\d*\.?\d+)", clean_response)
        if number_matches:
            # Use the last number
            predicted_number = number_matches[-1]
            extracted_patterns.append(f"Last resort: {predicted_number}")
        else:
            logger.warning(f"No number found in response for question: {question_id}")
            return False, None

    # 디버깅을 위한 로깅
    logger.debug(f"Extracted patterns for {question_id}: {extracted_patterns}")
    logger.debug(f"Expected: {expected_number}, Predicted: {predicted_number}")

    # Compare numbers
    try:
        expected_value = float(expected_number)
        predicted_value = float(predicted_number)
        is_correct = abs(expected_value - predicted_value) < 1e-6

        if not is_correct:
            logger.warning(
                f"Incorrect answer for {question_id}: expected {expected_number}, got {predicted_number}"
            )
        else:
            logger.debug(f"Correct answer for {question_id}: {predicted_number}")

        return is_correct, predicted_number
    except Exception as e:
        logger.warning(f"Error comparing numbers for {question_id}: {str(e)}")
        return False, predicted_number


def evaluate_commonsenseqa_response(test_item: Dict, response: str) -> tuple:
    """
    Evaluate CommonsenseQA response

    Args:
        test_item: Test item
        response: GPT response

    Returns:
        Tuple of (correctness (True/False), extracted_answer)
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
        return False, None

    # 정답 키를 대문자로 변환하고 필요 없는 문자 제거
    expected_key = expected_key.strip().upper()
    if expected_key not in "ABCDE":
        logger.warning(f"Invalid answer key: {expected_key}")
        return False, None

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
        return False, None

    # 디버깅 로깅
    logger.debug(f"Extracted patterns for {question_id}: {extracted_patterns}")
    logger.debug(f"Predicted key: {predicted_key}, Expected key: {expected_key}")

    # 답변 비교
    is_correct = predicted_key == expected_key
    if not is_correct:
        logger.warning(
            f"Incorrect answer for {question_id}: extracted {predicted_key} != expected {expected_key}"
        )

    return is_correct, predicted_key


def evaluate_arc_response(test_item: Dict, response: str) -> tuple:
    """
    Evaluate ARC multiple choice response

    Args:
        test_item: Test item
        response: GPT response

    Returns:
        Tuple of (correctness (True/False), extracted_answer)
    """
    # 질문 ID 로깅 (디버깅용)
    question_id = test_item.get("id", "unknown")
    logger.debug(f"Evaluating ARC response for question: {question_id}")
    logger.debug(f"Full response: {response}")

    # 응답에서 모든 answer/final answer 부분 추출 (디버깅용)
    debug_extract = []
    answer_debug_patterns = [
        r"(?:final answer|answer)[\s:]*(?:is)?[\s:#]*\s*([A-D1-4a-d])\b",
        r"(?:final answer|answer)(?:\s*:)?\s*([A-D1-4a-d])[^A-Za-z0-9]",
        r"(?:final answer|answer)(?:\s*:)?\s*([A-D1-4a-d])$",
    ]

    for pattern in answer_debug_patterns:
        matches = re.finditer(pattern, response.lower())
        for match in matches:
            debug_extract.append(
                f"Pattern '{pattern}' matched: '{match.group(0)}' -> '{match.group(1)}'"
            )

    # Get expected answer
    expected_answer = test_item.get("answer", "")
    if not expected_answer:
        logger.warning(f"No expected answer for question: {question_id}")
        return False, None

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
        return False, None

    # 질문에서 선택지 추출 (선택지 텍스트를 레이블과 매핑하기 위함)
    question_text = test_item.get("question", "")
    choice_text_map = {}

    # 선택지 추출 패턴 - A. text 또는 1. text 형식
    choices_pattern = r"(?:^|\n)([A-D]|[1-4])[.)\s]\s*([^\n]+)"
    choices_matches = re.finditer(choices_pattern, question_text)

    for match in choices_matches:
        label = match.group(1).upper()
        text = match.group(2).strip().lower()

        # 숫자 레이블을 알파벳으로 변환
        if label in ["1", "2", "3", "4"]:
            label = {"1": "A", "2": "B", "3": "C", "4": "D"}[label]

        choice_text_map[label] = text
        # 선택지 텍스트에서 앞뒤 공백과 마침표 제거 후 첫 단어만 저장
        first_word = text.split()[0].strip(".,:;") if text.split() else ""
        if first_word and len(first_word) > 2:  # 너무 짧은 단어는 무시
            choice_text_map[first_word] = label

    logger.debug(f"Extracted choice map: {choice_text_map}")
    logger.debug(
        f"Expected letter: {expected_letter}, Expected number: {expected_number}"
    )

    # 다양한 패턴으로 응답에서 답변 추출
    predicted_answer = None
    extracted_patterns = []  # 추출된 패턴들을 저장

    # 0) 소문자 응답도 인식하기 위해 정규화
    normalized_response = response.upper()

    # 1) 명확한 "Final answer: A" 패턴 우선 검사 (최우선)
    clear_answer_patterns = [
        r"FINAL\s*ANSWER\s*(?:IS)?[:=#\s]*\s*([A-D1-4])\b",  # FINAL ANSWER: A
        r"ANSWER\s*(?:IS)?[:=#\s]*\s*([A-D1-4])\b",  # ANSWER: A
        r"(?:^|\n)ANSWER\s*[:=#\s]*\s*([A-D1-4])\b",  # ANSWER: A (라인 시작)
        r"(?:^|\n)FINAL\s*ANSWER\s*[:=#\s]*\s*([A-D1-4])\b",  # FINAL ANSWER: A (라인 시작)
    ]

    for pattern in clear_answer_patterns:
        matches = re.search(pattern, normalized_response)
        if matches:
            predicted_answer = matches.group(1)
            extracted_patterns.append(f"Clear answer pattern: {predicted_answer}")
            break

    # 2) #### 뒤의 패턴 (높은 우선순위)
    if not predicted_answer and "####" in response:
        after_hash = response.split("####")[-1].strip().upper()  # 대문자로 정규화

        # #### 뒤에 단일 문자 A-D 또는 1-4 찾기
        hash_direct_pattern = r"([A-D1-4])\b"
        hash_matches = re.search(hash_direct_pattern, after_hash)
        if hash_matches:
            predicted_answer = hash_matches.group(1)
            extracted_patterns.append(f"Hash pattern: {predicted_answer}")
        # 선택지 텍스트 찾기 (예: "sunlight")
        elif choice_text_map:
            for text, label in choice_text_map.items():
                if text in after_hash.lower() and len(text) > 2:  # 짧은 단어는 무시
                    predicted_answer = label
                    extracted_patterns.append(f"Hash text match: {text} -> {label}")
                    break

    # 3) Final answer 패턴 (대소문자 모두 검색)
    if not predicted_answer:
        final_patterns = [
            r"(?:final answer|answer)[\s:]*(?:is)?[\s:#]*\s*([A-D1-4])\b",
            r"(?:final answer|answer)(?:\s*:)?\s*([A-D1-4])[^A-Za-z0-9]",
            r"(?:final answer|answer)(?:\s*:)?\s*([A-D1-4])$",
            r"(?:final answer|answer)(?:\s*:)?\s*option\s+([A-D1-4])",
            r"(?:final answer|answer)(?:\s*:)?\s*choice\s+([A-D1-4])",
        ]

        for pattern in final_patterns:
            matches = re.search(pattern, response.lower())
            if matches:
                predicted_answer = matches.group(1).upper()
                extracted_patterns.append(f"Final pattern: {predicted_answer}")
                break

        # 선택지 텍스트 포함된 패턴 찾기 (예: "Final answer: sunlight")
        if not predicted_answer and choice_text_map:
            final_text_pattern = (
                r"(?:final answer|answer)[\s:]*(?:is)?[\s:#]*\s*([a-zA-Z]+)"
            )
            text_matches = re.search(final_text_pattern, response.lower())
            if text_matches:
                answer_text = text_matches.group(1).strip().lower()
                # 선택지 텍스트와 일치하는지 확인
                for text, label in choice_text_map.items():
                    if text in answer_text or answer_text in text:
                        predicted_answer = label
                        extracted_patterns.append(
                            f"Final text match: {answer_text} -> {label}"
                        )
                        break

    # 4) 마지막 줄에서 정답 패턴 찾기
    if not predicted_answer:
        last_line = response.strip().split("\n")[-1].upper()  # 대문자로 정규화
        last_line_patterns = [
            r"^([A-D1-4])[^A-Za-z0-9]",  # 줄 시작 부분의 A-D, 1-4
            r"([A-D1-4])(?:\.|:|$|\s|,)",  # A-D, 1-4 다음에 구두점이나 공백
            r"(?:OPTION|CHOICE)\s*([A-D1-4])\b",  # "option A" 또는 "choice 1" 형식
            r"(?:ANSWER|SOLUTION)[\s:]*(?:IS)?[\s:]*([A-D1-4])\b",  # "answer: A" 형식
        ]

        for pattern in last_line_patterns:
            matches = re.search(pattern, last_line)
            if matches:
                predicted_answer = matches.group(1)
                extracted_patterns.append(f"Last line pattern: {predicted_answer}")
                break

        # 마지막 줄에서 선택지 텍스트 찾기
        if not predicted_answer and choice_text_map:
            for text, label in choice_text_map.items():
                if text in last_line.lower() and len(text) > 2:
                    predicted_answer = label
                    extracted_patterns.append(
                        f"Last line text match: {text} -> {label}"
                    )
                    break

    # 5) 체계적인 키워드 이후의 패턴 찾기
    if not predicted_answer:
        keyword_patterns = [
            r"(?:therefore|thus|so|hence)[^.!?]*(?:answer|choose|select|pick|option)[^.!?]*?([A-D1-4])\b",
            r"(?:answer|solution|option|choice)[^.!?]*(?:is|=)[^.!?]*?([A-D1-4])\b",
            r"(?:correct|right)[^.!?]*(?:answer|solution|option|choice)[^.!?]*?([A-D1-4])\b",
        ]

        for pattern in keyword_patterns:
            matches = re.search(pattern, response.lower())
            if matches:
                predicted_answer = matches.group(1).upper()
                extracted_patterns.append(f"Keyword pattern: {predicted_answer}")
                break

        # 키워드 후에 선택지 텍스트 찾기
        if not predicted_answer and choice_text_map:
            text_keyword_patterns = [
                r"(?:therefore|thus|so|hence)[^.!?]*(?:answer|choose|select|pick|option)[^.!?]*?([a-zA-Z]+)[\.!\?,]",
                r"(?:answer|solution|option|choice)[^.!?]*(?:is|=)[^.!?]*?([a-zA-Z]+)[\.!\?,]",
            ]

            for pattern in text_keyword_patterns:
                matches = re.search(pattern, response.lower())
                if matches:
                    answer_text = matches.group(1).strip().lower()
                    for text, label in choice_text_map.items():
                        if text in answer_text or answer_text in text:
                            predicted_answer = label
                            extracted_patterns.append(
                                f"Keyword text match: {answer_text} -> {label}"
                            )
                            break
                    if predicted_answer:
                        break

    # 6) 응답 전체에서 마지막 선택지 형식 찾기
    if not predicted_answer:
        options_pattern = r"\b([A-D])\s*(?:\)|\.)\s*([a-zA-Z].{5,})"
        all_options = list(re.finditer(options_pattern, response))
        if all_options:
            last_option_discussion = response.split(all_options[-1].group(0))[-1]
            option_conclusions = [
                r"(?:answer|pick|choose|select)[^.!?]*?([A-D1-4])\b",
                r"(?:option|choice)\s*([A-D1-4])\b",
                r"(?:answer|solution)\s*(?:is|=)\s*([A-D1-4])\b",
            ]

            for pattern in option_conclusions:
                matches = re.search(pattern, last_option_discussion.lower())
                if matches:
                    predicted_answer = matches.group(1).upper()
                    extracted_patterns.append(
                        f"Option discussion pattern: {predicted_answer}"
                    )
                    break

            # 텍스트 기반 결론 찾기
            if not predicted_answer and choice_text_map:
                for text, label in choice_text_map.items():
                    if text in last_option_discussion.lower() and len(text) > 2:
                        predicted_answer = label
                        extracted_patterns.append(
                            f"Option discussion text: {text} -> {label}"
                        )
                        break

    # 7) 응답 전체에서 A-D, 1-4 형태의 단독 패턴 찾기 (낮은 우선순위)
    if not predicted_answer:
        all_letters = re.findall(r"\b([A-D])\b", response.upper())
        if all_letters:
            predicted_answer = all_letters[-1]  # 마지막 문자 사용
            extracted_patterns.append(f"Last letter: {predicted_answer}")
        else:
            all_numbers = re.findall(r"\b([1-4])\b", response)
            if all_numbers:
                predicted_answer = all_numbers[-1]  # 마지막 숫자 사용
                extracted_patterns.append(f"Last number: {predicted_answer}")

        # 마지막 시도: 응답 전체에서 선택지 텍스트 찾기
        if not predicted_answer and choice_text_map:
            # 응답 전체에서 선택지 텍스트와 동일한 텍스트 찾기
            for text, label in choice_text_map.items():
                if text in response.lower() and len(text) > 2:
                    predicted_answer = label
                    extracted_patterns.append(f"Full text match: {text} -> {label}")
                    break

    # 결과 로깅
    if not predicted_answer:
        logger.warning(f"Could not extract answer from response for {question_id}")
        # 전체 내용 및 선택지 맵 출력
        logger.debug(f"Question options: {choice_text_map}")
        logger.debug(f"Full response: {response}")
        if debug_extract:
            logger.debug(f"Debug extracts: {debug_extract}")
        return False, None

    logger.debug(f"Extracted patterns: {extracted_patterns}")
    if debug_extract:
        logger.debug(f"Debug extracts: {debug_extract}")
    logger.debug(f"Predicted answer: {predicted_answer}")

    # 추출된 답변이 숫자면 해당하는 알파벳으로 변환
    if predicted_answer in ["1", "2", "3", "4"]:
        predicted_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}[predicted_answer]
        predicted_number = predicted_answer
    # 추출된 답변이 알파벳이면 해당하는 숫자로 변환
    elif predicted_answer in ["A", "B", "C", "D"]:
        predicted_letter = predicted_answer
        predicted_number = {"A": "1", "B": "2", "C": "3", "D": "4"}[predicted_answer]
    else:
        logger.warning(f"Invalid predicted answer format: {predicted_answer}")
        return False, None

    # 예상 답변과 실제 답변 비교 (숫자 또는 알파벳 형식 모두 고려)
    is_correct = (expected_letter and expected_letter == predicted_letter) or (
        expected_number and expected_number == predicted_number
    )

    logger.info(
        f"ARC evaluation: {question_id} | Expected: {expected_answer} | Predicted: {predicted_answer} | Correct: {is_correct}"
    )

    return is_correct, predicted_letter if expected_letter else predicted_number


def evaluate_response(test_item: Dict, response: str, dataset_name: str) -> tuple:
    """
    Evaluate GPT response

    Args:
        test_item: Test item
        response: GPT response
        dataset_name: Dataset name

    Returns:
        Tuple of (correctness (True/False), extracted_answer)
    """
    # For combined dataset, determine evaluation method based on the dataset field in the test item
    if dataset_name == "combined":
        item_dataset = test_item.get("dataset", "").lower()
        if item_dataset == "arc":
            # ARC multiple choice format evaluation
            return evaluate_arc_response(test_item, response)
        elif item_dataset == "commonsenseqa":
            return evaluate_commonsenseqa_response(test_item, response)
        elif item_dataset == "gsm8k":
            return evaluate_gsm8k_response(test_item, response)
        else:
            logger.warning(f"Unknown dataset in combined data: {item_dataset}")
            return False, None

    # Original dataset evaluation
    elif dataset_name == "gsm8k":
        return evaluate_gsm8k_response(test_item, response)
    elif dataset_name == "commonsenseqa":
        return evaluate_commonsenseqa_response(test_item, response)
    elif dataset_name == "arc":
        return evaluate_arc_response(test_item, response)
    else:
        logger.warning(f"No evaluation logic for dataset: {dataset_name}")
        return False, None


def run_experiment(args, repeat_index=0):
    """
    Run experiment

    Args:
        args: Experiment arguments
        repeat_index: Index of the current repeat (0-based)

    Returns:
        Tuple of (experiment_manager, results_summary)
    """
    # Create experiment name with timestamp and repeat index
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.dataset}_{args.method}_{timestamp}"

    if args.num_repeats > 1:
        experiment_name = f"{args.experiment_name}_repeat{repeat_index+1}"
    else:
        experiment_name = args.experiment_name

    # Initialize experiment manager
    experiment = ExperimentManager(args.results_dir, experiment_name)

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
    selector_kwargs = {
        "lambda_param": args.lambda_param,
        "n_clusters": args.n_clusters,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
    }

    selector = get_selector(
        args.method, example_pool, example_embeddings, **selector_kwargs
    )

    # 결과를 저장할 딕셔너리
    all_results = {}

    # 다양한 num_examples 값에 대해 실험 실행
    for num_ex in args.num_examples:
        logger.info(f"Running experiment with {num_ex} examples")

        # Create sub-experiment name
        sub_experiment_name = f"{args.experiment_name}_ex{num_ex}"
        sub_experiment = ExperimentManager(args.results_dir, sub_experiment_name)

        # Run experiment for each test item
        results = []
        total_latency = 0.0
        successful_tests = 0

        for i, test_item in enumerate(test_data):
            logger.info(
                f"Test ({i+1}/{len(test_data)}): {test_item['id']} with {num_ex} examples"
            )

            # Calculate test item embedding
            test_embedding = embedding_processor.model.encode(test_item["question"])

            # Select examples
            start_time = time.time()
            selected_examples = selector.select_examples(
                test_embedding, n_examples=num_ex
            )
            selection_time = time.time() - start_time

            # Extract selected example IDs
            selected_ids = [example["id"] for example in selected_examples]
            logger.info(f"Selected examples: {selected_ids}")

            # Create prompt
            prompt = gpt_client.create_cot_prompt(
                test_item["question"], selected_examples
            )

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
                is_correct, extracted_answer = evaluate_response(
                    test_item, response, args.dataset
                )

                # Log result
                result = {
                    "dataset": args.dataset,
                    "method": args.method,
                    "num_examples": num_ex,
                    "question_id": test_item["id"],
                    "question": test_item["question"],
                    "expected_answer": test_item.get("answer", ""),
                    "response": response,
                    "selected_examples": selected_ids,
                    "correct": is_correct,
                    "extracted_answer": extracted_answer,
                    "tokens": gpt_result["tokens"],
                    "response_time": response_time,
                    "selection_time": selection_time,
                    "total_latency": total_test_latency,
                    "timestamp": datetime.now().isoformat(),
                }

                results.append(result)
                sub_experiment.log_result(result)

            except Exception as e:
                logger.error(f"GPT call failed: {str(e)}")

                # Log error result
                result = {
                    "dataset": args.dataset,
                    "method": args.method,
                    "num_examples": num_ex,
                    "question_id": test_item["id"],
                    "question": test_item["question"],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

                sub_experiment.log_result(result)

            # Rate limiting
            time.sleep(0.5)

        # Save results for this num_examples
        sub_experiment.save_results()

        # Output summary
        summary = sub_experiment.summarize_results()
        sub_experiment.print_summary(summary)

        # Store results for this num_examples configuration
        all_results[num_ex] = {"results": results, "summary": summary}

        # Print latency metrics
        if successful_tests > 0:
            avg_latency = total_latency / successful_tests
            logger.info(
                f"Average total latency per test sample: {avg_latency:.2f} seconds"
            )

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
                    logger.info(
                        f"Average LLM response time: {avg_response:.2f} seconds"
                    )
        else:
            logger.warning("No successful tests to calculate average latency")

    # 모든 결과를 종합하여 저장
    experiment.save_meta(
        {
            "all_results_summary": {
                num_ex: all_results[num_ex]["summary"] for num_ex in args.num_examples
            },
            "args": vars(args),
            "repeat_index": repeat_index,
        }
    )

    logger.info(f"Experiment completed for repeat {repeat_index+1}/{args.num_repeats}")

    return experiment, all_results


def main():
    """Main function"""
    args = parse_args()

    logger.info(
        f"Starting experiments: dataset={args.dataset}, method={args.method}, repeats={args.num_repeats}"
    )

    # 반복 실험 결과를 저장할 변수
    all_repeat_results = {}

    # 반복 실험 실행
    for repeat_idx in range(args.num_repeats):
        logger.info(f"Starting experiment repeat {repeat_idx+1}/{args.num_repeats}")
        experiment, results = run_experiment(args, repeat_idx)

        # 결과 저장
        all_repeat_results[repeat_idx] = results

    # 반복 실험 결과 통계 계산
    if args.num_repeats > 1:
        logger.info("Computing statistics across repeated experiments...")

        # num_examples 별로 통계 계산
        for num_ex in args.num_examples:
            accuracies = []
            selection_times = []

            # 각 반복 실험의 결과 수집
            for repeat_idx in range(args.num_repeats):
                if num_ex in all_repeat_results[repeat_idx]:
                    summary = all_repeat_results[repeat_idx][num_ex]["summary"]
                    accuracies.append(summary.get("accuracy", 0) * 100)  # 퍼센트로 변환

                    # 선택 시간 통계 계산
                    results = all_repeat_results[repeat_idx][num_ex]["results"]
                    repeat_selection_times = [
                        r.get("selection_time", 0)
                        for r in results
                        if "selection_time" in r
                    ]
                    if repeat_selection_times:
                        avg_selection_time = sum(repeat_selection_times) / len(
                            repeat_selection_times
                        )
                        selection_times.append(avg_selection_time)

            # 통계 계산 및 출력
            if accuracies:
                min_accuracy = min(accuracies)
                max_accuracy = max(accuracies)
                avg_accuracy = sum(accuracies) / len(accuracies)

                logger.info(
                    f"\nStatistics for {num_ex} examples across {args.num_repeats} repeats:"
                )
                logger.info(f"  Min accuracy: {min_accuracy:.2f}%")
                logger.info(f"  Max accuracy: {max_accuracy:.2f}%")
                logger.info(f"  Avg accuracy: {avg_accuracy:.2f}%")

                if selection_times:
                    avg_selection_latency = sum(selection_times) / len(selection_times)
                    logger.info(
                        f"  Avg selection latency: {avg_selection_latency:.6f} seconds"
                    )

        # 전체 통계를 파일로 저장
        stats_file = os.path.join(
            args.results_dir, f"{args.dataset}_{args.method}_repeat_stats.json"
        )
        stats_data = {
            "dataset": args.dataset,
            "method": args.method,
            "num_repeats": args.num_repeats,
            "timestamp": datetime.now().isoformat(),
            "statistics": {},
        }

        for num_ex in args.num_examples:
            accuracies = []
            selection_times = []

            for repeat_idx in range(args.num_repeats):
                if num_ex in all_repeat_results[repeat_idx]:
                    summary = all_repeat_results[repeat_idx][num_ex]["summary"]
                    accuracies.append(summary.get("accuracy", 0) * 100)

                    results = all_repeat_results[repeat_idx][num_ex]["results"]
                    repeat_selection_times = [
                        r.get("selection_time", 0)
                        for r in results
                        if "selection_time" in r
                    ]
                    if repeat_selection_times:
                        avg_selection_time = sum(repeat_selection_times) / len(
                            repeat_selection_times
                        )
                        selection_times.append(avg_selection_time)

            if accuracies:
                stats_data["statistics"][str(num_ex)] = {
                    "min_accuracy": min(accuracies),
                    "max_accuracy": max(accuracies),
                    "avg_accuracy": sum(accuracies) / len(accuracies),
                }

                if selection_times:
                    stats_data["statistics"][str(num_ex)]["avg_selection_latency"] = (
                        sum(selection_times) / len(selection_times)
                    )

        # 통계 파일 저장
        with open(stats_file, "w") as f:
            json.dump(stats_data, f, indent=2)

        logger.info(f"Repeat experiment statistics saved to {stats_file}")

    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()
