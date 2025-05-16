#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import openai

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="LLM을 사용하여 데이터셋의 reasoning 향상"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="입력 JSON 파일 경로 (CommonsenseQA 또는 GSM8K 데이터셋)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="출력 JSON 파일 경로 (기본값: input_file에 '_enhanced' 추가)",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["commonsenseqa", "gsm8k", "arc"],
        required=True,
        help="데이터셋 타입 (commonsenseqa, gsm8k, 또는 arc)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-nano",
        help="사용할 OpenAI 모델 (기본값: gpt-4o)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="한 번에 처리할 항목 수 (기본값: 10)",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="처리 시작 인덱스 (기본값: 0)",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="처리 종료 인덱스 (기본값: None, 전체 처리)",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["ko", "en"],
        default="en",
        help="reasoning 언어 (ko: 한국어, en: 영어) (기본값: en)",
    )
    return parser.parse_args()


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """JSON 파일에서 데이터 로드"""
    logger.info(f"데이터 로드 중: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data(data: List[Dict[str, Any]], file_path: str):
    """데이터를 JSON 파일로 저장"""
    logger.info(f"데이터 저장 중: {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_commonsenseqa_reasoning(
    question: str, options: List[str], answer: str, model: str, language: str
) -> str:
    """
    CommonsenseQA에 대한 향상된 reasoning 생성

    Args:
        question: 질문 텍스트
        options: 선택지 리스트
        answer: 정답 (예: "A. option")
        model: 사용할 OpenAI 모델
        language: 출력 언어 (ko/en)

    Returns:
        생성된 reasoning 텍스트
    """
    # 프롬프트 작성
    if language == "ko":
        prompt = f"""당신은 상식 질문에 답하는 전문가입니다. 
다음 객관식 질문에 대해 논리적인 추론 과정을 단계별로 작성해주세요.
답변은 이미 주어져 있으니, 어떻게 그 답에 도달했는지 설명해주세요.

질문: {question}
선택지:
{chr(10).join(options)}

정답: {answer}

추론 과정을 3-5문장으로 작성해주세요. 다음 형식으로 답변하세요:
"이 질문은 [주제]에 관한 것입니다. [첫 번째 사고 과정]. [두 번째 사고 과정]. [세 번째 사고 과정]. 따라서 정답은 [정답 내용]입니다."

추론 과정만 답변해주세요. 다른 설명이나 부가 정보는 포함하지 마세요.
"""
    else:  # 영어
        prompt = f"""You are an expert at answering common sense questions.
Please provide a step-by-step logical reasoning process for the following multiple-choice question.
The answer is already given, so please explain how one would arrive at that answer.

Question: {question}
Options:
{chr(10).join(options)}

Answer: {answer}

Please provide your reasoning in 3-5 sentences in the following format:
"This question is about [topic]. [First thought process]. [Second thought process]. [Third thought process]. Therefore, the answer is [answer content]."

Only provide the reasoning process. Do not include any other explanations or additional information.
"""

    # API 호출 및 결과 반환
    return call_openai_api(prompt, model)


def generate_gsm8k_reasoning(
    question: str, answer: str, model: str, language: str
) -> str:
    """
    GSM8K 수학 문제에 대한 향상된 reasoning 생성

    Args:
        question: 문제 텍스트
        answer: 정답
        model: 사용할 OpenAI 모델
        language: 출력 언어 (ko/en)

    Returns:
        생성된 reasoning 텍스트
    """
    # 프롬프트 작성
    if language == "ko":
        prompt = f"""당신은 수학 문제를 푸는 전문가입니다.
다음 수학 문제에 대한 단계별 풀이 과정을 논리적으로 자세히 설명해주세요.
답은 이미 알려져 있으니, 그 답에 어떻게 도달했는지 설명해주세요.

문제: {question}
정답: {answer}

풀이 과정을 단계별로 명확하게 작성해주세요. 각 단계에서 사용한 수식과 계산을 포함해야 합니다.
최종 답변에는 수치적인 답만 포함해야 합니다.

풀이 과정만 답변해주세요. 다른 설명이나 부가 정보는 포함하지 마세요.
"""
    else:  # 영어
        prompt = f"""You are an expert at solving math problems.
Please provide a detailed step-by-step solution for the following math problem.
The answer is already given, so please explain how to arrive at that answer.

Problem: {question}
Answer: {answer}

Please provide your solution with clear steps, showing all calculations and formulas used.
The final answer should only include the numerical answer.

Only provide the solution process. Do not include any other explanations or additional information.
"""

    # API 호출 및 결과 반환
    return call_openai_api(prompt, model)


def generate_arc_reasoning(
    question: str, options: List[str], answer: str, model: str, language: str
) -> str:
    """
    ARC Challenge 과학 문제에 대한 향상된 reasoning 생성

    Args:
        question: 질문 텍스트
        options: 선택지 리스트
        answer: 정답 (예: "A")
        model: 사용할 OpenAI 모델
        language: 출력 언어 (ko/en)

    Returns:
        생성된 reasoning 텍스트
    """
    # 프롬프트 작성
    if language == "ko":
        prompt = f"""당신은 과학 문제를 푸는 전문가입니다. 
다음 객관식 과학 문제에 대해 단계별 과학적 추론 과정을 작성해주세요.
답변은 이미 주어져 있으니, 과학적 원리를 사용하여 어떻게 그 답에 도달했는지 설명해주세요.

질문: {question}
선택지:
{chr(10).join(options)}

정답: {answer}

추론 과정을 3-5문장으로 작성해주세요. 다음 형식으로 답변하세요:
"이 질문은 [과학 개념]에 관한 것입니다. [첫 번째 과학적 원리/사실]. [두 번째 사고 과정]. [세 번째 사고 과정]. 따라서 정답은 {answer}입니다."

추론 과정만 답변해주세요. 다른 설명이나 부가 정보는 포함하지 마세요.
"""
    else:  # 영어
        prompt = f"""You are an expert at solving scientific questions.
Please provide a step-by-step scientific reasoning process for the following multiple-choice science question.
The answer is already given, so please explain how one would arrive at that answer using scientific principles.

Question: {question}
Options:
{chr(10).join(options)}

Answer: {answer}

Please provide your reasoning in 3-5 sentences in the following format:
"This question is about [scientific concept]. [First scientific principle/fact]. [Second thought process]. [Third thought process]. Therefore, the answer is {answer}."

Only provide the reasoning process. Do not include any other explanations or additional information.
"""

    # API 호출 및 결과 반환
    return call_openai_api(prompt, model)


def call_openai_api(prompt: str, model: str) -> str:
    """
    OpenAI API 호출 및 결과 반환

    Args:
        prompt: 프롬프트 텍스트
        model: 사용할 OpenAI 모델

    Returns:
        생성된 텍스트
    """
    # API 호출
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )

            # 결과 반환
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"API 호출 실패 (시도 {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"{retry_delay}초 후 재시도...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 지수 백오프
            else:
                logger.error(f"최대 재시도 횟수 초과: {str(e)}")
                return "API 호출 오류로 reasoning을 생성할 수 없습니다."


def process_commonsenseqa_item(
    item: Dict[str, Any], model: str, language: str
) -> Dict[str, Any]:
    """
    CommonsenseQA 항목 처리

    Args:
        item: 처리할 항목
        model: 사용할 OpenAI 모델
        language: 출력 언어 (ko/en)

    Returns:
        업데이트된 항목
    """
    # 선택지 추출
    question_text = item["question"]

    # 선택지 추출 (A, B, C, D, E 형식)
    options = []
    if "\n" in question_text:
        # 질문과 선택지가 줄바꿈으로 구분되어 있는 경우
        parts = question_text.split("\n")
        question_only = parts[0]
        options = parts[1:]
    else:
        # 선택지가 포함되어 있지 않은 경우 (드문 경우)
        question_only = question_text
        logger.warning(f"선택지를 찾을 수 없습니다: {item['id']}")

    # 향상된 reasoning 생성
    answer = item["answer"]
    enhanced_reasoning = generate_commonsenseqa_reasoning(
        question_only, options, answer, model, language
    )

    # 기존 데이터 복사 및 reasoning 업데이트
    item["reasoning"] = enhanced_reasoning

    return item


def process_gsm8k_item(
    item: Dict[str, Any], model: str, language: str
) -> Dict[str, Any]:
    """
    GSM8K 항목 처리

    Args:
        item: 처리할 항목
        model: 사용할 OpenAI 모델
        language: 출력 언어 (ko/en)

    Returns:
        업데이트된 항목
    """
    # 향상된 reasoning 생성
    question = item["question"]
    answer = item.get("answer", "")

    enhanced_reasoning = generate_gsm8k_reasoning(question, answer, model, language)

    # 기존 데이터 복사 및 reasoning 업데이트
    item["reasoning"] = enhanced_reasoning

    return item


def process_arc_item(item: Dict[str, Any], model: str, language: str) -> Dict[str, Any]:
    """
    ARC Challenge 항목 처리

    Args:
        item: 처리할 항목
        model: 사용할 OpenAI 모델
        language: 출력 언어 (ko/en)

    Returns:
        업데이트된 항목
    """
    # 질문 텍스트 추출
    question_text = item["question"]

    # 선택지 추출 (A, B, C, D 또는 1, 2, 3, 4 형식)
    options = []
    if "\n" in question_text:
        # 질문과 선택지가 줄바꿈으로 구분되어 있는 경우
        parts = question_text.split("\n")
        question_only = parts[0]
        options = parts[1:]
    else:
        # 선택지가 포함되어 있지 않은 경우 (드문 경우)
        question_only = question_text
        logger.warning(f"선택지를 찾을 수 없습니다: {item['id']}")

    # 향상된 reasoning 생성
    answer = item["answer"]
    enhanced_reasoning = generate_arc_reasoning(
        question_only, options, answer, model, language
    )

    # 기존 데이터 복사 및 reasoning 업데이트
    item["reasoning"] = enhanced_reasoning

    return item


def process_batch(
    data: List[Dict[str, Any]],
    start_idx: int,
    end_idx: int,
    dataset_type: str,
    model: str,
    language: str,
) -> List[Dict[str, Any]]:
    """
    데이터 배치 처리

    Args:
        data: 처리할 데이터
        start_idx: 시작 인덱스
        end_idx: 종료 인덱스
        dataset_type: 데이터셋 타입 (commonsenseqa, gsm8k 또는 arc)
        model: 사용할 OpenAI 모델
        language: 출력 언어 (ko/en)

    Returns:
        업데이트된 데이터
    """
    batch_data = data[start_idx:end_idx]
    total = len(batch_data)

    for i, item in enumerate(batch_data):
        logger.info(
            f"처리 중: {i+1}/{total} (전체 인덱스: {start_idx+i+1}/{len(data)})"
        )

        # 데이터셋 타입에 따른 처리
        if dataset_type == "commonsenseqa":
            process_commonsenseqa_item(item, model, language)
        elif dataset_type == "gsm8k":
            process_gsm8k_item(item, model, language)
        elif dataset_type == "arc":
            process_arc_item(item, model, language)

        # 처리 간격 (API 속도 제한 방지)
        time.sleep(1)

    # 업데이트된 전체 데이터 반환
    return data


def main():
    """메인 함수"""
    args = parse_args()

    # 출력 파일 이름 설정
    if args.output_file is None:
        input_path = Path(args.input_file)
        output_dir = input_path.parent
        output_name = f"{input_path.stem}_enhanced{input_path.suffix}"
        args.output_file = str(output_dir / output_name)

    # 데이터 로드
    data = load_data(args.input_file)
    logger.info(f"총 {len(data)}개 항목 로드됨")

    # 처리 범위 설정
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(data)
    end_idx = min(end_idx, len(data))

    logger.info(f"인덱스 범위 {start_idx}에서 {end_idx-1}까지 처리")

    # 배치 처리
    current_idx = start_idx
    while current_idx < end_idx:
        batch_end = min(current_idx + args.batch_size, end_idx)
        logger.info(f"배치 처리: {current_idx}에서 {batch_end-1}까지")

        # 배치 처리
        data = process_batch(
            data, current_idx, batch_end, args.dataset_type, args.model, args.language
        )

        # 중간 결과 저장 (안전을 위해)
        save_data(data, args.output_file)
        logger.info(f"중간 결과 저장됨: {args.output_file}")

        # 다음 배치
        current_idx = batch_end

    # 최종 결과 저장
    save_data(data, args.output_file)
    logger.info(f"처리 완료! 결과 저장됨: {args.output_file}")


if __name__ == "__main__":
    main()
