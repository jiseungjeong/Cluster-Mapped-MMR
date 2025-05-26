#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_gsm8k_response(test_item, response):
    """GSM8K 응답 평가 함수 (단순화 버전)"""
    expected_answer = test_item.get("answer", "")
    if not expected_answer:
        return False, None

    # 정답에서 숫자만 추출
    expected_number = re.search(r"[-+]?\d*\.?\d+", expected_answer)
    if expected_number:
        expected_number = expected_number.group()
    else:
        return False, None

    # 예측된 답변 추출 로직 (단순화)
    predicted_number = None
    extraction_method = None

    # 1) #### 태그 이후의 텍스트
    if "####" in response:
        after_hash = response.split("####")[-1].strip()
        after_hash = after_hash.replace("$", "").strip()
        hash_direct_pattern = r"([-+]?\d*\.?\d+)"
        hash_direct_matches = re.findall(hash_direct_pattern, after_hash)
        if hash_direct_matches:
            predicted_number = hash_direct_matches[0]
            extraction_method = "#### 태그 이후"

    # 2) "Final answer: X" 패턴
    if not predicted_number:
        final_pattern = (
            r"(?:final answer|final result|answer)(?:[:\s])+\s*([-+]?\d*\.?\d+)"
        )
        final_matches = re.search(final_pattern, response.lower())
        if final_matches:
            predicted_number = final_matches.group(1)
            extraction_method = "Final answer 패턴"

    # 3) 굵은 글씨 패턴
    if not predicted_number:
        bold_pattern = r"\*\*([-+]?\d*\.?\d+)\*\*"
        bold_matches = re.search(bold_pattern, response)
        if bold_matches:
            predicted_number = bold_matches.group(1)
            extraction_method = "굵은 글씨 패턴"

    # 4) 달러 기호 제거 후 숫자 찾기
    if not predicted_number:
        clean_response = re.sub(r"\$\s*", "", response)
        number_matches = re.findall(r"([-+]?\d*\.?\d+)", clean_response)
        if number_matches:
            predicted_number = number_matches[-1]
            extraction_method = "달러 기호 제거 후 마지막 숫자"

    # 답변 비교
    if predicted_number:
        try:
            expected_value = float(expected_number)
            predicted_value = float(predicted_number)
            is_correct = abs(expected_value - predicted_value) < 1e-6
            return is_correct, (predicted_number, extraction_method)
        except:
            return False, (predicted_number, extraction_method)
    else:
        return False, (None, "추출 실패")


def test_gsm8k_extraction():
    """GSM8K 정답 추출 로직 테스트"""
    # 테스트 항목 정의
    test_item = {"id": "gsm8k_test_521", "answer": "6"}

    # 원본 응답 (실제 GPT 응답)
    original_response = """David and Dasha went to the carnival, where there are 9 rides.  Each ride costs 2 ride tickets at $2 per ticket.  You can also buy a ride bracelet for $30 which gives you 9 rides.  If David buys a ride bracelet and Dasha buys tickets, and they ride all 9 rides, how much money does David save?",6,"Let's analyze the problem step-by-step:

1. **Cost of the ride bracelet:** $30, which gives 9 rides.
2. **Cost of Dasha's tickets:** Each ride costs 2 tickets at $2 per ticket, so:
   - Cost per ride: 2 tickets * $2 = $4
   - For 9 rides: 9 * $4 = $36

3. **Total cost for David and Dasha:**
   - David: $30 (bracelet)
   - Dasha: $36 (tickets)
   - Total: $30 + $36 = $66

4. **If both bought tickets individually:**
   - Each would pay $4 per ride, so for 9 rides:
   - 2 people * $36 = $72

5. **Savings:**
   - Total if both paid individually: $72
   - Total with David's bracelet and Dasha's tickets: $66
   - Savings: $72 - $66 = $6

**Final answer:** David saves **$6** by buying the bracelet.

#### ####"""

    # 정답 추출 실행
    is_correct, (extracted_answer, method) = evaluate_gsm8k_response(
        test_item, original_response
    )

    # 결과 출력
    print("\n=== GSM8K 정답 추출 테스트 ===")
    print(f"예상 정답: {test_item['answer']}")
    print(f"추출한 정답: {extracted_answer} (방법: {method})")
    print(f"정답 여부: {is_correct}")

    # 다른 형식의 응답들도 테스트
    test_variations = [
        {
            "name": "#### 태그만 있는 경우",
            "response": "Let's solve this step by step... \n\n#### 39",
        },
        {
            "name": "Final answer 형식",
            "response": "Let's solve this...\n\nFinal answer: 39",
        },
        {"name": "굵은 글씨 형식", "response": "The remaining problems are **39**."},
        {
            "name": "마지막 3줄에 숫자만 있는 경우",
            "response": "Let's calculate...\n\nThe answer is:\n39",
        },
        {
            "name": "달러 기호가 있는 경우",
            "response": "After calculation, we get $39 problems remaining.",
        },
    ]

    print("\n=== 추가 테스트 케이스 ===")
    for i, test_case in enumerate(test_variations, 1):
        is_correct, (extracted, method) = evaluate_gsm8k_response(
            test_item, test_case["response"]
        )
        print(f"\n테스트 {i}: {test_case['name']}")
        print(f"응답: {test_case['response']}")
        print(f"추출한 정답: {extracted} (방법: {method})")
        print(f"정답 여부: {is_correct}")


if __name__ == "__main__":
    test_gsm8k_extraction()
