#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import random
from pathlib import Path
from tqdm import tqdm

# 데이터 경로 설정
ARC_CHALLENGE_TRAIN = (
    "data/raw/arc/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl"
)
ARC_CHALLENGE_DEV = (
    "data/raw/arc/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl"
)
ARC_CHALLENGE_TEST = (
    "data/raw/arc/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl"
)
ARC_EASY_TRAIN = "data/raw/arc/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Train.jsonl"
ARC_EASY_DEV = "data/raw/arc/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Dev.jsonl"
ARC_EASY_TEST = "data/raw/arc/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl"

OUTPUT_EXAMPLES = "data/examples/arc_examples.json"
OUTPUT_TEST = "data/processed/arc_test.json"
OUTPUT_TOTAL_EXAMPLES = "data/examples/total_examples_with_arc.json"
OUTPUT_TOTAL_TEST = "data/processed/total_test_with_arc.json"


def load_jsonl(file_path):
    """JSONL 파일을 로드하는 함수"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_question(item):
    """ARC 데이터의 질문 형식 변환"""
    choices_text = []
    for choice in item["question"]["choices"]:
        choices_text.append(f"{choice['label']}. {choice['text']}")

    return {
        "id": item["id"],
        "question": item["question"]["stem"] + "\n" + "\n".join(choices_text),
        "reasoning": f"This question requires analyzing the given scientific scenario. After careful consideration of the options, the answer is {item['answerKey']}.",
        "answer": f"{item['answerKey']}",
        "dataset": "arc",
    }


def main():
    # 디렉토리 생성
    Path(os.path.dirname(OUTPUT_EXAMPLES)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(OUTPUT_TEST)).mkdir(parents=True, exist_ok=True)

    print("ARC 데이터 로드 중...")
    # 모든 데이터셋 로드
    arc_challenge_train = load_jsonl(ARC_CHALLENGE_TRAIN)
    arc_challenge_dev = load_jsonl(ARC_CHALLENGE_DEV)
    arc_challenge_test = load_jsonl(ARC_CHALLENGE_TEST)
    arc_easy_train = load_jsonl(ARC_EASY_TRAIN)
    arc_easy_dev = load_jsonl(ARC_EASY_DEV)
    arc_easy_test = load_jsonl(ARC_EASY_TEST)

    # 학습 데이터와 개발 데이터를 합쳐서 examples 데이터셋 생성
    train_examples = (
        arc_challenge_train + arc_challenge_dev + arc_easy_train + arc_easy_dev
    )
    # 테스트 데이터 생성
    test_examples = arc_challenge_test + arc_easy_test

    print(
        f"Examples 데이터셋: {len(train_examples)} 항목, 테스트 데이터셋: {len(test_examples)} 항목"
    )

    # 무작위로 섞기
    random.seed(42)
    random.shuffle(train_examples)
    random.shuffle(test_examples)

    # 500개만 선택
    train_examples = train_examples[:500]
    # 100개만 선택
    test_examples = test_examples[:100]

    # 데이터 형식 변환
    print("데이터 형식 변환 중...")
    formatted_train = [format_question(item) for item in tqdm(train_examples)]
    formatted_test = [format_question(item) for item in tqdm(test_examples)]

    # 개별 ARC 데이터셋 저장
    print(f"ARC 데이터셋을 {OUTPUT_EXAMPLES}와 {OUTPUT_TEST}에 저장 중...")
    with open(OUTPUT_EXAMPLES, "w", encoding="utf-8") as f:
        json.dump(formatted_train, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_TEST, "w", encoding="utf-8") as f:
        json.dump(formatted_test, f, indent=2, ensure_ascii=False)

    # 기존 total_examples.json과 병합
    print("기존 데이터셋과 병합 중...")
    with open("data/examples/total_examples.json", "r", encoding="utf-8") as f:
        total_examples = json.load(f)

    with open("data/processed/total_test.json", "r", encoding="utf-8") as f:
        total_test = json.load(f)

    # 병합된 데이터셋 저장
    merged_examples = total_examples + formatted_train
    merged_test = total_test + formatted_test

    with open(OUTPUT_TOTAL_EXAMPLES, "w", encoding="utf-8") as f:
        json.dump(merged_examples, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_TOTAL_TEST, "w", encoding="utf-8") as f:
        json.dump(merged_test, f, indent=2, ensure_ascii=False)

    # 요약 정보 출력
    print("\n데이터 처리 완료:")
    print(f"ARC Examples: {len(formatted_train)} 항목")
    print(f"ARC Test: {len(formatted_test)} 항목")
    print(f"Total Examples with ARC: {len(merged_examples)} 항목")
    print(f"Total Test with ARC: {len(merged_test)} 항목")

    # 데이터셋 유형별 개수
    examples_by_type = {}
    test_by_type = {}

    for item in merged_examples:
        dataset = item["dataset"]
        examples_by_type[dataset] = examples_by_type.get(dataset, 0) + 1

    for item in merged_test:
        dataset = item["dataset"]
        test_by_type[dataset] = test_by_type.get(dataset, 0) + 1

    print("\nExamples 데이터셋 구성:")
    for dataset, count in examples_by_type.items():
        print(f"  - {dataset}: {count} 항목")

    print("\nTest 데이터셋 구성:")
    for dataset, count in test_by_type.items():
        print(f"  - {dataset}: {count} 항목")


if __name__ == "__main__":
    main()
