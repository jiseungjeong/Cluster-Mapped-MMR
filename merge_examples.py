#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

# 데이터 파일 경로
cqa_file = "data/examples/commonsenseqa_examples_enhanced.json"
gsm_file = "data/examples/gsm8k_examples.json"
output_file = "data/examples/total_examples.json"

# CommonsenseQA 데이터 로드
with open(cqa_file, "r", encoding="utf-8") as f:
    cqa_data = json.load(f)

# GSM8K 데이터 로드
with open(gsm_file, "r", encoding="utf-8") as f:
    gsm_data = json.load(f)

# 각 데이터셋 항목에 dataset 필드 추가하고 필요한 필드만 선택
cqa_processed = []
for item in cqa_data:
    cqa_processed.append(
        {
            "id": item["id"],
            "question": item["question"],
            "reasoning": item["reasoning"],
            "answer": item["answer"],
            "dataset": "commonsenseqa",
        }
    )

gsm_processed = []
for item in gsm_data:
    gsm_processed.append(
        {
            "id": item["id"],
            "question": item["question"],
            "reasoning": item["reasoning"],
            "answer": item["answer"],
            "dataset": "gsm8k",
        }
    )

# 두 데이터셋 병합
merged_data = cqa_processed + gsm_processed

# 결과 저장
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=2, ensure_ascii=False)

print(f"데이터셋이 성공적으로 병합되었습니다.")
print(f"CommonsenseQA 예제: {len(cqa_processed)}개")
print(f"GSM8K 예제: {len(gsm_processed)}개")
print(f"총 예제: {len(merged_data)}개")
print(f"결과 파일: {output_file}")
