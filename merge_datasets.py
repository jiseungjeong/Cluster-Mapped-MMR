#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

# 데이터 파일 경로
cqa_file = "data/processed/commonsenseqa_test_enhanced.json"
gsm_file = "data/processed/gsm8k_test.json"
output_file = "data/processed/total_test.json"

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

# 두 데이터셋 합치기
merged_data = cqa_processed + gsm_processed

# 병합된 데이터 저장
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(
    f"병합 완료: CommonsenseQA {len(cqa_processed)}개, GSM8K {len(gsm_processed)}개 = 총 {len(merged_data)}개 문제"
)
