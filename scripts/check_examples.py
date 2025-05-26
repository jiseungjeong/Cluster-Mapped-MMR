#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

# 데이터 파일 경로
total_file = "data/examples/total_examples.json"

# 병합된 데이터 로드
with open(total_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 각 데이터셋별 항목 수 확인
cqa_data = [item for item in data if item["dataset"] == "commonsenseqa"]
gsm_data = [item for item in data if item["dataset"] == "gsm8k"]

print(f"총 예제 수: {len(data)}")
print(f"CommonsenseQA 예제 수: {len(cqa_data)}")
print(f"GSM8K 예제 수: {len(gsm_data)}")

# 각 데이터셋의 첫 번째 항목 확인
if cqa_data:
    print("\nCommonsenseQA 첫 번째 예제:")
    print(f"ID: {cqa_data[0]['id']}")
    print(f"질문: {cqa_data[0]['question'][:50]}...")
    print(f"답변: {cqa_data[0]['answer']}")

if gsm_data:
    print("\nGSM8K 첫 번째 예제:")
    print(f"ID: {gsm_data[0]['id']}")
    print(f"질문: {gsm_data[0]['question'][:50]}...")
    print(f"답변: {gsm_data[0]['answer']}")
