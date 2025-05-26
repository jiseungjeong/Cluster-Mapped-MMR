import json
import os

# 입력 및 출력 파일 경로
input_file = "data/examples/total_examples_with_arc.json"
output_file = "data/examples/questions_only.json"


# JSON 파일 로드
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# question 필드만 추출
def extract_questions(data):
    questions_only = []
    for item in data:
        questions_only.append(
            {
                "id": item["id"],
                "question": item["question"],
                "dataset": item.get("dataset", ""),  # dataset 필드가 있으면 유지
            }
        )
    return questions_only


def main():
    # 데이터 로드
    print(f"데이터 로드 중: {input_file}")
    data = load_json(input_file)

    # question 필드만 추출
    print("question 필드 추출 중...")
    questions_only = extract_questions(data)

    # 결과 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(questions_only, f, ensure_ascii=False, indent=2)

    print(f"추출 완료: {len(questions_only)}개 항목이 {output_file}에 저장되었습니다.")


if __name__ == "__main__":
    main()
