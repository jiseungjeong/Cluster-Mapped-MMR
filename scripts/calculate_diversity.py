import json
import os
import numpy as np
import pickle
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


# 데이터셋에서 예시 ID 로드
def load_example_ids(file_path):
    with open(file_path, "r") as f:
        examples = json.load(f)
    return [ex["id"] for ex in examples]


# 임베딩 로드
def load_embeddings(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


# 주어진 예시들 간의 다양성 점수 계산
def calculate_diversity_score(example_indices, embeddings):
    if len(example_indices) < 2:
        return None

    # 임베딩 추출
    example_embeddings = embeddings[example_indices]

    # 모든 쌍의 코사인 유사도 계산
    similarities = []
    for i in range(len(example_embeddings)):
        for j in range(i + 1, len(example_embeddings)):
            similarity = cosine_similarity(
                [example_embeddings[i]], [example_embeddings[j]]
            )[0][0]
            similarities.append(similarity)

    # 평균 유사도 계산
    avg_similarity = np.mean(similarities)

    # 다양성 점수 = 1 - 평균 유사도
    diversity_score = 1 - avg_similarity

    return diversity_score


# 결과 파일 로드
def load_results(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# 예시 ID를 인덱스로 변환
def map_example_ids_to_indices(example_ids, all_example_ids):
    indices = []
    for ex_id in example_ids:
        try:
            index = all_example_ids.index(ex_id)
            indices.append(index)
        except ValueError:
            # ID가 목록에 없는 경우 건너뜀
            continue
    return indices


# 메인 함수
def main():
    # 데이터셋에서 예시 ID 로드
    example_ids = load_example_ids("data/examples/combined_examples.json")

    # 임베딩 로드
    embeddings_path = "data/embeddings/combined_questions_all-mpnet-base-v2.pkl"
    embeddings = load_embeddings(embeddings_path)

    print(f"로드된 예시 ID 수: {len(example_ids)}")
    print(f"임베딩 형태: {embeddings.shape}")

    # 방법론 디렉토리 경로
    method_dirs = [
        "saved_result/turbo/combined_random_20250517_190118",
        "saved_result/turbo/combined_kmeans_20250517_181808",
        "saved_result/turbo/combined_hdbscan_20250517_182924",
        "saved_result/turbo/combined_mmr_20250517_184000",
        "saved_result/turbo/combined_cm-mmr_20250517_185018",
        "saved_result/turbo_cm-hdbscan/combined_cm-hdbscan-mmr_20250517_191732",
        "saved_result/turbo_similarity/combined_similarity_20250517_195043",
    ]

    # 각 방법별 다양성 점수 계산
    results = {}

    for method_dir in method_dirs:
        try:
            # 방법론 이름 추출
            method_name = os.path.basename(method_dir)
            results_file = os.path.join(method_dir, "results.json")

            if not os.path.exists(results_file):
                print(f"결과 파일을 찾을 수 없습니다: {results_file}")
                continue

            # 결과 파일 로드
            method_results = load_results(results_file)

            # 다양성 점수 계산
            diversity_scores = []

            for iteration in method_results:
                if (
                    "selected_examples" in iteration
                    and len(iteration["selected_examples"]) > 1
                ):
                    # 예시 ID를 인덱스로 변환
                    example_indices = map_example_ids_to_indices(
                        iteration["selected_examples"], example_ids
                    )

                    if len(example_indices) >= 2:
                        diversity_score = calculate_diversity_score(
                            example_indices, embeddings
                        )
                        if diversity_score is not None:
                            diversity_scores.append(diversity_score)

            # 평균 다양성 점수 계산
            if diversity_scores:
                avg_diversity = np.mean(diversity_scores)
                std_diversity = np.std(diversity_scores)
                results[method_name] = {
                    "avg_diversity": avg_diversity,
                    "std_diversity": std_diversity,
                    "num_iterations": len(diversity_scores),
                }
                print(
                    f"{method_name}: 평균 다양성 점수 = {avg_diversity:.4f} ± {std_diversity:.4f} (이터레이션 수: {len(diversity_scores)})"
                )
            else:
                print(f"{method_name}: 다양성 점수를 계산할 수 없습니다.")

        except Exception as e:
            print(f"{method_dir} 처리 중 오류 발생: {e}")

    if not results:
        print("다양성 점수를 계산할 수 없습니다. 모든 방법론에서 오류가 발생했습니다.")
        return

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame(
        {
            "방법론": list(results.keys()),
            "평균 다양성 점수": [results[k]["avg_diversity"] for k in results],
            "표준편차": [results[k]["std_diversity"] for k in results],
            "이터레이션 수": [results[k]["num_iterations"] for k in results],
        }
    )

    # 다양성 점수에 따라 정렬
    results_df = results_df.sort_values("평균 다양성 점수", ascending=False)

    # 결과 저장
    results_df.to_csv("diversity_scores.csv", index=False)
    print("\n다양성 점수 순위:")
    print(results_df)


if __name__ == "__main__":
    main()
