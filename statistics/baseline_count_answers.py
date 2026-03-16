import json
import matplotlib.pyplot as plt
from helper_functions.helper_functions import (
    classify_answer_LLM,
    create_graph, 
    is_match, 
    classify_answer, 
    load_version,
    increment_version
)

GROUND_TRUTH_PATH = "../data/ground_truth_answers.csv"
GROUND_TRUTH_PAGE_PATH = "../data/hoh_question_pageid_map.csv"

results_version = load_version("Results", "Baseline")
figure_version = load_version("Answer_Count", "Baseline")

RESULTS_PATH = f"../results/rag_baseline_results_v3.jsonl"
FIGURE_PATH = f"../results/figures/rag_baseline_answer_counts_{figure_version}.png"


    
def count():
    with open(RESULTS_PATH, "r") as f:
        results = [json.loads(line) for line in f]

    with open(GROUND_TRUTH_PATH, "r") as f:
        f.readline()  # skip header
        ground_truth = [line.strip().split(";") for line in f]

    # for r in range(len(results)):
    #     print(f"Answer {r}: {results[r]['predicted_answer']}. Retrieved documents:")
    #     for doc in results[r]['retrieved_docs'][:5]:
    #         print(f"pageid: {doc['pageid']}, page-score: {doc['score']}")

    correct_count = 0
    outdated_count = 0
    wrong_count = 0
    unsure_count = 0
    BM_count = 0
    for r in range(5):
        predicted = results[r]['predicted_answer']
        question = results[r]['question']
        new = ground_truth[r][1]
        old = ground_truth[r][2]
    
        if is_match(predicted, "Unsure"):
            print(f"Answer {r} is unsure")
            unsure_count += 1
        elif is_match(predicted, new):
            print(f"Answer {r} is correct.")
            correct_count += 1
        elif is_match(predicted, old):
            print(f"Answer {r} is outdated")
            outdated_count += 1
        else:
            classification = classify_answer(predicted, new, old)
            # classification = classify_answer_LLM(predicted, new, old, question)
            BM_count += 1
            if classification == "correct":
                print(f"Answer {r} is correct (Bert).")
                correct_count += 1
            elif classification == "outdated":
                print(f"Answer {r} is outdated (Bert).")
                outdated_count += 1
            elif classification == "wrong":
                print(f"Answer '{predicted}' needs review.")
                review = input("correct (c), outdated (o), or wrong (w)? ")
                if review == "c":
                    correct_count += 1
                elif review == "o":
                    outdated_count += 1
                elif review == "w":
                    wrong_count += 1
    print(
        f"BERT: correct: {correct_count}, outdated: {outdated_count}, wrong: {wrong_count}, unsure: {unsure_count}"
    )
    # counts = {
    #     "Correct answers": correct_count,
    #     "Outdated answers": outdated_count,
    #     "Wrong answers": wrong_count,
    #     "Unsure": unsure_count,
    # }
    # print(BM_count)
    # 
    # create_graph(counts, title = "RAG Baseline Answer Evaluation (100 questions)", path = FIGURE_PATH)
    # 
    # increment_version("Answer_Count", "Baseline")

if __name__ == "__main__":
    count()