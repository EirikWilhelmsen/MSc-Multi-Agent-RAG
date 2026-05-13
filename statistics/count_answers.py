import json
import matplotlib.pyplot as plt
from helper_functions.helper_functions import (
    classify_answer_LLM,
    answer_graph, 
    is_match, 
    load_version,
    increment_version,
    numeric_ground_truth_match,
    is_unsure
)
LLM_AS_JUDGE = True
GROUND_TRUTH_PATH = "../data/500Q/500_hoh_questions.csv"
#GROUND_TRUTH_PAGE_PATH = "../data/hoh_question_pageid_map.csv"
ITERATION_COUNT = int(1)
    
def count(model):
    if model == "rta":
        results_version = load_version("Results", "RTA")
        figure_version = load_version("Answer_Count", "RTA")
        results_path = f"../results/rta_results_{results_version}.jsonl"
        figure_path = f"../results/figures/rta_answer_counts_{figure_version}.png"
        count_answers(results_path, figure_path, model = "RTA")
    elif model == "baseline":
        results_version = load_version("Results", "Baseline")
        figure_version = load_version("Answer_Count", "Baseline")
        results_path = f"../results/final_results/rag_baseline_results_{results_version}.jsonl"
        figure_path = f"../results/figures/rag_baseline_answer_counts_{figure_version}.png"
        count_answers(results_path, figure_path, model = "Baseline")
    elif model == "rca":
        results_version = load_version("Results", "RCA")
        figure_version = load_version("Answer_Count", "RCA")
        results_path = f"../results/rca_results_{results_version}.jsonl"
        figure_path = f"../results/figures/rca_answer_counts_{figure_version}.png"
        count_answers(results_path, figure_path, model = "RCA")
    elif model == "rco":
        results_version = load_version("Results", "RCO")
        figure_version = load_version("Answer_Count", "RCO")
        results_path = f"../results/rco_results_{results_version}.jsonl"
        figure_path = f"../results/figures/rco_answer_counts_{figure_version}.png"
        count_answers(results_path, figure_path, model = "RCO")
    elif model == "rco_v2":
        results_version = load_version("Results", "RCO_V2")
        figure_version = load_version("Answer_Count", "RCO_V2")
        results_path = f"../results/rco_v2_results_{results_version}.jsonl"
        figure_path = f"../results/figures/rco_v2_answer_counts_{figure_version}.png"
        count_answers(results_path, figure_path, model = "RCO_V2")

def count_answers(results_path, figure_path, model):
    global ITERATION_COUNT
    print(f"Running answer count for {results_path}...")
    with open(results_path, "r") as f:
        results = [json.loads(line) for line in f]

    with open(GROUND_TRUTH_PATH, "r") as f:
        f.readline()  # skip header
        ground_truth = [line.strip().split(";") for line in f]
        
    correct_count = 0
    outdated_count = 0
    wrong_count = 0
    unsure_count = 0
    unable_string_match_count = 0
    
    for r in range(len(results)):
        predicted = results[r]['predicted_answer']
        if model == "RCO_V2" and ITERATION_COUNT == 2: predicted = results[r]['preliminary_answer']
        question = ground_truth[r][3]
        # wierd workaround some questions have an error in the csv

        new = ground_truth[r][-2]
        old = ground_truth[r][-1]
        if LLM_AS_JUDGE:
            classification = classify_answer_LLM(predicted, new, old, question)
            if classification == "correct":
                print(f"Answer {r} is correct.")
                correct_count += 1
            else:
                wrong_count += 1
        else:
            numeric_result = numeric_ground_truth_match(predicted, new, old)

            if is_match(predicted, "Unsure") or is_unsure(predicted):
                print(f"predicted: {predicted} - new: {new} - old: {old} -> Wrong")
                wrong_count += 1
            elif is_match(predicted, new) or numeric_result == "current":
                print(f"predicted: {predicted} - new: {new} - old: {old} -> Correct")
                correct_count += 1
            elif is_match(predicted, old) or numeric_result == "outdated":
                print(f"predicted: {predicted} - new: {new} - old: {old} -> Wrong")
                wrong_count += 1
            elif numeric_result == "wrong":
                print(f"predicted: {predicted} - new: {new} - old: {old} -> Wrong")
                wrong_count += 1
            else:
                print(f"predicted: {predicted} - new: {new} - old: {old} -> No string match")
                unable_string_match_count += 1
    counts = {
        "Correct answers": correct_count,
        "Wrong answers": wrong_count,
    }
    
    #if ITERATION_COUNT == 1:
    #    create_graph(counts, LLM_counts, title = f"{model} Answer Evaluation (100 questions)", path = figure_path)
    #    increment_version("Answer_Count", model)
    answer_graph(counts, title = f"{model} alpha=0.3 Answer Evaluation ({500-unable_string_match_count} questions) Method: LLM-as-judge", path = figure_path)
    increment_version("Answer_Count", model)
    print(f"fullførte analyse av {results_path}, {model}")
if __name__ == "__main__":
    ITERATION_COUNT = 1
    model = input("enter rag model: ").lower()
    count(model)
    ITERATION_COUNT = 2
    if ITERATION_COUNT == 2 and model == "rco_v2":
        count(model)