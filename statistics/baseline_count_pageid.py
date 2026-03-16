import json
import matplotlib.pyplot as plt
from helper_functions.helper_functions import (
    create_graph, 
    is_match, 
    classify_answer,
    load_version,
    increment_version
)

GROUND_TRUTH_PATH = "../data/ground_truth_answers.csv"
GROUND_TRUTH_PAGE_PATH = "../data/hoh_question_pageid_map.csv"
results_version = load_version("Results", "Baseline")
figure_version = load_version("PageID_Count", "Baseline")
RESULTS_PATH = f"../results/rag_baseline_results_{results_version}.jsonl"
FIGURE_PATH = f"../results/figures/rag_baseline_pageid_counts_{figure_version}.png"

def correct_pageids():
    correct_id_correct_version_rank1 = 0
    correct_id_wrong_version_rank1 = 0
    correct_id_not_rank1 = 0
    correct_id_not_retrieved = 0

    with open(RESULTS_PATH, "r") as f:
        results = [json.loads(line) for line in f]

    for r in results:
        if not r['correct_article_retrieved']:
            correct_id_not_retrieved += 1
            continue

        if r['correct_article_rank'] != 1:
            correct_id_not_rank1 += 1
            continue

        # Riktig id er rank 1 — sjekk om versjonen er riktig
        top_doc = r['retrieved_docs'][0]
        correct_version = top_doc['date'] == r['new_date']

        if correct_version:
            correct_id_correct_version_rank1 += 1
        else:
            correct_id_wrong_version_rank1 += 1

    counts = {
        "Correct pageID, new version, rank 1": correct_id_correct_version_rank1,
        "Correct pageID, outdated version, rank 1":   correct_id_wrong_version_rank1,
        "Correct pageID, not rank 1":             correct_id_not_rank1,
        "Correct pageID not retrieved":              correct_id_not_retrieved,
    }

    
    create_graph(counts, title="RAG Baseline Page ID Evaluation (100 questions)", path=FIGURE_PATH)

    increment_version("PageID_Count", "Baseline")

if __name__ == "__main__":
    correct_pageids()