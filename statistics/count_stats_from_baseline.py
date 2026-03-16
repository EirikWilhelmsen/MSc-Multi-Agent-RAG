import json
import matplotlib.pyplot as plt
from helper_functions.helper_functions import create_graph, is_match, classify_answer

RESULTS_PATH = "../results/rag_baseline_results_v3.jsonl"
GROUND_TRUTH_PATH = "../data/ground_truth_answers.csv"
GROUND_TRUTH_PAGE_PATH = "../data/hoh_question_pageid_map.csv"
    
def count():
    with open(RESULTS_PATH, "r") as f:
        results = [json.loads(line) for line in f]

    with open(GROUND_TRUTH_PATH, "r") as f:
        f.readline()  # skip header
        ground_truth = [line.strip().split(";") for line in f]

    for r in range(len(results)):
        print(f"Answer {r}: {results[r]['predicted_answer']}. Retrieved documents:")
        for doc in results[r]['retrieved_docs'][:5]:
            print(f"pageid: {doc['pageid']}, page-score: {doc['score']}")

    correct_count = 0
    outdated_count = 0
    wrong_count = 0
    unsure_count = 0
    BM_count = 0
    for r in range(len(results)):
        predicted = results[r]['predicted_answer']
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
            BM_count += 1
            if classification == "correct":
                print(f"Answer {r} is correct (bert).")
                correct_count += 1
            elif classification == "outdated":
                print(f"Answer {r} is outdated (bert).")
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

    counts = {
        "Correct answers": correct_count,
        "Outdated answers": outdated_count,
        "Wrong answers": wrong_count,
        "Unsure": unsure_count,
    }
    print(BM_count)

    path = "../results/figures/rag_baseline_answer_counts_v3.png"

    create_graph(counts, title = "RAG Baseline Answer Evaluation (100 questions)", path = path)

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
        "Correct id, correct version, rank 1": correct_id_correct_version_rank1,
        "Correct id, wrong version, rank 1":   correct_id_wrong_version_rank1,
        "Correct id, not rank 1":             correct_id_not_rank1,
        "Correct id not retrieved":              correct_id_not_retrieved,
    }

    path = "../results/figures/rag_baseline_pageid_counts_v3.png"
    create_graph(counts, title="RAG Baseline Page ID Evaluation (100 questions)", path=path)

mode = input("count (c) or evaluate pageids (p) ? ")
if mode == "c":
    count()
elif mode == "p":
    correct_pageids()