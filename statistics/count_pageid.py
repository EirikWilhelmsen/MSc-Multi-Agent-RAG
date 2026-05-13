import json
import matplotlib.pyplot as plt
import numpy as np
from helper_functions.helper_functions import (
    is_match, 
    #classify_answer,
    load_version,
    increment_version,
    pageid_graph,
)

GROUND_TRUTH_PATH = "../data/ground_truth_answers.csv"
GROUND_TRUTH_PAGE_PATH = "../data/hoh_question_pageid_map.csv"

RADAR_PLOT = False

def correct_pageids(model):
    if model == "rta":
        results_version = load_version("Results", "RTA")
        figure_version = load_version("PageID_Count", "RTA")
        results_path = f"../results/rta_results_{results_version}.jsonl"
        figure_path = f"../results/figures/rta_pageid_counts_{figure_version}.png"
        count_RTA(results_path, figure_path, model="RTA")
    elif model == "rco_v2":
        results_version = load_version("Results", "RCO_V2")
        figure_version = load_version("PageID_Count", "RCO_V2")
        results_path = f"../results/rco_v2_results_{results_version}.jsonl"
        figure_path = f"../results/figures/rco_v2_pageid_counts_{figure_version}.png"
        count_RCO_V2(results_path, figure_path, model="RCO_V2")
    elif model == "baseline":
        results_version = load_version("Results", "Baseline")
        figure_version = load_version("PageID_Count", "Baseline")
        results_path = f"../results/final_results/rag_baseline_results_{results_version}.jsonl"
        print(f"Baseline results path: {results_path}")
        figure_path = f"../results/figures/rag_baseline_pageid_counts_{figure_version}.png"
        count_baseline(results_path, figure_path)
    elif model == "rca":
        results_version = load_version("Results", "RCA")
        figure_version = load_version("PageID_Count", "RCA")
        results_path = f"../results/rca_results_{results_version}.jsonl"
        figure_path = f"../results/figures/rca_pageid_counts_{figure_version}.png"
        analyze_rca_for_plot(results_path, model="RCA", figure_path=figure_path)
    elif model == "rco":
        results_version = load_version("Results", "RCO")
        figure_version = load_version("PageID_Count", "RCO")
        results_path = f"../results/rco_results_{results_version}.jsonl"
        figure_path = f"../results/figures/rco_pageid_counts_{figure_version}"
        with open(results_path, "r") as f:
            results = [json.loads(line) for line in f]

        create_rco_retrieval_graph(
            results,
            title="RCO Retrieval Analysis (100 questions)",
            path=figure_path
        )
        create_rco_selected_chunk_graph(
            results,
            title="RCO Coverage Analysis (100 questions)",
            path=figure_path
        )
        increment_version("PageID_Count", "RCO")

def count_RTA(results_path, figure_path, model):
    correct_id_correct_version_rank1 = 0
    correct_id_wrong_version_rank1 = 0
    neither = 0

    with open(results_path, "r") as f:
        results = [json.loads(line) for line in f]

    for r in results:
        if not r['correct_article_retrieved']:
            neither += 1
            continue

        top_doc = r['best_chunk_pageid']
        correct_version = r['best_chunk_date'] == r['new_date'] and top_doc == r['gold_pageid']
        wrong_version = r['best_chunk_date'] == r['old_date'] and top_doc == r['gold_pageid']

        if correct_version:
            correct_id_correct_version_rank1 += 1
        elif wrong_version:
            correct_id_wrong_version_rank1 += 1
        else:            
            neither += 1

    counts = {
        "Updated selected": correct_id_correct_version_rank1,
        "Outdated selected": correct_id_wrong_version_rank1,
        "Other selected": neither,
    }

    print(counts)
    
    pageid_graph(counts, title=f"{model} alpha=0.3 Page ID Evaluation (500 questions, rank 1 document)", path=figure_path)

    increment_version("PageID_Count", "RTA")

def count_baseline(results_path, figure_path):
    correct_id_correct_version_rank1 = 0
    correct_id_wrong_version_rank1 = 0
    neither = 0

    with open(results_path, "r") as f:
        results = [json.loads(line) for line in f]

    for r in results:
        if not r['correct_article_retrieved']:
            neither += 1
            continue

        top_doc = r['retrieved_docs'][0]
        correct_version = top_doc['date'] == r['new_date']
        wrong_version = top_doc['date'] == r['old_date']

        if correct_version:
            correct_id_correct_version_rank1 += 1
        elif wrong_version:
            correct_id_wrong_version_rank1 += 1
        else:            
            neither += 1

    counts = {
        "Updated selected": correct_id_correct_version_rank1,
        "Outdated selected": correct_id_wrong_version_rank1,
        "Other selected": neither,
    }

    print(counts)
    
    pageid_graph(counts, title=f"Pointwise {model} Page ID Evaluation (500 questions, rank 1 document)", path=figure_path)

    increment_version("PageID_Count", "Baseline")

def count_RCO_V2(results_path, figure_path, model):
    updated_selected = 0
    outdated_selected = 0
    neither = 0
    
    # Konflikt-statistikk
    total_conflicts = 0
    conflicts_with_revision = 0
    revision_to_updated = 0
    revision_to_outdated = 0

    with open(results_path, "r") as f:
        results = [json.loads(line) for line in f]

    for r in results:
        gold_pageid = r['gold_pageid']
        best_pageid = r.get('best_chunk_pageid')
        best_date = r.get('best_chunk_date')
        new_date = r['new_date']
        old_date = r['old_date']
        
        # Hovedkategorisering
        if best_pageid == gold_pageid and best_date == new_date:
            updated_selected += 1
        elif best_pageid == gold_pageid and best_date == old_date:
            outdated_selected += 1
        else:
            neither += 1
        
        # Konflikt-analyse
        if r.get('conflict_detected'):
            total_conflicts += 1
            
            # Sjekk om verification endret svaret
            decision = r.get('verification_decision')
            if decision == 'revise':
                conflicts_with_revision += 1
                
                # Hvilken versjon ble valgt etter revisjon?
                if best_pageid == gold_pageid and best_date == new_date:
                    revision_to_updated += 1
                elif best_pageid == gold_pageid and best_date == old_date:
                    revision_to_outdated += 1

    counts = {
        "Updated selected": updated_selected,
        "Outdated selected": outdated_selected,
        "Other selected": neither,
    }
    
    # Print konflikt-statistikk
    print(f"\n{'='*50}")
    print(f"CONFLICT ANALYSIS - {model}")
    print(f"{'='*50}")
    print(f"Total conflicts detected: {total_conflicts}/{len(results)} ({100*total_conflicts/len(results):.1f}%)")
    print(f"Conflicts with revision:  {conflicts_with_revision}/{total_conflicts} ({100*conflicts_with_revision/total_conflicts:.1f}%)" if total_conflicts > 0 else "Conflicts with revision:  0")
    if conflicts_with_revision > 0:
        print(f"  - Revised to updated:   {revision_to_updated}")
        print(f"  - Revised to outdated:  {revision_to_outdated}")
    print(f"{'='*50}\n")
    
    # Plot
    pageid_graph(counts, f"{model} Retrieval Results", figure_path)
    
    increment_version("PageID_Count", "RCO_V2")
    
    return counts
        

def create_rco_retrieval_graph(results, title, path):
    
    both_in_top3 = 0
    only_new_in_top3 = 0
    only_old_in_top3 = 0
    neither_in_top3 = 0
    conflict_detected = 0
    new_selected = 0
    old_selected = 0
    other_selected = 0

    for r in results:
        top3_chunks = r['top_chunks']
        new_in_top3 = any(c['pageid'] == r['gold_pageid'] and c['date'] == r['new_date'] for c in top3_chunks)
        old_in_top3 = any(c['pageid'] == r['gold_pageid'] and c['date'] == r['old_date'] for c in top3_chunks)

        if new_in_top3 and old_in_top3:
            both_in_top3 += 1
        elif new_in_top3:
            only_new_in_top3 += 1
        elif old_in_top3:
            only_old_in_top3 += 1
        else:
            neither_in_top3 += 1

        if r['conflict_detected']:
            conflict_detected += 1

        if r['best_chunk_pageid'] == r['gold_pageid']:
            if r['best_chunk_date'] == r['new_date']:
                new_selected += 1
            else:
                old_selected += 1
        else:
            other_selected += 1

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title)

    # Plot 1: Top 3 innhold
    axes[0].bar(
        ["Both", "Only updated", "Only outdated", "Neither"],
        [both_in_top3, only_new_in_top3, only_old_in_top3, neither_in_top3],
        color=["green", "steelblue", "tomato", "gray"]
    )
    axes[0].set_title("Gold versions in top 3")
    axes[0].set_ylabel("Count")

    # Plot 2: Konflikt detektert
    axes[1].bar(
        ["Conflict detected", "No conflict"],
        [conflict_detected, len(results) - conflict_detected],
        color=["orange", "steelblue"]
    )
    axes[1].set_title("Conflict detection")

    # Plot 3: Valgt chunk
    axes[2].bar(
        ["Updated selected", "Outdated selected", "Other selected"],
        [new_selected, old_selected, other_selected],
        color=["mediumseagreen", "tomato", "gray"]
    )
    axes[2].set_title("Selected chunk version")
    
    figure_path = f"{path}_rco_retrieval.png"

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()
    
def create_rco_selected_chunk_graph(results, title, path):
    figure_path = f"{path}_rco_selected_chunk.png"
    new_selected = 0
    old_selected = 0
    other_selected = 0

    for r in results:
        if r['best_chunk_pageid'] == r['gold_pageid']:
            if r['best_chunk_date'] == r['new_date']:
                new_selected += 1
            else:
                old_selected += 1
        else:
            other_selected += 1

    categories = ["Updated selected", "Outdated selected", "Other selected"]
    values = [new_selected, old_selected, other_selected]
    colors = ["mediumseagreen", "tomato", "gray"]

    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, color=colors)
    plt.title(title)
    plt.ylabel("Count")
    plt.ylim(0, max(values) * 1.2)
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

def analyze_rca_for_plot(results_path, model, figure_path):
    counts = {
        "Updated": 0,
        "Outdated": 0,
        "Neither": 0,
        "Tied": 0,
    }
    
    with open(results_path, "r") as f:
        results = [json.loads(line) for line in f]
    
    for r in results:
        docs = r["retrieved_docs"]
        candidates = r.get("candidates", [])
        gold_pageid = r["gold_pageid"]
        new_date = r["new_date"]
        old_date = r["old_date"]
        best_date = r.get("best_chunk_date")
        
        # Selection outcome
        if best_date == new_date and r["best_chunk_pageid"] == gold_pageid:
            counts["Updated"] += 1
        elif best_date == old_date and r["best_chunk_pageid"] == gold_pageid:
            counts["Outdated"] += 1
        else:
            counts["Neither"] += 1
        
        # Tied? (finn begge versjoner og sjekk scores)
        updated_idx = None
        outdated_idx = None
        for idx, d in enumerate(docs):
            if d["pageid"] == gold_pageid:
                if d["date"] == new_date:
                    updated_idx = idx
                elif d["date"] == old_date:
                    outdated_idx = idx
        
        if updated_idx is not None and outdated_idx is not None:
            bm25_tie = abs(docs[updated_idx]["score"] - docs[outdated_idx]["score"]) < 0.01
            
            updated_conf = next((c["confidence"] for c in candidates if c["chunk_id"] == updated_idx), None)
            outdated_conf = next((c["confidence"] for c in candidates if c["chunk_id"] == outdated_idx), None)
            
            conf_tie = (updated_conf is not None and outdated_conf is not None 
                        and abs(updated_conf - outdated_conf) < 0.01)
            
            if bm25_tie and conf_tie:
                counts["Tied"] += 1
    print(counts, model, figure_path)
    pageid_graph(counts, title=f"{model} Page ID Evaluation (100 questions)", path=figure_path)
    increment_version("PageID_Count", "RCA")

def create_star_plot():
    version = load_version("radarplot", "RADAR_PLOT")
    
    models = ["Baseline", "RTA", "RCA", "RCO"]
    data = {
        "Updated selected": [21, 38, 45, 33],
        "Outdated selected": [57, 41, 44, 52],
        "Other selected": [22, 21, 11, 15],
    }
    
    # Beregn vinkler for hver modell
    num_models = len(models)
    angles = np.linspace(0, 2 * np.pi, num_models, endpoint=False).tolist()
    angles += angles[:1]  # Lukk sirkelen
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ["mediumseagreen", "tomato", "gray"]
    
    for (category, values), color in zip(data.items(), colors):
        values_closed = values + values[:1]  # Lukk sirkelen
        ax.plot(angles, values_closed, 'o-', linewidth=2, label=category, color=color)
        ax.fill(angles, values_closed, alpha=0.15, color=color)
    
    # Sett modell-labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(models)
    
    ax.set_ylim(0, 70)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title("Page ID Selection by Model", y=1.08)
    
    plt.tight_layout()
    plt.savefig(f"../results/figures/pageid_selection_radar_chart_{version}.png", 
                bbox_inches='tight', dpi=150)
    plt.close()
    
    increment_version("radarplot", "RADAR_PLOT")

if __name__ == "__main__":
    if RADAR_PLOT:
        create_star_plot()
    model = input("Enter model to analyze (rta, rca, baseline, rco): ").strip().lower()
    correct_pageids(model)