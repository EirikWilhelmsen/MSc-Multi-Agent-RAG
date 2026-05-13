"""RCA.py
    R - Relevance Agent
    C - Candidate Agents (3 parallel)
    A - Aggregation
"""
import os
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from help_functions import get_es_client, increment_version, load_questions, load_version, search_documents
from Agents import relevance_agent, temporal_agent, candidate_agent
from help.aggregations import aggregate

increment_version("Results", "RCA")
rca_results_version = load_version("Results", "RCA")
OUTPUT_PATH = f"../../results/rca_results_{rca_results_version}.jsonl"
SLEEP_BETWEEN_REQUESTS = 1.0
TOP_CANDIDATES = 3

# -----------------------------------
# Adjustable aggregation methods:
AGGREGATION_METHOD = "Majority Vote"
#AGGREGATION_METHOD = "Confidence"
#AGGREGATION_METHOD = "Random"
# -----------------------------------


def run_candidate_agents(query: str, top_chunks: list[dict]) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=TOP_CANDIDATES) as executor:
        futures = {
            executor.submit(candidate_agent, query, chunk["content"], i): i
            for i, chunk in enumerate(top_chunks)
        }
        for future in as_completed(futures):
            results.append(future.result())
    return results


def main():
    es = get_es_client()
    questions = load_questions()
    total_tokens = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] {q['question'][:80]}...")

            # 1. Retrieve
            docs = search_documents(es, q["question"])
            if not docs:
                continue

            # 2. Relevance Agent -> top 3
            relevance_scores, tokens = relevance_agent(q["question"], [d["content"] for d in docs])
            total_tokens += tokens
            
            temporal_scores, tokens = temporal_agent(q["question"], [d["content"] for d in docs])
            total_tokens += tokens
            
            print(f"Relevance scores: {relevance_scores}")
            print(f"Temporal scores: {temporal_scores}")
            alpha = 0.3
            final_scores = {
                chunk_id: alpha * relevance_scores[chunk_id] + (1 - alpha) * temporal_scores[chunk_id]
                for chunk_id in relevance_scores
            }
            print(f"  Final scores: {final_scores}")

            # Top - N based on combined relevance and temporal scores
            top_indices = sorted(final_scores, key=final_scores.get, reverse=True)[:TOP_CANDIDATES]
            top_docs = [docs[i] for i in top_indices]

            candidates = run_candidate_agents(q["question"], top_docs)
            for c in candidates:
                total_tokens += c["tokens"]
                print(f"Chunk {c['chunk_id']}: answer='{c['answer']}', confidence={c['confidence']:.2f}, tokens={c['tokens']}")

            # 4. Aggregate
            if AGGREGATION_METHOD == "Majority Vote": best = aggregate(candidates, method="majority_vote")
            elif AGGREGATION_METHOD == "Confidence": best = aggregate(candidates, method="Confidence")
            elif AGGREGATION_METHOD == "Random": best = aggregate(candidates, method="Random")
            
            answer = best["answer"]
            best_doc = top_docs[best["chunk_id"]]
            print(f"Best answer: '{answer}' (confidence={best['confidence']:.2f})")

            result = {
                "question_id": q["id"],
                "question": q["question"],
                "gold_pageid": q["pageid"],
                "gold_title": q["title"],
                "new_date": q["new_date"],
                "old_date": q["old_date"],
                "predicted_answer": answer,
                "best_chunk_pageid": best_doc["pageid"],
                "best_chunk_date": best_doc["date"],
                "correct_article_retrieved": any(d["pageid"] == q["pageid"] for d in docs),
                "candidates": candidates,
                "retrieved_docs": [
                    {"pageid": d["pageid"], "date": d["date"], "score": d["score"]}
                    for d in docs
                ],
            }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            
    print(f"Total tokens used: {total_tokens}")
    print(f"method used for aggregation: {AGGREGATION_METHOD}")
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()