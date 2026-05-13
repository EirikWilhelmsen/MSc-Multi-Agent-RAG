"""RTA.py
    R - Relevance Agent
    T - Temporal Agent
    A - Aggregation Agent
"""
import time
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from help_functions import (
    get_es_client, load_questions, search_documents,
    load_version, increment_version
)
from Agents import temporal_agent, relevance_agent, generation_agent

increment_version("Results", "RTA")
rta_results_version = load_version("Results", "RTA")
OUTPUT_PATH = f"../../results/rta_results_{rta_results_version}.jsonl"
SLEEP_BETWEEN_REQUESTS = 1.0

# Adjust if needed
ALPHA = 0.3


def aggregate(relevance_scores: dict[int, float], temporal_scores: dict[int, float], alpha: float) -> int:
    """Combine relevance and temporal scores and return the best chunk_id."""
    final_scores = {}
    for chunk_id, r in relevance_scores.items():
        t = temporal_scores.get(chunk_id, 0.5)
        final_scores[chunk_id] = alpha * r + (1 - alpha) * t
    return max(final_scores, key=final_scores.get)


def main():
    es = get_es_client()
    questions = load_questions()
    token_count = 0
    alpha = ALPHA

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] {q['question'][:80]}...")

            # 1. Retrieve
            docs = search_documents(es, q["question"])
            print(f"Retrieved {len(docs)} documents")

            if not docs:
                print("No documents retrieved, skipping.")
                continue

            chunks = [d["content"] for d in docs]

            # 2. Relevance Agent
            relevance_scores, tokens = relevance_agent(q["question"], chunks)
            print(f"Relevance scores: {relevance_scores}")
            token_count += tokens

            # 3. Temporal Agent
            temporal_scores, tokens = temporal_agent(q["question"], chunks)
            print(f"Temporal scores: {temporal_scores}")
            token_count += tokens

            # 4. Aggregate
            if not relevance_scores:
                print("Could not parse relevance scores, falling back to chunk 0.")
                best_chunk_id = 0
            else:
                best_chunk_id = aggregate(relevance_scores, temporal_scores, alpha)

            best_doc = docs[best_chunk_id]
            print(f"Best chunk ID: {best_chunk_id}, PageID: {best_doc['pageid']}")

            # 5. Generate answer
            answer, tokens = generation_agent(q["question"], best_doc["content"])
            token_count += tokens
            print(f"Answer: {answer}")

            # 6. Log
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
                "retrieved_docs": [
                    {"pageid": d["pageid"], "date": d["date"], "score": d["score"]}
                    for d in docs
                ],
            }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    with open(OUTPUT_PATH, "r") as f:
        all_results = [json.loads(line) for line in f]

    total = len(all_results)
    retrieved = sum(1 for r in all_results if r["correct_article_retrieved"])
    print(f"Results for RTA with alpha = {alpha}")
    print(f"Total questions:           {total}")
    print(f"Correct article retrieved: {retrieved}/{total} ({100*retrieved/total:.1f}%)")
    print(f"Results saved to:          {OUTPUT_PATH}")
    print(f"Total LLM tokens used:    {token_count}")
    
    increment_version("Results", "RTA")


if __name__ == "__main__":
    main()