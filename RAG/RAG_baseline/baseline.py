import csv
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Agents import baseline_generation_agent
from help_functions import (
    get_es_client, load_questions, search_documents, load_version, increment_version
)

increment_version("Results", "Baseline")
results_version = load_version("Results", "Baseline")
OUTPUT_PATH = Path(f"../../data/rag_baseline_results_{results_version}.jsonl")
SLEEP_BETWEEN_REQUESTS = 1.0  # seconds between LLM calls


def main():
    tokens = 0
    es = get_es_client()
    print(f"Connected to Elasticsearch")

    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    results = []

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] {q['question'][:80]}...")

            # 1. Retrieve
            docs = search_documents(es, q["question"])
            print(f"Retrieved {len(docs)} documents")
            if not docs:
                print("No documents retrieved, skipping.")
                continue

            # 2. Generate
            try:
                answer, token_count = baseline_generation_agent(q["question"], docs)
                tokens += token_count
            except Exception as e:
                answer = f"[ERROR] {e}"
                print(f"LLM error: {e}")

            print(f"Answer: {answer[:100]}")

            # 4. Log result
            result = {
                "question_id": q["id"],
                "question": q["question"],
                "gold_pageid": q["pageid"],
                "gold_title": q["title"],
                "new_date": q["new_date"],
                "old_date": q["old_date"],
                "predicted_answer": answer,
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
    print("Top-1")
    print(f"Total tokens: {tokens}")
    print(f"Total questions: {total}")
    print(f"Correct article retrieved: {retrieved}/{total} ({100*retrieved/total:.1f}%)")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()