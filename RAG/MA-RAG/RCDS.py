"""RCDS.py
    R - Relevance Agent
    C - Candidate Agents (3 parallel)
    D - Debate (multi-round)
    S - Supervisor Agent
"""
import os
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from help_functions import (
    get_es_client, increment_version, load_questions, search_documents, load_version
)
from Agents import relevance_agent, candidate_agent, debate_round, supervisor_agent


increment_version("Results", "RCDS")
rcds_results_version = load_version("Results", "RCDS")
OUTPUT_PATH = f"../../results/rcds_results_{rcds_results_version}.jsonl"
SLEEP_BETWEEN_REQUESTS = 1.0
TOP_CANDIDATES = 3
DEBATE_ROUNDS = 2


def run_candidate_agents(query: str, top_chunks: list[dict]) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=TOP_CANDIDATES) as executor:
        futures = {
            executor.submit(candidate_agent, query, chunk["content"], i): i
            for i, chunk in enumerate(top_chunks)
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: x["chunk_id"])
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

            top_indices = sorted(relevance_scores, key=relevance_scores.get, reverse=True)[:TOP_CANDIDATES]
            top_docs = [docs[idx] for idx in top_indices]
            print(f"  Top chunks: {[docs[idx]['pageid'] for idx in top_indices]}")

            # 3. Candidate Agents (parallel)
            candidates = run_candidate_agents(q["question"], top_docs)
            for c in candidates:
                total_tokens += c["tokens"]
            debate_history = [candidates]  # round 0 = initial answers

            for c in candidates:
                print(f"  [Initial] Candidate {c['chunk_id']}: '{c['answer']}' "
                      f"(conf={c['confidence']:.2f})")

            # 4. Debate rounds
            for r in range(1, DEBATE_ROUNDS + 1):
                print(f"  --- Debate round {r} ---")
                candidates = debate_round(q["question"], candidates, top_docs, r, debate_history)
                debate_history.append(candidates)
                for c in candidates:
                    total_tokens += c["tokens"]
                    print(f"Candidate {c['chunk_id']}: '{c['answer']}' "
                          f"(conf={c['confidence']:.2f})")

            # 5. Supervisor
            result_sup = supervisor_agent(q["question"], candidates, top_docs, debate_history)
            for c in candidates:
                total_tokens += c["tokens"]
            answer = result_sup["answer"]
            winner_id = result_sup["winner_chunk_id"]
            winner_doc = top_docs[winner_id]
            print(f"Supervisor chose Candidate {winner_id}: '{answer}'")
            

            result = {
                "question_id": q["id"],
                "question": q["question"],
                "gold_pageid": q["pageid"],
                "gold_title": q["title"],
                "new_date": q["new_date"],
                "old_date": q["old_date"],
                "predicted_answer": answer,
                "best_chunk_pageid": winner_doc["pageid"],
                "best_chunk_date": winner_doc["date"],
                "correct_article_retrieved": any(d["pageid"] == q["pageid"] for d in docs),
                "supervisor_reasoning": result_sup["reasoning"],
                "debate_history": debate_history,
                "retrieved_docs": [
                    {"pageid": d["pageid"], "date": d["date"], "score": d["score"]}
                    for d in docs
                ],
            }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tokens used: {total_tokens}")
    print(f"Average debate rounds per question: {DEBATE_ROUNDS}")
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print(f"\nTotal time: {time_end - time_start:.2f} seconds")