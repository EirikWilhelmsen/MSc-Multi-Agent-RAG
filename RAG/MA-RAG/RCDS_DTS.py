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
    get_es_client, increment_version, load_questions, search_documents,
    load_version, normalize_answer, has_consensus, all_unsure
)
from Agents import relevance_agent, temporal_agent, supervisor_agent, debate_round, candidate_agent

increment_version("Results", "RCDS")
rcds_results_version = load_version("Results", "RCDS")
OUTPUT_PATH = f"../../results/rcds_results_{rcds_results_version}.jsonl"
SLEEP_BETWEEN_REQUESTS = 1.0
TOP_CANDIDATES = 3

MIN_DEBATE_ROUNDS = 2
MAX_DEBATE_ROUNDS = 6

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

def aggregate(relevance_scores: dict[int, float], temporal_scores: dict[int, float], alpha: float) -> dict[int, float]:
    """Combine relevance and temporal scores and return the best chunk_id."""
    final_scores = {}
    for chunk_id, r in relevance_scores.items():
        t = temporal_scores.get(chunk_id, 0.5)
        final_scores[chunk_id] = alpha * r + (1 - alpha) * t
    return final_scores

def main():
    es = get_es_client()
    questions = load_questions()
    total_tokens = 0
    total_rounds = 0
    stop_reasons_count = {"max_rounds": 0, "all_unsure": 0, "consensus": 0, "deadlock": 0}

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
            for d in docs:
                print(f"doc {d['pageid']} (date {d['date']}): relevance={relevance_scores.get(docs.index(d), 0.5):.2f}, temporal={temporal_scores.get(docs.index(d), 0.5):.2f}")
            
            # 4. Aggregate
            best_chunks = aggregate(relevance_scores, temporal_scores, alpha = 0.3)
            print(best_chunks)

            top_indices = sorted(best_chunks, key=best_chunks.get, reverse=True)[:TOP_CANDIDATES]
            top_docs = [docs[idx] for idx in top_indices]
            print(f"Top chunks: {[docs[idx]['pageid'] for idx in top_indices]}")

            # 3. Candidate Agents (parallel)
            candidates = run_candidate_agents(q["question"], top_docs)
            for c in candidates:
                total_tokens += c["tokens"]
            debate_history = [candidates]  # round 0 = initial answers

            for c in candidates:
                print(f"[Initial] Candidate {c['chunk_id']}: '{c['answer']}' "f"(conf={c['confidence']:.2f})")

            # 4. Debate rounds (adaptive)
            debate_history = [candidates]
            stop_reason = "max_rounds"
            rounds_run = 0
            rounds_without_change = 0

            for r in range(1, MAX_DEBATE_ROUNDS + 1):
                print(f"--- Debate round {r} ---")
                prev_candidates = candidates
                candidates = debate_round(q["question"], candidates, top_docs, r, debate_history)
                debate_history.append(candidates)
                rounds_run = r

                for c in candidates:
                    total_tokens += c["tokens"]
                    print(f"Candidate {c['chunk_id']}: '{c['answer']}' "f"(conf={c['confidence']:.2f})")

                if r < MIN_DEBATE_ROUNDS:
                    continue
                
                if all_unsure(candidates):
                    stop_reason = "all_unsure"
                    print(f"Stopping: all candidates are unsure after round {r}")
                    stop_reasons_count[stop_reason] += 1
                    break
                
                if has_consensus(candidates):
                    stop_reason = "consensus"
                    print(f"Stopping: consensus among real candidates after round {r}")
                    stop_reasons_count[stop_reason] += 1
                    break
                
                # Hard deadlock: no change in real candidates for 2 consecutive rounds
                prev_real = {c["chunk_id"]: normalize_answer(c["answer"]) for c in prev_candidates
                             if normalize_answer(c["answer"]) != "unsure"}
                curr_real = {c["chunk_id"]: normalize_answer(c["answer"]) for c in candidates
                             if normalize_answer(c["answer"]) != "unsure"}

                if prev_real == curr_real:
                    rounds_without_change += 1
                else:
                    rounds_without_change = 0

                if rounds_without_change >= 2:
                    stop_reason = "deadlock"
                    print(f"Stopping: no movement among real candidates for 2 rounds")
                    stop_reasons_count[stop_reason] += 1
                    break

            # 5. Supervisor
            result_sup = supervisor_agent(q["question"], candidates, top_docs, debate_history)
            for c in candidates:
                total_tokens += c["tokens"]
            answer = result_sup["answer"]
            winner_id = result_sup["winner_chunk_id"]
            winner_doc = top_docs[winner_id]
            print(f"Supervisor chose Candidate {winner_id}: '{answer}'")
            
            total_rounds += rounds_run

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
                "num_debate_rounds": rounds_run,
                "debate_stop_reason": stop_reason,
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
    print(f"Average debate rounds per question: {total_rounds / 500:.2f}")
    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"\nStop reasons:")
    for reason, count in stop_reasons_count.items():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()