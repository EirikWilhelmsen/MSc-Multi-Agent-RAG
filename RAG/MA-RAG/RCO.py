"""RCO_v2.py
    R  - Relevance Agent (listwise ranking)
    C  - Conflict Detection Agent (identifiserer og formulerer konflikt)
    V  - Verification Agent (utfordrer svaret med oppfølgingsspørsmål)
    O  - Output
"""
import time
import json
import re
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from help_functions import (
    get_es_client, load_questions, search_documents, load_version, increment_version
)
from Agents import relevance_agent, conflict_detection_agent, verification_agent, generation_agent

increment_version("Results", "RCO_V2")
rco_results_version = load_version("Results", "RCO_V2")
OUTPUT_PATH = f"../../results/rco_v2_results_{rco_results_version}.jsonl"
SLEEP_BETWEEN_REQUESTS = 1.0
TOP_CANDIDATES = 3
TRESHOLD_CONFLICT_DETECTION = 1.0

def main():
    es = get_es_client()
    questions = load_questions()
    total_tokens = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] {q['question'][:80]}...")

            docs = search_documents(es, q["question"])
            if not docs:
                continue

            chunks = [d["content"] for d in docs]

            # 1. Relevance ranking
            relevance_scores, tokens= relevance_agent(q["question"], chunks)
            top_indices = sorted(
                relevance_scores, key=relevance_scores.get, reverse=True
            )[:TOP_CANDIDATES] if relevance_scores else list(range(min(TOP_CANDIDATES, len(docs))))
            total_tokens += tokens
            top_docs = [docs[i] for i in top_indices]
            top_chunks = [d["content"] for d in top_docs]

            # 2. Conflict detection
            conflict_info, tokens = conflict_detection_agent(q["question"], top_chunks)
            total_tokens += tokens
            print(f"Conflict: {conflict_info.get('conflict_detected')}")
            formulation = conflict_info.get('conflict_formulation') or ''
            print(f"Formulation: {formulation[:80]}")

            # 3. Verification (kun hvis konflikt eller lav confidence)
            if conflict_info.get("conflict_detected") or conflict_info.get("confidence", 1.0) < TRESHOLD_CONFLICT_DETECTION:
                verification, tokens = verification_agent(q["question"], conflict_info, top_chunks)
                total_tokens += tokens
                source_chunk_id = verification.get("final_source_chunk_id")
                decision = verification.get("decision")
                followup_q = verification.get("followup_question", "")
                followup_reasoning = verification.get("followup_reasoning", "")
                print(f"Verification decision: {decision}")
                print(f"Follow-up question: {followup_q}")
                print(f"Follow-up reasoning: {followup_reasoning}")
                print(f"Source chunk ID after verification: {source_chunk_id}")
                if decision == "revise":
                    try:
                        source_chunk_id = int(source_chunk_id)
                        final_answer, tokens = generation_agent(q["question"], top_chunks[source_chunk_id])
                        print(f"Revised answer: {final_answer}")
                        total_tokens += tokens
                    except (TypeError, ValueError):
                        final_answer = "Unsure"
                else:
                    final_answer = conflict_info.get("preliminary_answer", "Unsure")
            else:
                verification = {}
                final_answer = conflict_info.get("preliminary_answer", "Unsure")
                source_chunk_id = conflict_info.get("source_chunk_id")
                decision = "no_conflict"

            print(f"Final answer: '{final_answer}'")
    
            if source_chunk_id is None or source_chunk_id == 'None':
                best_chunk = None
            elif isinstance(source_chunk_id, str):
                try:
                    source_chunk_id = int(source_chunk_id)
                    best_chunk = top_docs[source_chunk_id] if source_chunk_id < len(top_docs) else None
                except ValueError:
                    best_chunk = None
            else:
                best_chunk = top_docs[source_chunk_id] if source_chunk_id < len(top_docs) else None
                
            result_out = {
                "question_id": q["id"],
                "question": q["question"],
                "gold_pageid": q["pageid"],
                "gold_title": q["title"],
                "new_date": q["new_date"],
                "old_date": q["old_date"],
                "predicted_answer": final_answer,
                "conflict_detected": conflict_info.get("conflict_detected", False),
                "conflict_formulation": conflict_info.get("conflict_formulation", ""),
                "outdated_hypothesis": conflict_info.get("outdated_hypothesis", ""),
                "preliminary_answer": conflict_info.get("preliminary_answer", ""),
                "source_chunk_id": source_chunk_id,
                "verification_decision": decision,
                "followup_question": verification.get("followup_question", ""),
                "followup_reasoning": verification.get("followup_reasoning", ""),
                "preliminary_chunk_id": conflict_info.get("source_chunk_id"),
                "best_chunk_pageid": best_chunk["pageid"] if best_chunk else None,
                "best_chunk_date": best_chunk["date"] if best_chunk else None,
                "correct_article_retrieved": any(d["pageid"] == q["pageid"] for d in docs),
                "top_chunks": [{"pageid": d["pageid"], "date": d["date"]} for d in top_docs],
                "retrieved_docs": [{"pageid": d["pageid"], "date": d["date"], "score": d["score"]} for d in docs],
            }

            out_f.write(json.dumps(result_out, ensure_ascii=False) + "\n")
            out_f.flush()
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    with open(OUTPUT_PATH, "r") as f:
        all_results = [json.loads(line) for line in f]

    total = len(all_results)
    retrieved = sum(1 for r in all_results if r["correct_article_retrieved"])
    conflicts = sum(1 for r in all_results if r["conflict_detected"])
    
    print(f"Total questions:           {total}")
    print(f"Correct article retrieved: {retrieved}/{total} ({100*retrieved/total:.1f}%)")
    print(f"Conflicts detected:        {conflicts}/{total} ({100*conflicts/total:.1f}%)")
    print(f"Total tokens used:         {total_tokens}")
    print(f"Threshold conflict detection: {TRESHOLD_CONFLICT_DETECTION}")
    print(f"Results saved to:          {OUTPUT_PATH}")
    

if __name__ == "__main__":
    main()