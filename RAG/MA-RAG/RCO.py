"""RCO.py
    R - Relevance Agent
    C - Conflict Resolution Agent
    O - Output
"""
import time
import json
import re
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from help_functions import (
    get_es_client, load_questions, search_documents,
    format_chunks, parse_scores, call_llm, load_version, increment_version
)

increment_version("Results", "RCO")
rco_results_version = load_version("Results", "RCO")
OUTPUT_PATH = f"../../results/rco_results_{rco_results_version}.jsonl"
SLEEP_BETWEEN_REQUESTS = 1.0
TOP_CANDIDATES = 3


def relevance_agent(query: str, chunks: list[str]) -> dict[int, float]:
    prompt = (
        f"You are a relevance ranking agent.\n"
        f"Given the question and a list of text chunks, rank each chunk by how "
        f"directly and completely it answers the question.\n\n"
        f"Question: {query}\n\n"
        f"Chunks:\n{format_chunks(chunks)}\n\n"
        f"For each chunk, output a relevance score from 0.0 to 1.0.\n"
        f"Think step by step before scoring.\n"
        f"Output ONLY valid JSON, no extra text:\n"
        f"[{{\"chunk_id\": 0, \"score\": 0.9, \"reasoning\": \"...\"}}]"
    )
    response = call_llm(prompt, model="llama3.3:70b")
    return parse_scores(response)


def conflict_resolution_agent(query: str, chunks: list[str]) -> dict:
    chunks_str = format_chunks(chunks)
    prompt = (
        f"You are given a question and multiple text chunks about the same topic. "
        f"Some chunks may contain conflicting or outdated information.\n\n"
        f"# Question\n{query}\n\n"
        f"# Text Chunks\n{chunks_str}\n\n"
        f"# Instructions\n"
        f"1. Identify whether the chunks contain conflicting information relevant to the question.\n"
        f"2. If there is a conflict, reason about which chunk is most likely to contain "
        f"the most updated information based on the content.\n"
        f"3. Answer the question based on the chunk you trust most.\n\n"
        f"# Requirements\n"
        f"- Please give a SHORT ANSWER. Use as few words as possible.\n"
        f"- If the answer is a number with more than 4 digits, use commas as thousand separators "
        f"(e.g., \"1,000\" instead of \"1000\").\n"
        f"- Don't include period at the end of the answer.\n"
        f"- If you are not sure about the answer, you MUST reply \"Unsure\" as answer.\n\n"
        f"Output ONLY valid JSON:\n"
        f"{{\"conflict_detected\": true, \"reasoning\": \"...\", \"answer\": \"...\", \"source_chunk_id\": 0}}"
    )
    response = call_llm(prompt, model="llama3.3:70b")

    cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
    try:
        parsed = json.loads(cleaned)
        return {
            "answer": parsed.get("answer", "Unsure"),
            "conflict_detected": parsed.get("conflict_detected", False),
            "reasoning": parsed.get("reasoning", ""),
            "source_chunk_id": parsed.get("source_chunk_id", None),
        }
    except (json.JSONDecodeError, KeyError):
        return {"answer": "Unsure", "conflict_detected": False, "reasoning": "", "source_chunk_id": None}


def main():
    es = get_es_client()
    questions = load_questions()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] {q['question'][:80]}...")

            docs = search_documents(es, q["question"])
            print(f"  Retrieved {len(docs)} documents")

            if not docs:
                print("  No documents retrieved, skipping.")
                continue

            chunks = [d["content"] for d in docs]

            relevance_scores = relevance_agent(q["question"], chunks)
            print(f"  Relevance scores: {relevance_scores}")

            if not relevance_scores:
                print("  Could not parse relevance scores, falling back to top 3.")
                top_indices = list(range(min(TOP_CANDIDATES, len(docs))))
            else:
                top_indices = sorted(
                    relevance_scores, key=relevance_scores.get, reverse=True
                )[:TOP_CANDIDATES]

            top_docs = [docs[i] for i in top_indices]
            print(f"Top chunk pageIDs: {[d['pageid'] for d in top_docs]}")

            result = conflict_resolution_agent(q["question"], [d["content"] for d in top_docs])
            answer = result["answer"]
            print(f"Conflict detected: {result['conflict_detected']}")
            print(f"Answer: '{answer}'")
            
            source_chunk_id = result.get("source_chunk_id")
            best_chunk = top_docs[source_chunk_id] if source_chunk_id is not None else None
            
            result_out = {
                "question_id": q["id"],
                "question": q["question"],
                "gold_pageid": q["pageid"],
                "gold_title": q["title"],
                "new_date": q["new_date"],
                "old_date": q["old_date"],
                "predicted_answer": answer,
                "conflict_detected": result["conflict_detected"],
                "conflict_reasoning": result["reasoning"],
                "source_chunk_id": source_chunk_id,
                "best_chunk_pageid": best_chunk["pageid"] if best_chunk else None,
                "best_chunk_date": best_chunk["date"] if best_chunk else None,
                "correct_article_retrieved": any(d["pageid"] == q["pageid"] for d in docs),
                "top_chunks": [
                    {"pageid": d["pageid"], "date": d["date"]}
                    for d in top_docs
                ],
                "retrieved_docs": [
                    {"pageid": d["pageid"], "date": d["date"], "score": d["score"]}
                    for d in docs
                ],
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
    print(f"Results saved to:          {OUTPUT_PATH}")
    

if __name__ == "__main__":
    main()