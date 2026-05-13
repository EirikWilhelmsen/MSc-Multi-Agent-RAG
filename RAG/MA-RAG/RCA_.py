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

from help_functions import (
    get_es_client, increment_version, load_questions, search_documents,
    format_chunks, parse_scores, call_llm, load_version
)
from help.aggregations import aggregate

increment_version("Results", "RCA")
rca_results_version = load_version("Results", "RCA")
OUTPUT_PATH = f"../../results/rca_results_{rca_results_version}.jsonl"
SLEEP_BETWEEN_REQUESTS = 1.0
TOP_CANDIDATES = 3
AGGREGATION_METHOD = "Majority Vote"
#AGGREGATION_METHOD = "Confidence"
#AGGREGATION_METHOD = "Random"


def relevance_agent(query: str, chunks: list[str]) -> tuple[str, float]:
    prompt = f"""You are a relevance ranking agent.
        Given the question and a list of text chunks, rank each chunk by how
        directly and completely it answers the question.
        
        Question: {query}
        
        Chunks:
        {format_chunks(chunks)}
        
        For each chunk, output a relevance score from 0.0 to 1.0.
        Think step by step before scoring.
        Output ONLY valid JSON, no extra text:
        [{{"chunk_id": 0, "score": 0.9, "reasoning": "..."}}, ...]"""

    response, tokens = call_llm(prompt, model="gorina10.llama3.3:70b")
    return parse_scores(response), tokens

def temporal_agent(query: str, chunks: list[str]) -> tuple[dict[int, float], int]:
    prompt = f"""You are a temporal reasoning agent.
        Given a question and text chunks that may contain conflicting information,
        estimate which chunk contains the most RECENT information.
        
        Reason about the content of the text itself, including:
        - Explicit dates or years mentioned in the text
        - Language cues suggesting updates or changes ("now", "currently", 
          "recently", "replaced", "formerly", "previously")
        - Whether one chunk's facts imply the other is outdated
        
        Question: {query}
        
        Chunks:
        {format_chunks(chunks)}
        
        You MUST score ALL {len(chunks)} chunks (chunk_id 0 through {len(chunks)-1}).
        
        Think step by step. Output ONLY valid JSON, no extra text:
        [{{"chunk_id": 0, "recency_score": 0.8, "reasoning": "..."}}, ...]"""

    response, tokens = call_llm(prompt, model="gorina10.llama3.3:70b")
    return parse_scores(response), tokens


def candidate_agent(query: str, chunk: str, chunk_id: int) -> dict:
    prompt = (
        f"Given a question and a text chunk, answer the question based only "
        f"on the text. Then rate your confidence.\n\n"
        f"# Question\n{query}\n\n"
        f"# Text\n{chunk}\n\n"
        f"# Requirements\n"
        f"- Please give a SHORT ANSWER. Use as few words as possible.\n"
        f"- If the answer is a number with more than 4 digits, use commas as thousand separators.\n"
        f"- Don't include period at the end of the answer.\n"
        f"- If you are not sure about the answer, you MUST reply \"Unsure\" as answer.\n\n"
        f"Output ONLY valid JSON:\n"
        f"{{\"answer\": \"...\", \"confidence\": 0.85, \"reasoning\": \"...\"}}"
    )
    response, tokens = call_llm(prompt, model="gorina10.llama3.3:70b")
    

    cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
    try:
        parsed = json.loads(cleaned)
        return {
            "chunk_id": chunk_id,
            "answer": parsed.get("answer", "Unsure"),
            "confidence": float(parsed.get("confidence", 0.0)),
            "reasoning": parsed.get("reasoning", ""),
            "tokens": tokens
        }
    except (json.JSONDecodeError, KeyError):
        return {"chunk_id": chunk_id, "answer": "Unsure", "confidence": 0.0, "reasoning": "", "tokens": tokens}


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

            # 2. Relevance Agent → top 3
            relevance_scores, tokens = relevance_agent(q["question"], [d["content"] for d in docs])
            total_tokens += tokens

            top_indices = sorted(relevance_scores, key=relevance_scores.get, reverse=True)[:TOP_CANDIDATES]
            top_docs = [docs[i] for i in top_indices]
            print(f"Top chunks: {[docs[i]['pageid'] for i in top_indices]}")
            #print("token consumption after relevance", total_tokens)
            # 3. Candidate Agents
            candidates = run_candidate_agents(q["question"], top_docs)
            #tokens
            for c in candidates:
                total_tokens += c["tokens"]
                print(f"Chunk {c['chunk_id']}: answer='{c['answer']}', confidence={c['confidence']:.2f}, tokens={c['tokens']}")
            #print("token consumption after candidates", total_tokens)

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
    print(f"sequencer = False")
    print(f"Total tokens used: {total_tokens}")
    print(f"method used for aggregation: {AGGREGATION_METHOD}")
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()