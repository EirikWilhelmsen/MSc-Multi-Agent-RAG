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
import re
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from help_functions import (
    get_es_client, increment_version, load_questions, search_documents,
    format_chunks, parse_scores, call_llm, load_version
)


increment_version("Results", "RCDS")
rcds_results_version = load_version("Results", "RCDS")
OUTPUT_PATH = f"../../results/rcds_results_{rcds_results_version}.jsonl"
SLEEP_BETWEEN_REQUESTS = 1.0
TOP_CANDIDATES = 3
DEBATE_ROUNDS = 2


# ── 1. Relevance Agent (same as RCA) ─────────────────────────────────

def relevance_agent(query: str, chunks: list[str]) -> dict[int, float]:
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


# ── 2. Candidate Agent (initial answer) ──────────────────────────────

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
    results.sort(key=lambda x: x["chunk_id"])
    return results


# ── 3. Debate Round ──────────────────────────────────────────────────

def debate_round(query: str, candidates: list[dict], top_chunks: list[dict],
                 round_num: int, debate_history: list[list[dict]]) -> list[dict]:
    """Each candidate sees the others' answers and argues for their own."""

    updated = []

    for c in candidates:
        others = [o for o in candidates if o["chunk_id"] != c["chunk_id"]]
        others_text = "\n".join(
            f"- Candidate {o['chunk_id']}: \"{o['answer']}\" "
            f"(confidence {o['confidence']:.2f}, reasoning: {o['reasoning']})"
            for o in others
        )

        prev_rounds_text = ""
        if debate_history:
            for r_i, r_entries in enumerate(debate_history):
                prev_rounds_text += f"\n--- Previous debate round {r_i+1} ---\n"
                for entry in r_entries:
                    prev_rounds_text += (
                        f"Candidate {entry['chunk_id']}: \"{entry['answer']}\" "
                        f"(confidence {entry['confidence']:.2f}) — {entry['reasoning']}\n"
                    )

        chunk_text = top_chunks[c["chunk_id"]]["content"]

        prompt = (
            f"You are Candidate {c['chunk_id']} in a debate about the correct answer "
            f"to a question. You have access to your own source text.\n\n"
            f"# Question\n{query}\n\n"
            f"# Your source text\n{chunk_text}\n\n"
            f"# Your current answer\n\"{c['answer']}\" (confidence {c['confidence']:.2f})\n"
            f"Your reasoning: {c['reasoning']}\n\n"
            f"# Other candidates' answers\n{others_text}\n"
            f"{prev_rounds_text}\n"
            f"# Debate round {round_num}\n"
            f"Consider the other candidates' arguments carefully. They may have "
            f"more recent or more accurate information. You may update your answer "
            f"and confidence, or defend your current position with stronger arguments.\n\n"
            f"Think about:\n"
            f"- Does your source text contain more specific or more recent information?\n"
            f"- Are the other candidates' answers contradicting yours? If so, why might "
            f"your source be more reliable?\n"
            f"- If another candidate has a more convincing argument, you may change your answer.\n\n"
            f"# Requirements\n"
            f"- SHORT ANSWER. Use as few words as possible.\n"
            f"- If the answer is a number with more than 4 digits, use commas as thousand separators.\n"
            f"- Don't include period at the end of the answer.\n\n"
            f"Output ONLY valid JSON:\n"
            f"{{\"answer\": \"...\", \"confidence\": 0.85, \"reasoning\": \"...\"}}"
        )

        response, tokens = call_llm(prompt, model="gorina10.llama3.3:70b")
        cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
        try:
            parsed = json.loads(cleaned)
            updated.append({
                "chunk_id": c["chunk_id"],
                "answer": parsed.get("answer", c["answer"]),
                "confidence": float(parsed.get("confidence", c["confidence"])),
                "reasoning": parsed.get("reasoning", c["reasoning"]),
                "tokens": tokens
            })
        except (json.JSONDecodeError, KeyError):
            updated.append(c)  # keep previous if parsing fails

    return updated


# ── 4. Supervisor Agent ──────────────────────────────────────────────

def supervisor_agent(query: str, candidates: list[dict],
                     top_chunks: list[dict], debate_history: list[list[dict]]) -> dict:
    """Reviews the full debate and picks the best answer."""

    debate_log = ""
    # Initial positions
    debate_log += "=== Initial positions ===\n"
    for c in debate_history[0]:
        debate_log += (
            f"Candidate {c['chunk_id']}: "
            f"\"{c['answer']}\" (confidence {c['confidence']:.2f}) — {c['reasoning']}\n"
        )

    # Debate rounds
    for r_i, r_entries in enumerate(debate_history[1:], start=1):
        debate_log += f"\n=== Debate round {r_i} ===\n"
        for entry in r_entries:
            debate_log += (
                f"Candidate {entry['chunk_id']}: \"{entry['answer']}\" "
                f"(confidence {entry['confidence']:.2f}) — {entry['reasoning']}\n"
            )

    # Final positions
    debate_log += "\n=== Final positions ===\n"
    for c in candidates:
        debate_log += (
            f"Candidate {c['chunk_id']}: "
            f"\"{c['answer']}\" (confidence {c['confidence']:.2f}) — {c['reasoning']}\n"
        )

    prompt = (
        f"You are a supervisor agent. Your job is to review a debate between "
        f"candidates who each answered a question based on different source texts.\n\n"
        f"# Question\n{query}\n\n"
        f"# Debate transcript\n{debate_log}\n"
        f"# Your task\n"
        f"Select the best answer. Consider:\n"
        f"1. Quality of reasoning and evidence from the source text\n"
        f"2. How well the candidate defended their position during debate\n"
        f"3. Confidence levels and whether they were justified\n\n"
        f"# Requirements\n"
        f"- Pick the winning candidate's answer.\n"
        f"- SHORT ANSWER. Use as few words as possible.\n"
        f"- If the answer is a number with more than 4 digits, use commas as thousand separators.\n"
        f"- Don't include period at the end of the answer.\n\n"
        f"Output ONLY valid JSON:\n"
        f"{{\"winner_chunk_id\": 0, \"answer\": \"...\", \"reasoning\": \"...\"}}"
    )

    response, tokens = call_llm(prompt, model="gorina10.llama3.3:70b")
    cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
    try:
        parsed = json.loads(cleaned)
        winner_id = int(parsed.get("winner_chunk_id", 0))
        return {
            "winner_chunk_id": winner_id,
            "answer": parsed.get("answer", candidates[0]["answer"]),
            "reasoning": parsed.get("reasoning", ""),
            "tokens": tokens
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: pick highest confidence
        best = max(candidates, key=lambda x: x["confidence"])
        return {
            "winner_chunk_id": best["chunk_id"],
            "answer": best["answer"],
            "reasoning": "Fallback: highest confidence",
            "tokens": tokens
        }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    es = get_es_client()
    questions = load_questions()
    total_tokens = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions):
            if i < 290:
                continue
            print(f"\n[{i+1}/{len(questions)}] {q['question'][:80]}...")

            # 1. Retrieve
            docs = search_documents(es, q["question"])
            if not docs:
                continue

            # 2. Relevance Agent → top 3
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