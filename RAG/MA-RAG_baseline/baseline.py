import csv
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import os
import sys
sys.path.append("../../statistics")

import requests
from elasticsearch import Elasticsearch
from helper_functions.helper_functions import increment_version

load_dotenv()  # Load environment variables from .env file

# --- Configuration ---
CSV_PATH = Path("../../data/hoh_question_pageid_map.csv")
OUTPUT_PATH = Path("../../data/rag_baseline_results_v3.jsonl")
VERSION_CONTROL_PATH = Path("../../graph_version_control.json")

ES_HOST = "http://localhost:9200"
INDEX_NAME = "wikipedia_snapshots"
TOP_K = 5

LLM_URL = "https://openwebui.ux.uis.no/api/chat/completions"
LLM_MODEL = "llama3.3:70b"
TEST = True
LLM_API_KEY = os.getenv("OLLAMA_KEY")
if not LLM_API_KEY:
    raise ValueError("LLM_API_KEY is not set. Please set the OLLAMA_KEY environment variable.")

SLEEP_BETWEEN_REQUESTS = 1.0  # seconds between LLM calls

def get_es_client() -> Elasticsearch:
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise ConnectionError("Cannot connect to Elasticsearch")
    return es


def search_documents(es: Elasticsearch, query: str, top_k: int = TOP_K) -> list[dict]:
    """Search Elasticsearch and return top-k documents."""
    response = es.search(
        index=INDEX_NAME,
        query={"match": {"content": query}},
        size=top_k,
    )

    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "score": hit["_score"],
            "title": hit["_source"]["title"],
            "date": hit["_source"]["date"],
            "pageid": hit["_source"]["pageid"],
            "content": hit["_source"]["content"],  
        })
    return results


def build_prompt(question: str, documents: list[dict]) -> str:
    """Build the RAG prompt with retrieved documents."""
    doc_texts = []
    for i, doc in enumerate(documents, 1):
        doc_texts.append(
            f"## Document {i}: {doc['title']} (Last Modified: {doc['date']})\n"
            f"{doc['content']}"
        )

    documents_str = "\n\n".join(doc_texts)

    prompt = (
        f"Given a question and some relevant documents, generate a SHORT ANSWER "
        f"to the question based on the documents.\n\n"
        f"# Question\n{question}\n\n"
        f"# Documents\n{documents_str}\n\n"
        f"# Requirements\n"
        f"- Please give a SHORT ANSWER. Use as few words as possible.\n"
        f"- If your answer is a number under 10, always type it as a word (e.g., \"five\" instead of \"5\").\n"
        f"- If the answer is a number with more than 4 digits, use commas as thousand separators (e.g., \"1,000\" instead of \"1000\").\n"
        f"- Don't include period at the end of the answer.\n"
        f"- If you are not sure about the answer, you MUST reply \"Unsure\".\n"
        f"- Your answer should be based on the most up-to-date information available "
        f"in the documents.\n\n"
        f"# Answer"
    )
    return prompt


def query_llm(prompt: str) -> str:
    """Send prompt to LLM and return the response text."""
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    response = requests.post(LLM_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def load_questions(csv_path: Path) -> list[dict]:
    """Load questions from the CSV file."""
    questions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 6:
                continue
            questions.append({
                "id": row[0].strip(),
                "question": row[1].strip(),
                "pageid": row[2].strip(),
                "title": row[3].strip(),
                "new_date": row[4].strip().split("T")[0],
                "old_date": row[5].strip().split("T")[0],
            })
    return questions


def evaluate_answer(predicted: str, gold_title: str, gold_pageid: str,
                    retrieved_docs: list[dict]) -> dict:
    """
    Basic evaluation:
    - Did retrieval find the correct article?
    - Which version (new/old) was ranked highest?
    """
    correct_retrieved = False
    correct_rank = -1

    for i, doc in enumerate(retrieved_docs):
        if doc["pageid"] == gold_pageid:
            if not correct_retrieved:
                correct_rank = i + 1
            correct_retrieved = True

    return {
        "correct_article_retrieved": correct_retrieved,
        "correct_article_rank": correct_rank,
    }


def main():
    if TEST:
        answer = query_llm("hei")
        print(f"Test LLM response: {answer}")
    
    es = get_es_client()
    print(f"Connected to Elasticsearch")

    questions = load_questions(CSV_PATH)
    print(f"Loaded {len(questions)} questions")

    results = []

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions):
            print(f"\n[{i+1}/{len(questions)}] {q['question'][:80]}...")

            # 1. Retrieve
            docs = search_documents(es, q["question"])
            print(f"  Retrieved {len(docs)} documents")

            # 2. Generate
            prompt = build_prompt(q["question"], docs)
            try:
                answer = query_llm(prompt)
            except Exception as e:
                answer = f"[ERROR] {e}"
                print(f"  LLM error: {e}")

            print(f"  Answer: {answer[:100]}")

            # 3. Evaluate retrieval
            eval_result = evaluate_answer(answer, q["title"], q["pageid"], docs)
            print(f"  Correct article retrieved: {eval_result['correct_article_retrieved']}"
                  f" (rank: {eval_result['correct_article_rank']})")

            # 4. Log result
            result = {
                "question_id": q["id"],
                "question": q["question"],
                "gold_pageid": q["pageid"],
                "gold_title": q["title"],
                "new_date": q["new_date"],
                "old_date": q["old_date"],
                "predicted_answer": answer,
                "retrieved_docs": [
                    {"pageid": d["pageid"], "title": d["title"],
                     "date": d["date"], "score": d["score"]}
                    for d in docs
                ],
                **eval_result,
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
    print(f"Total questions: {total}")
    print(f"Correct article retrieved: {retrieved}/{total} ({100*retrieved/total:.1f}%)")
    print(f"Results saved to: {OUTPUT_PATH}")
    
    increment_version("Results", "Baseline")


if __name__ == "__main__":
    main()