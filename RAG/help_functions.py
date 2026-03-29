from elasticsearch import Elasticsearch
from pathlib import Path
from dotenv import load_dotenv
import os
import csv
import json
import re
import requests

load_dotenv()

ES_HOST = "http://localhost:9200"
LLM_API_KEY = os.getenv("OLLAMA_API")
LLM_URL = "https://openwebui.ux.uis.no/api/chat/completions"
INDEX_NAME = "wikipedia_snapshots"
TOP_K = 5


def get_es_client() -> Elasticsearch:
    """Connect to Elasticsearch and return the client."""
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise ConnectionError("Cannot connect to Elasticsearch")
    return es


def load_questions(csv_path="../../data/hoh_question_pageid_map.csv") -> list[dict]:
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
            "date": hit["_source"]["date"],
            "pageid": hit["_source"]["pageid"],
            "content": hit["_source"]["content"],
        })
    return results


def format_chunks(chunks: list[str]) -> str:
    """Format a list of chunks for inclusion in a prompt."""
    return "\n\n".join(
        f"[Chunk {i}]\n{chunk}" for i, chunk in enumerate(chunks)
    )


def parse_scores(response: str) -> dict[int, float]:
    """
    Parse LLM JSON response into a dict of {chunk_id: score}.
    Handles both 'score' and 'recency_score' keys.
    Falls back to regex if JSON parsing fails.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?|```", "", response).strip()

    try:
        parsed = json.loads(cleaned)
        scores = {}
        for item in parsed:
            chunk_id = int(item["chunk_id"])
            score = float(item.get("score", item.get("recency_score", 0.5)))
            scores[chunk_id] = score
        return scores
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: regex extraction
        scores = {}
        for match in re.finditer(r'"chunk_id"\s*:\s*(\d+).*?"(?:score|recency_score)"\s*:\s*([\d.]+)', cleaned):
            scores[int(match.group(1))] = float(match.group(2))
        if not scores:
            print(f"  [WARN] Could not parse scores from response: {response[:200]}")
        return scores


def call_llm(prompt: str, model: str) -> str:
    """Send prompt to LLM and return the response text."""
    if LLM_API_KEY is None:
        raise ValueError("LLM_API_KEY is not set in environment variables.")
    print("model", model)
    print(LLM_API_KEY[:10])
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    
    response = requests.post(LLM_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def load_version(type: str, model: str, path="../../graph_version_control.json") -> str:
    with open(path, "r") as f:
        version_control = json.load(f)
    return version_control[model][type]


def increment_version(type: str, model: str, path="../../graph_version_control.json"):
    with open(path, "r") as f:
        version_control = json.load(f)
    current = int(version_control[model][type][1:])
    version_control[model][type] = f"v{current + 1}"
    with open(path, "w") as f:
        json.dump(version_control, f, indent=4)