from elasticsearch import Elasticsearch
from pathlib import Path
from dotenv import load_dotenv
import os
import csv
import json
import re
import requests
import time

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


def load_questions(csv_path="../../data/500Q/500_hoh_questions.csv") -> list[dict]:
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
                "question": row[3].strip(),
                "pageid": row[1].strip(),
                "title": row[2].strip(),
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


def call_llm(prompt: str, model: str, max_retries: int = 5, backoff: float = 5.0) -> tuple[str, int]:
    """Send prompt to LLM and return the response text and token count.
    
    Retries on 5xx server errors with exponential backoff.
    """
    if LLM_API_KEY is None:
        raise ValueError("LLM_API_KEY is not set in environment variables.")
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(LLM_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            tokens = data["usage"]["total_tokens"]
            return data["choices"][0]["message"]["content"].strip(), tokens

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and 500 <= status < 600 and attempt < max_retries:
                wait = backoff * attempt
                print(f"  [LLM {status}] retry {attempt}/{max_retries - 1} in {wait:.1f}s")
                time.sleep(wait)
                continue
            print(e.response.text if e.response is not None else str(e))
            raise

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            if attempt < max_retries:
                wait = backoff * attempt
                print(f"  [LLM network error] retry {attempt}/{max_retries - 1} in {wait:.1f}s: {e}")
                time.sleep(wait)
                continue
            raise

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
        
def normalize_answer(answer: str) -> str:
    """Lowercase, strip whitespace og trailing skilletegn for sammenligning."""
    return re.sub(r"[.\s,;:!?]+$", "", answer.strip().lower())

def has_consensus(candidates: list[dict]) -> bool:
    """Konsensus blant kandidater med faktiske svar."""
    real_answers = {normalize_answer(c["answer"]) for c in candidates
                    if normalize_answer(c["answer"]) != "unsure"}
    return len(real_answers) <= 1

def has_stabilized(prev: list[dict], curr: list[dict]) -> bool:
    """Stable kun hvis alle kandidater med faktisk svar holder fast på samme svar."""
    prev_map = {c["chunk_id"]: normalize_answer(c["answer"]) for c in prev
                if normalize_answer(c["answer"]) != "unsure"}
    curr_map = {c["chunk_id"]: normalize_answer(c["answer"]) for c in curr
                if normalize_answer(c["answer"]) != "unsure"}

    if not curr_map:
        return False

    common = prev_map.keys() & curr_map.keys()
    if not common:
        return False

    return all(prev_map[cid] == curr_map[cid] for cid in common)

def all_unsure(candidates: list[dict]) -> bool:
    return all(normalize_answer(c["answer"]) == "unsure" for c in candidates)