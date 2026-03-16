#!/usr/bin/env python3
"""
Indexes cleaned Wikipedia snapshots into Elasticsearch with chunking.

Reads the sampled CSV file to identify which articles and dates to index.
Each article is split into chunks of ~256 tokens with 32-token overlap.
Each chunk is indexed as a separate document.

Usage:
    python index_elasticsearch.py
"""

import csv
import glob
from pathlib import Path
from elasticsearch import Elasticsearch

# --- Configuration ---
CSV_PATH = Path("../hoh_question_pageid_map.csv")
KB_CLEANED_ROOT = Path("../KB_cleaned")
ES_HOST = "http://localhost:9200"
INDEX_NAME = "wikipedia_snapshots"

# Chunking settings (matching HoH defaults)
CHUNK_SIZE = 256      # tokens per chunk
CHUNK_OVERLAP = 32    # token overlap between chunks


def get_es_client() -> Elasticsearch:
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        raise ConnectionError("Cannot connect to Elasticsearch at " + ES_HOST)
    print(f"Connected to Elasticsearch at {ES_HOST}")
    return es


def create_index(es: Elasticsearch) -> None:
    """Create the index with mapping. Deletes if it already exists."""
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index: {INDEX_NAME}")

    mapping = {
        "mappings": {
            "properties": {
                "pageid": {"type": "keyword"},
                "title": {"type": "text"},
                "date": {"type": "date", "format": "yyyy-MM-dd"},
                "content": {"type": "text"},
                "chunk_index": {"type": "integer"},
                "filename": {"type": "keyword"},
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Created index: {INDEX_NAME}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks of approximately chunk_size tokens
    with overlap tokens between consecutive chunks.
    Tokenization is whitespace-based.
    """
    tokens = text.split()

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))

        if end >= len(tokens):
            break

        start = end - overlap

    return chunks


def find_wikitext_file(pageid: str, date_str: str) -> Path | None:
    """Find the cleaned wikitext file for a given pageid and date."""
    date_dir = KB_CLEANED_ROOT / date_str
    if not date_dir.exists():
        return None

    pattern = str(date_dir / f"{pageid}_*.wikitext.txt")
    matches = glob.glob(pattern)

    if not matches:
        return None
    return Path(matches[0])


def parse_date(date_str: str) -> str:
    """Convert '2024-07-01T00:00:00Z' to '2024-07-01'."""
    return date_str.split("T")[0]


def collect_and_chunk_documents(csv_path: Path) -> list[dict]:
    """
    Read the CSV, collect unique (pageid, date) pairs,
    chunk each article, and return a list of chunk documents.
    """
    seen = set()
    documents = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 6:
                continue

            _, question, pageid, title, new_date_raw, old_date_raw = row[:6]
            new_date = parse_date(new_date_raw.strip())
            old_date = parse_date(old_date_raw.strip())

            for date in [new_date, old_date]:
                key = (pageid.strip(), date)
                if key in seen:
                    continue
                seen.add(key)

                filepath = find_wikitext_file(pageid.strip(), date)
                if filepath is None:
                    print(f"[WARN] File not found: pageid={pageid}, date={date}")
                    continue

                content = filepath.read_text(encoding="utf-8", errors="replace")
                if not content.strip():
                    print(f"[WARN] Empty file: {filepath}")
                    continue

                chunks = chunk_text(content)

                for chunk_idx, chunk in enumerate(chunks):
                    documents.append({
                        "pageid": pageid.strip(),
                        "title": title.strip(),
                        "date": date,
                        "content": chunk,
                        "chunk_index": chunk_idx,
                        "filename": filepath.name,
                    })

    return documents


def index_documents(es: Elasticsearch, documents: list[dict]) -> None:
    """Index all chunk documents into Elasticsearch."""
    success = 0
    failed = 0

    for doc in documents:
        doc_id = f"{doc['pageid']}_{doc['date']}_chunk{doc['chunk_index']}"
        try:
            es.index(index=INDEX_NAME, id=doc_id, document=doc)
            success += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] Failed to index {doc_id}: {e}")

    print(f"\nIndexing complete: {success} succeeded, {failed} failed")


def main():
    es = get_es_client()
    create_index(es)

    print(f"\nCollecting and chunking documents from {CSV_PATH}...")
    documents = collect_and_chunk_documents(CSV_PATH)

    unique_articles = len({(d["pageid"], d["date"]) for d in documents})
    print(f"Found {unique_articles} unique articles -> {len(documents)} chunks")
    print(f"Average chunks per article: {len(documents)/unique_articles:.1f}")

    print(f"\nIndexing into '{INDEX_NAME}'...")
    index_documents(es, documents)

    es.indices.refresh(index=INDEX_NAME)

    count = es.count(index=INDEX_NAME)["count"]
    print(f"\nTotal chunks in index: {count}")


if __name__ == "__main__":
    main()