from pathlib import Path
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from tqdm import tqdm

# --- Configuration ---
KB_CLEANED_ROOT = Path("../KB_cleaned")
ES_HOST = "http://localhost:9200"
INDEX_NAME = "wikipedia_snapshots"

# Chunking settings (matching HoH defaults)
CHUNK_SIZE = 256      # tokens per chunk
CHUNK_OVERLAP = 32    # token overlap between chunks

# Bulk indexing batch size
BULK_BATCH_SIZE = 500


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
                "pageid":       {"type": "keyword"},
                "date":         {"type": "date", "format": "yyyy-MM-dd"},
                "content":      {"type": "text"},
                "chunk_index":  {"type": "integer"},
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Created index: {INDEX_NAME}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
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
        chunks.append(" ".join(tokens[start:end]))

        if end >= len(tokens):
            break

        start = end - overlap

    return chunks


def generate_actions(kb_root: Path):
    """
    Generator that yields bulk index actions for all chunks across all date folders.
    Filename format: {pageid}_{revisionid}.wikitext.txt
    """
    date_dirs = sorted(
        d for d in kb_root.iterdir()
        if d.is_dir() and d.name.startswith("2024-")
    )
    for date_dir in date_dirs:
        date_str = date_dir.name  # e.g. "2024-06-01"
        files = list(date_dir.glob("*.wikitext.txt"))

        for filepath in tqdm(files, desc=date_str, unit="articles"):
            pageid = filepath.name.split("_")[0]

            content = filepath.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                continue

            for chunk_idx, chunk in enumerate(chunk_text(content)):
                doc_id = f"{pageid}_{date_str}_chunk{chunk_idx}"
                yield {
                    "_index": INDEX_NAME,
                    "_id": doc_id,
                    "_source": {
                        "pageid":      pageid,
                        "date":        date_str,
                        "content":     chunk,
                        "chunk_index": chunk_idx,
                    }
                }


def main():
    es = get_es_client()
    create_index(es)

    print(f"\nIndexing all documents from {KB_CLEANED_ROOT} into '{INDEX_NAME}'...")

    success, failed = 0, 0
    for ok, info in helpers.streaming_bulk(
        es,
        generate_actions(KB_CLEANED_ROOT),
        chunk_size=BULK_BATCH_SIZE,
        raise_on_error=False,
    ):
        if ok:
            success += 1
        else:
            failed += 1
            print(f"[ERROR] {info}")

    es.indices.refresh(index=INDEX_NAME)

    count = es.count(index=INDEX_NAME)["count"]
    print(f"\nIndexing complete: {success} succeeded, {failed} failed")
    print(f"Total chunks in index: {count}")


if __name__ == "__main__":
    main()