#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

IN_ROOT = Path("KB_cleaned")

OUT_ROOT = Path("KB_chunked")

RECURSIVE = True

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def iter_files(root: Path):
    pattern = "**/*.wikitext.txt" if RECURSIVE else "*.wikitext.txt"
    yield from root.glob(pattern)


def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def parse_ids_from_stem(fp: Path):
    """
    Forventet fil: <pageid>_<revid>.wikitext.txt
    fp.stem blir: <pageid>_<revid>.wikitext
    """
    stem = fp.stem
    if stem.endswith(".wikitext"):
        stem = stem[: -len(".wikitext")]

    parts = stem.split("_")
    pageid = parts[0] if len(parts) >= 2 else ""
    revid = parts[1] if len(parts) >= 2 else ""
    doc_id = f"{pageid}_{revid}" if pageid and revid else stem
    return doc_id, pageid, revid


def out_path_for(in_fp: Path) -> Path:
    """
    Behold samme relative path:
      data/KB_cleaned/2024-07-01/1000005_123.wikitext.txt
    ->data/KB_chunked/2024-07-01/1000005_123.jsonl
    """
    rel = in_fp.relative_to(IN_ROOT)
    # bytt suffix .txt -> .jsonl, men behold resten
    # .../xxx.wikitext.txt -> stem=xxx.wikitext -> vi vil xxx.jsonl
    out_dir = OUT_ROOT / rel.parent
    stem = in_fp.stem
    if stem.endswith(".wikitext"):
        stem = stem[: -len(".wikitext")]
    return out_dir / f"{stem}.jsonl"


def process_all():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    n_files = 0
    n_chunks = 0

    for fp in iter_files(IN_ROOT):
        text = fp.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(text)

        out_file = out_path_for(fp)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # snapshot date = første mappen under IN_ROOT (typisk YYYY-MM-DD)
        rel = fp.relative_to(IN_ROOT)
        snapshot_date = rel.parts[0] if len(rel.parts) >= 2 else ""

        doc_id, pageid, revid = parse_ids_from_stem(fp)

        with out_file.open("w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                row = {
                    "snapshot_date": snapshot_date,
                    "doc_id": doc_id,
                    "pageid": pageid,
                    "revid": revid,
                    "chunk_id": f"{doc_id}_{i}",
                    "chunk_index": i,
                    "text": chunk,
                    "length": len(chunk),
                    "source_path": str(rel),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        n_files += 1
        n_chunks += len(chunks)
        print(f"[OK] {fp.name} → {out_file.relative_to(OUT_ROOT)} ({len(chunks)} chunks)")

    print("\n[DONE]")
    print(f"Files processed: {n_files}")
    print(f"Chunks written:  {n_chunks}")
    print(f"Output root:     {OUT_ROOT}")


def main():
    if not IN_ROOT.exists() or not IN_ROOT.is_dir():
        raise SystemExit(f"Input folder missing: {IN_ROOT}")

    process_all()


if __name__ == "__main__":
    main()