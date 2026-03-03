from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import os

load_dotenv()

CSV_PATH = Path("data/hoh_question_pageid_map.csv")
KB_NEW_DIR = Path("data/KB_chunked_new")
KB_OLD_DIR = Path("data/KB_chunked_outdated")

OUT_CSV = Path("questions_with_retrieval.csv")

OLLAMA_API = os.getenv("OLLAMA_API")

TOP_K = 1

TOKEN_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]

@dataclass
class Doc:
    chunk_id: str
    text: str

class BM25:
    def __init__(self, docs: List[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b

        self.doc_tokens: List[List[str]] = [tokenize(d.text) for d in docs]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))

        # df / idf
        df: Dict[str, int] = {}
        for toks in self.doc_tokens:
            seen = set(toks)
            for term in seen:
                df[term] = df.get(term, 0) + 1

        N = len(self.docs)
        self.idf: Dict[str, float] = {}
        for term, f in df.items():
            self.idf[term] = math.log(1 + (N - f + 0.5) / (f + 0.5))

        self.tf: List[Dict[str, int]] = []
        for toks in self.doc_tokens:
            counts: Dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)

    def score(self, query: str) -> List[float]:
        q = tokenize(query)
        scores = [0.0] * len(self.docs)

        for i, tf_doc in enumerate(self.tf):
            dl = self.doc_lens[i]
            norm = self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 1.0))
            s = 0.0
            for term in q:
                if term not in tf_doc:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf_doc[term]
                s += idf * (freq * (self.k1 + 1)) / (freq + norm)
            scores[i] = s

        return scores

def parse_iso_to_ym(iso: str) -> str:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return f"{dt.year:04d}-{dt.month:02d}"

def find_doc_jsonl(kb_dir: Path, ym: str, pageid: str) -> Optional[Path]:
    """
    Finner fil basert på:
      - prefix YYYY-MM_
      - inneholder _<pageid>_
    """
    pattern = f"{ym}_*_{pageid}_*.jsonl"
    hits = sorted(kb_dir.glob(pattern))
    if hits:
        return hits[0]

    pattern2 = f"*_{pageid}_*.jsonl"
    hits2 = sorted(kb_dir.glob(pattern2))
    for h in hits2:
        if h.name.startswith(ym + "_"):
            return h
    return hits2[0] if hits2 else None

def load_chunks(jsonl_path: Path) -> List[Doc]:
    docs: List[Doc] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(Doc(
                chunk_id=row.get("chunk_id") or f"{row.get('doc_id','doc')}_{row.get('chunk_index',0)}",
                text=row.get("text", "")
            ))
    return docs

def top_k_chunks(question: str, jsonl_path: Path, k: int) -> List[Tuple[float, Doc]]:
    docs = load_chunks(jsonl_path)
    if not docs:
        return []
    bm25 = BM25(docs)
    scores = bm25.score(question)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    # filtrer bort 0-score hvis du vil
    return ranked[:k]


def format_hits(hits: List[Tuple[float, Doc]]) -> str:
    """
    Legger top-k i én streng (til CSV).
    """
    parts = []
    for score, doc in hits:
        txt = doc.text.replace("\n", " ").strip()
        if len(txt) > 800:
            txt = txt[:800] + "..."
        parts.append(f"[{doc.chunk_id} | {score:.3f}] {txt}")
    return " || ".join(parts)


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Finner ikke CSV: {CSV_PATH}")
    for d in (KB_NEW_DIR, KB_OLD_DIR):
        if not d.exists():
            raise SystemExit(f"Finner ikke mappe: {d}")

    rows_out: List[Dict[str, str]] = []

    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 10:
                print("hard stop after 10 QA")
                break
            qid = row.get("qid", "")
            question = row.get("question", "") or ""
            pageid = row.get("pageid", "") or ""
            current_time = row.get("current_time", "") or ""
            outdated_time = row.get("outdated_info_dates", "") or ""

            ym_new = parse_iso_to_ym(current_time) if current_time else ""
            ym_old = parse_iso_to_ym(outdated_time) if outdated_time else ""

            new_file = find_doc_jsonl(KB_NEW_DIR, ym_new, pageid) if ym_new else None
            old_file = find_doc_jsonl(KB_OLD_DIR, ym_old, pageid) if ym_old else None

            new_hits = top_k_chunks(question, new_file, TOP_K) if new_file else []
            old_hits = top_k_chunks(question, old_file, TOP_K) if old_file else []

            out = dict(row)  # behold originalkolonner
            out["new_chunks_file"] = str(new_file) if new_file else ""
            out["outdated_chunks_file"] = str(old_file) if old_file else ""
            out["new_top_chunks"] = format_hits(new_hits)
            out["outdated_top_chunks"] = format_hits(old_hits)

            rows_out.append(out)

            print(f"[Q{qid}] pageid={pageid} ym_new={ym_new} ym_old={ym_old} "
                  f"new_hits={len(new_hits)} old_hits={len(old_hits)}")

    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n[DONE] Skrev: {OUT_CSV}")


if __name__ == "__main__":
    main()