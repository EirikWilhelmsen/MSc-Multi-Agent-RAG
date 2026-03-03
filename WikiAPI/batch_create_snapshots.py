#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
import time
from pathlib import Path

from wiki_snapshots_by_pageID_v2 import fetch_snapshot, save_snapshot  # juster hvis nødvendig

DOC_TIMES_PATH = Path("../data/doc_times.json")  # pageid -> ["YYYY-MM-DD", ...]

OUT_ROOT = Path("../data/KB_raw")
CACHE_DIR = OUT_ROOT / "_cache"

SLEEP_SECONDS = 0.1
MAX_REDIRECT_HOPS = 5
USE_HARDLINKS = True


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def base_name(pageid: int, revid: int) -> str:
    return f"{pageid}_{revid}"


def safe_exists(out_dir: Path, base: str) -> bool:
    return (out_dir / f"{base}.wikitext.txt").exists() and (out_dir / f"{base}.json").exists()


def _link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return

    if USE_HARDLINKS:
        try:
            os.link(src, dst)
            return
        except Exception:
            pass

    shutil.copy2(src, dst)


def materialize_from_cache(cache_base: str, out_dir: Path, out_base: str) -> None:
    src_txt = CACHE_DIR / f"{cache_base}.wikitext.txt"
    src_js = CACHE_DIR / f"{cache_base}.json"

    if not src_txt.exists() or not src_js.exists():
        raise RuntimeError(f"Cache missing for {cache_base}")

    dst_txt = out_dir / f"{out_base}.wikitext.txt"
    dst_js = out_dir / f"{out_base}.json"

    _link_or_copy(src_txt, dst_txt)
    _link_or_copy(src_js, dst_js)


def save_snapshot_cached(res, pageid: int, snapshot_date_ymd: str) -> tuple[str, str, str]:
    """
    Lagrer snapshot til OUT_ROOT/YYYY-MM-DD/pageid_revid.*
    Cache per (pageid,revid) i OUT_ROOT/_cache
    """
    ymd = snapshot_date_ymd.strip()
    if not ymd or len(ymd) != 10:
        raise ValueError(f"Invalid date (expected YYYY-MM-DD): {snapshot_date_ymd}")

    out_dir = OUT_ROOT / ymd
    ensure_dir(out_dir)
    ensure_dir(CACHE_DIR)

    out_base = base_name(pageid, res.revid)
    cache_base = out_base

    if safe_exists(out_dir, out_base):
        return ymd, out_base, "skip"

    if safe_exists(CACHE_DIR, cache_base):
        materialize_from_cache(cache_base, out_dir, out_base)
        return ymd, out_base, "cache_hit"

    save_snapshot(res, str(CACHE_DIR), base=cache_base)
    materialize_from_cache(cache_base, out_dir, out_base)
    return ymd, out_base, "downloaded"


def load_doc_times(path: Path) -> dict[int, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    out: dict[int, list[str]] = {}
    for k, dates in raw.items():
        try:
            pid = int(k)
        except Exception:
            continue
        if not isinstance(dates, list):
            continue
        # normalize: unique + sorted (valgfritt)
        clean_dates = sorted({str(d).strip() for d in dates if str(d).strip()})
        out[pid] = clean_dates

    return out


def main():
    if not DOC_TIMES_PATH.exists():
        raise SystemExit(f"Finner ikke: {DOC_TIMES_PATH.resolve()}")

    doc_times = load_doc_times(DOC_TIMES_PATH)

    ensure_dir(OUT_ROOT)
    ensure_dir(CACHE_DIR)

    total_tasks = sum(len(v) for v in doc_times.values())
    print(f"Loaded doc_times: pageids={len(doc_times)}, total snapshots={total_tasks}")

    ok, skipped, failed = 0, 0, 0
    cache_hits, downloaded = 0, 0

    for pageid, dates in doc_times.items():
        for ymd in dates:
            try:
                # fetch_snapshot forventer typisk ISO-timestamp eller en dato.
                # Hvis din fetch_snapshot krever ISO, bruk f"{ymd}T00:00:00Z"
                ts = f"{ymd}T00:00:00Z"

                res = fetch_snapshot(pageid, ts, max_redirect_hops=MAX_REDIRECT_HOPS)
                out_ymd, b, mode = save_snapshot_cached(res, pageid, ymd)

                if mode == "skip":
                    skipped += 1
                    print(f"[SKIP] {pageid} @ {out_ymd} revid={res.revid}")
                elif mode == "cache_hit":
                    ok += 1
                    cache_hits += 1
                    print(f"[OK cached] {pageid} @ {out_ymd} revid={res.revid}")
                else:
                    ok += 1
                    downloaded += 1
                    print(f"[OK] {pageid} @ {out_ymd} revid={res.revid}")

                if SLEEP_SECONDS > 0:
                    time.sleep(SLEEP_SECONDS)

            except Exception as e:
                failed += 1
                print(f"[ERROR] {pageid} @ {ymd} -> {e}")

    print("\nDone.")
    print(f"ok={ok}, skipped={skipped}, failed={failed}")
    print(f"downloaded_from_api={downloaded}, materialized_from_cache={cache_hits}")
    print(f"Output root: {OUT_ROOT.resolve()}")
    print(f"Cache dir:   {CACHE_DIR.resolve()}")


if __name__ == "__main__":
    main()