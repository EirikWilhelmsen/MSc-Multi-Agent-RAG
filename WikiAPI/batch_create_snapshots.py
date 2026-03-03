import time
from pathlib import Path
import pandas as pd

from wiki_snapshots_by_pageID_v2 import fetch_snapshot, save_snapshot  # juster ved behov

CSV_PATH = "../data/hoh_question_pageid_map.csv"

OUT_ROOT = Path("../data/KB_raw")  # NY: én rotmappe med dato-submapper

SLEEP_SECONDS = 0.3
MAX_REDIRECT_HOPS = 5


def parse_dates(s: str) -> list[str]:
    if pd.isna(s) or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(";") if x.strip()]


def iso_to_ymd(ts: str) -> str:
    """
    Tar '2024-07-01T00:00:00Z' eller '2024-07-01' og returnerer '2024-07-01'
    """
    s = str(ts).strip()
    if not s:
        return ""
    if "T" in s:
        return s.split("T", 1)[0]
    return s[:10]


def safe_exists(out_dir: Path, base: str) -> bool:
    return (out_dir / f"{base}.wikitext.txt").exists() and (out_dir / f"{base}.json").exists()


def base_name(pageid: int, revid: int) -> str:
    return f"{pageid}_{revid}"


def fetch_and_save(pageid: int, ts: str, label: str) -> tuple[bool, str]:
    """
    label kun for logging.
    return (success, message)
    """
    res = fetch_snapshot(pageid, ts, max_redirect_hops=MAX_REDIRECT_HOPS)

    ymd = iso_to_ymd(ts)
    if not ymd:
        raise ValueError("Missing timestamp")

    out_dir = OUT_ROOT / ymd
    out_dir.mkdir(parents=True, exist_ok=True)

    b = base_name(pageid, res.revid)

    if safe_exists(out_dir, b):
        return (False, f"[{label} SKIP] {pageid} @ {ymd} revid={res.revid}")

    save_snapshot(res, str(out_dir), base=b)
    time.sleep(SLEEP_SECONDS)
    return (True, f"[{label} OK] {pageid} @ {ymd} revid={res.revid}")


def main():
    df = pd.read_csv(CSV_PATH)

    OUT_ROOT.mkdir(exist_ok=True)

    ok, skipped, failed = 0, 0, 0

    for _, row in df.iterrows():
        pageid = int(row["pageid"])
        current_time = str(row["current_time"]).strip()
        outdated_times = parse_dates(row.get("outdated_info_dates", ""))

        try:
            did_save, msg = fetch_and_save(pageid, current_time, "NEW")
            print(msg)
            if did_save:
                ok += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            print(f"[NEW ERROR] {pageid} @ {current_time} -> {e}")

        for ts in outdated_times:
            try:
                did_save, msg = fetch_and_save(pageid, ts, "OLD")
                print(msg)
                if did_save:
                    ok += 1
                else:
                    skipped += 1
            except Exception as e:
                failed += 1
                print(f"[OLD ERROR] {pageid} @ {ts} -> {e}")

    print(f"\nDone. ok={ok}, skipped={skipped}, failed={failed}")
    print(f"Output root: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()