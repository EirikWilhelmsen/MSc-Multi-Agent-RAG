This folder contains scripts for generating the raw KB from wikipedia snaphots using wikipedia API
---
* `wiki_snapshots_by_pageID.py` is an importable module that exposes `fetch_snapshot()` and `save_snapshot()` for downloading historical Wikipedia snapshots, picking the latest revision at or before a given timestamp and following redirect chains; returns a `SnapshotResult` dataclass and is used as the backend by `batch_create_snapshots.py`
* `batch_create_snapshots.py` iterates over all `(pageid, date)` pairs from `doc_times.json` and uses `wiki_snapshots_by_pageID` to download each snapshot into `KB_raw/YYYY-MM-DD/`, with a `_cache/` directory and hardlinks to avoid re-downloading shared revisions, rate-limiting between requests, and failures logged to `failed_snapshots.jsonl`
--- 