import json
from datetime import datetime, date, timezone
from typing import Any, Dict, Set

import pyarrow.parquet as pq

# ==========================
INPUT_PATH = "../clean_HoH_dataset/hoh_qas_240601_241201.parquet"
OUTPUT_PATH = "../doc_times.json"
SORT_DESCENDING = True  # newest first
# ==========================


def to_iso_date(value: Any):
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.date().isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
        return dt.date().isoformat()

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        return s[:10]  # YYYY-MM-DD

    try:
        s = str(value).strip()
        if len(s) >= 10:
            return s[:10]
    except Exception:
        pass

    return None


def main():
    doc_dates: Dict[str, Set[str]] = {}

    pf = pq.ParquetFile(INPUT_PATH)

    for batch in pf.iter_batches(batch_size=4096):
        table = batch.to_pydict()
        rows = len(next(iter(table.values()))) if table else 0

        for i in range(rows):
            row = {col: table[col][i] for col in table}

            document = row.get("document") or {}
            doc_id = document.get("id")
            if doc_id is None:
                continue
            doc_id = str(doc_id)

            doc_dates.setdefault(doc_id, set())

            new_time = to_iso_date(row.get("last_modified_time"))
            if new_time:
                doc_dates[doc_id].add(new_time)

            outdated = row.get("outdated_infos") or []
            if isinstance(outdated, list):
                for item in outdated:
                    if not isinstance(item, dict):
                        continue
                    old_time = to_iso_date(item.get("last_modified_time"))
                    if old_time:
                        doc_dates[doc_id].add(old_time)

    result = {doc_id: sorted(dates, reverse=SORT_DESCENDING) for doc_id, dates in doc_dates.items()}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Ferdig. Lagret {len(result)} document IDs i {OUTPUT_PATH}")


if __name__ == "__main__":
    main()