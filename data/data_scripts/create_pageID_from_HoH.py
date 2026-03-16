import pandas as pd
from pathlib import Path
import numpy as np

MAX_QUESTIONS = 100  # hard stop

def main():
    parquet_path = Path("../clean_HoH_dataset/hoh_qas_240601_241201.parquet")
    out_csv = Path("../hoh_question_pageid_map.csv")

    df = pd.read_parquet(parquet_path)

    # Extract nested fields
    def get_doc_id(doc):
        if isinstance(doc, dict):
            return doc.get("id")
        return None

    def get_doc_title(doc):
        if isinstance(doc, dict):
            return doc.get("title")
        return None

    def extract_outdated_dates(outdated_infos):
        if not isinstance(outdated_infos, (list, np.ndarray)):
            return ""
    
        dates = []
        for item in outdated_infos:
            if isinstance(item, dict) and item.get("last_modified_time"):
                date = str(item["last_modified_time"])[:10]
                dates.append(f"{date}T00:00:00Z")
    
        return ";".join(sorted(set(dates)))

    out = pd.DataFrame({
        "question": df["question"],
        "pageid": df["document"].apply(get_doc_id),
        "title": df["document"].apply(get_doc_title),
        "current_time": df["last_modified_time"].apply(lambda x: f"{str(x)[:10]}T00:00:00Z"),
        "outdated_info_dates": df["outdated_infos"].apply(extract_outdated_dates),
    })

    out["pageid"] = pd.to_numeric(out["pageid"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["pageid", "question"]).drop_duplicates(subset=["question", "pageid"])

    out = out.head(MAX_QUESTIONS).copy()

    out.insert(0, "qid", range(1, len(out) + 1))

    out.to_csv(out_csv, index=False, sep=";")
    print(f"Wrote {len(out)} rows (capped at {MAX_QUESTIONS}) to {out_csv}")

if __name__ == "__main__":
    main()