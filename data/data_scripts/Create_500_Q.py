import pandas as pd
from pathlib import Path
import numpy as np

N_SAMPLES = 500
SEED = 42  # reproducibility

def main():
    parquet_path = Path("../clean_HoH_dataset/hoh_qas_240601_241201.parquet")
    out_csv = Path("../500Q/500_hoh_questions.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)

    def get_doc_id(doc):
        return doc.get("id") if isinstance(doc, dict) else None

    def get_doc_title(doc):
        return doc.get("title") if isinstance(doc, dict) else None

    def extract_outdated_dates(outdated_infos):
        if not isinstance(outdated_infos, (list, np.ndarray)):
            return ""
        dates = []
        for item in outdated_infos:
            if isinstance(item, dict) and item.get("last_modified_time"):
                date = str(item["last_modified_time"])[:10]
                dates.append(f"{date}T00:00:00Z")
        return ";".join(sorted(set(dates)))

    def get_outdated_answer(outdated_infos):
        if not isinstance(outdated_infos, (list, np.ndarray)):
            return ""
        answers = []
        for item in outdated_infos:
            if isinstance(item, dict) and item.get("answer"):
                answers.append(item["answer"])
        return ";".join(answers)

    out = pd.DataFrame({
        "pageid": df["document"].apply(get_doc_id),
        "title": df["document"].apply(get_doc_title),
        "question": df["question"],
        "current_time": df["last_modified_time"].apply(lambda x: f"{str(x)[:10]}T00:00:00Z"),
        "outdated_info_dates": df["outdated_infos"].apply(extract_outdated_dates),
        "answer": df["answer"],
        "outdated_answer": df["outdated_infos"].apply(get_outdated_answer),
    })

    out["pageid"] = pd.to_numeric(out["pageid"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["pageid", "question", "answer"])
    out = out.drop_duplicates(subset=["question", "pageid"])

    n = min(N_SAMPLES, len(out))
    out = out.sample(n=n, random_state=SEED).reset_index(drop=True)

    out.insert(0, "qid", range(1, len(out) + 1))

    out.to_csv(out_csv, index=False, sep=";")
    print(f"Wrote {len(out)} rows to {out_csv}")

if __name__ == "__main__":
    main()