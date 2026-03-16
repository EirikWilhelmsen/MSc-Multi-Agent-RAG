import pandas as pd
from pathlib import Path
import numpy as np

MAX_QUESTIONS = 100

def main():
    parquet_path = Path("../clean_HoH_dataset/hoh_qas_240601_241201.parquet")
    out_csv = Path("../ground_truth_answers.csv")

    df = pd.read_parquet(parquet_path)

    def get_outdated_answer(outdated_infos):
        if not isinstance(outdated_infos, (list, np.ndarray)):
            return ""
        answers = []
        for item in outdated_infos:
            if isinstance(item, dict) and item.get("answer"):
                answers.append(item["answer"])
        return ";".join(answers)

    out = pd.DataFrame({
        "answer": df["answer"],
        "outdated_answer": df["outdated_infos"].apply(get_outdated_answer),
    })

    out = out.dropna(subset=["answer"]).head(MAX_QUESTIONS).copy()
    out.insert(0, "qid", range(1, len(out) + 1))

    out.to_csv(out_csv, index=False, sep=";")
    print(f"Wrote {len(out)} rows (capped at {MAX_QUESTIONS}) to {out_csv}")

if __name__ == "__main__":
    main()