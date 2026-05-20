import json
import csv
from helper_functions.helper_functions import classify_answer_LLM

class ModelResults:
    def __init__(self, model, version, exact_model, setup=""):
        self.model = model
        self.version = version
        self.exact_model = exact_model
        self.setup = setup
        self.data = []
        self.ground_truth = []
    
    def load(self):
        path = self._get_path()
        self.data = self._read_json(path)
        self.ground_truth = self._read_csv("../data/500Q/500_hoh_questions.csv")
    
    def _read_csv(self, path):
        with open(path, "r") as f:
            f.readline()  # skip header
            return [line.strip().split(";") for line in f]

    def _get_path(self):
        if self.model in ["Baseline", "RCDS", "RCO", "RTA"]:
            return f"{self.model}/{self.exact_model}/{self.model.lower()}_results_v{self.version}.jsonl"
        elif self.model == "RCA":
            return f"{self.model}/{self.exact_model}/{self.setup}/rca_results_v{self.version}.jsonl"
        else:
            raise ValueError(f"Unknown model: {self.model!r}")
    
    def _read_json(self, path):
        with open(path, "r") as f:
            return [json.loads(line) for line in f]

    def LLM_match(self):
        self.load()
        correct_count = 0
        outdated_count = 0
        wrong_count = 0
        
        for entry, gt_row in zip(self.data, self.ground_truth):
            predicted = entry["predicted_answer"]
            updated_answer = gt_row[-2]
            outdated_answer = gt_row[-1]
            question = gt_row[3]
            
            classification = classify_answer_LLM(predicted, updated_answer, outdated_answer, question)
            if classification == "correct":
                correct_count += 1
            elif classification == "outdated":
                outdated_count += 1
            elif classification == "wrong":
                wrong_count += 1
        
        return {
            "correct": correct_count,
            "outdated": outdated_count,
            "wrong": wrong_count,
        }


if __name__ == "__main__":
    with open("models_to_evaluate.json", "r") as f:
        config = json.load(f)
    
    rows = []
    for model_name, exact_models in config.items():
        for exact_model, value in exact_models.items():
            entries = value.items() if isinstance(value, dict) else [("", value)]
            for setup, version in entries:
                if version == 0:
                    continue
                results = ModelResults(model_name, version, exact_model, setup)
                counts = results.LLM_match()
                rows.append({
                    "model": model_name,
                    "exact_model": exact_model,
                    "setup": setup,
                    "version": version,
                    **counts,
                })
                print(f"{model_name}/{exact_model}" + (f"/{setup}" if setup else ""))
    
    with open("new_baselines.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nWrote {len(rows)} rows to new_baselines.csv")