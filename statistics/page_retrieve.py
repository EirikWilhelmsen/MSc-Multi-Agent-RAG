import json
import csv
from helper_functions.helper_functions import (
    count_baseline, count_RCA, count_RCDS, count_RCO, count_RTA
)

class ModelResults:
    def __init__(self, model, version, exact_model, setup=""):
        self.model = model
        self.version = version
        self.exact_model = exact_model
        self.setup = setup
        self.data = []

    
    def load(self):
        path = self._get_path()
        self.data = self._read_json(path)

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
    
    def selected_recall(self):
        self.load()
        found_updated_count = 0
        found_outdated_count = 0
        
        for entry in self.data:
            gold_pageid = entry["gold_pageid"]
            new_date = entry["new_date"]
            old_date = entry["old_date"]
            
            found_updated = False
            found_outdated = False

            if self.model == "Baseline": 
                found_updated, found_outdated = count_baseline(entry["retrieved_docs"], gold_pageid, new_date, old_date)
            else: 
                if gold_pageid == entry["best_chunk_pageid"]:
                    if new_date == entry["best_chunk_date"]:
                        found_updated = True
                    elif old_date == entry["best_chunk_date"]:
                        found_outdated = True
                
            if found_updated: found_updated_count += 1
            if found_outdated: found_outdated_count += 1

        return {
            "found_updated": found_updated_count,
            "found_outdated": found_outdated_count,
        }
    
    def exposed_recall(self):
        self.load()
        found_outdated_count = 0
        found_updated_count = 0
        count = 0
        print(self.exact_model,"-", self.setup)
        
        for entry in self.data:
            gold_pageid = entry["gold_pageid"]
            new_date = entry["new_date"]
            old_date = entry["old_date"]

            if self.model == "Baseline": 
                found_updated, found_outdated = count_baseline(entry["retrieved_docs"], gold_pageid, new_date, old_date)
            elif self.model == "RCA":
                found_updated, found_outdated = count_RCA(entry["candidates"], entry["retrieved_docs"], gold_pageid, new_date, old_date)
            elif self.model == "RCDS":
                found_updated, found_outdated = count_RCDS(entry["debate_history"], entry["retrieved_docs"], gold_pageid, new_date, old_date)
            elif self.model == "RCO":
                found_updated, found_outdated = count_RCO(entry, entry["retrieved_docs"], gold_pageid, new_date, old_date)
            elif self.model == "RTA":
                found_updated, found_outdated = count_RTA(entry, gold_pageid, new_date, old_date)
                
            if found_updated: found_updated_count += 1
            if found_outdated: found_outdated_count += 1
            count += 1
        return {
            "found_updated": found_updated_count,
            "found_outdated": found_outdated_count,
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
                counts = results.exposed_recall()
                rows.append({
                    "model": model_name,
                    "exact_model": exact_model,
                    "setup": setup,
                    "version": version,
                    **counts,
                })
                print(f"{model_name}/{exact_model}" + (f"/{setup}" if setup else ""))
    
    with open("test_used_results_2.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nWrote {len(rows)} rows to test_used_results_2.csv")