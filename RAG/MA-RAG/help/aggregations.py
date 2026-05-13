import json
import os
import sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from help_functions import call_llm


def aggregate(candidates: list[dict], method: str) -> dict:
    """Returner kandidaten med høyest confidence."""
    if method == "majority_vote":
        answer_counts = {}
        for c in candidates:
            ans = c["answer"]
            answer_counts[ans] = answer_counts.get(ans, 0) + 1

        majority_answer = max(answer_counts, key=answer_counts.get)

        if majority_answer == "Unsure":
            non_unsure = [c for c in candidates if c["answer"] != "Unsure"]
            if non_unsure:
                return max(non_unsure, key=lambda c: (c["confidence"], -c["chunk_id"]))

        if answer_counts[majority_answer] >= 2:
            winners = [c for c in candidates if c["answer"] == majority_answer]
            return max(winners, key=lambda c: (c["confidence"], -c["chunk_id"]))

        return max(candidates, key=lambda c: (c["confidence"], -c["chunk_id"]))

    elif method == "Confidence":
        return max(candidates, key=lambda c: (c["confidence"], -c["chunk_id"]))
    elif method == "Random":
        return random.choice(candidates)