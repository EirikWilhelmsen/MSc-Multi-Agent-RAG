import json
import time
import matplotlib.pyplot as plt
import re
from dotenv import load_dotenv
import os
import requests

load_dotenv()
LLM_API_KEY = os.getenv("OLLAMA_API")
LLM_URL = "https://openwebui.ux.uis.no/api/chat/completions"
LLM_MODEL = "qwen3:0.6b"

os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"

def load_version(type, model, path="../graph_version_control.json"):
    with open(path, "r") as f:
        version_control = json.load(f)
    version = version_control[model][type]
    return version

def increment_version(type, model, path="../graph_version_control.json"):
    with open(path, "r") as f:
        version_control = json.load(f)
    current_version = version_control[model][type]
    current_version = int(current_version[1:])
    current_version += 1
    new_version = f"v{current_version}"
    version_control[model][type] = new_version
    with open(path, "w") as f:
        json.dump(version_control, f, indent=4)
        
def pageid_graph(counts, title, path):
    plt.figure(figsize=(10, 5))
    plt.bar(
        list(counts.keys()), list(counts.values()),
        color = ["mediumseagreen", "tomato", "gray", "steelblue"][:len(counts)]
    )
    for i, v in enumerate(counts.values()):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.title(f"{title}")
    plt.ylabel("Count")
    plt.ylim(0, 500)
    plt.tight_layout()
    plt.savefig(path)
    
def answer_graph(counts, title, path):
    plt.figure(figsize=(10, 5))
    plt.bar(
        list(counts.keys()), list(counts.values()),
        color = ["mediumseagreen", "tomato", "gray", "steelblue"][:len(counts)]
    )
    for i, v in enumerate(counts.values()):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.title(f"{title}")
    plt.ylabel("Count")
    plt.ylim(0, 500)
    plt.tight_layout()
    plt.savefig(path)
    
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[,\.\-\(\)]", "", text)  # fjern spesialtegn
    text = re.sub(r"\s+", " ", text)          # normaliser whitespace
    return text

def extract_number(text: str) -> float | None:
    """Ekstraherer første tall fra tekst, støtter også tekstlige tall."""
    text_lower = text.lower().strip()

    if re.search(r'\d+:\d+', text_lower):
        return None
    
    word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12
    }
    
    for word, num in word_to_num.items():
        if re.search(rf'\b{word}\b', text_lower):
            return float(num)
    
    match = re.search(r'\b\d+(?:\.\d+)?\b', text_lower)
    if match:
        return float(match.group())
    
    return None

def numeric_ground_truth_match(generated, current_gt, outdated_gt):
    gen_num = extract_number(generated)
    if gen_num is None:
        return None  # Ingen tall → la LLM-judge håndtere

    curr_num = extract_number(current_gt)
    outd_num = extract_number(outdated_gt)

    curr_match = (curr_num is not None and gen_num == curr_num)
    outd_match = (outd_num is not None and gen_num == outd_num)

    if curr_match and not outd_match:
        return "current"
    elif outd_match and not curr_match:
        return "outdated"
    else:
        return "wrong"

def is_match(predicted: str, ground_truth: str) -> bool:
    return normalize(predicted) == normalize(ground_truth)

def classify_answer_LLM(predicted: str, new_answer: str, old_answer: str, question: str, max_retries: int = 5, backoff: float = 5.0) -> str:
    prompt = (
        f"You are evaluating whether a predicted answer is correct, outdated, or wrong.\n\n"
        f"Note: A predicted answer does not need to be an exact match. If it is a partial but unambiguous match (e.g. '10,000' matches '10,000 years'), classify it accordingly.\n\n"
        f"If the predicted answer is uncertain (e.g. '1949 or 1950'), classify it as 'wrong'.\n\n"
        f"Question: {question}\n"
        f"Definitions:\n"
        f"- 'correct': the predicted answer matches or semantically agrees with the new answer\n"
        f"- 'outdated': the predicted answer matches or semantically agrees with the old answer\n"
        f"- 'wrong': the predicted answer does not match either answer\n\n"
        f"Predicted: {predicted}\n"
        f"New Answer: {new_answer}\n"
        f"Old Answer: {old_answer}\n\n"
        f"After evaluating the predicted answer against the new and old answers, re evaluate your classification."
        f"Respond with exactly one word: 'correct', 'outdated', 'wrong'."
    )
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(LLM_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = json.loads(response.text)
            content = data["choices"][0]["message"]["content"].strip()
            cleaned = re.sub(r"```(?:json)?|```", "", content).strip()
            print(f"LLM response: {content}")
            try:
                parsed = json.loads(cleaned)
                classification = parsed["classification"].strip().lower()
                if classification in ("correct", "outdated", "wrong", "unsure"):
                    return classification
            except (json.JSONDecodeError, KeyError):
                pass
            
            for label in ("wrong", "outdated", "correct"):
                if label in content.lower():
                    return label

            print(f"  [WARN] Could not parse classification from: {content[:200]}")
            return "wrong"
        
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and 500 <= status < 600 and attempt < max_retries:
                wait = backoff * attempt
                print(f"  [LLM {status}] retry {attempt}/{max_retries - 1} in {wait:.1f}s")
                time.sleep(wait)
                continue
            print(e.response.text if e.response is not None else str(e))
            raise

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            if attempt < max_retries:
                wait = backoff * attempt
                print(f"  [LLM network error] retry {attempt}/{max_retries - 1} in {wait:.1f}s: {e}")
                time.sleep(wait)
                continue
            raise

def is_unsure(predicted: str) -> bool:
    predicted_lower = predicted.lower()
    patterns = [" or ", " and/or ", " either ", "not sure", "unsure", "unclear"]
    return any(p in predicted_lower for p in patterns)

