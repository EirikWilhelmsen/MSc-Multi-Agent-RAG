import json
import matplotlib.pyplot as plt
import re
from bert_score import score
from dotenv import load_dotenv
import os
import requests
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI
from deepeval.metrics import GEval

load_dotenv()
LLM_API_KEY = os.getenv("OLLAMA_API")
LLM_URL = "https://openwebui.ux.uis.no/api/chat/completions"
LLM_MODEL = "qwen3:0.6b"

os.environ["CONFIDENT_METRIC_LOGGING_VERBOSE"] = "0"

#print(repr(LLM_API_KEY))

class OpenWebUIModel(DeepEvalBaseLLM):
    def __init__(self):
        self.client = OpenAI(
            api_key=LLM_API_KEY,
            base_url="https://openwebui.ux.uis.no/api",
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip() # type: ignore

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return LLM_MODEL

custom_model = OpenWebUIModel()

correctness_metric = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check if the actual output matches or semantically agrees with the expected output",
        "Partial but unambiguous matches should score high (e.g. '10,000' matches '10,000 years')",
        "Penalize answers that contradict or are unrelated to the expected output",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model=custom_model,
)

def load_version(type, model):
    with open("../graph_version_control.json", "r") as f:
        version_control = json.load(f)
    version = version_control[model][type]
    return version

def increment_version(type, model):
    with open("../graph_version_control.json", "r") as f:
        version_control = json.load(f)
    current_version = version_control[model][type]
    current_version = int(current_version[1:])
    current_version += 1
    new_version = f"v{current_version}"
    version_control[model][type] = new_version
    with open("../graph_version_control.json", "w") as f:
        json.dump(version_control, f, indent=4)

def create_graph(counts, title, path):
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path)
    
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[,\.\-\(\)]", "", text)  # fjern spesialtegn
    text = re.sub(r"\s+", " ", text)          # normaliser whitespace
    return text

def is_match(predicted: str, ground_truth: str) -> bool:
    return normalize(predicted) == normalize(ground_truth)

def classify_answer(predicted: str, new_answer: str, old_answer: str) -> str:
    result_new = score([predicted], [new_answer], lang="en", verbose=False)
    result_old = score([predicted], [old_answer], lang="en", verbose=False)
    
    f_new = float(result_new[2][0])
    f_old = float(result_old[2][0])
    print(f"predicted: '{predicted}', new: '{new_answer}' score: {f_new}, old: '{old_answer}' score: {f_old}")
    
    threshold = 0.7
    if f_new < threshold and f_old < threshold:
        return "wrong"
    elif f_new > f_old:
        return "correct"
    else:
        return "outdated"

def classify_answer_LLM(predicted: str, new_answer: str, old_answer: str, question: str) -> str:
    predicted_norm = normalize(predicted)
    new_norm = normalize(new_answer)
    old_norm = normalize(old_answer)
    print(f"LLM classification for predicted: '{predicted_norm}', new: '{new_norm}', old: '{old_norm}'")
    prompt = (
        f"You are evaluating whether a predicted answer is correct, outdated, or wrong.\n\n"
        f"Note: A predicted answer does not need to be an exact match. If it is a partial but unambiguous match (e.g. '10,000' matches '10,000 years'), classify it accordingly.\n\n"
        f"Question: {question}\n"
        f"Definitions:\n"
        f"- 'correct': the predicted answer matches or semantically agrees with the new answer\n"
        f"- 'outdated': the predicted answer matches or semantically agrees with the old answer\n"
        f"- 'wrong': the predicted answer does not match either answer\n\n"
        f"Predicted: {predicted}\n"
        f"New Answer: {new_answer}\n"
        f"Old Answer: {old_answer}\n\n"
        f"Respond with exactly one word: 'correct', 'outdated' or 'wrong'."
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

    response = requests.post(LLM_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

def classify_answer_GEval(predicted: str, new_answer: str, old_answer: str, question: str) -> str | None:
    # Kall mot new_answer
    test_case_new = LLMTestCase(
        input=question,
        actual_output=predicted,
        expected_output=new_answer
    )
    correctness_metric.measure(test_case_new)
    score_new = correctness_metric.score
    print(f"predicted: '{predicted}', new: '{new_answer}' GEval score: {score_new}")
    # Kall mot old_answer
    test_case_old = LLMTestCase(
        input=question,
        actual_output=predicted,
        expected_output=old_answer
    )
    correctness_metric.measure(test_case_old)
    score_old = correctness_metric.score
    print(f"predicted: '{predicted}', old: '{old_answer}' GEval score: {score_old}")
    # Klassifiser
    if score_new is not None and score_old is not None:
        if score_new < 0.5 and score_old < 0.5:
            return "wrong"
        elif score_new >= score_old:
            return "correct"
        else:
            return "outdated"
    else:
        return None