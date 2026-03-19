BERT_PATH = "parsed/bert_parsed.txt"
LLM_PATH = "parsed/llm_parsed.txt"
MANUAL_PATH = "parsed/manual_parsed.txt"
GEVAL_PATH = "parsed/geval_parsed.txt"

ground_truth = []
LLM_score = 0
Bert_score = 0
GEVAL_score = 0

with open(MANUAL_PATH, "r") as f:
    lines = f.readlines()
    for line in lines:
        ans = line.split(" - ")[1].strip()
        ground_truth.append(ans)

with open(BERT_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            answer = line.split(" - ")[1].strip()
            if answer == ground_truth[lines.index(line)]:
                Bert_score += 1

with open(LLM_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            answer = line.split(" - ")[1].strip()
            if answer == ground_truth[lines.index(line)]:
                LLM_score += 1              

with open(GEVAL_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            answer = line.split(" - ")[1].strip()
            if answer == ground_truth[lines.index(line)]:
                GEVAL_score += 1

print(f"Bert score: {Bert_score}/100")
print(f"LLM score: {LLM_score}/100")
print(f"GEval score: {GEVAL_score}/100")