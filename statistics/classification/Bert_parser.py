BERT_PATH = "Bert.txt"
LLM_PATH = "LLM.txt"
MANUAL_PATH = "Manual.txt"
GEVAL_PATH = "GEval.txt"

BERT = False
LLM = False
MANUAL = False 
GEVAL = True

if BERT:
    with open(BERT_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Answer"):
                number = line.split("Answer ")[1].split(" is")[0].strip()
                answer = line.split(" is ")[1].strip()
                answer = answer.split(" ")[0]
                answer = answer.strip(".").strip()
                print(f"{number} - {answer}")

if LLM:
    with open(LLM_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("LLM classification"):
                continue
            if line.startswith("Answer") and "needs review" in line:
                number = line.split("Answer ")[1].split(" needs")[0].strip().strip("'")
            elif line.startswith("correct (c)"):
                answer_char = line.split("? ")[1].strip()
                mapping = {"c": "correct", "o": "outdated", "w": "wrong"}
                print(f"{number} - {mapping[answer_char]}")
            elif line.startswith("Answer") and " is " in line:
                number = line.split("Answer ")[1].split(" is")[0].strip()
                answer = line.split(" is ")[1].strip().split(" ")[0].strip(".").strip()
                print(f"{number} - {answer}")

if MANUAL:
    with open(MANUAL_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Ground truth"):
                continue
            if line.startswith("Answer") and "needs review" in line:
                number = line.split("Answer ")[1].split(" needs")[0].strip().strip("'")
            elif line.startswith("correct (c)"):
                answer_char = line.split("? ")[1].strip()
                mapping = {"c": "correct", "o": "outdated", "w": "wrong"}
                print(f"{number} - {mapping[answer_char]}")
            elif line.startswith("Answer") and " is " in line:
                number = line.split("Answer ")[1].split(" is")[0].strip()
                answer = line.split(" is ")[1].strip().split(" ")[0].strip(".").strip()
                print(f"{number} - {answer}")

if GEVAL:
    with open(GEVAL_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Answer"):
                number = line.split("Answer ")[1].split(" is")[0].strip()
                answer = line.split(" is ")[1].strip()
                answer = answer.split(" ")[0] 
                answer = answer.strip(".").strip()
                print(f"{number} - {answer}")