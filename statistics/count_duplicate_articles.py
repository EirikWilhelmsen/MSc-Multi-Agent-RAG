import json
from collections import Counter

INPUT_PATH = "../data/doc_times.json"

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    counter = Counter()

    for doc_id, dates in data.items():
        counter[len(dates)] += 1

    duplicate_counts = Counter(counter.values())
    with open("statistics.txt", "w", encoding="utf-8") as f:
        f.write("Article Counts:\n")
        for num_dates in sorted(counter.keys()):
            f.write(f"{num_dates} dato(er): {counter[num_dates]} dokumenter\n")


if __name__ == "__main__":
    main()