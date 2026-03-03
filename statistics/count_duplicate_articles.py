from collections import Counter
from pathlib import Path

# NY root i stedet for _new / _outdated
path_kb = "../data/KB_raw/"

def count_duplicate_articles(path):
    """
    Count the number of articles that appear multiple times in a given path.
    Works with new structure: KB_raw/YYYY-MM-DD/pageid_revid.wikitext.txt
    """

    file_counts = Counter()

    # Gå rekursivt gjennom alle dato-mapper
    for file in Path(path).glob("**/*.wikitext.txt"):
        parts = file.stem.split("_")

        # Filnavn er nå: pageid_revid.wikitext.txt
        # file.stem blir: pageid_revid.wikitext → vi vil ha pageid
        if len(parts) >= 1:
            page_id = parts[0]
            file_counts[page_id] += 1

    duplicate_counts = Counter(file_counts.values())

    with open("statistics.txt", "w") as f:
        f.write("Article Counts:\n")
        for count in sorted(duplicate_counts.keys()):
            f.write(f"{count} file(s): {duplicate_counts[count]} PageIDs\n")


if __name__ == "__main__":
    duplicate_counts = count_duplicate_articles(path_kb)
