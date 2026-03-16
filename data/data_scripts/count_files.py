from pathlib import Path

ROOT = Path("../KB_cleaned")

for d in sorted(ROOT.iterdir()):
    if d.is_dir() and d.name != "_cache":
        count = sum(1 for f in d.iterdir() if f.is_file())
        print(f"{d.name}: {count} files")