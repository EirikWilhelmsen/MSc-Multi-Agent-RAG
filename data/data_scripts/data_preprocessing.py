from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Tuple

import wikitextparser as wtp

RAW_ROOT = Path("KB_raw")
CLEAN_ROOT = Path("KB_cleaned")
RECURSIVE = True

CUTOFF_HEADINGS = {
    "references", "notes", "footnotes", "external links",
    "further reading", "bibliography", "see also"
}


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()

def strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

def strip_refs(text: str) -> str:
    text = re.sub(r"<ref\b[^>/]*?>.*?</ref\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<ref\b[^>]*?/>", "", text, flags=re.IGNORECASE)
    return text

def strip_file_links(text: str) -> str:
    return re.sub(r"\[\[(File|Image):[^\]]+\]\]", "", text, flags=re.IGNORECASE)

def strip_external_bracket_links(text: str) -> str:
    def repl(m: re.Match) -> str:
        inside = m.group(1).strip()
        parts = inside.split(None, 1)
        return parts[1].strip() if len(parts) == 2 else ""
    return re.sub(r"\[([a-z]+:\/\/[^\]]+)\]", repl, text, flags=re.IGNORECASE)

def cleanup_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"

def cut_tail_sections(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(=+)\s*(.+?)\s*\1\s*$", line)
        if not m:
            continue
        title = normalize(m.group(2))
        if title in CUTOFF_HEADINGS and i > 0 and lines[i - 1].strip() == "":
            return "\n".join(lines[:i]).rstrip() + "\n"
    return text

def wtp_plain_text(wikitext: str) -> str:
    parsed = wtp.parse(wikitext)
    pt = getattr(parsed, "plain_text", "")
    if callable(pt):
        return str(pt())
    return str(pt)

def strip_templates_aggressive(text: str) -> str:
    pattern = re.compile(r"\{\{[^{}]*\}\}")
    for _ in range(50):
        new = pattern.sub("", text)
        if new == text:
            break
        text = new
    return text

def extract_infobox_as_text(raw: str) -> Tuple[str, str]:
    parsed = wtp.parse(raw)
    infobox_lines: List[str] = []
    to_remove: List[str] = []

    for tpl in list(parsed.templates):
        name = normalize(tpl.name)
        if name.startswith("infobox"):
            to_remove.append(str(tpl))
            kvs: List[str] = []
            for arg in tpl.arguments:
                key = (arg.name or "").strip()
                val = (arg.value or "").strip()
                if not key or not val:
                    continue
                val = strip_comments(val)
                val = strip_refs(val)
                val = strip_file_links(val)
                val = strip_external_bracket_links(val)
                val = wtp_plain_text(val)
                val = re.sub(r"\s+", " ", val).strip()
                if val:
                    kvs.append(f"- {key}: {val}")
            if kvs:
                infobox_lines.append("[Infobox]")
                infobox_lines.extend(kvs)
                infobox_lines.append("")

    s = raw
    for t in to_remove:
        s = s.replace(t, "")

    return "\n".join(infobox_lines).strip(), s

def preprocess(raw: str) -> str:
    raw = raw.replace("\r\n", "\n")
    raw = strip_comments(raw)
    raw = strip_refs(raw)

    infobox_text, raw = extract_infobox_as_text(raw)

    raw = strip_file_links(raw)
    raw = strip_external_bracket_links(raw)
    raw = cut_tail_sections(raw)
    raw = strip_templates_aggressive(raw)

    body = wtp_plain_text(raw)
    body = cleanup_whitespace(body)

    if infobox_text:
        return cleanup_whitespace(infobox_text + "\n\n" + body)
    return body


def iter_wikitext_files(root: Path) -> List[Path]:
    pattern = "**/*.wikitext.txt" if RECURSIVE else "*.wikitext.txt"
    return sorted(root.glob(pattern))

def out_path_for(in_fp: Path) -> Path:
    rel = in_fp.relative_to(RAW_ROOT)
    return CLEAN_ROOT / rel

def copy_sidecar_json(in_wikitext_fp: Path, out_wikitext_fp: Path) -> None:
    """
    Kopierer JSON med samme base-navn:
      <base>.wikitext.txt  -> <base>.json
    """
    in_json = in_wikitext_fp.with_suffix("")  # fjerner .txt
    # in_json peker nå på .../<base>.wikitext
    # vi vil ha .../<base>.json
    in_json = in_json.with_suffix(".json")

    if in_json.exists():
        out_json = out_wikitext_fp.with_suffix("").with_suffix(".json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(in_json, out_json)


def process_all() -> Tuple[int, int]:
    changed = 0
    failed = 0

    for fp in iter_wikitext_files(RAW_ROOT):
        try:
            raw = fp.read_text(encoding="utf-8", errors="replace")
            cleaned = preprocess(raw)

            out_fp = out_path_for(fp)
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            out_fp.write_text(cleaned, encoding="utf-8")

            # kopier tilhørende json
            copy_sidecar_json(fp, out_fp)

            if cleaned != raw:
                changed += 1

        except Exception as e:
            failed += 1
            print(f"[FAIL] {fp}: {e}")

    return changed, failed

def main() -> None:
    if not RAW_ROOT.exists() or not RAW_ROOT.is_dir():
        raise SystemExit(f"folder does not exist or is not a directory: {RAW_ROOT}")

    CLEAN_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"RAW_ROOT:   {RAW_ROOT}")
    print(f"CLEAN_ROOT: {CLEAN_ROOT}")
    print(f"RECURSIVE={RECURSIVE}\n")

    changed, failed = process_all()

    print("\n[DONE]")
    print(f"Endret (cleaned != raw): {changed}")
    print(f"Feilet: {failed}")

if __name__ == "__main__":
    main()