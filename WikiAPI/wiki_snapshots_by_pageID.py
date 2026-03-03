#!/usr/bin/env python3
"""
Download exact historical snapshots of a Wikipedia page using PAGEID, at given timestamps.

Features:
- Works from a starting pageid (e.g., 1000011)
- For each timestamp: picks latest revision <= timestamp
- Downloads wikitext + (optional) rendered HTML
- If the selected revision is a redirect, follows the redirect chain as-of that timestamp
- Saves per-snapshot JSON + wikitext + SUMMARY.json

Usage:
  python wiki_snapshots_by_pageid.py --pageid 1000011 --out snaps

  # Custom timestamps
  python wiki_snapshots_by_pageid.py --pageid 1000011 --snapshots 2024-11-30 2024-12-31 --out snaps

Notes:
- Timestamps are UTC and must end with 'Z' if full ISO; YYYY-MM-DD is allowed and becomes 23:59:59Z.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://en.wikipedia.org/w/api.php"
UA = "wiki-snapshot-script/1.2 (+https://en.wikipedia.org/)"


# -----------------------------
# HTTP helpers
# -----------------------------
def http_get_json(url: str, headers: dict | None = None, timeout: int = 60) -> dict:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as r:
        data = r.read().decode("utf-8", errors="replace")
    return json.loads(data)


def http_get_text(url: str, headers: dict | None = None, timeout: int = 60) -> str:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sanitize_filename(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s


def parse_iso8601_z(ts: str) -> str:
    """
    Accepts:
      - '2024-11-30T23:59:59Z'
      - '2024-11-30'  -> interpreted as 23:59:59Z

    Returns canonical 'YYYY-MM-DDTHH:MM:SSZ'
    """
    ts = ts.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", ts):
        ts = ts + "T23:59:59Z"
    if not ts.endswith("Z"):
        raise ValueError("Timestamp must end with 'Z' (UTC), e.g. 2024-11-30T23:59:59Z")
    datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    return ts


# -----------------------------
# MediaWiki API calls
# -----------------------------
def get_revision_at_or_before_by_pageid(pageid: int, ts_z: str) -> dict:
    """
    Find latest revision of pageid with revision timestamp <= ts_z.
    Returns: {pageid, title, revid, parentid, timestamp, requested_ts}
    """
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "revisions",
        "pageids": str(pageid),
        "rvlimit": "1",
        "rvstart": ts_z,
        "rvdir": "older",
        "rvprop": "ids|timestamp",
    }
    url = API + "?" + urlencode(params)
    data = http_get_json(url, headers={"User-Agent": UA})

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        raise RuntimeError("No pages returned by API.")

    page = pages[0]
    if "missing" in page:
        raise RuntimeError(f"Page not found for pageid={pageid}")

    revs = page.get("revisions", [])
    if not revs:
        raise RuntimeError(f"No revision found at or before {ts_z} for pageid={pageid}")

    rev = revs[0]
    return {
        "pageid": page.get("pageid"),
        "title": page.get("title"),
        "revid": rev.get("revid"),
        "parentid": rev.get("parentid"),
        "timestamp": rev.get("timestamp"),
        "requested_ts": ts_z,
    }


def get_revision_at_or_before_by_title(title: str, ts_z: str) -> dict:
    """
    Same as above, but for titles (used when following redirects).
    """
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "revisions",
        "titles": title,
        "rvlimit": "1",
        "rvstart": ts_z,
        "rvdir": "older",
        "rvprop": "ids|timestamp",
    }
    url = API + "?" + urlencode(params)
    data = http_get_json(url, headers={"User-Agent": UA})

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        raise RuntimeError("No pages returned by API.")

    page = pages[0]
    if "missing" in page:
        raise RuntimeError(f"Page not found: {title}")

    revs = page.get("revisions", [])
    if not revs:
        raise RuntimeError(f"No revision found at or before {ts_z} for: {title}")

    rev = revs[0]
    return {
        "pageid": page.get("pageid"),
        "title": page.get("title"),
        "revid": rev.get("revid"),
        "parentid": rev.get("parentid"),
        "timestamp": rev.get("timestamp"),
        "requested_ts": ts_z,
    }


def get_wikitext_by_revid(revid: int) -> str:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "revisions",
        "revids": str(revid),
        "rvprop": "content",
        "rvslots": "main",
    }
    url = API + "?" + urlencode(params)
    data = http_get_json(url, headers={"User-Agent": UA})

    pages = data.get("query", {}).get("pages", [])
    if not pages or "revisions" not in pages[0] or not pages[0]["revisions"]:
        raise RuntimeError(f"Could not fetch wikitext for revid={revid}")

    rev = pages[0]["revisions"][0]
    slots = rev.get("slots", {})
    main = slots.get("main", {})
    content = main.get("content")
    if content is None:
        content = rev.get("content")
    if content is None:
        raise RuntimeError(f"Wikitext content missing for revid={revid}")
    return content


# def get_rendered_html_by_oldid(revid: int) -> str:
#     url = f"https://en.wikipedia.org/w/index.php?oldid={revid}"
#     return http_get_text(url, headers={"User-Agent": UA})


# -----------------------------
# Redirect resolution
# -----------------------------
def extract_redirect_target(wikitext: str) -> str | None:
    """
    Detect: #REDIRECT [[Target]]
    Returns "Target" (without fragment/pipe) or None.
    """
    m = re.match(r"(?is)^\s*#redirect\s*\[\[([^\]|#]+)", wikitext)
    return m.group(1).strip() if m else None


def resolve_redirect_chain_from_pageid(pageid: int, ts_z: str, max_hops: int = 5) -> tuple[dict, str, list[dict]]:
    """
    Starting from a pageid, find the revision <= ts_z and follow redirects (if any).
    Returns:
      (final_meta, final_wikitext, chain)

    chain entries include both pageid + title + revid info per hop.
    """
    chain: list[dict] = []

    # First hop is by pageid
    meta = get_revision_at_or_before_by_pageid(pageid, ts_z)
    wikitext = get_wikitext_by_revid(int(meta["revid"]))
    target = extract_redirect_target(wikitext)

    chain.append(
        {
            "hop": 0,
            "from_pageid": meta["pageid"],
            "from_title": meta["title"],
            "revid": meta["revid"],
            "rev_timestamp_utc": meta["timestamp"],
            "redirect_target": target,
            "method": "pageid",
        }
    )

    if not target:
        return meta, wikitext, chain

    # Follow redirects by title (because a redirect target is title-based)
    cur_title = target
    for hop in range(1, max_hops + 1):
        meta2 = get_revision_at_or_before_by_title(cur_title, ts_z)
        wikitext2 = get_wikitext_by_revid(int(meta2["revid"]))
        target2 = extract_redirect_target(wikitext2)

        chain.append(
            {
                "hop": hop,
                "from_pageid": meta2["pageid"],
                "from_title": meta2["title"],
                "revid": meta2["revid"],
                "rev_timestamp_utc": meta2["timestamp"],
                "redirect_target": target2,
                "method": "title",
            }
        )

        if not target2:
            return meta2, wikitext2, chain

        cur_title = target2

    raise RuntimeError(f"Redirect chain too long (>{max_hops}) starting from pageid={pageid}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pageid", required=True, type=int, help="Wikipedia pageid (numeric), e.g. 1000005")
    ap.add_argument("--out", default="wiki_snapshots", help="Output directory")
    ap.add_argument(
        "--snapshots",
        nargs="+",
        default=["2024-11-30T23:59:59Z", "2024-12-31T23:59:59Z"],
        help="One or more UTC timestamps (Z). You can also pass YYYY-MM-DD (assumes 23:59:59Z).",
    )
    # ap.add_argument("--no_html", action="store_true", help="Do not download rendered HTML (only JSON/wikitext).")
    ap.add_argument("--max_redirect_hops", type=int, default=5, help="Maximum redirect hops per snapshot.")
    args = ap.parse_args()

    ensure_dir(args.out)

    summary = {
        "pageid_requested": args.pageid,
        "snapshots": [],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": "For each snapshot time, selects latest revision <= timestamp. If revision is a redirect, follows redirect chain at that time.",
    }

    for snap in args.snapshots:
        ts_z = parse_iso8601_z(snap)

        final_meta, final_wikitext, chain = resolve_redirect_chain_from_pageid(
            args.pageid, ts_z, max_hops=args.max_redirect_hops
        )

        revid = int(final_meta["revid"])
        title = final_meta["title"]
        safe_title = sanitize_filename(title)

        snap_tag = ts_z[:4]
        base = f"{snap_tag}_{args.pageid}"

        json_path = os.path.join(args.out, base + ".json")
        txt_path = os.path.join(args.out, base + ".wikitext.txt")

        write_text(txt_path, final_wikitext)
        # html_path = os.path.join(args.out, base + ".html")

        record = {
            "title": title,
            "pageid": final_meta["pageid"],
            "requested_timestamp_utc": final_meta["requested_ts"],
            "revision_timestamp_utc": final_meta["timestamp"],
            "revid": revid,
            "parentid": final_meta["parentid"],
            "wikipedia_oldid_url": f"https://en.wikipedia.org/w/index.php?oldid={revid}",
            "redirect_chain": chain,
            
            "wikitext_file": os.path.basename(txt_path),

            "wikitext_chars": len(final_wikitext),
            "wikitext_lines": final_wikitext.count("\n") + 1,
        }

        write_json(json_path, record)

        summary["snapshots"].append(
            {
                "requested_timestamp_utc": ts_z,
                "resolved_title": title,
                "resolved_pageid": final_meta["pageid"],
                "revid": revid,
                "revision_timestamp_utc": final_meta["timestamp"],
                "redirect_hops": max(0, len(chain) - 1),
                "json": os.path.basename(json_path),
                "wikitext": os.path.basename(txt_path),
            }
        )

        print(
            f"[OK] pageid={args.pageid} @ <= {ts_z} -> '{title}' (pageid={final_meta['pageid']}) oldid={revid}"
        )

    summary_path = os.path.join(args.out, f"PAGEID_{args.pageid}__SUMMARY.json")
    write_json(summary_path, summary)
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)