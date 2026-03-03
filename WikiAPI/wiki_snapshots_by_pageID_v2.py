from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://en.wikipedia.org/w/api.php"
UA = "wiki-snapshot-script/1.3 (+https://en.wikipedia.org/)"


# -----------------------------
# HTTP helpers
# -----------------------------
def http_get_json(url: str, headers: dict | None = None, timeout: int = 60) -> dict:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as r:
        data = r.read().decode("utf-8", errors="replace")
    return json.loads(data)


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


# -----------------------------
# Redirect resolution
# -----------------------------
def extract_redirect_target(wikitext: str) -> str | None:
    m = re.match(r"(?is)^\s*#redirect\s*\[\[([^\]|#]+)", wikitext)
    return m.group(1).strip() if m else None


def resolve_redirect_chain_from_pageid(pageid: int, ts_z: str, max_hops: int = 5):
    chain: list[dict] = []

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
# Public API (import-friendly)
# -----------------------------
@dataclass(frozen=True)
class SnapshotResult:
    pageid_requested: int
    requested_timestamp_utc: str
    resolved_title: str
    resolved_pageid: int
    revid: int
    parentid: int | None
    revision_timestamp_utc: str
    wikipedia_oldid_url: str
    redirect_chain: list[dict]
    wikitext: str


def fetch_snapshot(pageid: int, snapshot_ts: str, *, max_redirect_hops: int = 5) -> SnapshotResult:
    ts_z = parse_iso8601_z(snapshot_ts)
    meta, wikitext, chain = resolve_redirect_chain_from_pageid(pageid, ts_z, max_hops=max_redirect_hops)

    revid = int(meta["revid"])
    return SnapshotResult(
        pageid_requested=int(pageid),
        requested_timestamp_utc=ts_z,
        resolved_title=str(meta["title"]),
        resolved_pageid=int(meta["pageid"]),
        revid=revid,
        parentid=meta.get("parentid"),
        revision_timestamp_utc=str(meta["timestamp"]),
        wikipedia_oldid_url=f"https://en.wikipedia.org/w/index.php?oldid={revid}",
        redirect_chain=chain,
        wikitext=wikitext,
    )


def save_snapshot(result: SnapshotResult, out_dir: str, *, base: str | None = None) -> dict[str, Any]:
    """
    Writes:
      - <base>.wikitext.txt
      - <base>.json (metadata-only + pointer to wikitext file)
    """
    ensure_dir(out_dir)

    # Default base: YYYY-MM_<pageid>_oldid_<revid>
    if base is None:
        snap_tag = result.requested_timestamp_utc[:7]  # YYYY-MM
        base = f"{snap_tag}_{result.pageid_requested}_oldid_{result.revid}"

    txt_path = os.path.join(out_dir, base + ".wikitext.txt")
    json_path = os.path.join(out_dir, base + ".json")

    write_text(txt_path, result.wikitext)

    record = {
        "title": result.resolved_title,
        "pageid_requested": result.pageid_requested,
        "resolved_pageid": result.resolved_pageid,
        "requested_timestamp_utc": result.requested_timestamp_utc,
        "revision_timestamp_utc": result.revision_timestamp_utc,
        "revid": result.revid,
        "parentid": result.parentid,
        "wikipedia_oldid_url": result.wikipedia_oldid_url,
        "redirect_chain": result.redirect_chain,
        "wikitext_file": os.path.basename(txt_path),
        "wikitext_chars": len(result.wikitext),
        "wikitext_lines": result.wikitext.count("\n") + 1,
    }
    write_json(json_path, record)

    return {"json": json_path, "wikitext": txt_path, "base": base}