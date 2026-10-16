"""
Automatic settlement of open picks.

For each open pick, fetch its TennisExplorer match-detail page and parse
the result. If the match has finished, settle the pick as won or lost.

Match-detail header format (TE convention, left-first):
    "Left Name 2 : 1 (4-6, 6-3, 6-2) Right Name"
"""
import re
import time

import requests
from bs4 import BeautifulSoup

import tracker

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/121.0.0.0 Safari/537.36")

# Matches: "Name A 2 : 1 (4-6, 6-3, 6-2) Name B"
# Also handles no-spaces and missing score string
HEADER_RE = re.compile(
    r"^(.+?)\s+(\d+)\s*:\s*(\d+)\s*(?:\(\s*([^)]*)\s*\))?\s+(.+?)\s*$"
)

# Rate limit between pick fetches (seconds)
FETCH_DELAY = 1.0


# ============= Helpers =============
def _surname_sig(name):
    if not name:
        return ""
    tokens = [t.strip(".").lower() for t in re.split(r"\s+", name.strip()) if t]
    tokens = [re.sub(r"[^a-z]", "", t) for t in tokens]
    tokens = [t for t in tokens if t]
    if len(tokens) <= 1:
        return tokens[0] if tokens else ""
    return "".join(tokens[:-1])


def _name_match(tracked, result_name):
    a = _surname_sig(tracked)
    b = _surname_sig(result_name)
    return bool(a) and a == b


def _fetch_te_match(te_id):
    url = f"https://www.tennisexplorer.com/match-detail/?id={te_id}"
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        if r.status_code != 200:
            return None
        return BeautifulSoup(r.text, "lxml")
    except Exception as e:
        print(f"[settler] {te_id}: {e}")
        return None


def _parse_result(soup):
    """
    Return (winner_name, loser_name) if the match has finished, else None.
    """
    # Match-detail page has a table with class 'result' near the top
    table = soup.find("table", class_="result")
    if not table:
        return None

    for tr in table.find_all("tr"):
        txt = tr.get_text(" ", strip=True)
        if not txt or ":" not in txt:
            continue

        m = HEADER_RE.match(txt)
        if not m:
            continue

        name_a = m.group(1).strip()
        sets_a = int(m.group(2))
        sets_b = int(m.group(3))
        score = (m.group(4) or "").strip()
        name_b = m.group(5).strip()

        # Sanity: best-of-3 gives max 3 sets; skip anything wild
        if sets_a > 5 or sets_b > 5:
            continue

        # Not started or still in progress: no completed sets
        if sets_a + sets_b == 0:
            continue

        # Check that the score string looks like completed sets
        # e.g. "4-6, 6-3, 6-2" — at least one completed set with a dash
        if score and "-" not in score:
            continue

        if sets_a > sets_b:
            return (name_a, name_b)
        elif sets_b > sets_a:
            return (name_b, name_a)

    return None


# ============= Main =============
def settle_open_picks(verbose=True):
    """
    Fetch result for every open pick, settle if determinable.
    Returns a dict with counts.
    """
    picks = tracker.list_picks(status="open")
    if not picks:
        return {"checked": 0, "settled": 0, "skipped": 0, "errors": 0}

    settled = 0
    skipped = 0
    errors = 0
    results = []

    for i, p in enumerate(picks):
        te_id = p.get("te_match_id")
        if not te_id:
            skipped += 1
            if verbose:
                print(f"[settler] pick {p['id']}: no te_match_id, skipping")
            continue

        soup = _fetch_te_match(te_id)
        if soup is None:
            errors += 1
            continue

        result = _parse_result(soup)
        if not result:
            skipped += 1
            if verbose:
                print(f"[settler] pick {p['id']}: match not finished")
            continue

        winner, loser = result

        if _name_match(p["pick_name"], winner):
            tracker.settle_pick(p["id"], "won")
            settled += 1
            results.append((p["id"], p["pick_name"], "won"))
            if verbose:
                print(f"[settler] pick {p['id']}: {p['pick_name']} WON")
        elif _name_match(p["pick_name"], loser):
            tracker.settle_pick(p["id"], "lost")
            settled += 1
            results.append((p["id"], p["pick_name"], "lost"))
            if verbose:
                print(f"[settler] pick {p['id']}: {p['pick_name']} LOST")
        else:
            skipped += 1
            if verbose:
                print(f"[settler] pick {p['id']}: no name match — "
                      f"pick={p['pick_name']}, result={winner}/{loser}")

        # Rate limit between fetches
        if i < len(picks) - 1:
            time.sleep(FETCH_DELAY)

    return {
        "checked": len(picks),
        "settled": settled,
        "skipped": skipped,
        "errors": errors,
        "results": results,
    }
