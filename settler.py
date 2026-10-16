"""
Automatic settlement of open MW picks.

- Skips non-MW picks (legacy Sets) — auto-voids them
- Auto-voids picks older than STALE_DAYS (unsettleable)
- Fetches in parallel (4 workers) to avoid blocking startup
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

import tracker

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/121.0.0.0 Safari/537.36")

HEADER_RE = re.compile(
    r"^(.+?)\s+(\d+)\s*:\s*(\d+)\s*(?:\(\s*([^)]*)\s*\))?\s+(.+?)\s*$"
)

FETCH_WORKERS = 4
FETCH_TIMEOUT = 15

STALE_DAYS = 5


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
        r = requests.get(url, headers={"User-Agent": UA},
                         timeout=FETCH_TIMEOUT)
        if r.status_code != 200:
            return None
        return BeautifulSoup(r.text, "lxml")
    except Exception:
        return None


def _parse_result(soup):
    """Return (winner_name, loser_name) if finished, else None."""
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

        if sets_a > 5 or sets_b > 5:
            continue
        if sets_a + sets_b == 0:
            continue
        if score and "-" not in score:
            continue

        if sets_a > sets_b:
            return (name_a, name_b)
        elif sets_b > sets_a:
            return (name_b, name_a)

    return None


def _age_days(pick):
    ts = pick.get("created_at")
    if not ts:
        return 0
    try:
        created = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - created).total_seconds() / 86400
    except Exception:
        return 0


def _process_pick(p):
    """
    Returns a tuple: (pick_id, status, detail)
    status: 'won' | 'lost' | 'void' | 'skip' | 'error'
    """
    market = (p.get("market") or "mw").lower()
    if market != "mw":
        return (p["id"], "void", "legacy non-MW pick")

    if _age_days(p) > STALE_DAYS:
        return (p["id"], "void", f"stale (>{STALE_DAYS}d old)")

    te_id = p.get("te_match_id")
    if not te_id:
        return (p["id"], "skip", "no te_match_id")

    soup = _fetch_te_match(te_id)
    if soup is None:
        return (p["id"], "error", "fetch failed")

    result = _parse_result(soup)
    if not result:
        return (p["id"], "skip", "match not finished")

    winner, loser = result

    if _name_match(p["pick_name"], winner):
        return (p["id"], "won", f"beat {loser}")
    if _name_match(p["pick_name"], loser):
        return (p["id"], "lost", f"lost to {winner}")
    return (p["id"], "skip", f"no name match (winner={winner})")


def settle_open_picks(verbose=True):
    picks = tracker.list_picks(status="open")
    if not picks:
        return {"checked": 0, "settled": 0, "skipped": 0,
                "errors": 0, "voided": 0, "results": [], "elapsed": 0.0}

    settled = 0
    skipped = 0
    errors = 0
    voided = 0
    results = []

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(_process_pick, p): p for p in picks}
        for f in as_completed(futures):
            p = futures[f]
            try:
                pid, status, detail = f.result()
            except Exception as e:
                pid, status, detail = p["id"], "error", str(e)

            if status == "won":
                tracker.settle_pick(pid, "won")
                settled += 1
                results.append((pid, p["pick_name"], "won"))
                if verbose:
                    print(f"[settler] pick {pid}: {p['pick_name']} WON ({detail})")
            elif status == "lost":
                tracker.settle_pick(pid, "lost")
                settled += 1
                results.append((pid, p["pick_name"], "lost"))
                if verbose:
                    print(f"[settler] pick {pid}: {p['pick_name']} LOST ({detail})")
            elif status == "void":
                tracker.settle_pick(pid, "void")
                voided += 1
                if verbose:
                    print(f"[settler] pick {pid}: VOID ({detail})")
            elif status == "error":
                errors += 1
                if verbose:
                    print(f"[settler] pick {pid}: error ({detail})")
            else:
                skipped += 1
                if verbose:
                    print(f"[settler] pick {pid}: {detail}")

    elapsed = time.time() - t0
    if verbose:
        print(f"[settler] done in {elapsed:.1f}s · "
              f"settled={settled} voided={voided} skipped={skipped} errors={errors}")

    return {
        "checked": len(picks),
        "settled": settled,
        "skipped": skipped,
        "errors": errors,
        "voided": voided,
        "results": results,
        "elapsed": elapsed,
    }
