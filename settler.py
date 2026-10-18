"""
Automatic settlement of open MW picks (tracker).

- Settles open picks (won/lost/void)
- When a result is determined, also stores it in results.db
  for future cross-reference (H2H, historical accuracy, etc.)
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

import tracker
import results

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
            return {"winner": name_a, "loser": name_b,
                    "score": score, "sets_a": sets_a, "sets_b": sets_b}
        elif sets_b > sets_a:
            return {"winner": name_b, "loser": name_a,
                    "score": score, "sets_a": sets_b, "sets_b": sets_a}

    return None


def _extract_meta(soup):
    meta = {"tournament": "", "round": "", "surface": "", "date": ""}
    try:
        title = soup.find("title")
        if title:
            meta["tournament"] = title.get_text(strip=True)[:80]

        tour_link = soup.find("a", href=re.compile(r"/tournament/"))
        if tour_link:
            meta["tournament"] = tour_link.get_text(strip=True)

        date_m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})",
                           soup.get_text(" ", strip=True)[:2000])
        if date_m:
            meta["date"] = f"{date_m.group(3)}-{date_m.group(2)}-{date_m.group(1)}"
    except Exception:
        pass
    return meta


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
    market = (p.get("market") or "mw").lower()
    if market != "mw":
        return (p["id"], "void", "legacy non-MW pick", None)

    if _age_days(p) > STALE_DAYS:
        return (p["id"], "void", f"stale (>{STALE_DAYS}d old)", None)

    te_id = p.get("te_match_id")
    if not te_id:
        return (p["id"], "skip", "no te_match_id", None)

    soup = _fetch_te_match(te_id)
    if soup is None:
        return (p["id"], "error", "fetch failed", None)

    result = _parse_result(soup)
    if not result:
        return (p["id"], "skip", "match not finished", None)

    winner = result["winner"]
    loser = result["loser"]
    score = result["score"]

    meta = _extract_meta(soup)
    event_date = meta.get("date") or ""

    result_dict = {
        "te_match_id": te_id,
        "event_date": event_date,
        "tournament": p.get("tournament") or meta.get("tournament") or "",
        "round_": meta.get("round", ""),
        "surface": p.get("surface") or meta.get("surface") or "",
        "p1_name": p.get("pick_name", ""),
        "p2_name": p.get("opponent_name", ""),
        "winner_name": winner,
        "loser_name": loser,
        "score": score,
    }

    if _name_match(p["pick_name"], winner):
        return (p["id"], "won", f"beat {loser}", result_dict)
    if _name_match(p["pick_name"], loser):
        return (p["id"], "lost", f"lost to {winner}", result_dict)
    return (p["id"], "skip", f"no name match (winner={winner})", result_dict)


def settle_open_picks(verbose=True):
    picks = tracker.list_picks(status="open")
    if not picks:
        return {"checked": 0, "settled": 0, "skipped": 0,
                "errors": 0, "voided": 0, "results_logged": 0,
                "results": [], "elapsed": 0.0}

    settled = 0
    skipped = 0
    errors = 0
    voided = 0
    results_logged = 0
    results_summary = []

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(_process_pick, p): p for p in picks}
        for f in as_completed(futures):
            p = futures[f]
            try:
                pid, status, detail, result_dict = f.result()
            except Exception as e:
                pid, status, detail, result_dict = p["id"], "error", str(e), None

            if result_dict:
                rid = results.add_result(
                    te_match_id=result_dict["te_match_id"],
                    event_date=result_dict["event_date"],
                    tournament=result_dict["tournament"],
                    round_=result_dict["round_"],
                    surface=result_dict["surface"],
                    p1_name=result_dict["p1_name"],
                    p2_name=result_dict["p2_name"],
                    winner_name=result_dict["winner_name"],
                    loser_name=result_dict["loser_name"],
                    score=result_dict["score"],
                    source="settler",
                )
                if rid:
                    results_logged += 1

            if status == "won":
                tracker.settle_pick(pid, "won")
                settled += 1
                results_summary.append((pid, p["pick_name"], "won"))
                if verbose:
                    print(f"[settler] pick {pid}: {p['pick_name']} WON ({detail})")
            elif status == "lost":
                tracker.settle_pick(pid, "lost")
                settled += 1
                results_summary.append((pid, p["pick_name"], "lost"))
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
              f"settled={settled} voided={voided} "
              f"skipped={skipped} errors={errors} "
              f"results_logged={results_logged}")

    return {
        "checked": len(picks),
        "settled": settled,
        "skipped": skipped,
        "errors": errors,
        "voided": voided,
        "results_logged": results_logged,
        "results": results_summary,
        "elapsed": elapsed,
    }
