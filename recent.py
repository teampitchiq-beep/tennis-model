"""
Keeps Elo ratings current when the Sackmann archive lags behind.

Backtest: ratings that are 4.5 months stale cost ~0.027 log loss and ~2.5pp
accuracy, as much as everything else tuned combined. The archive copy in this
project stops months before today, so finished matches are pulled from
TennisExplorer's daily results pages, stored in recent_results.db, and fed into
the Elo calculation after the archive's last match.

    python recent.py              # sync everything missing up to yesterday
    python recent.py --since 2026-06-01

Only clearly finished singles matches are stored: the winner must have won 2
sets (3 in men's Grand Slams). Retirements/walkovers with an undecided set
count are skipped on purpose. Doubles and UTR/exhibition series are ignored.
"""
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

import requests
from bs4 import BeautifulSoup

import scrapers
import surfaces

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "recent_results.db")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
URL = "https://www.tennisexplorer.com/results/?type=all&year={y}&month={m}&day={d}"
REQUEST_DELAY = 1.0
MAX_DAYS_PER_RUN = 220

_SLAMS = ("australian open", "roland garros", "french open", "wimbledon", "us open")
_SKIP_TOURNAMENT = re.compile(r"\butr\b|exhibition|ntrp|\bliga\b|team", re.I)


# ----------------------------------------------------------------- storage
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            key TEXT PRIMARY KEY,
            day TEXT, tour TEXT, tournament TEXT, level TEXT, surface TEXT,
            best_of INTEGER, winner_name TEXT, loser_name TEXT, score TEXT
        )""")
    conn.execute("CREATE TABLE IF NOT EXISTS days_done (day TEXT PRIMARY KEY)")
    return conn


# ----------------------------------------------------------------- parsing
def _cell_games(td):
    """Games in a set cell; tiebreak points are in a <sup> and must not be
    glued onto the games number ('7' + '4' != 74)."""
    td = BeautifulSoup(str(td), "lxml").find("td")
    for sup in td.find_all("sup"):
        sup.extract()
    txt = td.get_text(strip=True)
    return int(txt) if txt.isdigit() else None


def _int(td):
    t = td.get_text(strip=True) if td else ""
    return int(t) if t.isdigit() else None


def parse_results_page(html, day):
    """-> list of finished singles matches (dicts) for that day."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        tournament, tour, skip_block = "", None, False
        i = 0
        while i < len(rows):
            tr = rows[i]
            if "head" in (tr.get("class") or []):
                info = scrapers.head_info(tr)
                if info:
                    tournament, tour, doubles = info
                    skip_block = (doubles
                                  or bool(_SKIP_TOURNAMENT.search(tournament))
                                  or tour is None)
                i += 1
                continue
            if skip_block or i + 1 >= len(rows):
                i += 1
                continue

            r1, r2 = tr, rows[i + 1]
            n1 = r1.find("td", class_="t-name")
            n2 = r2.find("td", class_="t-name")
            a1 = n1.find("a", href=re.compile(r"/player/")) if n1 else None
            a2 = n2.find("a", href=re.compile(r"/player/")) if n2 else None
            if not a1 or not a2:
                i += 1
                continue
            i += 2

            name1, name2 = a1.get_text(strip=True), a2.get_text(strip=True)
            if not name1 or not name2 or "/" in name1 or "/" in name2:
                continue

            s1, s2 = _int(r1.find("td", class_="result")), _int(r2.find("td", class_="result"))
            if s1 is None or s2 is None or s1 == s2:
                continue
            best_of = 5 if (tour == "atp" and any(
                x in tournament.lower() for x in _SLAMS)) else 3
            need = 3 if best_of == 5 else 2
            if max(s1, s2) < need:
                continue                       # unfinished / retirement: skip

            g1 = [_cell_games(td) for td in r1.find_all("td", class_="score")]
            g2 = [_cell_games(td) for td in r2.find_all("td", class_="score")]
            winner_first = s1 > s2
            parts = []
            for x, y in zip(g1, g2):
                if x is None or y is None:
                    continue
                parts.append(f"{x}-{y}" if winner_first else f"{y}-{x}")
            winner, loser = (name1, name2) if winner_first else (name2, name1)

            level = surfaces.level_from_name(tournament)
            level = "tour" if level == "other" else level
            month = int(day[5:7])
            out.append({
                "day": day, "tour": tour, "tournament": tournament,
                "level": level, "surface": surfaces.guess(tournament, month),
                "best_of": best_of, "winner_name": winner, "loser_name": loser,
                "score": " ".join(parts),
            })
    return out


# ----------------------------------------------------------------- fetching
def _fetch_day(day):
    y, m, d = int(day[:4]), int(day[5:7]), int(day[8:10])
    url = URL.format(y=y, m=f"{m:02d}", d=f"{d:02d}")
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            if r.status_code == 200 and "result" in r.text:
                return r.text
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def sync(since=None, until=None, verbose=True):
    """Fetch and store every missing day in [since, until]. Returns #matches added."""
    today = datetime.now(timezone.utc).date()
    until = until or (today - timedelta(days=1))       # today is incomplete
    if since is None:
        return 0
    conn = _conn()
    done = {r["day"] for r in conn.execute("SELECT day FROM days_done")}
    days, d = [], since
    while d <= until:
        iso = d.isoformat()
        if iso not in done:
            days.append(iso)
        d += timedelta(days=1)
    days = days[:MAX_DAYS_PER_RUN]
    added = 0
    for n, iso in enumerate(days, 1):
        html = _fetch_day(iso)
        if html is None:
            if verbose:
                print(f"[recent] {iso}: fetch failed (will retry next run)")
            continue
        try:
            matches = parse_results_page(html, iso)
        except Exception as e:
            print(f"[recent] {iso}: parse error {e}")
            continue
        for m in matches:
            key = f"{iso}|{m['tour']}|{m['winner_name']}|{m['loser_name']}"
            cur = conn.execute(
                "INSERT OR IGNORE INTO results VALUES (?,?,?,?,?,?,?,?,?,?)",
                (key, iso, m["tour"], m["tournament"], m["level"], m["surface"],
                 m["best_of"], m["winner_name"], m["loser_name"], m["score"]))
            added += cur.rowcount
        conn.execute("INSERT OR IGNORE INTO days_done VALUES (?)", (iso,))
        conn.commit()
        if verbose and (n % 10 == 0 or n == len(days)):
            print(f"[recent] {n}/{len(days)} days · {added} matches stored")
        time.sleep(REQUEST_DELAY)
    conn.close()
    return added


# ----------------------------------------------------------------- to Elo
def load_extra_rows(dataset_rows, index, base_ratings):
    """Stored results -> Elo rows (resolved to player ids) that come after the
    archive. Matches already present in the archive are dropped."""
    import players
    if not dataset_rows or not os.path.exists(DB_PATH):
        return []
    last = dataset_rows[-1]["_date"]
    seen = {(int(r["winner_id"]), int(r["loser_id"]))
            for r in dataset_rows
            if r["_date"] >= last - timedelta(days=21)
            and r.get("winner_id") and r.get("loser_id")}
    conn = _conn()
    recs = conn.execute(
        "SELECT * FROM results WHERE day >= ? ORDER BY day",
        ((last - timedelta(days=21)).strftime("%Y-%m-%d"),)).fetchall()
    conn.close()

    out, unresolved = [], 0
    for r in recs:
        w = players.resolve(r["winner_name"], index, tour=r["tour"], ratings=base_ratings)
        l = players.resolve(r["loser_name"], index, tour=r["tour"], ratings=base_ratings)
        if not w or not l or w == l:
            unresolved += 1
            continue
        day = datetime.strptime(r["day"], "%Y-%m-%d")
        if day <= last + timedelta(days=21) and (w, l) in seen:
            continue                                   # already in the archive
        out.append({
            "winner_id": w, "loser_id": l, "surface": r["surface"] or "",
            "score": r["score"], "best_of": r["best_of"], "_date": day,
            "_tour": r["tour"], "_level": r["level"], "round": "", "match_num": 0,
        })
    print(f"[recent] {len(out)} stored results added to Elo "
          f"({unresolved} skipped: unresolved names)")
    return out


def main():
    import dataset
    rows = dataset.load_all_matches()
    since = None
    if "--since" in sys.argv:
        since = datetime.strptime(sys.argv[sys.argv.index("--since") + 1], "%Y-%m-%d").date()
    elif rows:
        since = (rows[-1]["_date"] - timedelta(days=21)).date()
    print(f"[recent] syncing from {since}")
    n = sync(since)
    print(f"[recent] done, {n} new matches stored")


if __name__ == "__main__":
    main()
