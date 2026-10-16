import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
ODDS_CACHE = os.path.join(HERE, "odds_cache.json")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/121.0.0.0 Safari/537.36")

BOOKMAKERS = [
    "Pinnacle", "bet365", "Betfair", "1xBet",
    "Unibet", "William Hill", "bwin", "Betway",
    "BetVictor", "Coral", "Ladbrokes", "Interwetten",
    "Marathonbet", "Betsson", "10bet",
]

PREFERRED = ["Pinnacle", "bet365", "Betfair", "1xBet"]

IMPLIED_MIN = 0.95
IMPLIED_MAX = 1.15

DEFAULT_CACHE_TTL = 1800


def _load_cache():
    if not os.path.exists(ODDS_CACHE):
        return {}
    try:
        with open(ODDS_CACHE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    try:
        with open(ODDS_CACHE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"[odds] cache write failed: {e}")


def _parse_float(s):
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None


def _validate_pair(o1, o2):
    if o1 is None or o2 is None:
        return False
    if not (1.01 <= o1 <= 50.0 and 1.01 <= o2 <= 50.0):
        return False
    implied = 1.0 / o1 + 1.0 / o2
    return IMPLIED_MIN <= implied <= IMPLIED_MAX


def _fetch_page(te_match_id):
    url = f"https://www.tennisexplorer.com/match-detail/?id={te_match_id}"
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        if r.status_code != 200:
            return None
        return BeautifulSoup(r.text, "lxml")
    except Exception as e:
        print(f"[odds] {te_match_id}: {e}")
        return None


def _pick_bookmarker(all_books):
    for pref in PREFERRED:
        if pref in all_books:
            return pref, all_books[pref]
    if all_books:
        first = next(iter(all_books))
        return first, all_books[first]
    return None, None


def _extract_book_and_odds(tr):
    cells = tr.find_all("td")
    if len(cells) < 3:
        return None, []

    book_name = None
    book_idx = None
    for i, c in enumerate(cells[:3]):
        t = c.get_text(strip=True)
        for bk in BOOKMAKERS:
            if bk.lower() in t.lower():
                book_name = bk
                book_idx = i
                break
        if book_name:
            break
    if not book_name:
        return None, []

    nums = []
    for j, c in enumerate(cells):
        if j == book_idx:
            continue
        v = _parse_float(c.get_text(strip=True))
        if v is not None and 1.01 <= v <= 50.0:
            nums.append(v)
    return book_name, nums


def _parse_mw(soup):
    all_books = {}
    for table in soup.find_all("table", class_="result"):
        classes = " ".join(table.get("class", []))
        if "odds-ou" in classes or "odds-ah" in classes or "odds-cs" in classes:
            continue
        for tr in table.find_all("tr"):
            if tr.find("table"):
                continue
            bk, nums = _extract_book_and_odds(tr)
            if not bk:
                continue
            for i in range(len(nums) - 1):
                o1, o2 = nums[i], nums[i + 1]
                if _validate_pair(o1, o2):
                    if bk not in all_books:
                        all_books[bk] = [o1, o2]
                    else:
                        old = all_books[bk]
                        old_imp = 1.0 / old[0] + 1.0 / old[1]
                        new_imp = 1.0 / o1 + 1.0 / o2
                        if abs(new_imp - 1.05) < abs(old_imp - 1.05):
                            all_books[bk] = [o1, o2]
                    break

    if not all_books:
        return None
    bk, pair = _pick_bookmarker(all_books)
    return {
        "p1_odds": pair[0],
        "p2_odds": pair[1],
        "bookmaker": bk,
        "all_books": all_books,
    }


def _parse_sets(soup, line=2.5):
    all_books = {}
    seen_bookmakers = set()
    in_section = False

    for tr in soup.find_all("tr"):
        if tr.find("table"):
            continue
        row_text = tr.get_text(" ", strip=True)
        if not row_text:
            continue
        row_lower = row_text.lower()

        m = re.search(r"over/under\s+(\d+(?:\.\d+)?)\s+sets", row_lower)
        if m:
            try:
                section_line = float(m.group(1))
            except ValueError:
                section_line = None
            if section_line == line:
                in_section = True
                seen_bookmakers = set()
            else:
                in_section = False
            continue

        if in_section:
            if (("over/under" in row_lower and "games" in row_lower)
                    or "asian handicap" in row_lower
                    or "correct score" in row_lower):
                in_section = False
                continue

        if not in_section:
            continue

        bk, nums = _extract_book_and_odds(tr)
        if not bk or bk in seen_bookmakers:
            continue

        for i in range(len(nums) - 1):
            o1, o2 = nums[i], nums[i + 1]
            if _validate_pair(o1, o2):
                all_books[bk] = [o1, o2]
                seen_bookmakers.add(bk)
                break

    if not all_books:
        return None
    bk, pair = _pick_bookmarker(all_books)
    return {
        "over_odds": pair[0],
        "under_odds": pair[1],
        "bookmaker": bk,
        "line": line,
        "all_books": all_books,
    }


def _fetch_both(te_match_id):
    soup = _fetch_page(te_match_id)
    if not soup:
        return None, None
    return _parse_mw(soup), _parse_sets(soup, line=2.5)


def fetch_match_odds_cached(te_match_id, ttl=DEFAULT_CACHE_TTL):
    if not te_match_id:
        return None, None
    cache = _load_cache()
    key = str(te_match_id)
    now = time.time()
    entry = cache.get(key)
    if entry and (now - entry.get("ts", 0) < ttl):
        return entry.get("mw"), entry.get("sets")

    mw, sets = _fetch_both(te_match_id)
    cache[key] = {"ts": now, "mw": mw, "sets": sets}

    if len(cache) > 500:
        sorted_items = sorted(cache.items(),
                              key=lambda kv: kv[1].get("ts", 0),
                              reverse=True)
        cache = dict(sorted_items[:500])

    _save_cache(cache)
    return mw, sets


def clear_cache():
    if os.path.exists(ODDS_CACHE):
        os.remove(ODDS_CACHE)
    print("[odds] cache cleared")
