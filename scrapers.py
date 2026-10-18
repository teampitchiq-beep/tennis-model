import hashlib
import json
import os
import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

import surfaces

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/121.0.0.0 Safari/537.36")

HERE = os.path.dirname(os.path.abspath(__file__))
RANK_CACHE = os.path.join(HERE, "rankings.json")
FORM_CACHE = os.path.join(HERE, "form.json")
RANK_TTL = 24 * 3600
FORM_TTL = 12 * 3600
CACHE_VERSION = 8

_MEM_RANKS = None
_MEM_FORM = None


def _stable_id(*parts):
    s = "|".join(str(p) for p in parts)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def _get(url, timeout=25):
    r = requests.get(url, headers={"User-Agent": UA, "Accept": "text/html"},
                     timeout=timeout)
    r.raise_for_status()
    return r.text


# =================== NAME MATCHING ===================
def _surname_sig(name):
    if not name:
        return ""
    tokens = [t.strip(".").lower() for t in re.split(r"\s+", name.strip()) if t]
    tokens = [re.sub(r"[^a-z]", "", t) for t in tokens]
    tokens = [t for t in tokens if t]
    if len(tokens) <= 1:
        return tokens[0] if tokens else ""
    return "".join(tokens[:-1])


def _name_match(full, display):
    a = _surname_sig(full)
    b = _surname_sig(display)
    return bool(a) and a == b


# =================== RANKINGS ===================
def _parse_ranking_page(html, tour):
    soup = BeautifulSoup(html, "lxml")
    out = {}
    for tr in soup.find_all("tr"):
        t_name = tr.find("td", class_="t-name")
        rank_td = tr.find("td", class_="rank")
        if not t_name or not rank_td:
            continue
        a = t_name.find("a", href=re.compile(r"/player/"))
        if not a:
            continue
        rank_txt = rank_td.get_text(strip=True).rstrip(".")
        if not rank_txt.isdigit():
            continue
        name = a.get_text(strip=True)
        if name:
            out[name] = {"rank": int(rank_txt), "tour": tour,
                         "url": a.get("href", "")}
    return out


def _fetch_rankings(max_pages=6):
    results = {}
    for tour in ("atp", "wta"):
        base = "atp-men" if tour == "atp" else "wta-women"
        for page in range(1, max_pages + 1):
            url = f"https://www.tennisexplorer.com/ranking/{base}/"
            if page > 1:
                url += f"?page={page}"
            try:
                html = _get(url)
            except Exception as e:
                print(f"[rankings] {tour} p{page}: {e}")
                break
            parsed = _parse_ranking_page(html, tour)
            if not parsed:
                break
            new = {k: v for k, v in parsed.items() if k not in results}
            if not new:
                break
            results.update(new)
            print(f"[rankings] {tour} p{page}: +{len(new)} (total {len(results)})")
    return results


def load_rankings(force=False):
    global _MEM_RANKS
    if _MEM_RANKS and not force:
        return _MEM_RANKS
    if not force and os.path.exists(RANK_CACHE):
        age = time.time() - os.path.getmtime(RANK_CACHE)
        if age < RANK_TTL:
            try:
                with open(RANK_CACHE) as f:
                    _MEM_RANKS = json.load(f)
                sample = next(iter(_MEM_RANKS.values()), {})
                if "url" not in sample:
                    print("[rankings] cache missing URLs — refetching")
                else:
                    print(f"[rankings] loaded {len(_MEM_RANKS)} from cache")
                    return _MEM_RANKS
            except Exception as e:
                print(f"[rankings] cache read failed: {e}")
    print("[rankings] fetching fresh")
    fresh = _fetch_rankings()
    if fresh:
        try:
            with open(RANK_CACHE, "w") as f:
                json.dump(fresh, f)
        except Exception as e:
            print(f"[rankings] cache write failed: {e}")
        _MEM_RANKS = fresh
    else:
        _MEM_RANKS = _MEM_RANKS or {}
    return _MEM_RANKS


def _player_url_from_rankings(name, rankings):
    if not name or not rankings:
        return None
    sig = _surname_sig(name)
    if not sig:
        return None
    if name in rankings and rankings[name].get("url"):
        return rankings[name]["url"]
    for k, v in rankings.items():
        if _surname_sig(k) == sig and v.get("url"):
            return v["url"]
    return None


# =================== FORM ===================
def _target_won_score(score_text):
    if not score_text:
        return False
    sets = re.split(r",\s*", score_text.strip())
    target_sets = 0
    opp_sets = 0
    for s in sets:
        m = re.match(r"^(\d+)-(\d+)", s.strip())
        if not m:
            continue
        t, o = int(m.group(1)), int(m.group(2))
        if t > o:
            target_sets += 1
        else:
            opp_sets += 1
    return target_sets > opp_sets


def _parse_player_form(html, player_name):
    soup = BeautifulSoup(html, "lxml")
    matches = []
    seen = set()

    for tr in soup.find_all("tr"):
        date_td = tr.find("td", class_="first")
        if not date_td:
            continue
        date_txt = date_td.get_text(strip=True)
        if not re.match(r"^\d{1,2}\.\d{1,2}\.?$", date_txt):
            continue

        tname = tr.find("td", class_="t-name")
        if not tname:
            continue
        links = tname.find_all("a", href=re.compile(r"/player/"))
        if len(links) < 2:
            continue

        p1_link, p2_link = links[0], links[1]
        p1 = p1_link.get_text(strip=True)
        p2 = p2_link.get_text(strip=True)
        if not p1 or not p2 or "/" in p1 or "/" in p2:
            continue

        if _name_match(player_name, p1):
            opponent = p2
        elif _name_match(player_name, p2):
            opponent = p1
        else:
            continue

        tl = tr.find("td", class_="tl")
        if not tl:
            continue
        score_text = tl.get_text(strip=True)
        if not score_text:
            continue

        is_winner = _target_won_score(score_text)

        key = (date_txt, opponent)
        if key in seen:
            continue
        seen.add(key)

        matches.append({
            "date": date_txt,
            "opponent": opponent,
            "result": "W" if is_winner else "L",
            "score": score_text,
        })

        if len(matches) >= 10:
            break

    return matches


def _fetch_form_for_player(name, rankings):
    url = _player_url_from_rankings(name, rankings)
    if not url:
        print(f"[form] {name}: no URL found")
        return None
    if not url.startswith("http"):
        url = "https://www.tennisexplorer.com" + url
    try:
        html = _get(url, timeout=20)
        matches = _parse_player_form(html, name)
        if not matches:
            print(f"[form] {name}: 0 matches")
            return None
        w = sum(1 for m in matches if m["result"] == "W")
        seq = "".join(m["result"] for m in matches[:10])
        print(f"[form] {name}: {seq} ({w}-{len(matches)-w})")
        return {"matches": matches[:10]}
    except Exception as e:
        print(f"[form] {name}: {e}")
        return None


def load_form(force=False):
    global _MEM_FORM
    if _MEM_FORM and not force:
        return _MEM_FORM
    if not force and os.path.exists(FORM_CACHE):
        age = time.time() - os.path.getmtime(FORM_CACHE)
        if age < FORM_TTL:
            try:
                with open(FORM_CACHE) as f:
                    data = json.load(f)
                if data.get("__v") != CACHE_VERSION:
                    print("[form] cache version mismatch — refetching")
                    _MEM_FORM = {}
                    return _MEM_FORM
                _MEM_FORM = data.get("players", {})
                print(f"[form] loaded {len(_MEM_FORM)} from cache")
                return _MEM_FORM
            except Exception as e:
                print(f"[form] cache read failed: {e}")
    _MEM_FORM = _MEM_FORM or {}
    return _MEM_FORM


def save_form(cache):
    global _MEM_FORM
    _MEM_FORM = cache
    try:
        with open(FORM_CACHE, "w") as f:
            json.dump({"__v": CACHE_VERSION, "players": cache}, f)
    except Exception as e:
        print(f"[form] cache write failed: {e}")


# =================== TENNISEXPLORER MATCHES ===================
def head_info(tr):
    """Parse a tournament header row -> (name, tour, is_doubles).

    Normal headers carry a link (/<slug>/<year>/atp-men/). The aggregate
    "Futures <year>" header has no link at all (all ITF men's matches are
    listed under it), so fall back to the cell text and the type-men/women icon.
    """
    td = tr.find("td", class_="t-name")
    if td is None:
        return None
    a = td.find("a")
    name = (a.get_text(strip=True) if a else td.get_text(strip=True)) or "Tennis"
    href = a.get("href", "") if a else ""
    if "wta-women" in href:
        tour = "wta"
    elif "atp-men" in href:
        tour = "atp"
    elif td.find("span", class_=re.compile(r"type-men")):
        tour = "atp"
    elif td.find("span", class_=re.compile(r"type-women")):
        tour = "wta"
    else:
        tour = None
    doubles = "type=double" in href or bool(
        tr.find("span", class_=re.compile(r"type-(men|women)4")))
    return name, tour, doubles


def _parse_te_page(html, month=None):
    """Parse a TennisExplorer day page into singles matches.

    Tournament headers are `<tr class="head ..."><td class="t-name"><a href=
    "/<slug>/<year>/atp-men/">Name</a>`. The previous parser looked for a
    /tournament/ link on a row WITHOUT t-name, which never matched, so every
    match was labelled "Tennis". The header also tells us the tour
    (atp-men / wta-women), which is needed to resolve player names.
    """
    soup = BeautifulSoup(html, "lxml")
    matches = []
    seen = set()

    for table in soup.find_all("table", class_="result"):
        rows = table.find_all("tr")
        current_tournament = "Tennis"
        current_tour = None
        skip_block = False
        i = 0
        while i < len(rows):
            tr = rows[i]

            if "head" in (tr.get("class") or []):
                info = head_info(tr)
                if info:
                    current_tournament, current_tour, skip_block = info
                i += 1
                continue

            if skip_block:
                i += 1
                continue

            t_name = tr.find("td", class_="t-name")
            if not t_name:
                i += 1
                continue
            a1 = t_name.find("a", href=re.compile(r"/player/"))
            if not a1:
                i += 1
                continue

            if i + 1 >= len(rows):
                break
            next_tr = rows[i + 1]
            next_t = next_tr.find("td", class_="t-name")
            if not next_t:
                i += 1
                continue
            a2 = next_t.find("a", href=re.compile(r"/player/"))
            if not a2:
                i += 1
                continue

            n1 = a1.get_text(strip=True)
            n2 = a2.get_text(strip=True)

            te_match_id = None
            detail_link = tr.find("a", href=re.compile(r"/match-detail/"))
            if not detail_link:
                detail_link = next_tr.find("a", href=re.compile(r"/match-detail/"))
            if detail_link:
                m = re.search(r"id=(\d+)", detail_link.get("href", ""))
                if m:
                    te_match_id = int(m.group(1))

            i += 2

            if not n1 or not n2 or "/" in n1 or "/" in n2 or n1 == n2:
                continue

            key = (n1, n2, current_tournament)
            if key in seen:
                continue
            seen.add(key)

            time_td = tr.find("td", class_="time")
            time_str = time_td.get_text(strip=True) if time_td else ""

            surface = surfaces.guess(current_tournament, month) or "unknown"

            matches.append({
                "id": f"te_{_stable_id('te', n1, n2, current_tournament)}",
                "source": "tennisexplorer",
                "tournament": current_tournament,
                "tour": current_tour,
                "level": surfaces.level_from_name(current_tournament),
                "surface": surface.lower(), "round": "", "status": time_str,
                "p1_name": n1, "p2_name": n2,
                "p1_rank": None, "p2_rank": None,
                "p1_url": a1.get("href", ""),
                "p2_url": a2.get("href", ""),
                "te_match_id": te_match_id,
            })
    return matches


def from_tennisexplorer(target_date=None):
    d = target_date or date.today()
    url = ("https://www.tennisexplorer.com/matches/"
           f"?type=all&year={d.year}&month={d.month:02d}&day={d.day:02d}")
    print(f"[tennisexplorer] fetching {url}")
    html = _get(url)
    print(f"[tennisexplorer] {len(html)} bytes")
    parsed = _parse_te_page(html, month=d.month)
    names = sorted({m["tournament"] for m in parsed})
    print(f"[tennisexplorer] tournaments: {names[:8]}{'...' if len(names) > 8 else ''}")
    return parsed


# =================== AGGREGATOR ===================
def get_todays_matches(target_date=None, enrich_ranks=True,
                       ranked_only=True, enrich_form=True,
                       max_form=25):
    try:
        matches = from_tennisexplorer(target_date)
        print(f"[scrapers] tennisexplorer: {len(matches)} matches")
    except Exception as e:
        print(f"[scrapers] tennisexplorer failed: {e}")
        return []

    if not matches:
        return []

    rankings = load_rankings()
    idx = build_rank_index(rankings)

    for m in matches:
        if m.get("p1_rank") is None:
            m["p1_rank"] = get_rank(m["p1_name"], idx)
        if m.get("p2_rank") is None:
            m["p2_rank"] = get_rank(m["p2_name"], idx)
    hits = sum(1 for m in matches if m.get("p1_rank") and m.get("p2_rank"))
    print(f"[scrapers] ranked both sides: {hits}/{len(matches)}")

    if ranked_only:
        before = len(matches)
        matches = [m for m in matches
                   if m.get("p1_rank") and m.get("p2_rank")]
        print(f"[scrapers] ranked-only filter: {len(matches)}/{before}")

    if enrich_form and matches:
        form_cache = load_form()
        to_fetch = []
        seen_names = set()
        for m in matches:
            for nm in (m["p1_name"], m["p2_name"]):
                if nm in seen_names or nm in form_cache:
                    continue
                seen_names.add(nm)
                to_fetch.append(nm)
        to_fetch = to_fetch[:max_form]
        fetched = 0
        for nm in to_fetch:
            data = _fetch_form_for_player(nm, rankings)
            if data:
                form_cache[nm] = data
                fetched += 1
        if fetched:
            save_form(form_cache)
        print(f"[form] fetched {fetched}/{len(to_fetch)}, cache={len(form_cache)}")

    return matches


# =================== RANK HELPERS ===================
def _name_keys(display_name):
    if not display_name:
        return []
    cleaned = re.sub(r"[^A-Za-z .\-']", "", display_name).strip()
    tokens = [t for t in re.split(r"\s+", cleaned) if t]
    if not tokens:
        return []
    drop = {"de", "van", "del", "della", "di", "da", "dos"}
    if len(tokens) > 1 and tokens[0].lower() in drop:
        tokens = tokens[1:]
    first = tokens[0].lower()
    initial = ""
    for t in tokens[1:]:
        t = t.strip(".")
        if t:
            initial = t[0].lower()
            break
    keys = [(first, initial)]
    if initial:
        keys.append((first, ""))
    return keys


def build_rank_index(rankings):
    idx = {}
    for full, info in rankings.items():
        for key in _name_keys(full):
            existing = idx.get(key)
            if existing is None or info["rank"] < existing:
                idx[key] = info["rank"]
    return idx


def get_rank(display_name, rank_index):
    for key in _name_keys(display_name):
        if key in rank_index:
            return rank_index[key]
    return None
