"""
Infer a tournament's playing surface from historical data.

TennisExplorer's match list does not state the surface, and the old scraper
hardcoded "hard" for everything, which made the surface-specific Elo useless.
Here we build a city -> surface lookup from the Sackmann archive (most recent
editions weigh most) and use it to guess the surface from a tournament name.

guess() returns "Hard" / "Clay" / "Grass" / "Carpet", or None when unknown.
None is handled downstream by using the surface-neutral (overall) rating
instead of guessing wrong.
"""
import json
import os
import re
import unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "surface_lookup.json")

_SURF = {"hard": "Hard", "clay": "Clay", "grass": "Grass", "carpet": "Carpet"}

# Words that describe the event tier, not the place.
_NOISE = {
    "challenger", "ch", "itf", "wta", "atp", "masters", "open", "international",
    "classic", "championships", "championship", "cup", "trophy", "tour", "finals",
    "qualifying", "qualification", "final", "series", "tournament", "women",
    "men", "m15", "m25", "w15", "w25", "w35", "w50", "w60", "w75", "w100",
    "m15h", "m25h", "w15h", "w25h", "w35h", "w50h", "w60h", "w75h", "w100h",
}

_LOOKUP = None  # {city_key: {"all": [surface weights], ...}}


def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def normalize_name(name):
    """'Santa Cruz De La Sierra challenger' -> 'santa cruz de la sierra'."""
    if not name:
        return ""
    s = _strip_accents(str(name)).lower()
    s = re.sub(r"\+h\b", "", s)
    s = re.sub(r"[^a-z ]", " ", s)          # digits/punctuation out
    toks = [t for t in s.split() if t and t not in _NOISE]
    return " ".join(toks)


def level_from_name(name):
    """Classify a TennisExplorer tournament label: 'chall', 'itf', 'other'."""
    n = (name or "").lower()
    if "challenger" in n or re.search(r"\bch\b", n):
        return "chall"
    if "itf" in n or "futures" in n or re.search(r"\b[mw](15|25|35|50|60|75|100)\b", n):
        return "itf"
    return "other"


def _row_level(r):
    if r.get("_tour") == "wta":
        # WTA tour files only hold tour-level events
        return "tour"
    lv = r.get("tourney_level")
    if lv in ("G", "M", "A", "F", "D", "O"):
        return "tour"
    if lv == "C":
        return "chall"
    return "itf"


def build(rows, force=False):
    """Build (and cache) the lookup from dataset rows.

    lookup[city] = list of [level, month, surface, weight]
    Each tournament edition counted once (not once per match).
    """
    global _LOOKUP
    if _LOOKUP is not None and not force:
        return _LOOKUP
    if not rows:
        _LOOKUP = {}
        return _LOOKUP
    last_date = rows[-1]["_date"]
    seen = set()
    acc = defaultdict(float)
    for r in rows:
        surf = _SURF.get(str(r.get("surface", "")).strip().lower())
        if not surf:
            continue
        key = normalize_name(r.get("tourney_name"))
        if not key:
            continue
        ed = (r.get("tourney_id"), r["_tour"])
        if ed in seen:
            continue
        seen.add(ed)
        age_days = (last_date - r["_date"]).days
        weight = 0.5 ** (age_days / 730.0)      # 2-year half-life
        acc[(key, _row_level(r), r["_date"].month, surf)] += weight
    lookup = defaultdict(list)
    for (key, lvl, mon, surf), w in acc.items():
        lookup[key].append([lvl, mon, surf, w])
    _LOOKUP = dict(lookup)
    try:
        with open(CACHE, "w") as f:
            json.dump(_LOOKUP, f)
    except Exception:
        pass
    return _LOOKUP


def _load():
    global _LOOKUP
    if _LOOKUP is None:
        try:
            with open(CACHE) as f:
                _LOOKUP = json.load(f)
        except Exception:
            _LOOKUP = {}
    return _LOOKUP


def _month_gap(a, b):
    d = abs(a - b)
    return min(d, 12 - d)


def _decide(entries, level, month):
    if not entries:
        return None
    tot = defaultdict(float)
    for lvl, mon, surf, w in entries:
        f = 1.0 if lvl == level else 0.35
        if month:
            gap = _month_gap(mon, month)
            if gap > 1:
                continue    # edition from another season: not evidence
            f *= 1.0 if gap == 0 else 0.6
        tot[surf] += w * f
    total = sum(tot.values())
    if total <= 0:
        return None
    surf, w = max(tot.items(), key=lambda kv: kv[1])
    return surf if w / total >= 0.6 else None


def guess(tournament_name, month=None):
    """Best-guess surface for a tournament label, or None if unknown.

    month: 1-12 of the match date (editions recur at the same time every year,
    which separates e.g. Mallorca grass (June) from Mallorca clay).
    """
    lk = _load()
    key = normalize_name(tournament_name)
    if not key or not lk:
        return None
    level = level_from_name(tournament_name)
    level = "tour" if level == "other" else level
    entries = list(lk.get(key, []))
    if not entries:
        # 'edmond' vs 'edmond ok', 'santa cruz' vs 'santa cruz de la sierra'
        for k, v in lk.items():
            if k.startswith(key + " ") or key.startswith(k + " "):
                entries.extend(v)
    return _decide(entries, level, month)
