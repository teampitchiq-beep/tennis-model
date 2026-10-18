"""
Elo rating engine (tuned by walk-forward evaluation, see evaluate.py).

What changed vs. the original:
  * Parameters were tuned on out-of-sample log loss (decaying K per player,
    lower K for lower-tier events, best-of-5 weighting, surface weight 0.4).
  * Surface names are normalised (the old code treated 'clay'/'carpet' from
    some CSV rows as Hard).
  * Matches are processed in true chronological order within a tournament.
  * The on-disk cache is versioned and its keys are converted back to int.
    (The old cache loaded with string keys, so every lookup by integer
    player_id missed and the live app silently fell back to rank-only.)
  * Each player keeps `n` (matches played) and `last` (last match date).
  * surface=None means "unknown surface": the overall rating is used.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ELO_CACHE = os.path.join(HERE, "elo_ratings.json")

MODEL_VERSION = 3
SURFACES = ("Hard", "Clay", "Grass", "Carpet")

PARAMS = {
    "k": 350.0,          # K = k / (n + koff) ** kshape, floored at kmin
    "koff": 3.0,
    "kshape": 0.4,
    "kmin": 12.0,
    "surface_k_ratio": 0.4,   # surface K as a fraction of overall K
    "surface_weight": 0.4,    # blended = w*surface + (1-w)*overall
    "margin": 1.0,            # games-margin multiplier strength
    "bo5": 1.4,               # best-of-5 matches carry more information
    "k_tour": 0.8,
    "k_chall": 0.8,
    "k_itf": 0.7,
}

ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "RR": 3, "R128": 4, "R64": 5,
               "R32": 6, "R16": 7, "QF": 8, "SF": 9, "BR": 9, "F": 10}


# ---------------------------------------------------------------- helpers
def normalize_surface(s):
    s = str(s or "").strip().lower()
    return {"hard": "Hard", "clay": "Clay", "grass": "Grass",
            "carpet": "Carpet"}.get(s)


def elo_expected(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def games_from_score(score):
    if not isinstance(score, str):
        return (0, 0)
    w = l = 0
    for part in score.split():
        part = part.split("(")[0]
        if "-" not in part:
            continue
        try:
            a, b = part.split("-")[:2]
            w += int(a)
            l += int(b)
        except ValueError:
            continue
    return (w, l)


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def level_class(row):
    """'tour' | 'chall' | 'itf' for a dataset (or recent-result) row."""
    if row.get("_level"):
        return row["_level"]
    if row.get("_tour") == "wta":
        return "tour"
    lv = row.get("tourney_level")
    if lv in ("G", "M", "A", "F", "D", "O"):
        return "tour"
    if lv == "C":
        return "chall"
    return "itf"


def sort_key(row):
    try:
        mn = int(row.get("match_num") or 0)
    except (TypeError, ValueError):
        mn = 0
    return (row["_date"], ROUND_ORDER.get(row.get("round"), 5), mn)


def _new_entry(date_iso):
    return {"overall": 1500.0, "Hard": 1500.0, "Clay": 1500.0,
            "Grass": 1500.0, "Carpet": 1500.0, "n": 0, "last": date_iso}


# ---------------------------------------------------------------- engine
class EloSystem:
    def __init__(self, params=None, ratings=None):
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        self.ratings = ratings if ratings is not None else {}

    def blended(self, pid, surface):
        e = self.ratings.get(pid)
        if e is None:
            return 1500.0
        if surface not in SURFACES:
            return e["overall"]
        w = self.p["surface_weight"]
        return w * e[surface] + (1 - w) * e["overall"]

    def _k(self, n):
        p = self.p
        return max(p["kmin"], p["k"] / (n + p["koff"]) ** p["kshape"])

    def update(self, w_id, l_id, surface, score, best_of, level, date):
        p = self.p
        date_iso = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
        ew = self.ratings.setdefault(w_id, _new_entry(date_iso))
        el = self.ratings.setdefault(l_id, _new_entry(date_iso))

        gw, gl = games_from_score(score)
        if gw + gl == 0:
            gw, gl = 12, 6
        mult = 1.0 + p["margin"] * abs(gw - gl) / (gw + gl)
        if str(best_of) == "5":
            mult *= p["bo5"]
        mult *= p.get("k_" + level, 1.0)

        exp_o = elo_expected(ew["overall"], el["overall"])
        kw, kl = self._k(ew["n"]), self._k(el["n"])
        ew["overall"] += kw * mult * (1 - exp_o)
        el["overall"] -= kl * mult * (1 - exp_o)

        if surface in SURFACES:
            exp_s = elo_expected(ew[surface], el[surface])
            r = p["surface_k_ratio"]
            ew[surface] += kw * r * mult * (1 - exp_s)
            el[surface] -= kl * r * mult * (1 - exp_s)

        ew["n"] += 1
        el["n"] += 1
        ew["last"] = date_iso
        el["last"] = date_iso


# ---------------------------------------------------------------- cache
def _signature(n_matches, last_date):
    blob = json.dumps(PARAMS, sort_keys=True) + f"|v{MODEL_VERSION}|{n_matches}|{last_date}"
    return hashlib.md5(blob.encode()).hexdigest()


def _load_cache(sig):
    try:
        with open(ELO_CACHE) as f:
            cached = json.load(f)
        if cached.get("__sig") != sig:
            return None
        return {int(k): v for k, v in cached["ratings"].items()}
    except Exception:
        return None


def compute_ratings(matches, extra_matches=None, force=False, save=True):
    """Walk through every match chronologically and return {int pid: entry}.

    matches:       Sackmann dataset rows.
    extra_matches: optional newer rows (same keys; see recent.py) that are
                   appended after the dataset, so ratings stay current when
                   the archive lags behind.
    save:          write the result to the cache (False for intermediate
                   calls so they don't evict the final cache).
    """
    rows = list(matches) + list(extra_matches or [])
    rows.sort(key=sort_key)
    last_date = rows[-1]["_date"].strftime("%Y-%m-%d") if rows else ""
    sig = _signature(len(rows), last_date)

    if not force:
        cached = _load_cache(sig)
        if cached is not None:
            print(f"[elo] loaded {len(cached)} players from cache "
                  f"(data through {last_date})")
            return cached

    system = EloSystem()
    used = 0
    for row in rows:
        w_id, l_id = _to_int(row.get("winner_id")), _to_int(row.get("loser_id"))
        if not w_id or not l_id:
            continue
        system.update(
            w_id, l_id,
            normalize_surface(row.get("surface")),
            row.get("score", ""),
            row.get("best_of"),
            level_class(row),
            row["_date"],
        )
        used += 1

    try:
        if not save:
            raise StopIteration
        tmp = ELO_CACHE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"__sig": sig, "__version": MODEL_VERSION,
                       "ratings": {str(k): v for k, v in system.ratings.items()}}, f)
        os.replace(tmp, ELO_CACHE)
    except StopIteration:
        pass
    except Exception as e:
        print(f"[elo] cache write failed: {e}")

    print(f"[elo] computed {len(system.ratings)} players from {used} matches "
          f"(data through {last_date})")
    return system.ratings


# ---------------------------------------------------------------- lookups
def _entry(player_id, ratings):
    if player_id is None or not ratings:
        return None
    e = ratings.get(player_id)
    if e is None:
        try:
            e = ratings.get(int(player_id))
        except (TypeError, ValueError):
            e = None
    return e


def has_rating(player_id, ratings, min_matches=1):
    e = _entry(player_id, ratings)
    return e is not None and e.get("n", 0) >= min_matches


def get_player_elo(player_id, surface, ratings):
    """Blended surface/overall rating. surface=None -> overall only."""
    e = _entry(player_id, ratings)
    if e is None:
        return 1500.0
    surface = normalize_surface(surface)
    if surface not in SURFACES:
        return e["overall"]
    w = PARAMS["surface_weight"]
    return w * e[surface] + (1 - w) * e["overall"]
