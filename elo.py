import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ELO_CACHE = os.path.join(HERE, "elo_ratings.json")

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

BASE_K = 32
SURFACE_K = 24
SURFACE_WEIGHT = 0.75


def _elo_expected(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def _margin_multiplier(games_won, games_lost):
    total = games_won + games_lost
    if total == 0:
        return 1.0
    margin = abs(games_won - games_lost) / total
    return 1.0 + margin


def _games_from_score(score):
    if not isinstance(score, str):
        return (0, 0)
    w_games = l_games = 0
    for set_score in score.split():
        set_score = set_score.split("(")[0]
        if "-" not in set_score:
            continue
        parts = set_score.split("-")
        try:
            a, b = int(parts[0]), int(parts[1])
            w_games += a
            l_games += b
        except ValueError:
            continue
    return (w_games, l_games)


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def compute_ratings(matches, force=False):
    if not force and os.path.exists(ELO_CACHE):
        try:
            with open(ELO_CACHE) as f:
                cached = json.load(f)
            if cached.get("__matches") == len(matches):
                print(f"[elo] loaded {len(cached['ratings'])} players from cache")
                return cached["ratings"]
        except Exception:
            pass

    ratings = {}

    def get_rating(pid, surface):
        if pid not in ratings:
            ratings[pid] = {"overall": 1500.0,
                            "Hard": 1500.0, "Clay": 1500.0,
                            "Grass": 1500.0, "Carpet": 1500.0}
        return ratings[pid].get(surface, ratings[pid]["overall"])

    def set_rating(pid, surface, value):
        if pid not in ratings:
            ratings[pid] = {"overall": 1500.0,
                            "Hard": 1500.0, "Clay": 1500.0,
                            "Grass": 1500.0, "Carpet": 1500.0}
        ratings[pid][surface] = value

    processed = 0
    for row in matches:
        w_id = _to_int(row.get("winner_id"))
        l_id = _to_int(row.get("loser_id"))
        if not w_id or not l_id:
            continue

        surface = str(row.get("surface", "Hard")).strip()
        if surface not in SURFACES:
            surface = "Hard"

        w_games, l_games = _games_from_score(str(row.get("score", "")))
        if w_games + l_games == 0:
            w_games, l_games = 12, 6

        r_w = get_rating(w_id, "overall")
        r_l = get_rating(l_id, "overall")
        exp_w = _elo_expected(r_w, r_l)
        mult = _margin_multiplier(w_games, l_games)
        delta = BASE_K * mult * (1 - exp_w)
        set_rating(w_id, "overall", r_w + delta)
        set_rating(l_id, "overall", r_l - delta)

        r_ws = get_rating(w_id, surface)
        r_ls = get_rating(l_id, surface)
        exp_ws = _elo_expected(r_ws, r_ls)
        delta_s = SURFACE_K * mult * (1 - exp_ws)
        set_rating(w_id, surface, r_ws + delta_s)
        set_rating(l_id, surface, r_ls - delta_s)

        processed += 1

    with open(ELO_CACHE, "w") as f:
        json.dump({"__matches": len(matches), "ratings": ratings}, f)

    print(f"[elo] computed ratings for {len(ratings)} players "
          f"from {processed} matches")
    return ratings


def get_player_elo(player_id, surface, ratings):
    if player_id is None or player_id not in ratings:
        return 1500.0
    entry = ratings[player_id]
    overall = entry.get("overall", 1500.0)
    surface_r = entry.get(surface, overall)
    return SURFACE_WEIGHT * surface_r + (1 - SURFACE_WEIGHT) * overall
