"""
Most likely score line for a match.

Set-score model
---------------
Treating sets as independent coin flips is wrong: backtests show even matches
end in straight sets ~63% of the time, whereas independent sets give 50%.
Players have good and bad days that carry across all sets of a match. So each
match gets one shared "form shift" d ~ N(0, sigma^2) applied to the log-odds
of every set. The base set log-odds are solved so the match win probability
equals the model's probability; the shift then creates the right amount of
set-to-set correlation. sigma is fitted by evaluate.py (best-of-3 and best-of-5
separately) and was checked against actual straight-set rates by confidence
band, tour and surface.

Game scores
-----------
Typical game score of a set, from historical sets: for each bucket of set win
probability, the most common result when the favourite wins a set and when the
underdog wins one. This is a "typical" scoreline, not an exact-score forecast.

Pure Python (no numpy) so it runs on the phone.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "scoreline_params.json")

# 9-point Gauss-Hermite rule for a standard normal.
_X = [-4.512746, -3.205429, -2.076848, -1.023256, 0.0,
      1.023256, 2.076848, 3.205429, 4.512746]
_W = [2.2e-05, 0.002789, 0.049916, 0.244098, 0.406349,
      0.244098, 0.049916, 0.002789, 2.2e-05]

DEFAULTS = {
    "sigma": {"3": 1.25, "5": 0.93},
    # upper edge of favourite's set-win-probability bucket -> typical scores
    "game_buckets": [0.55, 0.60, 0.65, 0.70, 0.80, 1.01],
    "fav_wins_set": ["6-4", "6-4", "6-3", "6-3", "6-2", "6-1"],
    "dog_wins_set": ["6-4", "6-4", "6-4", "6-4", "6-4", "6-3"],
}

_SLAMS = ("australian open", "roland garros", "french open", "wimbledon",
          "us open")

_PARAMS = None


def get_params():
    global _PARAMS
    if _PARAMS is None:
        p = json.loads(json.dumps(DEFAULTS))
        try:
            with open(PARAMS_PATH) as f:
                p.update(json.load(f))
        except Exception:
            pass
        _PARAMS = p
    return _PARAMS


def reload_params():
    global _PARAMS
    _PARAMS = None


def best_of_for(tournament, tour):
    """Men's Grand Slam main draws are best-of-5; everything else best-of-3."""
    t = (tournament or "").lower()
    if tour == "atp" and any(s in t for s in _SLAMS) and "challenger" not in t \
            and "itf" not in t and "qualif" not in t:
        return 5
    return 3


def _sig(z):
    z = max(-40.0, min(40.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _match_win(q, bo):
    if bo == 3:
        return q * q * (3 - 2 * q)
    return q ** 3 * (10 - 15 * q + 6 * q * q)


def _marginal_match_win(z, sigma, bo):
    return sum(w * _match_win(_sig(z + sigma * x), bo) for x, w in zip(_X, _W))


def _base_logit(p, sigma, bo):
    """Solve for the set log-odds whose match win probability equals p."""
    lo, hi = -14.0, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _marginal_match_win(mid, sigma, bo) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def set_score_probs(p1, best_of=3, sigma=None):
    """P(player 1 wins the match), best_of -> {'p1': {'2-0':..}, 'p2': {...}}."""
    p1 = max(0.001, min(0.999, p1))
    if sigma is None:
        sigma = get_params()["sigma"][str(best_of)]
    z = _base_logit(p1, sigma, best_of)
    qs = [(_sig(z + sigma * x), w) for x, w in zip(_X, _W)]

    def E(fn):
        return sum(w * fn(q) for q, w in qs)

    if best_of == 3:
        d = {
            "p1": {"2-0": E(lambda q: q * q),
                   "2-1": E(lambda q: 2 * q * q * (1 - q))},
            "p2": {"2-0": E(lambda q: (1 - q) ** 2),
                   "2-1": E(lambda q: 2 * q * (1 - q) ** 2)},
        }
    else:
        d = {
            "p1": {"3-0": E(lambda q: q ** 3),
                   "3-1": E(lambda q: 3 * q ** 3 * (1 - q)),
                   "3-2": E(lambda q: 6 * q ** 3 * (1 - q) ** 2)},
            "p2": {"3-0": E(lambda q: (1 - q) ** 3),
                   "3-1": E(lambda q: 3 * (1 - q) ** 3 * q),
                   "3-2": E(lambda q: 6 * (1 - q) ** 3 * q ** 2)},
        }
    mean_q = E(lambda q: q)
    return d, mean_q


def _typical_games(mean_q, fav_is_p1):
    """(games when p1 wins a set, games when p2 wins a set)."""
    prm = get_params()
    c = max(mean_q, 1 - mean_q)
    idx = len(prm["game_buckets"]) - 1
    for i, edge in enumerate(prm["game_buckets"]):
        if c < edge:
            idx = i
            break
    fav, dog = prm["fav_wins_set"][idx], prm["dog_wins_set"][idx]
    return (fav, dog) if fav_is_p1 else (dog, fav)


def _flip(score):
    a, b = score.split("-")
    return f"{b}-{a}"


def _sequence(w_sets, l_sets):
    """Typical order of sets from the winner's view: W L W (L W ...)."""
    seq, wins_left, losses_left = ["W"], w_sets - 1, l_sets
    while losses_left:
        seq.append("L")
        losses_left -= 1
        if wins_left:
            seq.append("W")
            wins_left -= 1
    seq.extend("W" * wins_left)
    return seq


def predict(p1_prob, best_of=3):
    """Score-line forecast for the match.

    Returns a dict with
      outcomes   [{side, score, prob, games}] sorted by probability, where
                 score is sets (winner first) and games are the typical game
                 scores of each set from the match winner's point of view
      top        the most likely outcome
      p_straight probability the match ends without the loser taking a set
      p_decider  probability it goes the full distance
    """
    d, mean_q = set_score_probs(p1_prob, best_of)
    p1_wins_set, p2_wins_set = _typical_games(mean_q, mean_q >= 0.5)
    outcomes = []
    for side in ("p1", "p2"):
        mine = p1_wins_set if side == "p1" else p2_wins_set      # I win a set
        theirs = p2_wins_set if side == "p1" else p1_wins_set    # they win a set
        for sc, pr in d[side].items():
            w_sets, l_sets = (int(x) for x in sc.split("-"))
            games = [mine if r == "W" else _flip(theirs)
                     for r in _sequence(w_sets, l_sets)]
            outcomes.append({"side": side, "score": sc, "prob": pr,
                             "games": " ".join(games)})
    outcomes.sort(key=lambda o: -o["prob"])
    need = 2 if best_of == 3 else 3
    return {
        "outcomes": outcomes, "top": outcomes[0], "best_of": best_of,
        "p_straight": sum(o["prob"] for o in outcomes if o["score"].endswith("-0")),
        "p_decider": sum(o["prob"] for o in outcomes
                         if int(o["score"][2]) == need - 1),
    }


# --------------------------------------------------------------- fitting
def _parse_sets(score):
    """[(a, b), ...] winner-first game pairs, or None if not a clean finish."""
    if not isinstance(score, str):
        return None
    if any(x in score.upper() for x in ("RET", "W/O", "DEF", "UNP", "ABD")):
        return None
    out = []
    for part in score.split():
        part = part.split("(")[0]
        if "-" not in part:
            return None
        try:
            a, b = (int(x) for x in part.split("-")[:2])
        except ValueError:
            return None
        out.append((a, b))
    return out or None


def fit(samples):
    """samples: [(p_winner, best_of, score_str)] from walk-forward predictions.

    Fits sigma per best-of and the typical game-score tables, writes
    scoreline_params.json, and returns a small report dict.
    """
    from scipy.optimize import minimize_scalar
    import random
    from collections import Counter, defaultdict

    clean = []
    for p, bo, sc in samples:
        sets = _parse_sets(sc)
        if not sets:
            continue
        w = sum(1 for a, b in sets if a > b)
        l = sum(1 for a, b in sets if b > a)
        need = 2 if bo == 3 else 3
        if w == need and l < need:
            clean.append((min(max(p, 1e-3), 1 - 1e-3), bo, f"{w}-{l}", sets))

    prm = json.loads(json.dumps(DEFAULTS))
    rng = random.Random(0)
    report = {}
    for bo in (3, 5):
        rows = [r for r in clean if r[1] == bo]
        if len(rows) < 300:
            continue
        sub = rng.sample(rows, min(5000, len(rows)))

        def nll(s):
            t = 0.0
            for p, _, key, _ in sub:
                d, _ = set_score_probs(p, bo, sigma=s)
                t -= math.log(max(d["p1"][key], 1e-9))   # p is the winner's prob
            return t / len(sub)

        res = minimize_scalar(nll, bounds=(0.05, 2.5), method="bounded",
                              options={"xatol": 0.01})
        prm["sigma"][str(bo)] = round(float(res.x), 3)
        report[f"sigma_bo{bo}"] = prm["sigma"][str(bo)]
        report[f"n_bo{bo}"] = len(rows)

    # typical game scores per favourite-set-probability bucket
    edges = prm["game_buckets"]
    cnt = {k: defaultdict(Counter) for k in ("fav", "dog")}
    valid = {(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (7, 5), (7, 6)}
    for p, bo, key, sets in clean:
        _, mean_q = set_score_probs(p, bo, sigma=prm["sigma"][str(bo)])
        winner_is_fav = p >= 0.5
        c = max(mean_q, 1 - mean_q)
        idx = next((i for i, e in enumerate(edges) if c < e), len(edges) - 1)
        for a, b in sets:
            if (max(a, b), min(a, b)) not in valid:
                continue
            set_winner_is_match_winner = a > b
            fav_won_set = (set_winner_is_match_winner == winner_is_fav)
            cnt["fav" if fav_won_set else "dog"][idx][f"{max(a, b)}-{min(a, b)}"] += 1
    for kind, field in (("fav", "fav_wins_set"), ("dog", "dog_wins_set")):
        for i in range(len(edges)):
            if cnt[kind][i]:
                prm[field][i] = cnt[kind][i].most_common(1)[0][0]
    with open(PARAMS_PATH, "w") as f:
        json.dump(prm, f, indent=2)
    reload_params()
    report["game_tables"] = {k: prm[k] for k in ("fav_wins_set", "dog_wins_set")}
    return report
