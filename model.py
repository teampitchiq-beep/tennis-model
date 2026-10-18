"""
Match win-probability model.

A logistic regression on (Elo difference, log rank ratio), fitted by
evaluate.py on walk-forward data and stored in model_params.json. It replaces
the hand-weighted 60% Elo / 20% form / 20% rank blend plus the fixed 0.7
shrink. Form is not an input: its fitted weight was ~0 once Elo is used
(Elo already contains recent results), so it only added noise.

Regimes (all symmetric in p1/p2, no intercept):
  both players rated (>= min_matches) and both ranked -> Elo + rank
  both rated, ranks missing                            -> Elo only
  otherwise, both ranked                               -> rank only
  nothing usable                                       -> 0.5
"""
import json
import math
import os

import elo

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "model_params.json")

# Fallback if model_params.json is missing (fitted on 2023-01..2026-06).
DEFAULTS = {
    "w_elo": 1.9655, "w_rank": 0.1678,
    "w_elo_only": 2.2906, "w_rank_only": 0.6808,
    "min_matches": 3,
}

_PARAMS = None
_MTIME = None


def get_params():
    global _PARAMS, _MTIME
    try:
        mt = os.path.getmtime(PARAMS_PATH)
    except OSError:
        mt = None
    if _PARAMS is None or mt != _MTIME:
        p = dict(DEFAULTS)
        if mt is not None:
            try:
                with open(PARAMS_PATH) as f:
                    p.update(json.load(f))
            except Exception as e:
                print(f"[model] could not read model_params.json: {e}")
        _PARAMS, _MTIME = p, mt
    return _PARAMS


def _sig(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _log_rank_ratio(r1, r2):
    try:
        r1, r2 = float(r1), float(r2)
    except (TypeError, ValueError):
        return None
    if r1 <= 0 or r2 <= 0:
        return None
    return math.log(r2 / r1)       # >0 means player 1 is ranked better


def win_prob(elo1, elo2, rank1, rank2, n1=0, n2=0):
    """Probability that player 1 beats player 2.

    elo1/elo2: blended rating, or None if the player is unrated.
    n1/n2:     matches behind each rating (low counts are unreliable).
    Returns (prob, regime).
    """
    p = get_params()
    lr = _log_rank_ratio(rank1, rank2)
    rated = (elo1 is not None and elo2 is not None
             and n1 >= p["min_matches"] and n2 >= p["min_matches"])
    if rated and lr is not None:
        z = p["w_elo"] * (elo1 - elo2) / 400.0 + p["w_rank"] * lr
        return _sig(z), "elo+rank"
    if rated:
        return _sig(p["w_elo_only"] * (elo1 - elo2) / 400.0), "elo"
    if lr is not None:
        return _sig(p["w_rank_only"] * lr), "rank"
    return 0.5, "none"


def rank_prob(rank1, rank2):
    """Rank-only probability (for display)."""
    lr = _log_rank_ratio(rank1, rank2)
    if lr is None:
        return 0.5
    return _sig(get_params()["w_rank_only"] * lr)


def elo_prob(elo1, elo2):
    """Elo-only probability (for display)."""
    return _sig(get_params()["w_elo_only"] * (elo1 - elo2) / 400.0)
