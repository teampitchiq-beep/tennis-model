"""
Walk-forward evaluation, probability-model fitting and (optional) Elo tuning.
Replaces backtest.py.

Why backtest.py was replaced: it priced "bookmaker odds" from the model's OWN
probability (odds = 1/p * (1 - margin)), so ROI was just the margin by
construction and could never reveal an edge or a lack of one. This script
instead measures what can honestly be measured from match results alone:

  * log loss / Brier / accuracy / calibration, predicting every match using
    only information available BEFORE it was played (walk-forward);
  * a held-out final year the model was never fitted on.

Usage:
    python evaluate.py            # report + refit blend -> model_params.json
    python evaluate.py --tune     # coordinate-descent tune Elo parameters
                                  # (slow; prints a PARAMS block for elo.py)

Edge vs. real bookmaker prices needs real historical odds; log your own
(predictions.db already stores book odds) and judge with closing-line value.
"""
import json
import math
import os
import sys
from datetime import timedelta

import numpy as np
from sklearn.linear_model import LogisticRegression

import dataset
import elo
import scoreline

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PARAMS = os.path.join(HERE, "model_params.json")
FIT_START_YEARS_BACK = 3.4      # first ~1.6y of data is Elo burn-in
HOLDOUT_DAYS = 365
EVAL_LEVELS = ("tour", "chall")


def load_rows():
    rows = [r for r in dataset.load_all_matches()
            if r.get("winner_id") and r.get("loser_id")]
    rows.sort(key=elo.sort_key)
    return rows


def _rank_feature(row):
    try:
        rw, rl = float(row.get("winner_rank")), float(row.get("loser_rank"))
        if rw > 0 and rl > 0:
            return math.log(rl / rw)          # >0: winner better ranked
    except (TypeError, ValueError):
        pass
    return None


def walk_forward(rows, params=None, record_from=None):
    """Predict each match from pre-match state, then update. Winner-oriented."""
    system = elo.EloSystem(params)
    out = []
    for r in rows:
        w, l = int(r["winner_id"]), int(r["loser_id"])
        surf = elo.normalize_surface(r.get("surface"))
        lvl = elo.level_class(r)
        if record_from is not None and r["_date"] >= record_from and lvl in EVAL_LEVELS:
            ew, el = system.ratings.get(w), system.ratings.get(l)
            out.append({
                "d": r["_date"], "lvl": lvl,
                "diff": (system.blended(w, surf) - system.blended(l, surf)) / 400.0,
                "lrank": _rank_feature(r),
                "n": min(ew["n"] if ew else 0, el["n"] if el else 0),
                "score": r.get("score"),
                "bo": 5 if str(r.get("best_of")) == "5" else 3,
            })
        system.update(w, l, surf, r.get("score"), r.get("best_of"), lvl, r["_date"])
    return out


# ------------------------------------------------------------- metrics
def _ll(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(np.log(p)))


def _brier(p):
    return float(np.mean((1 - np.asarray(p, float)) ** 2))


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def _fit(X, w=None):
    X = np.asarray(X, float)
    Xs = np.vstack([X, -X])                     # mirrored: symmetric, no intercept
    y = np.r_[np.ones(len(X)), np.zeros(len(X))]
    return LogisticRegression(fit_intercept=False, C=100.0).fit(Xs, y).coef_[0]


def fit_blend(feats, min_matches=3):
    full = [f for f in feats if f["lrank"] is not None and f["n"] >= min_matches]
    elo_only = [f for f in feats if f["n"] >= min_matches]
    rank_only = [f for f in feats if f["lrank"] is not None]
    w_full = _fit([[f["diff"], f["lrank"]] for f in full])
    w_elo = _fit([[f["diff"]] for f in elo_only])
    w_rank = _fit([[f["lrank"]] for f in rank_only])
    return {
        "w_elo": float(w_full[0]), "w_rank": float(w_full[1]),
        "w_elo_only": float(w_elo[0]), "w_rank_only": float(w_rank[0]),
        "min_matches": min_matches,
    }


def apply_blend(f, m):
    """Winner-oriented probability the winner is picked (matches model.py)."""
    if f["n"] >= m["min_matches"] and f["lrank"] is not None:
        return float(_sig(m["w_elo"] * f["diff"] + m["w_rank"] * f["lrank"]))
    if f["n"] >= m["min_matches"]:
        return float(_sig(m["w_elo_only"] * f["diff"]))
    if f["lrank"] is not None:
        return float(_sig(m["w_rank_only"] * f["lrank"]))
    return 0.5


def old_blend(f_old):
    """The original hand-weighted blend (60% Elo, 20% form~0.5, 20% rank, x0.7)."""
    p_elo = 1 / (1 + 10 ** (-f_old["diff"] * 400 / 400))
    p_rank = 0.5
    if f_old["lrank"] is not None:
        # rank_to_rating: 2100 - 200*log10(rank); diff = 200*log10(rl/rw)
        p_rank = 1 / (1 + 10 ** (-(200 * f_old["lrank"] / math.log(10)) / 400))
    raw = 0.6 * p_elo + 0.2 * 0.5 + 0.2 * p_rank
    return 0.5 + (raw - 0.5) * 0.7


def report(feats_test, blend, old_feats):
    print("\n=== HELD-OUT RESULTS (never used for fitting) ===")
    print(f"{'model':<26}{'level':<7}{'n':>7}{'logloss':>9}{'brier':>8}{'acc':>7}")
    for lvl in EVAL_LEVELS:
        sub = [f for f in feats_test if f["lvl"] == lvl]
        osub = [f for f in old_feats if f["lvl"] == lvl]
        if not sub:
            continue
        rows = {
            "NEW fitted blend": [apply_blend(f, blend) for f in sub],
            "Elo only (tuned)": [float(_sig(blend["w_elo_only"] * f["diff"])) for f in sub],
            "OLD hand blend": [old_blend(f) for f in osub],
        }
        for name, ps in rows.items():
            print(f"{name:<26}{lvl:<7}{len(ps):>7}{_ll(ps):>9.4f}{_brier(ps):>8.4f}"
                  f"{np.mean(np.array(ps) > 0.5):>7.3f}")
    print("\nCalibration (NEW blend, tour+challenger): picked-side confidence vs. hit rate")
    ps = np.array([apply_blend(f, blend) for f in feats_test])
    conf = np.maximum(ps, 1 - ps)
    hit = (ps > 0.5).astype(float)
    for lo, hi in [(0.5, .55), (.55, .6), (.6, .65), (.65, .7), (.7, .8), (.8, 1.01)]:
        mk = (conf >= lo) & (conf < hi)
        if mk.sum():
            print(f"  {lo:.2f}-{min(hi, 1):.2f}  n={int(mk.sum()):<6} "
                  f"predicted={conf[mk].mean():.3f}  actual={hit[mk].mean():.3f}")


def score_report(train, test, blend):
    """Fit the set-score model on the training window, check it on hold-out."""
    sc_train = [(apply_blend(f, blend), f["bo"], f["score"]) for f in train]
    rep = scoreline.fit(sc_train)
    print("\n=== SCORE-LINE MODEL ===")
    print({k: v for k, v in rep.items() if k != "game_tables"})
    print("typical games when favourite wins a set:", rep.get("game_tables", {}).get("fav_wins_set"))
    print("typical games when underdog wins a set: ", rep.get("game_tables", {}).get("dog_wins_set"))
    print("Hold-out check (best-of-3): straight-set rate, predicted vs actual")
    rows = []
    for f in test:
        sets = scoreline._parse_sets(f["score"])
        if not sets or f["bo"] != 3:
            continue
        w = sum(1 for a, b in sets if a > b)
        l = sum(1 for a, b in sets if b > a)
        if w == 2 and l < 2:
            p = apply_blend(f, blend)
            rows.append((max(p, 1 - p), scoreline.predict(p, 3)["p_straight"], l == 0))
    for lo, hi in [(0.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]:
        sub = [r for r in rows if lo <= r[0] < hi]
        if sub:
            print(f"  conf {lo:.1f}-{min(hi, 1):.1f} n={len(sub):<6} "
                  f"predicted={np.mean([r[1] for r in sub]):.3f} "
                  f"actual={np.mean([r[2] for r in sub]):.3f}")


# ------------------------------------------------------------- tuning
def tune(rows, record_from, split):
    base = dict(elo.PARAMS)
    grid = {
        "k": [200, 250, 350, 500], "koff": [3, 5, 10], "kshape": [0.3, 0.4, 0.5, 0.6],
        "kmin": [8, 12, 16, 20], "surface_k_ratio": [0.3, 0.4, 0.5, 0.7],
        "surface_weight": [0.2, 0.3, 0.4, 0.5, 0.65], "margin": [0.0, 0.5, 1.0, 2.0],
        "bo5": [1.0, 1.4, 1.8], "k_tour": [0.6, 0.8, 1.0], "k_chall": [0.6, 0.8, 1.0],
        "k_itf": [0.4, 0.55, 0.7, 0.85],
    }

    def score(params):
        feats = walk_forward(rows, params, record_from)
        train = [f for f in feats if f["d"] < split]
        return _ll([_sig(f["diff"] * 2.3) for f in train])   # fixed 2.3 slope

    best_s = score(base)
    print(f"start train log loss {best_s:.5f}")
    for sweep in range(2):
        for key, vals in grid.items():
            for v in vals:
                if v == base[key]:
                    continue
                trial = dict(base)
                trial[key] = v
                s = score(trial)
                if s < best_s - 1e-5:
                    best_s, base = s, trial
                    print(f"  sweep {sweep}: {key}={v} -> {s:.5f}")
    print("\nPARAMS = " + json.dumps(base, indent=4))
    print("Paste into elo.py (and bump MODEL_VERSION), then re-run evaluate.py.")


# ------------------------------------------------------------- main
def main():
    rows = load_rows()
    if not rows:
        print("No data. Run dataset.load_all_matches() first.")
        return
    last = rows[-1]["_date"]
    print(f"Data: {len(rows)} matches, {rows[0]['_date'].date()} -> {last.date()}")
    record_from = last - timedelta(days=int(365 * FIT_START_YEARS_BACK))
    split = last - timedelta(days=HOLDOUT_DAYS)

    if "--tune" in sys.argv:
        tune(rows, record_from, split)
        return

    feats = walk_forward(rows, None, record_from)
    train = [f for f in feats if f["d"] < split]
    test = [f for f in feats if f["d"] >= split]
    print(f"Fit window: {train[0]['d'].date()} -> {split.date()} ({len(train)} matches); "
          f"hold-out: {len(test)} matches")

    blend_holdout = fit_blend(train)
    old_feats = walk_forward(rows, {                       # original Elo params
        "k": 32.0, "kshape": 0.0, "kmin": 0.0, "surface_k_ratio": 0.75,
        "surface_weight": 0.75, "margin": 1.0, "bo5": 1.0,
        "k_tour": 1.0, "k_chall": 1.0, "k_itf": 1.0}, split)
    old_test = [f for f in old_feats if f["d"] >= split]   # same matches, same order
    report(test, blend_holdout, old_test)

    score_report(train, test, blend_holdout)

    final = fit_blend(feats)                                # refit on everything
    final["trained_through"] = last.strftime("%Y-%m-%d")
    final["fit_rows"] = len(feats)
    final["holdout_logloss"] = round(_ll([apply_blend(f, blend_holdout) for f in test]), 5)
    final["elo_params_version"] = elo.MODEL_VERSION
    sc_all = [(apply_blend(f, final), f["bo"], f["score"]) for f in feats]
    scoreline.fit(sc_all)                                   # final score-line fit
    with open(MODEL_PARAMS, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nWrote {MODEL_PARAMS}:\n{json.dumps(final, indent=2)}")


if __name__ == "__main__":
    main()
