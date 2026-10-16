"""
Backtest the "best single pick" strategy with a CONFIDENCE BAND filter.

The previous run showed that "highest confidence" always picks the shortest
odds (1.05-1.10), where bookmaker margin exceeds the edge. This version
restricts candidates to a band where odds are meaningful (1.30-1.70).

Tests three bands per market:
  - Narrow:  60-70% confidence  (odds ~1.43-1.67)
  - Wide:    55-75% confidence  (odds ~1.33-1.82)
  - All:     ≥55% confidence    (sanity baseline)

For each band, picks the highest-confidence candidate each day and
compares to a random pick from the same band.

Markets: Moneyline (MW), Sets O/U 2.5
"""
import math
import random
from collections import defaultdict
from datetime import timedelta
from itertools import groupby

import dataset

SURFACES = ("Hard", "Clay", "Grass")
BASE_K = 32
SURFACE_K = 24
SURFACE_WEIGHT = 0.75
DAYS_BACK = 400

REFERENCE_MARGIN = 0.04
MARGIN_LEVELS = [
    ("Sharp (2%)", 0.02),
    ("Mid (4%)", 0.04),
    ("Soft (6%)", 0.06),
]

BANDS = [
    ("Narrow 60-70%", 0.60, 0.70),
    ("Wide 55-75%", 0.55, 0.75),
    ("All >=55%", 0.55, 1.01),
]


# ============= Math =============
def elo_prob(e1, e2):
    return 1 / (1 + 10 ** ((e2 - e1) / 400))


def p_3sets_model(elo_diff):
    return 0.35 * math.exp(-abs(elo_diff) / 250)


def margin_mult(g_won, g_lost):
    total = g_won + g_lost
    if total == 0:
        return 1.0
    return 1.0 + abs(g_won - g_lost) / total


def games_from_score(s):
    if not isinstance(s, str):
        return (0, 0)
    w = l = 0
    for set_s in s.split():
        set_s = set_s.split("(")[0]
        if "-" not in set_s:
            continue
        try:
            a, b = set_s.split("-")[:2]
            w += int(a)
            l += int(b)
        except (ValueError, IndexError):
            continue
    return (w, l)


def sets_in_score(s):
    if not isinstance(s, str):
        return 0
    n = 0
    for part in s.split():
        part = part.split("(")[0]
        if "-" in part:
            n += 1
    return n


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def sim_odds(p, margin):
    if p is None or p < 0.01 or p > 0.99:
        return None
    return (1.0 / p) * (1.0 - margin)


def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    rate = wins / n
    denom = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denom
    margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ============= Elo =============
class Elo:
    def __init__(self):
        self.ratings = {}

    def _ensure(self, pid):
        if pid not in self.ratings:
            self.ratings[pid] = {
                "overall": 1500.0, "Hard": 1500.0,
                "Clay": 1500.0, "Grass": 1500.0,
            }

    def get(self, pid, surface):
        self._ensure(pid)
        return self.ratings[pid].get(surface,
                                     self.ratings[pid]["overall"])

    def set(self, pid, surface, value):
        self._ensure(pid)
        self.ratings[pid][surface] = value

    def blended(self, pid, surface):
        if pid not in self.ratings:
            return 1500.0
        entry = self.ratings[pid]
        overall = entry["overall"]
        surface_r = entry.get(surface, overall)
        return SURFACE_WEIGHT * surface_r + (1 - SURFACE_WEIGHT) * overall

    def update(self, w_id, l_id, surface, score):
        w_g, l_g = games_from_score(score)
        if w_g + l_g == 0:
            w_g, l_g = 12, 6
        mult = margin_mult(w_g, l_g)

        r_w = self.get(w_id, "overall")
        r_l = self.get(l_id, "overall")
        delta = BASE_K * mult * (1 - elo_prob(r_w, r_l))
        self.set(w_id, "overall", r_w + delta)
        self.set(l_id, "overall", r_l - delta)

        r_ws = self.get(w_id, surface)
        r_ls = self.get(l_id, surface)
        delta_s = SURFACE_K * mult * (1 - elo_prob(r_ws, r_ls))
        self.set(w_id, surface, r_ws + delta_s)
        self.set(l_id, surface, r_ls - delta_s)


def normalize_surface(s):
    s = str(s or "Hard").strip().title()
    return s if s in SURFACES else "Hard"


# ============= Candidate builders =============
def build_mw_candidate(elo, m):
    w_id = _to_int(m.get("winner_id"))
    l_id = _to_int(m.get("loser_id"))
    if not w_id or not l_id:
        return None

    surface = normalize_surface(m.get("surface"))
    e_w = elo.blended(w_id, surface)
    e_l = elo.blended(l_id, surface)
    p_w = elo_prob(e_w, e_l)

    conf = max(p_w, 1 - p_w)
    picked_winner = p_w >= 0.5
    o = sim_odds(conf, REFERENCE_MARGIN)
    if o is None:
        return None

    return {
        "confidence": conf,
        "correct": picked_winner,
        "ref_odds": o,
    }


def build_sets_candidate(elo, m):
    w_id = _to_int(m.get("winner_id"))
    l_id = _to_int(m.get("loser_id"))
    if not w_id or not l_id:
        return None

    surface = normalize_surface(m.get("surface"))
    e_w = elo.blended(w_id, surface)
    e_l = elo.blended(l_id, surface)

    p_3 = p_3sets_model(e_w - e_l)
    p_under = 1 - p_3
    p_over = p_3

    n_sets = sets_in_score(str(m.get("score", "")))

    if p_under >= p_over:
        conf = p_under
        correct = (n_sets <= 2)
    else:
        conf = p_over
        correct = (n_sets >= 3)

    o = sim_odds(conf, REFERENCE_MARGIN)
    if o is None:
        return None

    return {
        "confidence": conf,
        "correct": correct,
        "ref_odds": o,
    }


# ============= Main =============
def main():
    random.seed(42)

    print("Loading dataset...")
    matches = dataset.load_all_matches()
    if not matches:
        print("No matches loaded.")
        return

    matches.sort(key=lambda m: m["_date"])
    end_date = matches[-1]["_date"]
    start_date = end_date - timedelta(days=DAYS_BACK)

    print(f"Backtest window: {start_date.date()} → {end_date.date()}")
    print(f"Total matches in dataset: {len(matches)}")
    print()

    elo = Elo()

    print("Seeding ratings from history...")
    seeded = 0
    for m in matches:
        if m["_date"] >= start_date:
            break
        w_id = _to_int(m.get("winner_id"))
        l_id = _to_int(m.get("loser_id"))
        if not w_id or not l_id:
            continue
        elo.update(w_id, l_id, normalize_surface(m.get("surface")),
                   str(m.get("score", "")))
        seeded += 1
        if seeded % 25000 == 0:
            print(f"  ...{seeded} matches processed")

    print(f"Seeded {seeded} prior matches, {len(elo.ratings)} players rated")
    print()

    window = [m for m in matches if m["_date"] >= start_date]
    window.sort(key=lambda m: m["_date"])
    print(f"Scoring {len(window)} matches...")

    # For each market × band × strategy: list of picks
    results = {}
    for market in ("mw", "sets"):
        for band_label, _, _ in BANDS:
            results[(market, band_label, "best")] = []
            results[(market, band_label, "random")] = []

    days_processed = 0
    for day, group in groupby(window, key=lambda m: m["_date"].date()):
        day_list = list(group)

        # Build all candidates for today
        mw_candidates = []
        sets_candidates = []
        for m in day_list:
            mw = build_mw_candidate(elo, m)
            if mw:
                mw_candidates.append(mw)
            sets = build_sets_candidate(elo, m)
            if sets:
                sets_candidates.append(sets)

        # Per market, per band: apply band filter, pick best, pick random
        for market, candidates in (("mw", mw_candidates),
                                    ("sets", sets_candidates)):
            for band_label, lo, hi in BANDS:
                band = [c for c in candidates
                        if lo <= c["confidence"] < hi]
                if not band:
                    continue
                best = max(band, key=lambda c: c["confidence"])
                results[(market, band_label, "best")].append(best)
                results[(market, band_label, "random")].append(
                    random.choice(band))

        # Update Elo
        for m in day_list:
            w_id = _to_int(m.get("winner_id"))
            l_id = _to_int(m.get("loser_id"))
            if not w_id or not l_id:
                continue
            elo.update(w_id, l_id, normalize_surface(m.get("surface")),
                       str(m.get("score", "")))

        days_processed += 1
        if days_processed % 60 == 0:
            print(f"  ...{days_processed} days processed")

    print(f"Done — {days_processed} days scored")
    print()

    # ============ Report ============
    for market, market_label in (("mw", "MONEYLINE"),
                                   ("sets", "SETS O/U 2.5")):
        print(f"{'='*74}")
        print(f"{market_label}")
        print(f"{'='*74}")

        for band_label, _, _ in BANDS:
            best_picks = results[(market, band_label, "best")]
            rand_picks = results[(market, band_label, "random")]

            print()
            print(f"  BAND: {band_label}")
            print(f"  {'-'*70}")

            for strategy, picks in (("BEST", best_picks),
                                     ("RANDOM", rand_picks)):
                n = len(picks)
                if n == 0:
                    print(f"    {strategy}: no picks")
                    continue

                wins = sum(1 for p in picks if p["correct"])
                strike = wins / n
                avg_conf = sum(p["confidence"] for p in picks) / n
                ci_lo, ci_hi = wilson_ci(wins, n)

                print(f"    {strategy:<7} n={n:<4} "
                      f"strike={strike*100:>5.1f}% "
                      f"[{ci_lo*100:.0f}-{ci_hi*100:.0f}%] "
                      f"model={avg_conf*100:>5.1f}% "
                      f"gap={(strike-avg_conf)*100:>+5.1f}pp")

                # Reprice at each margin
                for mlabel, margin in MARGIN_LEVELS:
                    total_pl = 0.0
                    total_o = 0.0
                    for p in picks:
                        o = sim_odds(p["confidence"], margin)
                        if o is None:
                            continue
                        total_o += o
                        if p["correct"]:
                            total_pl += (o - 1.0)
                        else:
                            total_pl -= 1.0
                    avg_o = total_o / n
                    be = 1.0 / avg_o
                    edge = strike - be
                    roi = total_pl / n
                    print(f"      {mlabel:<12} "
                          f"odds={avg_o:>5.2f}  "
                          f"BE={be*100:>5.1f}%  "
                          f"edge={edge*100:>+6.2f}pp  "
                          f"ROI={roi*100:>+6.2f}%")
                print()


if __name__ == "__main__":
    main()
