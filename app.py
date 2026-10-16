import math
import os
import random
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from flask import (Flask, redirect, render_template_string, request, url_for)

import scrapers
import dataset
import elo
import players
import odds_scraper
import tracker
import settler

app = Flask(__name__)

SURFACES = {"hard", "clay", "grass"}
fixtures_cache = {}
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "odds.db")

ELO_RATINGS = {}

# ===== Odds fetch tuning =====
ODDS_WORKERS = 10
ODDS_FETCH_MIN_PROB = 0.55
ODDS_FETCH_DEADLINE = 45

_ODDS_FETCH_LOCK = threading.Lock()

MODEL_SUPPORT_MIN_PROB = 0.60
AUTO_TRACK_MARKETS = ("mw",)
AUTO_TRACK_MIN_EDGE_MW = 0.03

AUTO_SETTLE_INTERVAL = 900
AUTO_SETTLE_START_DELAY = 180

NAIROBI_TZ = ZoneInfo("Africa/Nairobi")
BERLIN_TZ = ZoneInfo("Europe/Berlin")


# =================== SQLite (odds) ===================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA user_version = 6")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds (
            match_id TEXT PRIMARY KEY,
            mw1 REAL, mw2 REAL,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def ensure_schema():
    if not os.path.exists(DB_PATH):
        init_db()
        return
    conn = sqlite3.connect(DB_PATH)
    if conn.execute("PRAGMA user_version").fetchone()[0] < 6:
        conn.execute("DROP TABLE IF EXISTS odds")
        conn.commit()
    conn.close()
    init_db()


def load_all_odds():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM odds").fetchall()
    conn.close()
    return {r["match_id"]: dict(r) for r in rows}


def convert_te_time_to_nairobi(time_str, match_date=None):
    if not time_str:
        return ""
    m = re.match(r"^(\d{1,2}):(\d{2})", time_str.strip())
    if not m:
        return time_str

    h, mn = int(m.group(1)), int(m.group(2))

    if match_date is None:
        match_date = datetime.now(NAIROBI_TZ).date()

    naive = datetime(match_date.year, match_date.month, match_date.day,
                     h, mn, tzinfo=BERLIN_TZ)
    nairobi_dt = naive.astimezone(NAIROBI_TZ)
    return nairobi_dt.strftime("%H:%M")


# =================== Model ===================
def elo_prob(e1, e2):
    return 1 / (1 + 10 ** ((e2 - e1) / 400))


def rank_to_rating(rank):
    if rank is None:
        return 1500
    r = max(1, min(int(rank), 2000))
    return 2100 - 200 * math.log10(r)


def fair_odds(p):
    p = max(0.02, min(0.98, p))
    return round(1 / p, 2)


def elo_weighted_form(player_id, recent_matches, surface, player_index):
    """
    Elo-weighted form: uses opponent strength via Elo to compute surprise.
    Returns (form_score, sample_size) where sample_size is # of resolved
    opponents used.
    """
    if not recent_matches or player_id is None:
        return 0.5, 0
    if player_id not in ELO_RATINGS:
        return 0.5, 0

    surface_key = surface.title()
    my_elo = elo.get_player_elo(player_id, surface_key, ELO_RATINGS)

    recency_weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
    total_weight = 0.0
    weighted_surprise = 0.0
    used = 0

    for i, match in enumerate(recent_matches[:10]):
        opp_name = match.get("opponent", "")
        opp_id = players.resolve(opp_name, player_index)
        if not opp_id or opp_id not in ELO_RATINGS:
            continue

        opp_elo = elo.get_player_elo(opp_id, surface_key, ELO_RATINGS)
        expected = elo_prob(my_elo, opp_elo)
        actual = 1.0 if match.get("result") == "W" else 0.0
        surprise = actual - expected

        w = recency_weights[i] if i < len(recency_weights) else 0.15
        weighted_surprise += surprise * w
        total_weight += w
        used += 1

    if total_weight == 0:
        return 0.5, 0

    avg_surprise = weighted_surprise / total_weight
    form = 0.5 + avg_surprise * 1.5
    form = max(0.1, min(0.9, form))
    return form, used


def recency_weighted_wl(matches):
    """
    Fallback form: recency-weighted win/loss ratio.
    Returns (form_score, sample_size).
    """
    if not matches:
        return 0.5, 0

    weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
    total_w = 0.0
    wins = 0.0
    n = 0
    for i, m in enumerate(matches[:10]):
        w = weights[i] if i < len(weights) else 0.15
        total_w += w
        if m.get("result") == "W":
            wins += w
        n += 1

    if total_w == 0:
        return 0.5, 0
    return wins / total_w, n


def compute_form(recent_matches, player_id, surface, player_index):
    """
    Compute form for a player:
      1. If Elo-weighted resolves for ≥3 matches, use it
      2. Otherwise fall back to recency-weighted W/L
    """
    if not recent_matches:
        return 0.5, 0

    # Try Elo-weighted first
    if player_id and player_id in ELO_RATINGS:
        elo_form, elo_n = elo_weighted_form(
            player_id, recent_matches, surface, player_index)
        if elo_n >= 3:
            return elo_form, elo_n

    # Fallback: plain recency-weighted W/L
    return recency_weighted_wl(recent_matches)


def predict(p1, p2, surface, form1=None, form2=None,
            form1_n=0, form2_n=0):
    surface_key = surface.title()
    pid1 = p1.get("player_id")
    pid2 = p2.get("player_id")

    if ELO_RATINGS and pid1 and pid2 and pid1 in ELO_RATINGS and pid2 in ELO_RATINGS:
        e1 = elo.get_player_elo(pid1, surface_key, ELO_RATINGS)
        e2 = elo.get_player_elo(pid2, surface_key, ELO_RATINGS)
    else:
        e1 = rank_to_rating(p1.get("ranking"))
        e2 = rank_to_rating(p2.get("ranking"))
    p_elo = elo_prob(e1, e2)

    f1 = form1 if form1 is not None else 0.5
    f2 = form2 if form2 is not None else 0.5

    # Confidence based on sample size, per side.
    conf1 = min(form1_n / 5.0, 1.0) if form1_n > 0 else 0.0
    conf2 = min(form2_n / 5.0, 1.0) if form2_n > 0 else 0.0

    # If NEITHER side has data, form contributes nothing.
    # If AT LEAST ONE side has data, use max confidence but discount slightly.
    if form1_n == 0 and form2_n == 0:
        confidence = 0.0
    elif form1_n > 0 and form2_n > 0:
        confidence = min(conf1, conf2)
    else:
        confidence = max(conf1, conf2) * 0.7

    form_delta = (f1 - f2) * confidence
    p_form = 0.5 + form_delta * 0.15
    p_form = max(0.15, min(0.85, p_form))

    r1 = rank_to_rating(p1.get("ranking"))
    r2 = rank_to_rating(p2.get("ranking"))
    p_rank = elo_prob(r1, r2)

    p_match = 0.60 * p_elo + 0.20 * p_form + 0.20 * p_rank
    p_match = max(0.02, min(0.98, p_match))

    return {
        "prob": p_match,
        "p_elo": p_elo, "p_form": p_form, "p_rank": p_rank,
        "e1": e1, "e2": e2,
        "f1": f1, "f2": f2,
        "f1_n": form1_n, "f2_n": form2_n,
        "form_conf": confidence,
    }


# =================== Load matches ===================
def load_matches(ranked_only=True, force=False, fetch_odds=True,
                 target_date=None):
    global ELO_RATINGS
    if target_date is None:
        target_date = datetime.now(NAIROBI_TZ).date()

    date_str = target_date.isoformat()
    key = f"{date_str}_{'ranked' if ranked_only else 'all'}"

    if key in fixtures_cache and not force:
        return fixtures_cache[key]

    if not ELO_RATINGS:
        print("[app] loading historical dataset for Elo...")
        matches_list = dataset.load_all_matches()
        if matches_list:
            ELO_RATINGS = elo.compute_ratings(matches_list)
        else:
            print("[app] no historical data — using rank-based fallback")

    player_index = players.build_index()
    form_cache = scrapers.load_form()

    raw = scrapers.get_todays_matches(ranked_only=ranked_only,
                                     target_date=target_date)

    staged = []
    resolved_hits = 0
    form_resolved = 0

    for m in raw:
        surface = m.get("surface", "hard")
        if surface not in SURFACES:
            surface = "hard"

        pid1 = players.resolve(m["p1_name"], player_index)
        pid2 = players.resolve(m["p2_name"], player_index)
        if pid1 and pid2:
            resolved_hits += 1

        p1 = {"name": m["p1_name"], "ranking": m.get("p1_rank"),
              "player_id": pid1}
        p2 = {"name": m["p2_name"], "ranking": m.get("p2_rank"),
              "player_id": pid2}

        fm1 = form_cache.get(m["p1_name"], {}).get("matches", [])
        fm2 = form_cache.get(m["p2_name"], {}).get("matches", [])
        form1, form1_n = compute_form(fm1, pid1, surface, player_index)
        form2, form2_n = compute_form(fm2, pid2, surface, player_index)

        if form1_n >= 3 and form2_n >= 3:
            form_resolved += 1

        pred = predict(p1, p2, surface,
                       form1=form1, form2=form2,
                       form1_n=form1_n, form2_n=form2_n)

        display_time = convert_te_time_to_nairobi(
            m.get("status") or "", match_date=target_date)

        staged.append({
            "match": {
                "id": m["id"],
                "te_match_id": m.get("te_match_id"),
                "tournament": m.get("tournament") or "Tennis",
                "round": m.get("round", ""),
                "surface": surface,
                "source": m.get("source", ""),
                "time": display_time,
                "bookmaker_odds": None,
            },
            "p1": p1, "p2": p2,
            "p1_recent": m.get("p1_recent", []),
            "p2_recent": m.get("p2_recent", []),
            "surface": surface,
            "pred": pred,
        })

    # -------- Pass 2: only fetch odds for interesting matches --------
    if fetch_odds:
        interesting = []
        for s in staged:
            if not s["match"]["te_match_id"]:
                continue
            p = s["pred"]["prob"]
            if max(p, 1 - p) >= ODDS_FETCH_MIN_PROB:
                interesting.append(s)

        print(f"[app] odds needed for {len(interesting)}/{len(staged)} matches "
              f"(model prob ≥ {ODDS_FETCH_MIN_PROB*100:.0f}%)")

        if interesting:
            acquired = _ODDS_FETCH_LOCK.acquire(blocking=False)
            if not acquired:
                print("[app] another odds fetch in progress — using cache")
            else:
                try:
                    _fetch_odds_for(interesting)
                finally:
                    _ODDS_FETCH_LOCK.release()

    out = staged
    out.sort(key=lambda m: (m["match"]["time"] == "", m["match"]["time"]))

    print(f"[app] [{date_str}] resolved: {resolved_hits}/{len(out)} · "
          f"form: {form_resolved}/{len(out)} · "
          f"odds: {sum(1 for s in out if s['match']['bookmaker_odds'])}/{len(out)}")

    fixtures_cache[key] = out
    return out


def _fetch_odds_for(staged_matches):
    t0 = time.time()
    odds_map = {}

    futures = {}
    with ThreadPoolExecutor(max_workers=ODDS_WORKERS) as pool:
        for s in staged_matches:
            tid = s["match"]["te_match_id"]
            if tid:
                futures[pool.submit(odds_scraper.fetch_match_odds_cached, tid)] = s

        try:
            for f in as_completed(futures, timeout=ODDS_FETCH_DEADLINE):
                s = futures[f]
                try:
                    mw, _ = f.result(timeout=1)
                except Exception:
                    mw = None
                odds_map[s["match"]["te_match_id"]] = mw
        except TimeoutError:
            print(f"[app] odds fetch hit deadline ({ODDS_FETCH_DEADLINE}s), "
                  f"got {len(odds_map)}/{len(futures)}")
            for f in futures:
                f.cancel()

    for s in staged_matches:
        tid = s["match"]["te_match_id"]
        if tid in odds_map:
            s["match"]["bookmaker_odds"] = odds_map[tid]

    hits = sum(1 for s in staged_matches if s["match"]["bookmaker_odds"])
    print(f"[app] odds fetch done in {time.time() - t0:.1f}s · "
          f"{hits}/{len(staged_matches)} matched")


# =================== Auto-track ===================
def _is_valuable(prob, edge):
    return (prob >= MODEL_SUPPORT_MIN_PROB
            and edge >= AUTO_TRACK_MIN_EDGE_MW)


def _auto_track_picks(matches):
    added = 0

    for m in matches:
        bo = m["match"].get("bookmaker_odds")
        if not bo:
            continue

        p1 = m["pred"]["prob"]
        p2 = 1 - p1
        e1 = p1 - (1.0 / bo["p1_odds"])
        e2 = p2 - (1.0 / bo["p2_odds"])

        for side, prob, odds, edge, name, opp in (
            ("p1", p1, bo["p1_odds"], e1, m["p1"]["name"], m["p2"]["name"]),
            ("p2", p2, bo["p2_odds"], e2, m["p2"]["name"], m["p1"]["name"]),
        ):
            if not _is_valuable(prob, edge):
                continue
            pid = tracker.add_pick(
                match_id=m["match"]["id"],
                te_match_id=str(m["match"]["te_match_id"] or ""),
                market="mw", line=None, pick_side=side,
                pick_name=name, opponent_name=opp,
                tournament=m["match"]["tournament"],
                surface=m["match"]["surface"],
                match_time=m["match"]["time"],
                model_prob=prob, bookmaker=bo["bookmaker"],
                book_odds=odds, edge=edge,
            )
            if pid:
                added += 1
                print(f"[auto-track] {name} "
                      f"p={prob*100:.0f}% edge=+{edge*100:.1f}% @ {odds}")

    return added


# =================== Background auto-settle ===================
def _auto_settle_loop():
    time.sleep(AUTO_SETTLE_START_DELAY)

    while True:
        try:
            result = settler.settle_open_picks(verbose=True)
            if result["settled"] > 0 or result["voided"] > 0:
                print(f"[auto-settle] settled={result['settled']} "
                      f"voided={result['voided']} "
                      f"in {result['elapsed']:.1f}s")
        except Exception as e:
            print(f"[auto-settle] error: {e}")

        jitter = random.randint(-60, 60)
        time.sleep(AUTO_SETTLE_INTERVAL + jitter)


# =================== Template (main) ===================
PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tennis Betting Model</title>
<style>
  :root{
    --bg0:#0a0d1a;--bg2:#1a1f36;
    --fg:#e8eaf2;--dim:#8892b0;--dim2:#b8c0dc;
    --line:rgba(255,255,255,0.08);--line2:rgba(255,255,255,0.04);
    --p1:#7fd4ff;--p1b:#4fa3ff;
    --p2:#a78bfa;--p2b:#8b5cf6;
    --hit:#6ee7b7;--hitbg:rgba(52,211,153,0.13);
    --miss:#94a3b8;--missbg:rgba(148,163,184,0.07);
    --neg:#f87171;--negbg:rgba(248,113,113,0.11);
    --w:#6ee7b7;--l:#f87171;
    --bm:#fbbf24;--bmbg:rgba(251,191,36,0.14);
    --bmline:rgba(251,191,36,0.25);
    --dual:#a78bfa;
    --muted:#7a85a8;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,"Segoe UI",Roboto,sans-serif;
    background:radial-gradient(circle at top,var(--bg2) 0%,var(--bg0) 60%);
    color:var(--fg);min-height:100vh;padding:22px 12px 60px;
    -webkit-font-smoothing:antialiased}
  .wrap{max-width:680px;margin:0 auto}
  h1{font-size:22px;letter-spacing:-0.4px;
    background:linear-gradient(90deg,var(--p1),var(--p2));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .sub{color:var(--dim);font-size:12.5px;margin-top:4px}

  .toolbar{display:flex;gap:8px;margin:12px 0 8px;flex-wrap:wrap}
  .toolbar input[type=text]{flex:1;min-width:200px;
    background:rgba(255,255,255,0.05);
    border:1px solid var(--line);color:var(--fg);
    padding:9px 12px;border-radius:9px;font-size:13px;font-family:inherit}
  .toolbar input[type=text]:focus{outline:none;
    border-color:rgba(127,212,255,0.55);
    background:rgba(127,212,255,0.06)}
  .toolbar input[type=date]{
    background:rgba(255,255,255,0.05);
    border:1px solid var(--line);color:var(--fg);
    padding:9px 12px;border-radius:9px;font-size:13px;font-family:inherit;
    color-scheme:dark;
  }
  .toolbar input[type=date]:focus{outline:none;
    border-color:rgba(127,212,255,0.55);
    background:rgba(127,212,255,0.06)}
  .toolbar a{background:rgba(255,255,255,0.05);border:1px solid var(--line);
    color:var(--dim2);padding:9px 14px;border-radius:9px;
    font-size:12px;text-decoration:none;display:flex;align-items:center}
  .toolbar a:hover{border-color:rgba(127,212,255,0.4);color:var(--p1)}
  .toolbar a.primary{background:linear-gradient(90deg,var(--p1),var(--p2));
    color:#0a0d1a;font-weight:800;border-color:transparent}
  .toolbar .today-btn{
    background:rgba(110,231,183,0.12);border:1px solid rgba(110,231,183,0.35);
    color:var(--hit);padding:9px 12px;border-radius:9px;
    font-size:12px;font-weight:700;text-decoration:none;
    display:flex;align-items:center;gap:4px;
  }
  .toolbar .today-btn:hover{background:rgba(110,231,183,0.18)}

  .date-info{
    font-size:11.5px;color:var(--dim);
    margin-bottom:12px;display:flex;align-items:center;gap:8px;
  }
  .date-info .badge{
    background:rgba(255,255,255,0.06);
    padding:2px 8px;border-radius:6px;
    color:var(--dim2);font-weight:600;
    font-variant-numeric:tabular-nums;
  }
  .date-info .badge.today{
    background:rgba(110,231,183,0.13);
    color:var(--hit);
  }
  .date-info .badge.past{
    background:rgba(167,139,250,0.13);
    color:var(--dual);
  }

  .filter-bar{
    background:rgba(167,139,250,0.08);
    border:1px solid rgba(167,139,250,0.25);
    border-radius:10px;padding:10px 14px;margin-bottom:16px;
    font-size:11.5px;color:var(--dim2);
    display:flex;align-items:center;gap:10px;flex-wrap:wrap;
  }
  .filter-bar b{color:var(--fg);font-weight:700}
  .filter-bar .dot{color:var(--dual);font-size:14px}

  .cards-grid{display:block}

  .card{background:linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.012));
    border:1px solid var(--line);border-radius:16px;margin-bottom:12px;overflow:hidden}
  .card.dual{border-color:rgba(167,139,250,0.35)}
  .card-head{padding:14px 16px 12px;border-bottom:1px solid var(--line2)}
  .tourney{font-size:10.5px;letter-spacing:1.3px;text-transform:uppercase;
    color:var(--p1);font-weight:700;margin-bottom:8px;
    display:flex;align-items:center;gap:7px;flex-wrap:wrap}
  .tourney .dot{width:4px;height:4px;border-radius:50%;background:var(--p1)}
  .tourney .time-badge{
    background:rgba(127,212,255,0.1);color:var(--p1);
    padding:2px 6px;border-radius:5px;font-size:9.5px;
    letter-spacing:0.5px;font-weight:700;
    font-variant-numeric:tabular-nums;
    border:1px solid rgba(127,212,255,0.2);
  }
  .tourney .bm-badge{
    background:var(--bmbg);color:var(--bm);
    padding:2px 6px;border-radius:5px;font-size:9.5px;
    letter-spacing:0.3px;font-weight:700;
    border:1px solid var(--bmline);
  }
  .tourney .dual-badge{
    background:rgba(167,139,250,0.15);color:var(--dual);
    padding:2px 6px;border-radius:5px;font-size:9.5px;
    letter-spacing:0.3px;font-weight:800;
    border:1px solid rgba(167,139,250,0.4);
  }
  .tourney .src{margin-left:auto;color:#4a5270;font-size:9.5px;
    letter-spacing:0.5px;font-weight:600;text-transform:none}
  .matchup{display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center}
  .side{min-width:0}.side.right{text-align:right}
  .pname{font-size:15px;font-weight:600;letter-spacing:-0.2px;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .pmeta{font-size:10.5px;color:var(--dim);margin-top:3px;
    font-variant-numeric:tabular-nums}
  .pmeta b{color:var(--dim2);font-weight:600}
  .form-strip{display:flex;gap:3px;margin-top:5px;justify-content:flex-start}
  .form-strip.right{justify-content:flex-end}
  .form-dot{width:16px;height:16px;border-radius:4px;font-size:9px;
    display:flex;align-items:center;justify-content:center;
    font-weight:700;font-family:monospace}
  .form-dot.w{background:rgba(110,231,183,0.18);color:var(--w);
    border:1px solid rgba(110,231,183,0.35)}
  .form-dot.l{background:rgba(248,113,113,0.15);color:var(--l);
    border:1px solid rgba(248,113,113,0.3)}
  .vs-div{color:#3a4160;font-size:10px;font-weight:800;letter-spacing:1px}
  .pbar{display:flex;height:3px;border-radius:2px;overflow:hidden;
    margin-top:9px;background:rgba(255,255,255,0.05)}
  .pbar .a{background:linear-gradient(90deg,var(--p1),var(--p1b))}
  .pbar .b{background:linear-gradient(90deg,var(--p2),var(--p2b))}

  .market{padding:10px 16px 12px;border-bottom:1px solid var(--line2)}
  .market:last-of-type{border-bottom:none}
  .mkt-title{display:flex;justify-content:space-between;align-items:center;
    font-size:10px;letter-spacing:1.3px;text-transform:uppercase;
    color:var(--dim);font-weight:700;margin-bottom:6px;gap:8px}
  .mkt-title .left{display:flex;align-items:center;gap:8px;min-width:0}
  .mkt-title .hint{color:#4a5270;letter-spacing:0;text-transform:none;
    font-weight:500;font-size:10.5px;white-space:nowrap}
  .mrow{display:grid;grid-template-columns:minmax(0,1fr) 50px 60px 52px 26px;
    align-items:center;gap:6px;padding:4px 0}
  .mrow + .mrow{border-top:1px dashed var(--line2)}
  .label{display:flex;align-items:center;gap:6px;min-width:0}
  .swatch{width:3px;height:13px;border-radius:2px;flex-shrink:0}
  .swatch.p1{background:linear-gradient(180deg,var(--p1),var(--p1b))}
  .swatch.p2{background:linear-gradient(180deg,var(--p2),var(--p2b))}
  .lbl-text{font-size:12.5px;color:var(--fg);
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .prob{font-size:12.5px;font-weight:700;text-align:right;
    font-variant-numeric:tabular-nums;color:var(--fg)}
  .prob.soft{color:var(--dim2);font-weight:600}
  input.odds{width:100%;background:rgba(255,255,255,0.05);
    border:1px solid var(--line);color:var(--fg);padding:5px 7px;
    border-radius:6px;font-size:12px;font-variant-numeric:tabular-nums;
    text-align:right;font-family:inherit}
  input.odds.default{color:#94a3b8;font-style:italic}
  input.odds.bookmaker{color:var(--bm);font-weight:600}
  input.odds.soft{color:var(--dim2)}
  .edge{font-size:11px;font-weight:700;text-align:right;
    padding:3px 6px;border-radius:5px;font-variant-numeric:tabular-nums;
    border:1px solid transparent}
  .edge.hit{color:var(--hit);background:var(--hitbg);border-color:rgba(52,211,153,0.3)}
  .edge.miss{color:var(--miss);background:var(--missbg)}
  .edge.neg{color:var(--neg);background:var(--negbg)}
  .edge.empty{color:transparent;background:transparent;border-color:transparent}
  .edge.empty::after{content:"—";color:var(--muted);font-weight:500}
  .auto-dot{
    width:24px;height:24px;border-radius:5px;
    display:flex;align-items:center;justify-content:center;
    font-size:11px;color:var(--dual);
    background:rgba(167,139,250,0.15);
    border:1px solid rgba(167,139,250,0.45);
  }
  .auto-dot.placeholder{background:transparent;border-color:transparent}

  .card-foot{padding:10px 16px 14px;background:rgba(0,0,0,0.15);
    border-top:1px solid var(--line2)}
  .breakdown{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}
  .bd{font-size:9.5px;color:var(--dim);text-align:center;letter-spacing:0.4px;
    text-transform:uppercase}
  .bd b{display:block;color:var(--dim2);font-size:12px;font-weight:700;
    margin-top:1px;font-variant-numeric:tabular-nums;text-transform:none}
  .bd .n{display:block;font-size:8px;color:#4a5270;margin-top:1px;
    font-weight:500;letter-spacing:0}
  .empty{background:rgba(255,255,255,0.03);border:1px solid var(--line);
    border-radius:14px;padding:22px;text-align:center;color:var(--dim2);
    font-size:13px;line-height:1.6}
  a{color:var(--p1)}

  @media (min-width: 900px) {
    body { padding: 32px 24px 80px; }
    .wrap { max-width: 1280px; }
    .toolbar, .filter-bar, .date-info { max-width: 900px; }
    .cards-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      align-items: start;
    }
    .cards-grid .card { margin-bottom: 0; }
  }

  @media (min-width: 1400px) {
    .wrap { max-width: 1440px; }
    .cards-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>🎾 Tennis Betting Model</h1>
  <p class="sub">{{ matches|length }} matches · {{ selected_date_str }} · Elo + weighted form · {{ 'ranked only' if ranked_only else 'all' }}</p>

  <div class="toolbar">
    <input id="q" type="text" placeholder="Filter by player or tournament…" autocomplete="off">
    <input type="date" id="date-picker" value="{{ selected_date_str }}"
           max="{{ max_date_str }}"
           onchange="changeDate(this.value)">
    {% if not is_today %}
      <a href="/?filter={{ 'ranked' if ranked_only else 'all' }}"
         class="today-btn">↺ Today</a>
    {% endif %}
    <a href="/tracker" class="primary">📋 Tracker</a>
  </div>

  <div class="date-info">
    <span class="badge {% if is_today %}today{% else %}past{% endif %}">
      {% if is_today %}● Today{% else %}◷ {{ selected_date_str }}{% endif %}
    </span>
    <span>{{ weekday_name }}</span>
  </div>

  <div class="filter-bar">
    <span class="dot">◆</span>
    Show edge when model ≥ <b>{{ '%.0f'|format(MODEL_SUPPORT_MIN_PROB*100) }}%</b>
    · auto-track if edge ≥ <b>{{ '%.0f'|format(AUTO_TRACK_MIN_EDGE_MW*100) }}%</b>
    <a href="/?date={{ selected_date_str }}&filter={{ 'all' if not ranked_only else 'ranked' }}"
       style="margin-left:auto;font-size:11px;text-decoration:none;color:var(--p1)">
      {% if ranked_only %}All tours{% else %}Ranked only{% endif %}
    </a>
    <a href="/?date={{ selected_date_str }}&refresh=1&filter={{ 'ranked' if ranked_only else 'all' }}"
       style="font-size:11px;text-decoration:none;color:var(--dim)">↻</a>
  </div>

  {% if matches|length == 0 %}
    <div class="empty">
      No matches on {{ selected_date_str }}.
      {% if not is_today %}
        <a href="/?filter={{ 'ranked' if ranked_only else 'all' }}">Back to today →</a>
      {% else %}
        Try <a href="/?refresh=1">refreshing</a>.
      {% endif %}
    </div>
  {% endif %}

  <div class="cards-grid">
    {% for m in matches %}
      {% set s = odds_map.get(m.match.id, {}) %}
      {% set bo = m.match.bookmaker_odds %}
      {% set mw1 = s.get('mw1') if s.get('mw1') else (bo.p1_odds if bo else m.pred.fair_mw1) %}
      {% set mw2 = s.get('mw2') if s.get('mw2') else (bo.p2_odds if bo else m.pred.fair_mw2) %}
      {% set is_default = not (s.get('mw1')) %}
      {% set is_book = bo and not (s.get('mw1')) %}
      {% set p1p = m.pred.prob %}
      {% set p2p = 1 - p1p %}
      {% set e_mw1 = p1p - (1/mw1) %}
      {% set e_mw2 = p2p - (1/mw2) %}

      {% set sup_mw1 = p1p >= MODEL_SUPPORT_MIN_PROB %}
      {% set sup_mw2 = p2p >= MODEL_SUPPORT_MIN_PROB %}

      {% set tracked_mw1 = sup_mw1 and e_mw1 >= AUTO_TRACK_MIN_EDGE_MW %}
      {% set tracked_mw2 = sup_mw2 and e_mw2 >= AUTO_TRACK_MIN_EDGE_MW %}

      {% set card_dual = tracked_mw1 or tracked_mw2 %}

      <div class="card {% if card_dual %}dual{% endif %}" data-match="{{ m.match.id }}">
        <div class="card-head">
          <div class="tourney">
            <span class="dot"></span>{{ m.match.tournament }} · {{ m.match.surface|title }}{% if m.match.round %} · {{ m.match.round }}{% endif %}
            {% if m.match.time %}<span class="time-badge">🕐 {{ m.match.time }} GMT+3</span>{% endif %}
            {% if bo %}<span class="bm-badge">💰 {{ bo.bookmaker }}</span>{% endif %}
            {% if card_dual %}<span class="dual-badge">◆ auto</span>{% endif %}
            <span class="src">{{ m.match.source }}</span>
          </div>
          <div class="matchup">
            <div class="side">
              <div class="pname">{{ m.p1.name }}</div>
              <div class="pmeta">Rank <b>{% if m.p1.ranking %}#{{ m.p1.ranking }}{% else %}NR{% endif %}</b> · Elo <b>{{ '%.0f'|format(m.pred.e1) }}</b></div>
              {% if m.p1_recent %}
                <div class="form-strip">
                  {% for r in m.p1_recent %}<div class="form-dot {{ 'w' if r.result == 'W' else 'l' }}">{{ r.result }}</div>{% endfor %}
                </div>
              {% endif %}
            </div>
            <div class="vs-div">VS</div>
            <div class="side right">
              <div class="pname">{{ m.p2.name }}</div>
              <div class="pmeta">Rank <b>{% if m.p2.ranking %}#{{ m.p2.ranking }}{% else %}NR{% endif %}</b> · Elo <b>{{ '%.0f'|format(m.pred.e2) }}</b></div>
              {% if m.p2_recent %}
                <div class="form-strip right">
                  {% for r in m.p2_recent %}<div class="form-dot {{ 'w' if r.result == 'W' else 'l' }}">{{ r.result }}</div>{% endfor %}
                </div>
              {% endif %}
            </div>
          </div>
          <div class="pbar">
            <div class="a" style="width: {{ p1p*100 }}%"></div>
            <div class="b" style="width: {{ p2p*100 }}%"></div>
          </div>
        </div>

        <div class="market">
          <div class="mkt-title"><span>Match Winner</span>
            <span class="hint">{{ '%.1f'|format(p1p*100) }} / {{ '%.1f'|format(p2p*100) }}</span></div>
          <div class="mrow">
            <div class="label"><span class="swatch p1"></span><span class="lbl-text">{{ m.p1.name }}</span></div>
            <div class="prob {% if not sup_mw1 %}soft{% endif %}">{{ '%.1f'|format(p1p*100) }}%</div>
            <input class="odds {% if not sup_mw1 %}soft{% elif is_book %}bookmaker{% elif is_default %}default{% endif %}" value="{{ mw1 }}" readonly>
            {% if sup_mw1 %}
              <div class="edge {{ 'hit' if e_mw1>0.03 else ('neg' if e_mw1<-0.03 else 'miss') }}">{{ '%+.1f'|format(e_mw1*100) }}%</div>
            {% else %}
              <div class="edge empty"></div>
            {% endif %}
            {% if tracked_mw1 %}<div class="auto-dot">◆</div>
            {% else %}<div class="auto-dot placeholder"></div>{% endif %}
          </div>
          <div class="mrow">
            <div class="label"><span class="swatch p2"></span><span class="lbl-text">{{ m.p2.name }}</span></div>
            <div class="prob {% if not sup_mw2 %}soft{% endif %}">{{ '%.1f'|format(p2p*100) }}%</div>
            <input class="odds {% if not sup_mw2 %}soft{% elif is_book %}bookmaker{% elif is_default %}default{% endif %}" value="{{ mw2 }}" readonly>
            {% if sup_mw2 %}
              <div class="edge {{ 'hit' if e_mw2>0.03 else ('neg' if e_mw2<-0.03 else 'miss') }}">{{ '%+.1f'|format(e_mw2*100) }}%</div>
            {% else %}
              <div class="edge empty"></div>
            {% endif %}
            {% if tracked_mw2 %}<div class="auto-dot">◆</div>
            {% else %}<div class="auto-dot placeholder"></div>{% endif %}
          </div>
        </div>

        <div class="card-foot">
          <div class="breakdown">
            <div class="bd">
              Elo<b>{{ '%.0f'|format(m.pred.p_elo*100) }}%</b>
            </div>
            <div class="bd">
              Form<b>{{ '%.0f'|format(m.pred.p_form*100) }}%</b>
              <span class="n">n={{ m.pred.f1_n }}/{{ m.pred.f2_n }}</span>
            </div>
            <div class="bd">
              Rank<b>{{ '%.0f'|format(m.pred.p_rank*100) }}%</b>
            </div>
          </div>
        </div>
      </div>
    {% endfor %}
  </div>
</div>

<script>
  function changeDate(val) {
    if (!val) return;
    const url = new URL(window.location.href);
    url.searchParams.set('date', val);
    window.location = url.toString();
  }

  const q = document.getElementById('q');
  if (q) {
    q.addEventListener('input', e => {
      const term = e.target.value.toLowerCase();
      document.querySelectorAll('.card').forEach(card => {
        card.style.display = card.textContent.toLowerCase().includes(term) ? '' : 'none';
      });
    });
  }
</script>
</body>
</html>
"""


# =================== Template (tracker) ===================
TRACKER_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tracker</title>
<style>
  :root{
    --bg0:#0a0d1a;--bg2:#1a1f36;
    --fg:#e8eaf2;--dim:#8892b0;--dim2:#b8c0dc;
    --line:rgba(255,255,255,0.08);--line2:rgba(255,255,255,0.04);
    --p1:#7fd4ff;--p2:#a78bfa;
    --hit:#6ee7b7;--hitbg:rgba(52,211,153,0.13);
    --miss:#94a3b8;--missbg:rgba(148,163,184,0.07);
    --neg:#f87171;--negbg:rgba(248,113,113,0.11);
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,"Segoe UI",Roboto,sans-serif;
    background:radial-gradient(circle at top,var(--bg2) 0%,var(--bg0) 60%);
    color:var(--fg);min-height:100vh;padding:22px 12px 60px;
    -webkit-font-smoothing:antialiased}
  .wrap{max-width:680px;margin:0 auto}
  h1{font-size:22px;letter-spacing:-0.4px;
    background:linear-gradient(90deg,var(--p1),var(--p2));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .sub{color:var(--dim);font-size:12.5px;margin-top:4px}
  .nav{display:flex;gap:8px;margin:12px 0 16px}
  .nav a{background:rgba(255,255,255,0.05);border:1px solid var(--line);
    color:var(--dim2);padding:9px 14px;border-radius:9px;
    font-size:12px;text-decoration:none}
  .nav a:hover{border-color:rgba(127,212,255,0.4);color:var(--p1)}
  .nav form{display:inline}
  .nav button{background:linear-gradient(90deg,var(--p1),var(--p2));
    border:none;color:#0a0d1a;padding:9px 14px;border-radius:9px;
    font-size:12px;font-weight:800;cursor:pointer}
  .stats{
    background:linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.012));
    border:1px solid var(--line);border-radius:16px;
    padding:14px 16px;margin-bottom:16px;
  }
  .stat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
  .stat{text-align:center}
  .stat .label{font-size:9.5px;color:var(--dim);letter-spacing:0.5px;
    text-transform:uppercase;font-weight:700}
  .stat .value{font-size:18px;color:var(--fg);font-weight:800;
    margin-top:3px;font-variant-numeric:tabular-nums}
  .stat .value.hit{color:var(--hit)}
  .stat .value.neg{color:var(--neg)}
  .divider{height:1px;background:var(--line2);margin:12px 0}
  .sub-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
  .sub-stat{text-align:center}
  .sub-stat .label{font-size:9px;color:var(--dim);letter-spacing:0.4px;
    text-transform:uppercase}
  .sub-stat .value{font-size:13px;color:var(--dim2);font-weight:700;
    margin-top:2px;font-variant-numeric:tabular-nums}
  .filters{display:flex;gap:6px;margin-bottom:12px}
  .filters a{font-size:11.5px;padding:6px 12px;border-radius:8px;
    text-decoration:none;border:1px solid var(--line);color:var(--dim2)}
  .filters a.active{background:linear-gradient(90deg,var(--p1),var(--p2));
    color:#0a0d1a;font-weight:700;border-color:transparent}
  .picks-grid{display:block}
  .pick{
    background:linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.012));
    border:1px solid var(--line);border-radius:14px;
    padding:12px 14px;margin-bottom:10px;
  }
  .pick.open{border-color:rgba(127,212,255,0.25)}
  .pick.won{border-color:rgba(110,231,183,0.35)}
  .pick.lost{border-color:rgba(248,113,113,0.3)}
  .pick.void{opacity:0.5}
  .pick-head{display:flex;justify-content:space-between;align-items:baseline;
    margin-bottom:6px}
  .pick-name{font-size:14px;font-weight:600}
  .pick-status{font-size:10px;font-weight:800;letter-spacing:0.5px;
    text-transform:uppercase;padding:2px 6px;border-radius:4px}
  .pick-status.open{color:var(--p1);background:rgba(127,212,255,0.1)}
  .pick-status.won{color:var(--hit);background:var(--hitbg)}
  .pick-status.lost{color:var(--neg);background:var(--negbg)}
  .pick-status.void{color:var(--miss);background:var(--missbg)}
  .pick-meta{font-size:10.5px;color:var(--dim);margin-bottom:10px}
  .pick-row{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;
    font-size:11px;margin-bottom:10px}
  .pick-row .kv{text-align:center}
  .pick-row .kv .k{color:var(--dim);font-size:9px;letter-spacing:0.4px;
    text-transform:uppercase}
  .pick-row .kv .v{color:var(--fg);font-weight:700;margin-top:2px;
    font-variant-numeric:tabular-nums}
  .actions{display:flex;gap:6px}
  .actions form{flex:1}
  .actions button{width:100%;border:none;padding:8px;border-radius:7px;
    font-size:12px;font-weight:800;cursor:pointer;letter-spacing:0.3px}
  .btn-won{background:rgba(110,231,183,0.15);color:var(--hit);
    border:1px solid rgba(110,231,183,0.4)}
  .btn-lost{background:rgba(248,113,113,0.15);color:var(--neg);
    border:1px solid rgba(248,113,113,0.4)}
  .btn-void{background:rgba(148,163,184,0.1);color:var(--miss);
    border:1px solid rgba(148,163,184,0.3)}
  .empty{background:rgba(255,255,255,0.03);border:1px dashed var(--line);
    border-radius:14px;padding:22px;text-align:center;color:var(--dim2);
    font-size:13px;line-height:1.6}
  a{color:var(--p1)}

  @media (min-width: 900px) {
    body { padding: 32px 24px 80px; }
    .wrap { max-width: 1280px; }
    .nav, .stats, .filters { max-width: 900px; }
    .picks-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }
    .picks-grid .pick { margin-bottom: 0; }
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>📋 Singles Tracker</h1>
  <p class="sub">Auto-settled from TennisExplorer · refreshes every 15 min</p>

  <div class="nav">
    <a href="/">← Back to matches</a>
    <form method="post" action="/settle_now">
      <button type="submit">↻ Settle now</button>
    </form>
  </div>

  {% if settle_summary %}
    <div style="background:rgba(127,212,255,0.08);border:1px solid rgba(127,212,255,0.25);
                border-radius:10px;padding:10px 14px;margin-bottom:14px;
                font-size:12px;color:var(--dim2)">
      Settle run: <b style="color:var(--fg)">{{ settle_summary.checked }}</b> checked ·
      <b style="color:var(--hit)">{{ settle_summary.settled }}</b> settled ·
      <b style="color:var(--dim)">{{ settle_summary.voided }}</b> voided ·
      {{ settle_summary.skipped }} pending ·
      {{ settle_summary.errors }} errors
      · {{ '%.1f'|format(settle_summary.elapsed) }}s
    </div>
  {% endif %}

  <div class="stats">
    <div class="stat-grid">
      <div class="stat">
        <div class="label">Open</div>
        <div class="value">{{ stats.open }}</div>
      </div>
      <div class="stat">
        <div class="label">Settled</div>
        <div class="value">{{ stats.settled }}</div>
      </div>
      <div class="stat">
        <div class="label">Strike Rate</div>
        <div class="value {% if stats.strike and stats.strike > 0.5 %}hit{% endif %}">
          {% if stats.strike is not none %}{{ '%.1f'|format(stats.strike*100) }}%{% else %}—{% endif %}
        </div>
      </div>
    </div>
    <div class="divider"></div>
    <div class="sub-grid">
      <div class="sub-stat"><div class="label">Won</div><div class="value">{{ stats.won }}</div></div>
      <div class="sub-stat"><div class="label">Lost</div><div class="value">{{ stats.lost }}</div></div>
      <div class="sub-stat"><div class="label">Avg Odds</div><div class="value">{% if stats.avg_odds %}{{ '%.2f'|format(stats.avg_odds) }}{% else %}—{% endif %}</div></div>
      <div class="sub-stat"><div class="label">Avg Edge</div><div class="value">{% if stats.avg_edge %}{{ '%+.1f'|format(stats.avg_edge*100) }}%{% else %}—{% endif %}</div></div>
    </div>
    <div class="divider"></div>
    <div class="sub-grid">
      <div class="sub-stat"><div class="label">Avg Model</div><div class="value">{% if stats.avg_model_prob %}{{ '%.1f'|format(stats.avg_model_prob*100) }}%{% else %}—{% endif %}</div></div>
      <div class="sub-stat"><div class="label">Break-even</div><div class="value">{% if stats.breakeven %}{{ '%.1f'|format(stats.breakeven*100) }}%{% else %}—{% endif %}</div></div>
      <div class="sub-stat"><div class="label">Edge vs Book</div><div class="value {% if stats.edge_vs_book and stats.edge_vs_book > 0 %}hit{% elif stats.edge_vs_book and stats.edge_vs_book < 0 %}neg{% endif %}">{% if stats.edge_vs_book is not none %}{{ '%+.1f'|format(stats.edge_vs_book*100) }}pp{% else %}—{% endif %}</div></div>
      <div class="sub-stat"><div class="label">ROI</div><div class="value {% if stats.roi and stats.roi > 0 %}hit{% elif stats.roi and stats.roi < 0 %}neg{% endif %}">{% if stats.roi is not none %}{{ '%+.1f'|format(stats.roi*100) }}%{% else %}—{% endif %}</div></div>
    </div>
  </div>

  <div class="filters">
    <a href="/tracker" class="{{ 'active' if not active_filter else '' }}">All</a>
    <a href="/tracker?status=open" class="{{ 'active' if active_filter == 'open' else '' }}">Open</a>
    <a href="/tracker?status=won" class="{{ 'active' if active_filter == 'won' else '' }}">Won</a>
    <a href="/tracker?status=lost" class="{{ 'active' if active_filter == 'lost' else '' }}">Lost</a>
  </div>

  {% if picks|length == 0 %}
    <div class="empty">
      No tracked picks yet.<br>Open the main page to trigger auto-tracking.
    </div>
  {% endif %}

  <div class="picks-grid">
    {% for p in picks %}
      <div class="pick {{ p.status }}">
        <div class="pick-head">
          <div class="pick-name">{{ p.pick_name }}</div>
          <div class="pick-status {{ p.status }}">{{ p.status }}</div>
        </div>
        <div class="pick-meta">
          vs {{ p.opponent_name }} · {{ p.tournament }} · {{ p.surface|title }}
          {% if p.match_time %} · {{ p.match_time }} GMT+3{% endif %}
          · {{ p.bookmaker or 'n/a' }}
        </div>
        <div class="pick-row">
          <div class="kv"><div class="k">Model</div><div class="v">{{ '%.1f'|format(p.model_prob*100) }}%</div></div>
          <div class="kv"><div class="k">Odds</div><div class="v">{{ '%.2f'|format(p.book_odds) }}</div></div>
          <div class="kv"><div class="k">Implied</div><div class="v">{{ '%.1f'|format(100/p.book_odds) }}%</div></div>
          <div class="kv"><div class="k">Edge</div><div class="v" style="color:var(--hit)">{{ '%+.1f'|format(p.edge*100) }}%</div></div>
        </div>
        {% if p.status == 'open' %}
          <div class="actions">
            <form method="post" action="/settle/{{ p.id }}/won">
              <button class="btn-won" type="submit">✓ Won</button>
            </form>
            <form method="post" action="/settle/{{ p.id }}/lost">
              <button class="btn-lost" type="submit">✗ Lost</button>
            </form>
            <form method="post" action="/settle/{{ p.id }}/void">
              <button class="btn-void" type="submit">Void</button>
            </form>
          </div>
        {% endif %}
      </div>
    {% endfor %}
  </div>
</div>
</body>
</html>
"""


# =================== Routes ===================
def build_matches_with_markets(ranked_only=True, force=False,
                                target_date=None):
    matches = load_matches(ranked_only=ranked_only, force=force,
                           target_date=target_date)
    for m in matches:
        p = m["pred"]
        p["fair_mw1"] = fair_odds(p["prob"])
        p["fair_mw2"] = fair_odds(1 - p["prob"])
    return matches


@app.route("/")
def index():
    ranked_only = request.args.get("filter", "ranked") != "all"
    force = request.args.get("refresh") == "1"

    today = datetime.now(NAIROBI_TZ).date()
    date_str = request.args.get("date", "").strip()
    try:
        selected_date = (datetime.strptime(date_str, "%Y-%m-%d").date()
                        if date_str else today)
    except ValueError:
        selected_date = today

    is_today = selected_date == today
    selected_date_str = selected_date.isoformat()
    max_date_str = (today + timedelta(days=30)).isoformat()
    weekday_name = selected_date.strftime("%A, %d %B %Y")

    try:
        matches = build_matches_with_markets(
            ranked_only=ranked_only, force=force,
            target_date=selected_date,
        )
    except Exception as e:
        return f"<pre style='color:#f87171;padding:24px'>Error: {e}</pre>", 500

    auto_added = 0
    if is_today:
        try:
            auto_added = _auto_track_picks(matches)
        except Exception as e:
            print(f"[auto-track] error: {e}")

    return render_template_string(
        PAGE,
        matches=matches,
        odds_map=load_all_odds(),
        ranked_only=ranked_only,
        selected_date_str=selected_date_str,
        max_date_str=max_date_str,
        is_today=is_today,
        weekday_name=weekday_name,
        MODEL_SUPPORT_MIN_PROB=MODEL_SUPPORT_MIN_PROB,
        AUTO_TRACK_MIN_EDGE_MW=AUTO_TRACK_MIN_EDGE_MW,
        AUTO_TRACK_MARKETS=AUTO_TRACK_MARKETS,
        auto_track_added=auto_added,
    )


@app.route("/tracker")
def tracker_page():
    status = request.args.get("status")
    if status not in (None, "open", "won", "lost", "void"):
        status = None
    picks = tracker.list_picks(status=status)
    stats = tracker.stats()
    return render_template_string(
        TRACKER_PAGE,
        picks=picks, stats=stats,
        active_filter=status, settle_summary=None,
    )


@app.route("/settle_now", methods=["POST"])
def settle_now_route():
    summary = settler.settle_open_picks(verbose=True)
    picks = tracker.list_picks()
    stats = tracker.stats()
    return render_template_string(
        TRACKER_PAGE,
        picks=picks, stats=stats,
        active_filter=None, settle_summary=summary,
    )


@app.route("/settle/<int:pick_id>/<result>", methods=["POST"])
def settle_route(pick_id, result):
    if result not in ("won", "lost", "void"):
        return ("bad result", 400)
    tracker.settle_pick(pick_id, result)
    return redirect(url_for("tracker_page"))


@app.route("/clear_odds_cache")
def clear_odds_cache_route():
    odds_scraper.clear_cache()
    return "Odds cache cleared. <a href='/'>Back</a>"


@app.route("/debug")
def debug():
    import json as _json
    out = {}
    try:
        out["now_berlin"] = datetime.now(BERLIN_TZ).strftime("%H:%M %Z (UTC%z)")
        out["now_nairobi"] = datetime.now(NAIROBI_TZ).strftime("%H:%M %Z (UTC%z)")
        out["odds_cache_size"] = len(odds_scraper._load_cache())
        out["tracker_stats"] = tracker.stats()
        out["filters"] = {
            "model_support_min_prob": MODEL_SUPPORT_MIN_PROB,
            "odds_fetch_min_prob": ODDS_FETCH_MIN_PROB,
            "odds_fetch_deadline": ODDS_FETCH_DEADLINE,
            "mw_edge_floor": AUTO_TRACK_MIN_EDGE_MW,
            "markets": list(AUTO_TRACK_MARKETS),
        }
    except Exception as e:
        out["error"] = str(e)
    return ("<pre style='color:#e8eaf2;background:#0a0d1a;padding:20px;"
            "white-space:pre-wrap;word-break:break-word;font-size:12px'>"
            f"{_json.dumps(out, indent=2, default=str)}</pre>")


if __name__ == "__main__":
    ensure_schema()
    tracker.init_db()

    settle_thread = threading.Thread(target=_auto_settle_loop, daemon=True)
    settle_thread.start()
    print(f"[app] market: Moneyline only · "
          f"edge ≥ {AUTO_TRACK_MIN_EDGE_MW*100:.0f}% · "
          f"model ≥ {MODEL_SUPPORT_MIN_PROB*100:.0f}%")
    print(f"[app] odds fetch: only for matches with model ≥ "
          f"{ODDS_FETCH_MIN_PROB*100:.0f}% · "
          f"timeout={ODDS_FETCH_DEADLINE}s · workers={ODDS_WORKERS}")
    print(f"[app] auto-settle: first run in {AUTO_SETTLE_START_DELAY}s, "
          f"then every {AUTO_SETTLE_INTERVAL}s ± 60s")
    print(f"[app] timezone: Berlin {datetime.now(BERLIN_TZ).strftime('%H:%M %Z')} "
          f"→ Nairobi {datetime.now(NAIROBI_TZ).strftime('%H:%M %Z')}")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
