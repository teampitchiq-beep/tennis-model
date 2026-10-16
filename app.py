import math
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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

DEFAULT_SETS_LINE = 2.5
ODDS_WORKERS = 8

# ===== Filters =====
MODEL_SUPPORT_MIN_PROB = 0.60
AUTO_TRACK_MARKETS = ("mw", "sets")
AUTO_TRACK_MIN_EDGE_MW = 0.03
AUTO_TRACK_MIN_EDGE_SETS = 0.04

# Top picks dashboard
TOP_PICKS_MIN_ODDS = 1.50
TOP_PICKS_LIMIT = 5
TOP_PICKS_MIN_EDGE = 0.01

AUTO_SETTLE_INTERVAL = 900

# ===== Timezone =====
NAIROBI_TZ = ZoneInfo("Africa/Nairobi")
BERLIN_TZ = ZoneInfo("Europe/Berlin")


# =================== SQLite (odds) ===================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA user_version = 5")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds (
            match_id TEXT PRIMARY KEY,
            mw1 REAL, mw2 REAL,
            so REAL, su REAL,
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
    if conn.execute("PRAGMA user_version").fetchone()[0] < 5:
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


def convert_te_time_to_nairobi(time_str):
    """
    TennisExplorer shows times in Europe/Berlin (CET/CEST).
    Convert to Africa/Nairobi (EAT, GMT+3, no DST).

    Uses zoneinfo so the offset automatically adjusts for European DST
    (last Sunday of March → last Sunday of October).
    """
    if not time_str:
        return ""
    m = re.match(r"^(\d{1,2}):(\d{2})", time_str.strip())
    if not m:
        return time_str

    h, mn = int(m.group(1)), int(m.group(2))

    # Anchor to today's date in Nairobi so we know which DST period applies
    today_nairobi = datetime.now(NAIROBI_TZ).date()
    naive = datetime(today_nairobi.year, today_nairobi.month, today_nairobi.day,
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


def p_3sets_model(elo_diff):
    return 0.35 * math.exp(-abs(elo_diff) / 250)


def fair_odds(p):
    p = max(0.02, min(0.98, p))
    return round(1 / p, 2)


def elo_weighted_form(player_id, recent_matches, surface, player_index):
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
    conf1 = min(form1_n / 5.0, 1.0)
    conf2 = min(form2_n / 5.0, 1.0)
    confidence = min(conf1, conf2) if (form1_n > 0 and form2_n > 0) else 0.0

    form_delta = (f1 - f2) * confidence
    p_form = 0.5 + form_delta * 0.15
    p_form = max(0.15, min(0.85, p_form))

    r1 = rank_to_rating(p1.get("ranking"))
    r2 = rank_to_rating(p2.get("ranking"))
    p_rank = elo_prob(r1, r2)

    p_match = 0.60 * p_elo + 0.20 * p_form + 0.20 * p_rank
    p_match = max(0.02, min(0.98, p_match))

    p3 = p_3sets_model(e1 - e2)

    return {
        "prob": p_match,
        "p_elo": p_elo, "p_form": p_form, "p_rank": p_rank,
        "e1": e1, "e2": e2,
        "f1": f1, "f2": f2,
        "f1_n": form1_n, "f2_n": form2_n,
        "form_conf": confidence,
        "p_3sets": p3,
    }


# =================== Load matches ===================
def load_matches(ranked_only=True, force=False, fetch_odds=True):
    global ELO_RATINGS
    key = "ranked" if ranked_only else "all"
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

    raw = scrapers.get_todays_matches(ranked_only=ranked_only)

    odds_map = {}
    if fetch_odds:
        with_ids = [m for m in raw if m.get("te_match_id")]
        print(f"[app] fetching odds for {len(with_ids)} matches "
              f"({ODDS_WORKERS} workers)...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=ODDS_WORKERS) as pool:
            futures = {
                pool.submit(odds_scraper.fetch_match_odds_cached, m["te_match_id"]): m
                for m in with_ids
            }
            for f in as_completed(futures):
                m = futures[f]
                try:
                    mw, sets = f.result()
                except Exception:
                    mw, sets = None, None
                odds_map[m["te_match_id"]] = (mw, sets)
        print(f"[app] odds fetch done in {time.time() - t0:.1f}s")

    out = []
    resolved_hits = 0
    form_resolved = 0
    odds_hits = 0
    sets_hits = 0

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
        form1, form1_n = elo_weighted_form(pid1, fm1, surface, player_index)
        form2, form2_n = elo_weighted_form(pid2, fm2, surface, player_index)

        if form1_n >= 3 and form2_n >= 3:
            form_resolved += 1

        pred = predict(p1, p2, surface,
                       form1=form1, form2=form2,
                       form1_n=form1_n, form2_n=form2_n)

        bookmaker_odds = None
        sets_odds = None
        if fetch_odds and m.get("te_match_id") in odds_map:
            bookmaker_odds, sets_odds = odds_map[m["te_match_id"]]
            if bookmaker_odds:
                odds_hits += 1
            if sets_odds:
                sets_hits += 1

        display_time = convert_te_time_to_nairobi(m.get("status") or "")

        out.append({
            "match": {
                "id": m["id"],
                "te_match_id": m.get("te_match_id"),
                "tournament": m.get("tournament") or "Tennis",
                "round": m.get("round", ""),
                "surface": surface,
                "source": m.get("source", ""),
                "time": display_time,
                "bookmaker_odds": bookmaker_odds,
                "sets_odds": sets_odds,
            },
            "p1": p1, "p2": p2,
            "p1_recent": m.get("p1_recent", []),
            "p2_recent": m.get("p2_recent", []),
            "surface": surface,
            "pred": pred,
        })

    out.sort(key=lambda m: (m["match"]["time"] == "", m["match"]["time"]))

    print(f"[app] resolved IDs: {resolved_hits}/{len(out)} · "
          f"Elo-weighted form: {form_resolved}/{len(out)} · "
          f"MW odds: {odds_hits}/{len(out)} · Sets odds: {sets_hits}/{len(out)}")

    fixtures_cache[key] = out
    return out


# =================== Top picks ===================
def compute_top_picks(matches, limit=TOP_PICKS_LIMIT,
                      min_odds=TOP_PICKS_MIN_ODDS,
                      min_edge=TOP_PICKS_MIN_EDGE):
    picks = []

    for m in matches:
        bo = m["match"].get("bookmaker_odds")
        if bo:
            p1 = m["pred"]["prob"]
            p2 = 1 - p1
            e1 = p1 - (1.0 / bo["p1_odds"])
            e2 = p2 - (1.0 / bo["p2_odds"])

            if (p1 >= MODEL_SUPPORT_MIN_PROB
                    and e1 >= min_edge
                    and bo["p1_odds"] >= min_odds):
                picks.append({
                    "match": m["match"], "market": "mw",
                    "pick_name": m["p1"]["name"],
                    "opp_name": m["p2"]["name"],
                    "prob": p1, "odds": bo["p1_odds"], "edge": e1,
                    "bookmaker": bo["bookmaker"],
                    "implied": 1.0 / bo["p1_odds"],
                })

            if (p2 >= MODEL_SUPPORT_MIN_PROB
                    and e2 >= min_edge
                    and bo["p2_odds"] >= min_odds):
                picks.append({
                    "match": m["match"], "market": "mw",
                    "pick_name": m["p2"]["name"],
                    "opp_name": m["p1"]["name"],
                    "prob": p2, "odds": bo["p2_odds"], "edge": e2,
                    "bookmaker": bo["bookmaker"],
                    "implied": 1.0 / bo["p2_odds"],
                })

        so = m["match"].get("sets_odds")
        if so:
            p_over = m["pred"]["p_3sets"]
            p_under = 1 - p_over
            e_over = p_over - (1.0 / so["over_odds"])
            e_under = p_under - (1.0 / so["under_odds"])
            label_opp = f"{m['p1']['name']} vs {m['p2']['name']}"

            if (p_over >= MODEL_SUPPORT_MIN_PROB
                    and e_over >= min_edge
                    and so["over_odds"] >= min_odds):
                picks.append({
                    "match": m["match"], "market": "sets",
                    "pick_name": f"Over {so['line']}",
                    "opp_name": label_opp,
                    "prob": p_over, "odds": so["over_odds"], "edge": e_over,
                    "bookmaker": so["bookmaker"],
                    "implied": 1.0 / so["over_odds"],
                })

            if (p_under >= MODEL_SUPPORT_MIN_PROB
                    and e_under >= min_edge
                    and so["under_odds"] >= min_odds):
                picks.append({
                    "match": m["match"], "market": "sets",
                    "pick_name": f"Under {so['line']}",
                    "opp_name": label_opp,
                    "prob": p_under, "odds": so["under_odds"], "edge": e_under,
                    "bookmaker": so["bookmaker"],
                    "implied": 1.0 / so["under_odds"],
                })

    picks.sort(key=lambda x: x["edge"], reverse=True)
    return picks[:limit]


# =================== Auto-track ===================
def _edge_floor_for(market):
    return (AUTO_TRACK_MIN_EDGE_MW if market == "mw"
            else AUTO_TRACK_MIN_EDGE_SETS)


def _is_valuable(prob, edge, market):
    return (prob >= MODEL_SUPPORT_MIN_PROB
            and edge >= _edge_floor_for(market))


def _auto_track_picks(matches):
    added = 0

    for m in matches:
        if "mw" in AUTO_TRACK_MARKETS:
            bo = m["match"].get("bookmaker_odds")
            if bo:
                p1 = m["pred"]["prob"]
                p2 = 1 - p1
                e1 = p1 - (1.0 / bo["p1_odds"])
                e2 = p2 - (1.0 / bo["p2_odds"])

                for side, prob, odds, edge, name, opp in (
                    ("p1", p1, bo["p1_odds"], e1, m["p1"]["name"], m["p2"]["name"]),
                    ("p2", p2, bo["p2_odds"], e2, m["p2"]["name"], m["p1"]["name"]),
                ):
                    if not _is_valuable(prob, edge, "mw"):
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
                        print(f"[auto-track] MW {name} "
                              f"p={prob*100:.0f}% edge=+{edge*100:.1f}% @ {odds}")

        if "sets" in AUTO_TRACK_MARKETS:
            so = m["match"].get("sets_odds")
            if so:
                p_over = m["pred"]["p_3sets"]
                p_under = 1 - p_over
                e_over = p_over - (1.0 / so["over_odds"])
                e_under = p_under - (1.0 / so["under_odds"])
                label_opp = f"{m['p1']['name']} vs {m['p2']['name']}"

                for side, prob, odds, edge, name in (
                    ("over", p_over, so["over_odds"], e_over, f"Over {so['line']}"),
                    ("under", p_under, so["under_odds"], e_under, f"Under {so['line']}"),
                ):
                    if not _is_valuable(prob, edge, "sets"):
                        continue
                    pid = tracker.add_pick(
                        match_id=m["match"]["id"],
                        te_match_id=str(m["match"]["te_match_id"] or ""),
                        market="sets", line=so["line"], pick_side=side,
                        pick_name=name, opponent_name=label_opp,
                        tournament=m["match"]["tournament"],
                        surface=m["match"]["surface"],
                        match_time=m["match"]["time"],
                        model_prob=prob, bookmaker=so["bookmaker"],
                        book_odds=odds, edge=edge,
                    )
                    if pid:
                        added += 1
                        print(f"[auto-track] Sets {name} {label_opp} "
                              f"p={prob*100:.0f}% edge=+{edge*100:.1f}% @ {odds}")

    return added


# =================== Background auto-settle ===================
def _auto_settle_loop():
    time.sleep(30)
    while True:
        try:
            result = settler.settle_open_picks(verbose=True)
            if result["settled"] > 0 or result["checked"] > 0:
                print(f"[auto-settle] {result}")
        except Exception as e:
            print(f"[auto-settle] error: {e}")
        time.sleep(AUTO_SETTLE_INTERVAL)


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
    --muted:#3a4160;
    --top:#f59e0b;
    --sets:#a78bfa;
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
  .toolbar{display:flex;gap:8px;margin:12px 0 8px}
  .toolbar input{flex:1;background:rgba(255,255,255,0.05);
    border:1px solid var(--line);color:var(--fg);
    padding:9px 12px;border-radius:9px;font-size:13px;font-family:inherit}
  .toolbar input:focus{outline:none;border-color:rgba(127,212,255,0.55);
    background:rgba(127,212,255,0.06)}
  .toolbar a{background:rgba(255,255,255,0.05);border:1px solid var(--line);
    color:var(--dim2);padding:9px 14px;border-radius:9px;
    font-size:12px;text-decoration:none;display:flex;align-items:center}
  .toolbar a:hover{border-color:rgba(127,212,255,0.4);color:var(--p1)}
  .toolbar a.primary{background:linear-gradient(90deg,var(--p1),var(--p2));
    color:#0a0d1a;font-weight:800;border-color:transparent}

  .view-toggle{
    display:flex;gap:0;margin-bottom:14px;
    background:rgba(255,255,255,0.03);
    border:1px solid var(--line);
    border-radius:11px;padding:3px;
  }
  .view-toggle a{
    flex:1;text-align:center;
    font-size:12.5px;font-weight:700;
    padding:9px 12px;border-radius:8px;
    text-decoration:none;color:var(--dim2);
    transition:all .15s;
  }
  .view-toggle a:hover{color:var(--fg)}
  .view-toggle a.active{
    background:linear-gradient(90deg,var(--top),#f97316);
    color:#0a0d1a;
    box-shadow:0 2px 8px rgba(245,158,11,0.25);
  }
  .view-toggle a.active.all-view{
    background:linear-gradient(90deg,var(--p1),var(--p2));
    box-shadow:0 2px 8px rgba(127,212,255,0.25);
  }
  .view-toggle .badge{
    display:inline-block;
    background:rgba(255,255,255,0.15);
    color:inherit;
    font-size:10px;font-weight:800;
    padding:1px 6px;border-radius:8px;
    margin-left:6px;
    font-variant-numeric:tabular-nums;
  }
  .view-toggle a.active .badge{
    background:rgba(10,13,26,0.25);
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
  .filter-bar .add{margin-left:auto;color:var(--hit);font-weight:700}

  .top-header{
    display:flex;align-items:baseline;justify-content:space-between;
    margin-bottom:12px;gap:10px;flex-wrap:wrap;
  }
  .top-header h2{
    font-size:16px;letter-spacing:-0.3px;color:var(--fg);font-weight:800;
  }
  .top-header .criteria{
    font-size:11px;color:var(--dim);
  }
  .top-header .criteria b{color:var(--top);font-weight:700}

  .top-card{
    background:linear-gradient(180deg,rgba(245,158,11,0.08),rgba(245,158,11,0.015));
    border:1px solid rgba(245,158,11,0.35);
    border-radius:14px;
    padding:14px 16px;margin-bottom:10px;
    position:relative;
  }
  .top-card.sets{
    background:linear-gradient(180deg,rgba(167,139,250,0.08),rgba(167,139,250,0.015));
    border-color:rgba(167,139,250,0.35);
  }
  .top-card-rank{
    position:absolute;top:-8px;left:14px;
    background:linear-gradient(90deg,var(--top),#f97316);
    color:#0a0d1a;font-weight:800;font-size:10px;
    letter-spacing:0.5px;
    padding:2px 8px;border-radius:5px;
  }
  .top-card.sets .top-card-rank{
    background:linear-gradient(90deg,var(--sets),#7c3aed);
  }
  .top-tourney{
    font-size:10px;letter-spacing:1.2px;text-transform:uppercase;
    color:var(--top);font-weight:800;margin-bottom:6px;
    display:flex;align-items:center;gap:8px;
  }
  .top-card.sets .top-tourney{color:var(--sets)}
  .top-tourney .mkt-badge{
    background:rgba(255,255,255,0.1);
    color:inherit;
    padding:1px 5px;border-radius:4px;
    font-size:9px;letter-spacing:0.5px;font-weight:800;
  }
  .top-pick-row{
    display:flex;justify-content:space-between;align-items:flex-start;
    gap:12px;margin-bottom:10px;
  }
  .top-pick-name{
    font-size:17px;font-weight:800;color:var(--fg);
    letter-spacing:-0.3px;line-height:1.2;
  }
  .top-pick-opp{
    font-size:11.5px;color:var(--dim);margin-top:3px;
  }
  .top-pick-edge{text-align:right;font-variant-numeric:tabular-nums}
  .top-pick-edge .num{
    font-size:22px;font-weight:800;color:var(--hit);
    letter-spacing:-0.5px;line-height:1;
  }
  .top-pick-edge .lbl{
    font-size:9px;letter-spacing:0.6px;text-transform:uppercase;
    color:var(--dim);font-weight:700;margin-top:3px;
  }
  .top-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
  .top-stat{text-align:center}
  .top-stat .k{
    font-size:9px;color:var(--dim);letter-spacing:0.4px;
    text-transform:uppercase;font-weight:700;
  }
  .top-stat .v{
    font-size:13px;color:var(--fg);font-weight:800;
    margin-top:3px;font-variant-numeric:tabular-nums;
  }
  .top-stat .v.prob{color:var(--p1)}
  .top-stat .v.odds{color:var(--bm)}

  .top-empty{
    background:rgba(255,255,255,0.03);
    border:1px dashed var(--line);
    border-radius:14px;
    padding:26px 20px;text-align:center;
    color:var(--dim2);font-size:13px;line-height:1.7;
  }
  .top-empty b{color:var(--fg)}

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
  .mkt-title .bm-inline{
    background:var(--bmbg);color:var(--bm);
    padding:1px 5px;border-radius:4px;font-size:9px;
    letter-spacing:0.3px;font-weight:700;
    border:1px solid var(--bmline);text-transform:none;
  }
  .mrow{display:grid;grid-template-columns:minmax(0,1fr) 50px 60px 52px 26px;
    align-items:center;gap:6px;padding:4px 0}
  .mrow + .mrow{border-top:1px dashed var(--line2)}
  .label{display:flex;align-items:center;gap:6px;min-width:0}
  .swatch{width:3px;height:13px;border-radius:2px;flex-shrink:0}
  .swatch.p1{background:linear-gradient(180deg,var(--p1),var(--p1b))}
  .swatch.p2{background:linear-gradient(180deg,var(--p2),var(--p2b))}
  .swatch.n{background:linear-gradient(180deg,#64748b,#475569)}
  .lbl-text{font-size:12.5px;color:var(--fg);
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .prob{font-size:12px;font-weight:700;text-align:right;
    font-variant-numeric:tabular-nums}
  .prob.muted{color:var(--muted);font-weight:500}
  input.odds{width:100%;background:rgba(255,255,255,0.05);
    border:1px solid var(--line);color:var(--fg);padding:5px 7px;
    border-radius:6px;font-size:12px;font-variant-numeric:tabular-nums;
    text-align:right;font-family:inherit}
  input.odds.default{color:#94a3b8;font-style:italic}
  input.odds.bookmaker{color:var(--bm);font-weight:600}
  input.odds.muted{color:var(--muted)}
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
  .empty{background:rgba(255,255,255,0.03);border:1px solid var(--line);
    border-radius:14px;padding:22px;text-align:center;color:var(--dim2);
    font-size:13px;line-height:1.6}
  a{color:var(--p1)}
</style>
</head>
<body>
<div class="wrap">
  <h1>🎾 Tennis Betting Model</h1>
  <p class="sub">{{ matches|length }} matches · Elo + weighted form · {{ 'ranked only' if ranked_only else 'all' }}</p>

  <div class="toolbar">
    <input id="q" placeholder="Filter by player or tournament…" autocomplete="off">
    <a href="/tracker" class="primary">📋 Tracker</a>
  </div>

  <div class="view-toggle">
    <a href="/?view=top&filter={{ 'all' if not ranked_only else 'ranked' }}"
       class="{% if view == 'top' %}active{% endif %}">
      🏆 Top {{ TOP_PICKS_LIMIT }} Picks
      <span class="badge">{{ top_picks|length }}</span>
    </a>
    <a href="/?view=all&filter={{ 'all' if not ranked_only else 'ranked' }}"
       class="all-view {% if view == 'all' %}active{% endif %}">
      📋 All Matches
      <span class="badge">{{ matches|length }}</span>
    </a>
  </div>

  {% if view == 'all' %}
    <div class="filter-bar">
      <span class="dot">◆</span>
      Show edge when model ≥ <b>{{ '%.0f'|format(MODEL_SUPPORT_MIN_PROB*100) }}%</b>
      · auto-track MW ≥ <b>{{ '%.0f'|format(AUTO_TRACK_MIN_EDGE_MW*100) }}%</b>
      · Sets ≥ <b>{{ '%.0f'|format(AUTO_TRACK_MIN_EDGE_SETS*100) }}%</b>
      <a href="/?view=all&filter={{ 'all' if not ranked_only else 'ranked' }}"
         style="margin-left:auto;font-size:11px;text-decoration:none;color:var(--p1)">
        {% if ranked_only %}All tours{% else %}Ranked only{% endif %}
      </a>
      <a href="/?view=all&refresh=1&filter={{ 'ranked' if ranked_only else 'all' }}"
         style="font-size:11px;text-decoration:none;color:var(--dim)">↻</a>
    </div>
  {% endif %}

  {% if view == 'top' %}
    <div class="top-header">
      <h2>🏆 Today's Top Picks</h2>
      <div class="criteria">
        model ≥ <b>{{ '%.0f'|format(MODEL_SUPPORT_MIN_PROB*100) }}%</b>
        · odds ≥ <b>{{ '%.2f'|format(TOP_PICKS_MIN_ODDS) }}</b>
        · sorted by edge
      </div>
    </div>

    {% if top_picks|length == 0 %}
      <div class="top-empty">
        <b>No qualifying picks today.</b><br>
        Nothing meets: model ≥ {{ '%.0f'|format(MODEL_SUPPORT_MIN_PROB*100) }}% ·
        odds ≥ {{ '%.2f'|format(TOP_PICKS_MIN_ODDS) }}<br><br>
        <a href="/?view=all&filter={{ 'ranked' if ranked_only else 'all' }}"
           style="color:var(--p1)">Browse all matches →</a>
      </div>
    {% else %}
      {% for tp in top_picks %}
        <div class="top-card {% if tp.market == 'sets' %}sets{% endif %}">
          <div class="top-card-rank">#{{ loop.index }}</div>
          <div class="top-tourney">
            <span class="mkt-badge">{{ 'MW' if tp.market == 'mw' else 'SETS' }}</span>
            {{ tp.match.tournament }} · {{ tp.match.surface|title }}
            {% if tp.match.round %} · {{ tp.match.round }}{% endif %}
            {% if tp.match.time %} · {{ tp.match.time }} GMT+3{% endif %}
          </div>
          <div class="top-pick-row">
            <div>
              <div class="top-pick-name">{{ tp.pick_name }}</div>
              <div class="top-pick-opp">{{ tp.opp_name }} · {{ tp.bookmaker }}</div>
            </div>
            <div class="top-pick-edge">
              <div class="num">{{ '%+.1f'|format(tp.edge*100) }}%</div>
              <div class="lbl">Edge</div>
            </div>
          </div>
          <div class="top-stats">
            <div class="top-stat">
              <div class="k">Model</div>
              <div class="v prob">{{ '%.1f'|format(tp.prob*100) }}%</div>
            </div>
            <div class="top-stat">
              <div class="k">Implied</div>
              <div class="v">{{ '%.1f'|format(tp.implied*100) }}%</div>
            </div>
            <div class="top-stat">
              <div class="k">Odds</div>
              <div class="v odds">{{ '%.2f'|format(tp.odds) }}</div>
            </div>
            <div class="top-stat">
              <div class="k">Edge</div>
              <div class="v" style="color:var(--hit)">{{ '%+.1f'|format(tp.edge*100) }}%</div>
            </div>
          </div>
        </div>
      {% endfor %}
    {% endif %}

  {% else %}
    {% if matches|length == 0 %}
      <div class="empty">
        No matches. Try <a href="/?view=all&refresh=1">refreshing</a>.
      </div>
    {% endif %}

    {% for m in matches %}
      {% set s = odds_map.get(m.match.id, {}) %}
      {% set bo = m.match.bookmaker_odds %}
      {% set so_data = m.match.sets_odds %}
      {% set mw1 = s.get('mw1') if s.get('mw1') else (bo.p1_odds if bo else m.pred.fair_mw1) %}
      {% set mw2 = s.get('mw2') if s.get('mw2') else (bo.p2_odds if bo else m.pred.fair_mw2) %}
      {% set so = s.get('so') if s.get('so') else (so_data.over_odds if so_data else m.pred.fair_so) %}
      {% set su = s.get('su') if s.get('su') else (so_data.under_odds if so_data else m.pred.fair_su) %}
      {% set is_default = not (s.get('mw1')) %}
      {% set is_book = bo and not (s.get('mw1')) %}
      {% set is_sets_book = so_data and not (s.get('so')) %}
      {% set p1p = m.pred.prob %}
      {% set p2p = 1 - p1p %}
      {% set p_over_sets = m.pred.p_3sets %}
      {% set p_under_sets = 1 - p_over_sets %}
      {% set e_mw1 = p1p - (1/mw1) %}
      {% set e_mw2 = p2p - (1/mw2) %}
      {% set e_so  = p_over_sets - (1/so) %}
      {% set e_su  = p_under_sets - (1/su) %}

      {% set sup_mw1 = p1p >= MODEL_SUPPORT_MIN_PROB %}
      {% set sup_mw2 = p2p >= MODEL_SUPPORT_MIN_PROB %}
      {% set sup_so  = p_over_sets >= MODEL_SUPPORT_MIN_PROB %}
      {% set sup_su  = p_under_sets >= MODEL_SUPPORT_MIN_PROB %}

      {% set tracked_mw1 = sup_mw1 and e_mw1 >= AUTO_TRACK_MIN_EDGE_MW and ('mw' in AUTO_TRACK_MARKETS) %}
      {% set tracked_mw2 = sup_mw2 and e_mw2 >= AUTO_TRACK_MIN_EDGE_MW and ('mw' in AUTO_TRACK_MARKETS) %}
      {% set tracked_so  = sup_so and e_so >= AUTO_TRACK_MIN_EDGE_SETS and ('sets' in AUTO_TRACK_MARKETS) %}
      {% set tracked_su  = sup_su and e_su >= AUTO_TRACK_MIN_EDGE_SETS and ('sets' in AUTO_TRACK_MARKETS) %}

      {% set card_dual = tracked_mw1 or tracked_mw2 or tracked_so or tracked_su %}

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
            <div class="prob {% if not sup_mw1 %}muted{% endif %}">{{ '%.1f'|format(p1p*100) }}%</div>
            <input class="odds {% if not sup_mw1 %}muted{% elif is_book %}bookmaker{% elif is_default %}default{% endif %}" value="{{ mw1 }}" readonly>
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
            <div class="prob {% if not sup_mw2 %}muted{% endif %}">{{ '%.1f'|format(p2p*100) }}%</div>
            <input class="odds {% if not sup_mw2 %}muted{% elif is_book %}bookmaker{% elif is_default %}default{% endif %}" value="{{ mw2 }}" readonly>
            {% if sup_mw2 %}
              <div class="edge {{ 'hit' if e_mw2>0.03 else ('neg' if e_mw2<-0.03 else 'miss') }}">{{ '%+.1f'|format(e_mw2*100) }}%</div>
            {% else %}
              <div class="edge empty"></div>
            {% endif %}
            {% if tracked_mw2 %}<div class="auto-dot">◆</div>
            {% else %}<div class="auto-dot placeholder"></div>{% endif %}
          </div>
        </div>

        <div class="market">
          <div class="mkt-title">
            <span class="left">Sets · O/U {{ so_data.line if so_data else DEFAULT_SETS_LINE }}
              {% if so_data %}<span class="bm-inline">💰 {{ so_data.bookmaker }}</span>{% else %}<span class="hint" style="color:#4a5270;font-size:9.5px"> · fair only</span>{% endif %}
            </span>
            <span class="hint">{{ '%.1f'|format(p_over_sets*100) }} / {{ '%.1f'|format(p_under_sets*100) }}</span>
          </div>
          <div class="mrow">
            <div class="label"><span class="swatch n"></span><span class="lbl-text">Over {{ so_data.line if so_data else DEFAULT_SETS_LINE }}</span></div>
            <div class="prob {% if not sup_so %}muted{% endif %}">{{ '%.1f'|format(p_over_sets*100) }}%</div>
            <input class="odds {% if not sup_so %}muted{% elif is_sets_book %}bookmaker{% else %}default{% endif %}" value="{{ so }}" readonly>
            {% if sup_so %}
              <div class="edge {{ 'hit' if e_so>0.03 else ('neg' if e_so<-0.03 else 'miss') }}">{{ '%+.1f'|format(e_so*100) }}%</div>
            {% else %}
              <div class="edge empty"></div>
            {% endif %}
            {% if tracked_so %}<div class="auto-dot">◆</div>
            {% else %}<div class="auto-dot placeholder"></div>{% endif %}
          </div>
          <div class="mrow">
            <div class="label"><span class="swatch n"></span><span class="lbl-text">Under {{ so_data.line if so_data else DEFAULT_SETS_LINE }}</span></div>
            <div class="prob {% if not sup_su %}muted{% endif %}">{{ '%.1f'|format(p_under_sets*100) }}%</div>
            <input class="odds {% if not sup_su %}muted{% elif is_sets_book %}bookmaker{% else %}default{% endif %}" value="{{ su }}" readonly>
            {% if sup_su %}
              <div class="edge {{ 'hit' if e_su>0.03 else ('neg' if e_su<-0.03 else 'miss') }}">{{ '%+.1f'|format(e_su*100) }}%</div>
            {% else %}
              <div class="edge empty"></div>
            {% endif %}
            {% if tracked_su %}<div class="auto-dot">◆</div>
            {% else %}<div class="auto-dot placeholder"></div>{% endif %}
          </div>
        </div>

        <div class="card-foot">
          <div class="breakdown">
            <div class="bd">Elo<b>{{ '%.0f'|format(m.pred.p_elo*100) }}%</b></div>
            <div class="bd">Form<b>{{ '%.0f'|format(m.pred.p_form*100) }}%</b></div>
            <div class="bd">Rank<b>{{ '%.0f'|format(m.pred.p_rank*100) }}%</b></div>
          </div>
        </div>
      </div>
    {% endfor %}
  {% endif %}
</div>

<script>
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
  .pick-market{
    display:inline-block;font-size:9px;letter-spacing:0.4px;
    text-transform:uppercase;font-weight:700;
    padding:1px 5px;border-radius:4px;margin-right:6px;
    background:rgba(255,255,255,0.06);color:var(--dim2)
  }
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
      {{ settle_summary.skipped }} pending ·
      {{ settle_summary.errors }} errors
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

  {% for p in picks %}
    <div class="pick {{ p.status }}">
      <div class="pick-head">
        <div class="pick-name">
          <span class="pick-market">{{ (p.market or 'mw')|upper }}</span>
          {{ p.pick_name }}
        </div>
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
</body>
</html>
"""


# =================== Routes ===================
def build_matches_with_markets(ranked_only=True, force=False):
    matches = load_matches(ranked_only=ranked_only, force=force)
    for m in matches:
        p = m["pred"]
        p["fair_mw1"] = fair_odds(p["prob"])
        p["fair_mw2"] = fair_odds(1 - p["prob"])
        p["fair_so"] = fair_odds(p["p_3sets"])
        p["fair_su"] = fair_odds(1 - p["p_3sets"])
    return matches


@app.route("/")
def index():
    ranked_only = request.args.get("filter", "ranked") != "all"
    force = request.args.get("refresh") == "1"
    view = request.args.get("view", "top")
    if view not in ("top", "all"):
        view = "top"

    try:
        matches = build_matches_with_markets(ranked_only=ranked_only, force=force)
    except Exception as e:
        return f"<pre style='color:#f87171;padding:24px'>Error: {e}</pre>", 500

    auto_added = 0
    try:
        auto_added = _auto_track_picks(matches)
    except Exception as e:
        print(f"[auto-track] error: {e}")

    top_picks = compute_top_picks(matches)

    return render_template_string(
        PAGE,
        matches=matches,
        odds_map=load_all_odds(),
        ranked_only=ranked_only,
        view=view,
        top_picks=top_picks,
        DEFAULT_SETS_LINE=DEFAULT_SETS_LINE,
        MODEL_SUPPORT_MIN_PROB=MODEL_SUPPORT_MIN_PROB,
        AUTO_TRACK_MIN_EDGE_MW=AUTO_TRACK_MIN_EDGE_MW,
        AUTO_TRACK_MIN_EDGE_SETS=AUTO_TRACK_MIN_EDGE_SETS,
        AUTO_TRACK_MARKETS=AUTO_TRACK_MARKETS,
        TOP_PICKS_MIN_ODDS=TOP_PICKS_MIN_ODDS,
        TOP_PICKS_LIMIT=TOP_PICKS_LIMIT,
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
        out["tracker_stats"] = tracker.stats()
        out["filters"] = {
            "model_support_min_prob": MODEL_SUPPORT_MIN_PROB,
            "mw_edge_floor": AUTO_TRACK_MIN_EDGE_MW,
            "sets_edge_floor": AUTO_TRACK_MIN_EDGE_SETS,
            "top_picks_min_odds": TOP_PICKS_MIN_ODDS,
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
    print(f"[app] markets: {AUTO_TRACK_MARKETS} · "
          f"MW edge ≥ {AUTO_TRACK_MIN_EDGE_MW*100:.0f}% · "
          f"Sets edge ≥ {AUTO_TRACK_MIN_EDGE_SETS*100:.0f}% · "
          f"model ≥ {MODEL_SUPPORT_MIN_PROB*100:.0f}%")
    print(f"[app] top picks: odds ≥ {TOP_PICKS_MIN_ODDS}, "
          f"limit {TOP_PICKS_LIMIT}, sorted by edge desc")
    print(f"[app] timezone: Berlin {datetime.now(BERLIN_TZ).strftime('%H:%M %Z')} "
          f"→ Nairobi {datetime.now(NAIROBI_TZ).strftime('%H:%M %Z')}")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
