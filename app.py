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
import model
import scoreline
import surfaces
import recent
import odds_scraper
import tracker
import settler
import results
import predictions

app = Flask(__name__)

SURFACES = {"hard", "clay", "grass"}
fixtures_cache = {}
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "odds.db")

ELO_RATINGS = {}

ODDS_WORKERS = 10
ODDS_FETCH_MIN_PROB = 0.55
ODDS_FETCH_DEADLINE = 45

_ODDS_FETCH_LOCK = threading.Lock()

MODEL_SUPPORT_MIN_PROB = 0.60
AUTO_TRACK_MARKETS = ("mw",)
AUTO_TRACK_MIN_EDGE_MW = 0.03
AUTO_TRACK_MAX_EDGE_MW = 0.12   # bigger gaps vs. the market are far more likely
                                 # a data/name problem than a real edge

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
    return naive.astimezone(NAIROBI_TZ).strftime("%H:%M")


# =================== Model ===================
def fair_odds(p):
    p = max(0.02, min(0.98, p))
    return round(1 / p, 2)


def _surface_key(surface):
    """'hard'/'clay'/'grass' -> Elo surface; anything else -> None (overall)."""
    return surface.title() if surface in SURFACES else None


def elo_weighted_form(player_id, recent_matches, surface, player_index, tour=None):
    """Form = recent results vs. what Elo expected. DISPLAY/LOGGING ONLY:
    its fitted weight in the win-probability model is ~0 (Elo already
    reflects recent results), so it no longer moves the prediction."""
    if not recent_matches or player_id is None:
        return 0.5, 0
    if player_id not in ELO_RATINGS:
        return 0.5, 0
    sk = _surface_key(surface)
    my_elo = elo.get_player_elo(player_id, sk, ELO_RATINGS)
    weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
    tw = ws = 0.0
    used = 0
    for i, match in enumerate(recent_matches[:10]):
        opp_name = match.get("opponent", "")
        opp_id = players.resolve(opp_name, player_index, tour=tour,
                                 ratings=ELO_RATINGS)
        if not opp_id or opp_id not in ELO_RATINGS:
            continue
        opp_elo = elo.get_player_elo(opp_id, sk, ELO_RATINGS)
        expected = model.elo_prob(my_elo, opp_elo)
        actual = 1.0 if match.get("result") == "W" else 0.0
        w = weights[i] if i < len(weights) else 0.15
        ws += (actual - expected) * w
        tw += w
        used += 1
    if tw == 0:
        return 0.5, 0
    form = 0.5 + (ws / tw) * 1.5
    return max(0.1, min(0.9, form)), used


def recency_weighted_wl(matches):
    if not matches:
        return 0.5, 0
    weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
    tw = wins = 0.0
    n = 0
    for i, m in enumerate(matches[:10]):
        w = weights[i] if i < len(weights) else 0.15
        tw += w
        if m.get("result") == "W":
            wins += w
        n += 1
    return (wins / tw, n) if tw else (0.5, 0)


def compute_form(recent_matches, player_id, surface, player_index, tour=None):
    if not recent_matches:
        return 0.5, 0
    if player_id and player_id in ELO_RATINGS:
        ef, en = elo_weighted_form(player_id, recent_matches, surface,
                                    player_index, tour)
        if en >= 3:
            return ef, en
    return recency_weighted_wl(recent_matches)


def predict(p1, p2, surface, form1=None, form2=None,
            form1_n=0, form2_n=0):
    """Win probability from the fitted model (see model.py / evaluate.py).

    No manual shrink: the logistic fit is calibrated on held-out data, so the
    old fixed 0.7 shrink (a patch from a 52-match sample) is gone.
    """
    sk = _surface_key(surface)
    pid1, pid2 = p1.get("player_id"), p2.get("player_id")
    e1 = e2 = None
    n1 = n2 = 0
    if ELO_RATINGS and pid1 and pid2 and pid1 in ELO_RATINGS and pid2 in ELO_RATINGS:
        e1 = elo.get_player_elo(pid1, sk, ELO_RATINGS)
        e2 = elo.get_player_elo(pid2, sk, ELO_RATINGS)
        n1, n2 = ELO_RATINGS[pid1]["n"], ELO_RATINGS[pid2]["n"]

    prob, regime = model.win_prob(e1, e2, p1.get("ranking"), p2.get("ranking"),
                                  n1, n2)
    prob = max(0.02, min(0.98, prob))

    f1 = form1 if form1 is not None else 0.5
    f2 = form2 if form2 is not None else 0.5
    if form1_n == 0 and form2_n == 0:
        conf = 0.0
    elif form1_n > 0 and form2_n > 0:
        conf = min(min(form1_n, form2_n) / 5.0, 1.0)
    else:
        conf = min(max(form1_n, form2_n) / 5.0, 1.0) * 0.7

    return {
        "prob": prob, "raw_prob": prob, "regime": regime,
        "p_elo": model.elo_prob(e1, e2) if e1 is not None else 0.5,
        "p_form": max(0.15, min(0.85, 0.5 + (f1 - f2) * conf * 0.15)),
        "p_rank": model.rank_prob(p1.get("ranking"), p2.get("ranking")),
        "e1": e1, "e2": e2, "f1": f1, "f2": f2,
        "f1_n": form1_n, "f2_n": form2_n, "form_conf": conf,
    }


# =================== Load matches ===================
def _init_elo():
    """Archive Elo + results scraped since the archive ends (recent.py)."""
    global ELO_RATINGS
    print("[app] loading historical dataset for Elo...")
    ml = dataset.load_all_matches()
    if not ml:
        return
    surfaces.build(ml)
    base = elo.compute_ratings(ml)
    try:
        index = players.build_index()
        extra = recent.load_extra_rows(ml, index, base)
    except Exception as e:
        print(f"[app] recent results unavailable: {e}")
        extra = []
    ELO_RATINGS = elo.compute_ratings(ml, extra) if extra else base


def _recent_sync_loop():
    """Pull newly finished matches into recent_results.db and refresh Elo."""
    global ELO_RATINGS
    time.sleep(30)
    while True:
        try:
            ml = dataset.load_all_matches()
            if ml:
                since = (ml[-1]["_date"] - timedelta(days=21)).date()
                added = recent.sync(since, verbose=False)
                if added:
                    print(f"[recent] {added} new finished matches stored")
                    ELO_RATINGS = {}          # rebuilt on next request
                    fixtures_cache.clear()
        except Exception as e:
            print(f"[recent] sync error: {e}")
        time.sleep(6 * 3600)


def load_matches(ranked_only=True, force=False, fetch_odds=True,
                 target_date=None, sort_by="prob"):
    global ELO_RATINGS
    if target_date is None:
        target_date = datetime.now(NAIROBI_TZ).date()
    date_str = target_date.isoformat()
    key = f"{date_str}_{'ranked' if ranked_only else 'all'}"
    if key in fixtures_cache and not force:
        return fixtures_cache[key]

    if not ELO_RATINGS:
        _init_elo()

    player_index = players.build_index()
    form_cache = scrapers.load_form()
    raw = scrapers.get_todays_matches(ranked_only=ranked_only,
                                     target_date=target_date)

    staged = []
    resolved_hits = form_resolved = 0
    for m in raw:
        surface = (m.get("surface") or "").lower()
        if surface not in SURFACES:
            surface = "unknown"        # Elo then uses the surface-neutral rating
        tour = m.get("tour")
        pid1 = players.resolve(m["p1_name"], player_index, tour=tour,
                               ratings=ELO_RATINGS)
        pid2 = players.resolve(m["p2_name"], player_index, tour=tour,
                               ratings=ELO_RATINGS)
        if pid1 and pid2:
            resolved_hits += 1
        p1 = {"name": m["p1_name"], "ranking": m.get("p1_rank"),
              "player_id": pid1}
        p2 = {"name": m["p2_name"], "ranking": m.get("p2_rank"),
              "player_id": pid2}
        fm1 = form_cache.get(m["p1_name"], {}).get("matches", [])
        fm2 = form_cache.get(m["p2_name"], {}).get("matches", [])
        f1, f1n = compute_form(fm1, pid1, surface, player_index, tour)
        f2, f2n = compute_form(fm2, pid2, surface, player_index, tour)
        if f1n >= 3 and f2n >= 3:
            form_resolved += 1
        pred = predict(p1, p2, surface, form1=f1, form2=f2,
                       form1_n=f1n, form2_n=f2n)
        pred["score"] = scoreline.predict(
            pred["prob"],
            scoreline.best_of_for(m.get("tournament"), m.get("tour")))
        display_time = convert_te_time_to_nairobi(
            m.get("status") or "", match_date=target_date)
        staged.append({
            "match": {"id": m["id"], "te_match_id": m.get("te_match_id"),
                      "tournament": m.get("tournament") or "Tennis",
                      "round": m.get("round", ""), "surface": surface,
                      "source": m.get("source", ""), "time": display_time,
                      "bookmaker_odds": None},
            "p1": p1, "p2": p2,
            "p1_recent": m.get("p1_recent", []),
            "p2_recent": m.get("p2_recent", []),
            "surface": surface, "pred": pred,
        })

    if fetch_odds:
        interesting = [s for s in staged if s["match"]["te_match_id"]
                       and max(s["pred"]["prob"], 1 - s["pred"]["prob"])
                       >= ODDS_FETCH_MIN_PROB]
        if interesting:
            acquired = _ODDS_FETCH_LOCK.acquire(blocking=False)
            if acquired:
                try:
                    _fetch_odds_for(interesting)
                finally:
                    _ODDS_FETCH_LOCK.release()

    out = staged
    if sort_by == "time":
        out.sort(key=lambda m: (m["match"]["time"] == "", m["match"]["time"]))
    else:
        out.sort(key=lambda m: max(m["pred"]["prob"], 1 - m["pred"]["prob"]),
                 reverse=True)
    fixtures_cache[key] = out
    return out


def _fetch_odds_for(staged):
    t0 = time.time()
    odds_map = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=ODDS_WORKERS) as pool:
        for s in staged:
            tid = s["match"]["te_match_id"]
            if tid:
                futures[pool.submit(odds_scraper.fetch_match_odds_cached,
                                    tid)] = s
        try:
            for f in as_completed(futures, timeout=ODDS_FETCH_DEADLINE):
                s = futures[f]
                try:
                    mw, _ = f.result(timeout=1)
                except Exception:
                    mw = None
                odds_map[s["match"]["te_match_id"]] = mw
        except TimeoutError:
            print(f"[app] odds deadline hit, got {len(odds_map)}/{len(futures)}")
            for f in futures:
                f.cancel()
    for s in staged:
        tid = s["match"]["te_match_id"]
        if tid in odds_map:
            s["match"]["bookmaker_odds"] = odds_map[tid]
    hits = sum(1 for s in staged if s["match"]["bookmaker_odds"])
    print(f"[app] odds done in {time.time() - t0:.1f}s · {hits}/{len(staged)}")


def _log_predictions(matches, event_date):
    added = 0
    for m in matches:
        p1p = m["pred"]["prob"]
        p2p = 1 - p1p
        if max(p1p, p2p) < predictions.MIN_SIGNAL:
            continue
        
        pick_side, pick_name, pick_prob = ("p1", m["p1"]["name"], p1p) if p1p >= p2p else ("p2", m["p2"]["name"], p2p)
        bo = m["match"].get("bookmaker_odds")
        
        rid = predictions.add_prediction(
            match_id=m["match"]["id"], te_match_id=str(m["match"]["te_match_id"] or ""),
            event_date=event_date, tournament=m["match"]["tournament"], round=m["match"]["round"],
            surface=m["match"]["surface"], match_time=m["match"]["time"],
            p1_name=m["p1"]["name"], p2_name=m["p2"]["name"], p1_ranking=m["p1"].get("ranking"), p2_ranking=m["p2"].get("ranking"),
            model_prob_p1=p1p, model_prob_p2=p2p, model_pick=pick_side, model_pick_name=pick_name, model_pick_prob=pick_prob,
            book_odds_p1=(bo["p1_odds"] if bo else None), book_odds_p2=(bo["p2_odds"] if bo else None),
            book_pick_name=(m["p1"]["name"] if bo and bo["p1_odds"] <= bo["p2_odds"] else (m["p2"]["name"] if bo else None)),
            book_pick_odds=(bo["p1_odds"] if bo and bo["p1_odds"] <= bo["p2_odds"] else (bo["p2_odds"] if bo else None)),
            elo_p1=m["pred"]["e1"], elo_p2=m["pred"]["e2"], form_p1=m["pred"]["f1"], form_p2=m["pred"]["f2"], rank_p1=m["pred"]["p_rank"], rank_p2=1 - m["pred"]["p_rank"]
        )
        if rid:
            added += 1
    return added


def _auto_track_picks(matches):
    added = 0
    for m in matches:
        bo = m["match"].get("bookmaker_odds")
        if not bo:
            continue
        p1, p2 = m["pred"]["prob"], 1 - m["pred"]["prob"]
        # Remove the bookmaker margin before comparing with the model
        q1, q2 = 1.0 / bo["p1_odds"], 1.0 / bo["p2_odds"]
        e1 = p1 - q1 / (q1 + q2)
        e2 = p2 - q2 / (q1 + q2)
        for side, prob, odds, edge, name, opp in (
            ("p1", p1, bo["p1_odds"], e1, m["p1"]["name"], m["p2"]["name"]),
            ("p2", p2, bo["p2_odds"], e2, m["p2"]["name"], m["p1"]["name"]),
        ):
            if not (prob >= MODEL_SUPPORT_MIN_PROB
                    and AUTO_TRACK_MIN_EDGE_MW <= edge <= AUTO_TRACK_MAX_EDGE_MW
                    and prob * odds > 1.0):        # must be +EV after the margin
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
    return added


def _auto_settle_loop():
    time.sleep(AUTO_SETTLE_START_DELAY)
    while True:
        try:
            r1 = settler.settle_open_picks(verbose=False)
            r2 = predictions.settle_open_predictions(verbose=False)
            if r1["settled"] > 0 or r1["voided"] > 0 or \
               r2["settled"] > 0 or r2["voided"] > 0:
                print(f"[auto-settle] picks: {r1['settled']} settled, "
                      f"{r1['voided']} voided · "
                      f"preds: {r2['settled']} settled, "
                      f"{r2['voided']} voided")
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
  :root{--bg0:#0a0d1a;--bg2:#1a1f36;--fg:#e8eaf2;--dim:#8892b0;--dim2:#b8c0dc;
    --line:rgba(255,255,255,0.08);--line2:rgba(255,255,255,0.04);
    --p1:#7fd4ff;--p1b:#4fa3ff;--p2:#a78bfa;--p2b:#8b5cf6;
    --hit:#6ee7b7;--hitbg:rgba(52,211,153,0.13);--miss:#94a3b8;
    --missbg:rgba(148,163,184,0.07);--neg:#f87171;--negbg:rgba(248,113,113,0.11);
    --w:#6ee7b7;--l:#f87171;--bm:#fbbf24;--bmbg:rgba(251,191,36,0.14);
    --bmline:rgba(251,191,36,0.25);--dual:#a78bfa;--muted:#7a85a8;}
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
  .toolbar input[type=text]{flex:1;min-width:180px;
    background:rgba(255,255,255,0.05);border:1px solid var(--line);
    color:var(--fg);padding:9px 12px;border-radius:9px;
    font-size:13px;font-family:inherit}
  .toolbar input[type=date]{background:rgba(255,255,255,0.05);
    border:1px solid var(--line);color:var(--fg);padding:9px 12px;
    border-radius:9px;font-size:13px;font-family:inherit;color-scheme:dark}
  .toolbar a{background:rgba(255,255,255,0.05);border:1px solid var(--line);
    color:var(--dim2);padding:9px 14px;border-radius:9px;
    font-size:12px;text-decoration:none;display:flex;align-items:center}
  .toolbar a:hover{border-color:rgba(127,212,255,0.4);color:var(--p1)}
  .toolbar a.primary{background:linear-gradient(90deg,var(--p1),var(--p2));
    color:#0a0d1a;font-weight:800;border-color:transparent}
  .toolbar a.results-link,.toolbar a.pred-link{
    background:rgba(110,231,183,0.12);border-color:rgba(110,231,183,0.35);
    color:var(--hit);font-weight:700}
  .toolbar .today-btn{background:rgba(110,231,183,0.12);
    border:1px solid rgba(110,231,183,0.35);color:var(--hit);
    padding:9px 12px;border-radius:9px;font-size:12px;font-weight:700;
    text-decoration:none}
  .date-info{font-size:11.5px;color:var(--dim);margin-bottom:12px;
    display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  .date-info .badge{background:rgba(255,255,255,0.06);padding:2px 8px;
    border-radius:6px;color:var(--dim2);font-weight:600;
    font-variant-numeric:tabular-nums}
  .date-info .badge.today{background:rgba(110,231,183,0.13);color:var(--hit)}
  .sort-toggle{display:flex;margin-bottom:12px;background:rgba(255,255,255,0.03);
    border:1px solid var(--line);border-radius:10px;padding:3px;max-width:320px}
  .sort-toggle a{flex:1;text-align:center;font-size:11.5px;font-weight:700;
    padding:7px 10px;border-radius:7px;text-decoration:none;color:var(--dim2)}
  .sort-toggle a.active{background:linear-gradient(90deg,var(--p1),var(--p2));
    color:#0a0d1a}
  .filter-bar{background:rgba(167,139,250,0.08);
    border:1px solid rgba(167,139,250,0.25);border-radius:10px;
    padding:10px 14px;margin-bottom:16px;font-size:11.5px;color:var(--dim2);
    display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .filter-bar b{color:var(--fg);font-weight:700}
  .filter-bar .dot{color:var(--dual);font-size:14px}
  .cards-grid{display:block}
  .card{background:linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.012));
    border:1px solid var(--line);border-radius:16px;margin-bottom:12px;overflow:hidden}
  .card.dual{border-color:rgba(167,139,250,0.35)}
  .card-head{padding:14px 16px 12px;border-bottom:1px solid var(--line2)}
  .tourney{font-size:10.5px;letter-spacing:1.3px;text-transform:uppercase;
    color:var(--p1);font-weight:700;margin-bottom:8px;display:flex;
    align-items:center;gap:7px;flex-wrap:wrap}
  .tourney .dot{width:4px;height:4px;border-radius:50%;background:var(--p1)}
  .tourney .time-badge{background:rgba(127,212,255,0.1);color:var(--p1);
    padding:2px 6px;border-radius:5px;font-size:9.5px;font-weight:700;
    font-variant-numeric:tabular-nums;border:1px solid rgba(127,212,255,0.2)}
  .tourney .bm-badge{background:var(--bmbg);color:var(--bm);padding:2px 6px;
    border-radius:5px;font-size:9.5px;font-weight:700;
    border:1px solid var(--bmline)}
  .tourney .dual-badge{background:rgba(167,139,250,0.15);color:var(--dual);
    padding:2px 6px;border-radius:5px;font-size:9.5px;font-weight:800;
    border:1px solid rgba(167,139,250,0.4)}
  .tourney .src{margin-left:auto;color:#4a5270;font-size:9.5px;
    font-weight:600;text-transform:none}
  .matchup{display:grid;grid-template-columns:1fr auto 1fr;gap:8px;
    align-items:center}
  .side{min-width:0}.side.right{text-align:right}
  .pname{font-size:15px;font-weight:600;white-space:nowrap;overflow:hidden;
    text-overflow:ellipsis}
  .pmeta{font-size:10.5px;color:var(--dim);margin-top:3px;
    font-variant-numeric:tabular-nums}
  .pmeta b{color:var(--dim2);font-weight:600}
  .form-strip{display:flex;gap:3px;margin-top:5px}
  .form-strip.right{justify-content:flex-end}
  .form-dot{width:16px;height:16px;border-radius:4px;font-size:9px;
    display:flex;align-items:center;justify-content:center;font-weight:700;
    font-family:monospace}
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
  .mkt-title .hint{color:#4a5270;text-transform:none;font-weight:500;
    font-size:10.5px}
  .mrow{display:grid;grid-template-columns:minmax(0,1fr) 50px 60px 52px 26px;
    align-items:center;gap:6px;padding:4px 0}
  .mrow + .mrow{border-top:1px dashed var(--line2)}
  .label{display:flex;align-items:center;gap:6px;min-width:0}
  .swatch{width:3px;height:13px;border-radius:2px;flex-shrink:0}
  .swatch.p1{background:linear-gradient(180deg,var(--p1),var(--p1b))}
  .swatch.p2{background:linear-gradient(180deg,var(--p2),var(--p2b))}
  .lbl-text{font-size:12.5px;color:var(--fg);white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis}
  .prob{font-size:12.5px;font-weight:700;text-align:right;
    font-variant-numeric:tabular-nums}
  .prob.soft{color:var(--dim2);font-weight:600}
  input.odds{width:100%;background:rgba(255,255,255,0.05);
    border:1px solid var(--line);color:var(--fg);padding:5px 7px;
    border-radius:6px;font-size:12px;font-variant-numeric:tabular-nums;
    text-align:right;font-family:inherit}
  input.odds.default{color:#94a3b8;font-style:italic}
  input.odds.bookmaker{color:var(--bm);font-weight:600}
  input.odds.soft{color:var(--dim2)}
  .edge{font-size:11px;font-weight:700;text-align:right;padding:3px 6px;
    border-radius:5px;font-variant-numeric:tabular-nums;
    border:1px solid transparent}
  .edge.hit{color:var(--hit);background:var(--hitbg);
    border-color:rgba(52,211,153,0.3)}
  .edge.miss{color:var(--miss);background:var(--missbg)}
  .edge.neg{color:var(--neg);background:var(--negbg)}
  .edge.empty{color:transparent;background:transparent;border-color:transparent}
  .edge.empty::after{content:"—";color:var(--muted);font-weight:500}
  .auto-dot{width:24px;height:24px;border-radius:5px;display:flex;
    align-items:center;justify-content:center;font-size:11px;color:var(--dual);
    background:rgba(167,139,250,0.15);
    border:1px solid rgba(167,139,250,0.45)}
  .auto-dot.placeholder{background:transparent;border-color:transparent}
  .sl-top{font-size:14px;color:var(--fg);margin-top:2px}
  .sl-top b{color:var(--p1)}
  .sl-games{color:var(--dim2);font-size:12.5px}
  .sl-pct{float:right;color:var(--hit);font-weight:700}
  .sl-alt{display:flex;flex-wrap:wrap;gap:6px 10px;margin-top:6px;font-size:11.5px;color:var(--dim)}
  .sl-alt .sl-sum{margin-left:auto;color:var(--muted)}
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
    .toolbar, .filter-bar, .date-info, .sort-toggle { max-width: 900px; }
    .cards-grid { display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px; align-items: start; }
    .cards-grid .card { margin-bottom: 0; }
  }
  @media (min-width: 1400px) {
    .wrap { max-width: 1440px; }
    .cards-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>🎾 Tennis Betting Model</h1>
  <p class="sub">{{ matches|length }} matches · {{ selected_date_str }} · Fitted Elo + rank model · {{ 'ranked only' if ranked_only else 'all' }}</p>

  <div class="toolbar">
    <input id="q" type="text" placeholder="Filter…" autocomplete="off">
    <input type="date" value="{{ selected_date_str }}" max="{{ max_date_str }}"
           onchange="changeDate(this.value)">
    {% if not is_today %}
      <a href="/?filter={{ 'ranked' if ranked_only else 'all' }}&sort={{ sort_by }}"
         class="today-btn">↺ Today</a>
    {% endif %}
    <a href="/odds" class="primary">💰 Odds</a>
    <a href="/tracker" class="primary">📋 Tracker</a>
    <a href="/predictions" class="pred-link">📈 Predictions</a>
    <a href="/results" class="results-link">📊 Results</a>
  </div>

  <div class="date-info">
    <span class="badge {% if is_today %}today{% endif %}">
      {% if is_today %}● Today{% else %}◷ {{ selected_date_str }}{% endif %}
    </span>
    <span>{{ weekday_name }}</span>
  </div>

  <div class="sort-toggle">
    <a href="/?date={{ selected_date_str }}&sort=prob&filter={{ 'ranked' if ranked_only else 'all' }}"
       class="{% if sort_by == 'prob' %}active{% endif %}">🏆 By confidence</a>
    <a href="/?date={{ selected_date_str }}&sort=time&filter={{ 'ranked' if ranked_only else 'all' }}"
       class="{% if sort_by == 'time' %}active{% endif %}">🕐 By time</a>
  </div>

  <div class="filter-bar">
    <span class="dot">◆</span>
    Show edge when model ≥ <b>{{ '%.0f'|format(MODEL_SUPPORT_MIN_PROB*100) }}%</b>
    · auto-track if edge ≥ <b>{{ '%.0f'|format(AUTO_TRACK_MIN_EDGE_MW*100) }}%</b>
    {% if logged_count %}
      · <b style="color:var(--hit)">+{{ logged_count }} predictions logged</b>
    {% endif %}
    <a href="/?date={{ selected_date_str }}&sort={{ sort_by }}&filter={{ 'all' if not ranked_only else 'ranked' }}"
       style="margin-left:auto;font-size:11px;text-decoration:none;color:var(--p1)">
      {% if ranked_only %}All tours{% else %}Ranked only{% endif %}
    </a>
    <a href="/?date={{ selected_date_str }}&sort={{ sort_by }}&refresh=1&filter={{ 'ranked' if ranked_only else 'all' }}"
       style="font-size:11px;text-decoration:none;color:var(--dim)">↻</a>
  </div>

  {% if matches|length == 0 %}
    <div class="empty">
      No matches on {{ selected_date_str }}.
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
            <span class="dot"></span>{{ m.match.tournament }} · {{ m.match.surface|title }}
            {% if m.match.round %} · {{ m.match.round }}{% endif %}
            {% if m.match.time %}<span class="time-badge">🕐 {{ m.match.time }} GMT+3</span>{% endif %}
            {% if bo %}<span class="bm-badge">💰 {{ bo.bookmaker }}</span>{% endif %}
            {% if card_dual %}<span class="dual-badge">◆ auto</span>{% endif %}
            <span class="src">{{ m.match.source }}</span>
          </div>
          <div class="matchup">
            <div class="side">
              <div class="pname">{{ m.p1.name }}</div>
              <div class="pmeta">Rank <b>#{{ m.p1.ranking or 'NR' }}</b> · Elo <b>{{ '%.0f'|format(m.pred.e1) if m.pred.e1 is not none else '—' }}</b></div>
              {% if m.p1_recent %}
                <div class="form-strip">
                  {% for r in m.p1_recent %}<div class="form-dot {{ 'w' if r.result == 'W' else 'l' }}">{{ r.result }}</div>{% endfor %}
                </div>
              {% endif %}
            </div>
            <div class="vs-div">VS</div>
            <div class="side right">
              <div class="pname">{{ m.p2.name }}</div>
              <div class="pmeta">Rank <b>#{{ m.p2.ranking or 'NR' }}</b> · Elo <b>{{ '%.0f'|format(m.pred.e2) if m.pred.e2 is not none else '—' }}</b></div>
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
            {% if tracked_mw1 %}<div class="auto-dot">◆</div>{% else %}<div class="auto-dot placeholder"></div>{% endif %}
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
            {% if tracked_mw2 %}<div class="auto-dot">◆</div>{% else %}<div class="auto-dot placeholder"></div>{% endif %}
          </div>
        </div>

        {% set sc = m.pred.score %}
        {% set top = sc.top %}
        <div class="market sl">
          <div class="mkt-title"><span>Likely score</span>
            <span class="hint">best of {{ sc.best_of }}</span></div>
          <div class="sl-top">
            <b>{{ (m.p1.name if top.side == 'p1' else m.p2.name) }}</b>
            {{ top.score }} <span class="sl-games">({{ top.games }})</span>
            <span class="sl-pct">{{ '%.0f'|format(top.prob*100) }}%</span>
          </div>
          <div class="sl-alt">
            {% for o in sc.outcomes[1:3] %}
              <span>{{ (m.p1.name if o.side == 'p1' else m.p2.name).split(' ')[0] }} {{ o.score }} {{ '%.0f'|format(o.prob*100) }}%</span>
            {% endfor %}
            <span class="sl-sum">Straight sets {{ '%.0f'|format(sc.p_straight*100) }}% · Distance {{ '%.0f'|format(sc.p_decider*100) }}%</span>
          </div>
        </div>

        <div class="card-foot">
          <div class="breakdown">
            <div class="bd">Elo<b>{{ '%.0f'|format(m.pred.p_elo*100) }}%</b></div>
            <div class="bd">Form<b>{{ '%.0f'|format(m.pred.p_form*100) }}%</b>
              <span class="n">n={{ m.pred.f1_n }}/{{ m.pred.f2_n }}</span></div>
            <div class="bd">Rank<b>{{ '%.0f'|format(m.pred.p_rank*100) }}%</b></div>
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
  if (q) q.addEventListener('input', e => {
    const term = e.target.value.toLowerCase();
    document.querySelectorAll('.card').forEach(c => {
      c.style.display = c.textContent.toLowerCase().includes(term) ? '' : 'none';
    });
  });
</script>
</body>
</html>
"""


# =================== Predictions page template ===================
PREDICTIONS_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictions</title>
<style>
  :root{--bg0:#0a0d1a;--bg2:#1a1f36;--fg:#e8eaf2;--dim:#8892b0;--dim2:#b8c0dc;
    --line:rgba(255,255,255,0.08);--line2:rgba(255,255,255,0.04);
    --p1:#7fd4ff;--p2:#a78bfa;--hit:#6ee7b7;--hitbg:rgba(52,211,153,0.13);
    --miss:#94a3b8;--missbg:rgba(148,163,184,0.07);--neg:#f87171;
    --negbg:rgba(248,113,113,0.11);--warn:#fbbf24;}
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
  .nav{display:flex;gap:8px;margin:12px 0 16px;flex-wrap:wrap}
  .nav a{background:rgba(255,255,255,0.05);border:1px solid var(--line);
    color:var(--dim2);padding:9px 14px;border-radius:9px;font-size:12px;
    text-decoration:none}
  .nav a:hover{border-color:rgba(127,212,255,0.4);color:var(--p1)}
  .nav a.active{background:linear-gradient(90deg,var(--p1),var(--p2));
    color:#0a0d1a;font-weight:800;border-color:transparent}

  .range-toggle{display:flex;gap:0;margin-bottom:16px;
    background:rgba(255,255,255,0.03);border:1px solid var(--line);
    border-radius:10px;padding:3px;max-width:400px}
  .range-toggle a{flex:1;text-align:center;font-size:12px;font-weight:700;
    padding:8px 10px;border-radius:7px;text-decoration:none;color:var(--dim2)}
  .range-toggle a:hover{color:var(--fg)}
  .range-toggle a.active{background:linear-gradient(90deg,var(--p1),var(--p2));
    color:#0a0d1a}

  .hero{
    background:linear-gradient(180deg,rgba(127,212,255,0.06),rgba(167,139,250,0.03));
    border:1px solid rgba(127,212,255,0.25);
    border-radius:18px;padding:18px 20px;margin-bottom:16px;
  }
  .hero-title{font-size:11px;letter-spacing:1.4px;text-transform:uppercase;
    color:var(--p1);font-weight:800;margin-bottom:14px}
  .hero-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;
    margin-bottom:14px}
  .hero-stat{text-align:center}
  .hero-stat .k{font-size:9.5px;color:var(--dim);letter-spacing:0.4px;
    text-transform:uppercase;font-weight:700}
  .hero-stat .v{font-size:24px;font-weight:800;margin-top:4px;
    font-variant-numeric:tabular-nums;letter-spacing:-0.5px;
    color:var(--fg)}
  .hero-stat .v.hit{color:var(--hit)}
  .hero-stat .v.neg{color:var(--neg)}
  .hero-stat .v.warn{color:var(--warn)}
  .hero-stat .s{font-size:10px;color:var(--dim);margin-top:2px}
  .hero-sub{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;
    padding-top:12px;border-top:1px solid var(--line2)}
  .hero-sub .kv{text-align:center}
  .hero-sub .kv .k{font-size:9px;color:var(--dim);letter-spacing:0.4px;
    text-transform:uppercase}
  .hero-sub .kv .v{font-size:13px;color:var(--dim2);font-weight:700;
    margin-top:2px;font-variant-numeric:tabular-nums}

  .section{margin-bottom:20px}
  .section-title{font-size:11px;letter-spacing:1.4px;text-transform:uppercase;
    color:var(--p1);font-weight:800;margin-bottom:10px;
    display:flex;align-items:baseline;justify-content:space-between;gap:10px}
  .section-title .hint{font-size:10.5px;color:var(--dim);text-transform:none;
    letter-spacing:0;font-weight:500}

  .metric-table{
    background:linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.012));
    border:1px solid var(--line);border-radius:14px;
    padding:12px 14px;overflow-x:auto;
  }
  .metric-row{
    display:grid;grid-template-columns:1fr 44px 44px 60px 60px;
    gap:8px;align-items:center;padding:6px 0;
    font-size:11.5px;font-variant-numeric:tabular-nums;
  }
  .metric-row.head{color:var(--dim);font-weight:700;font-size:9.5px;
    letter-spacing:0.5px;text-transform:uppercase;border-bottom:1px solid var(--line2)}
  .metric-row + .metric-row:not(.head){border-top:1px dashed var(--line2)}
  .metric-row .lbl{color:var(--fg);font-weight:600}
  .metric-row .num{text-align:right;color:var(--dim2);font-weight:600}
  .metric-row .num.strong{color:var(--fg);font-weight:700}
  .metric-row .gap{text-align:right;font-weight:700}
  .metric-row .gap.pos{color:var(--hit)}
  .metric-row .gap.neg{color:var(--neg)}
  .metric-row .gap.neutral{color:var(--dim)}

  .pred{
    background:linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.012));
    border:1px solid var(--line);border-radius:12px;
    padding:10px 12px;margin-bottom:8px;
  }
  .pred.won{border-left:3px solid var(--hit)}
  .pred.lost{border-left:3px solid var(--neg)}
  .pred.void{border-left:3px solid var(--dim);opacity:0.6}
  .pred.open{border-left:3px solid var(--p1)}
  .pred-head{display:flex;justify-content:space-between;align-items:baseline;
    margin-bottom:4px;gap:8px}
  .pred-name{font-size:13px;font-weight:600;overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap}
  .pred-status{font-size:9px;font-weight:800;letter-spacing:0.5px;
    text-transform:uppercase;padding:2px 6px;border-radius:4px;flex-shrink:0}
  .pred-status.open{color:var(--p1);background:rgba(127,212,255,0.1)}
  .pred-status.won{color:var(--hit);background:var(--hitbg)}
  .pred-status.lost{color:var(--neg);background:var(--negbg)}
  .pred-status.void{color:var(--dim);background:var(--missbg)}
  .pred-meta{font-size:10px;color:var(--dim);margin-bottom:6px}
  .pred-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;
    font-size:10.5px}
  .pred-stats .kv{text-align:center}
  .pred-stats .kv .k{color:var(--dim);font-size:9px;letter-spacing:0.3px;
    text-transform:uppercase}
  .pred-stats .kv .v{color:var(--fg);font-weight:700;margin-top:1px;
    font-variant-numeric:tabular-nums}
  .pred-stats .kv .v.muted{color:var(--dim);font-weight:500}

  .empty{background:rgba(255,255,255,0.03);border:1px dashed var(--line);
    border-radius:14px;padding:26px 22px;text-align:center;color:var(--dim2);
    font-size:13px;line-height:1.7}
  a{color:var(--p1)}

  @media (min-width: 900px) {
    body { padding: 32px 24px 80px; }
    .wrap { max-width: 1280px; }
    .nav, .range-toggle, .hero, .section { max-width: 900px; }
    .pred-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .pred-grid .pred { margin-bottom: 0; }
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>📈 Prediction Archive</h1>
  <p class="sub">Every Moneyline model call, auto-logged and auto-settled</p>

  <div class="nav">
    <a href="/">← Matches</a>
    <a href="/tracker">📋 Tracker</a>
    <a href="/predictions" class="active">📈 Predictions</a>
    <a href="/results">📊 Results</a>
  </div>

  <div class="range-toggle">
    <a href="/predictions?days=7"  class="{% if days_back == 7 %}active{% endif %}">Last 7 days</a>
    <a href="/predictions?days=30" class="{% if days_back == 30 %}active{% endif %}">Last 30 days</a>
    <a href="/predictions?days=0"  class="{% if days_back == 0 %}active{% endif %}">All time</a>
  </div>

  <div class="hero">
    <div class="hero-title">Overall performance</div>
    <div class="hero-grid">
      <div class="hero-stat">
        <div class="k">Strike rate</div>
        <div class="v {% if stats.strike and stats.strike >= 0.55 %}hit{% elif stats.strike and stats.strike < 0.5 %}neg{% endif %}">
          {% if stats.strike is not none %}{{ '%.1f'|format(stats.strike*100) }}%{% else %}—{% endif %}
        </div>
        <div class="s">{{ stats.settled }} settled</div>
      </div>
      <div class="hero-stat">
        <div class="k">Avg model</div>
        <div class="v">{% if stats.avg_model %}{{ '%.1f'|format(stats.avg_model*100) }}%{% else %}—{% endif %}</div>
        <div class="s">what we expected</div>
      </div>
      <div class="hero-stat">
        <div class="k">Calibration</div>
        <div class="v {% if stats.gap and stats.gap > 0.01 %}hit{% elif stats.gap and stats.gap < -0.03 %}neg{% elif stats.gap %}warn{% endif %}">
          {% if stats.gap is not none %}{{ '%+.1f'|format(stats.gap*100) }}pp{% else %}—{% endif %}
        </div>
        <div class="s">strike − model</div>
      </div>
    </div>
    <div class="hero-sub">
      <div class="kv"><div class="k">Total</div><div class="v">{{ stats.total }}</div></div>
      <div class="kv"><div class="k">Open</div><div class="v">{{ stats.open }}</div></div>
      <div class="kv"><div class="k">With odds</div><div class="v">{{ stats.with_odds_n }}</div></div>
    </div>
    {% if stats.roi is not none %}
      <div class="hero-sub" style="margin-top:8px;padding-top:10px">
        <div class="kv" style="grid-column:1/-1">
          <div class="k">Flat-bet ROI (all picks with odds)</div>
          <div class="v" style="font-size:16px;color:{{ 'var(--hit)' if stats.roi > 0 else 'var(--neg)' }}">
            {{ '%+.1f'|format(stats.roi*100) }}%
          </div>
        </div>
      </div>
    {% endif %}
  </div>

  {% if stats.by_conf %}
    <div class="section">
      <div class="section-title">
        <span>Calibration by confidence</span>
        <span class="hint">actual vs expected</span>
      </div>
      <div class="metric-table">
        <div class="metric-row head">
          <div>Confidence</div>
          <div class="num">N</div>
          <div class="num">Won</div>
          <div class="num">Actual</div>
          <div class="num">Gap</div>
        </div>
        {% for b in stats.by_conf %}
          <div class="metric-row">
            <div class="lbl">{{ b.label }}</div>
            <div class="num">{{ b.n }}</div>
            <div class="num">{{ b.won }}</div>
            <div class="num strong">{{ '%.1f'|format(b.strike*100) }}%</div>
            <div class="gap {{ 'pos' if b.gap > 0.01 else ('neg' if b.gap < -0.03 else 'neutral') }}">
              {{ '%+.1f'|format(b.gap*100) }}pp
            </div>
          </div>
        {% endfor %}
      </div>
    </div>
  {% endif %}

  {% if stats.by_edge %}
    <div class="section">
      <div class="section-title">
        <span>Edge buckets (with bookmaker odds)</span>
        <span class="hint">does edge predict wins?</span>
      </div>
      <div class="metric-table" style="overflow-x:auto">
        <div class="metric-row head" style="grid-template-columns:1fr 40px 40px 55px 65px 70px">
          <div>Edge</div>
          <div class="num">N</div>
          <div class="num">Won</div>
          <div class="num">Strike</div>
          <div class="num">Break</div>
          <div class="num">vs Book</div>
        </div>
        {% for b in stats.by_edge %}
          <div class="metric-row" style="grid-template-columns:1fr 40px 40px 55px 65px 70px">
            <div class="lbl">{{ b.label }}</div>
            <div class="num">{{ b.n }}</div>
            <div class="num">{{ b.won }}</div>
            <div class="num strong">{{ '%.1f'|format(b.strike*100) }}%</div>
            <div class="num">{{ '%.1f'|format(b.breakeven*100) }}%</div>
            <div class="gap {{ 'pos' if b.edge_vs_book > 0.01 else ('neg' if b.edge_vs_book < -0.03 else 'neutral') }}">
              {{ '%+.1f'|format(b.edge_vs_book*100) }}pp
            </div>
          </div>
        {% endfor %}
      </div>
    </div>
  {% endif %}

  {% if stats.by_surface %}
    <div class="section">
      <div class="section-title">
        <span>By surface</span>
        <span class="hint">does surface matter?</span>
      </div>
      <div class="metric-table">
        <div class="metric-row head">
          <div>Surface</div>
          <div class="num">N</div>
          <div class="num">Won</div>
          <div class="num">Strike</div>
          <div class="num">Model</div>
        </div>
        {% for b in stats.by_surface %}
          <div class="metric-row">
            <div class="lbl">{{ b.label }}</div>
            <div class="num">{{ b.n }}</div>
            <div class="num">{{ b.won }}</div>
            <div class="num strong">{{ '%.1f'|format(b.strike*100) }}%</div>
            <div class="num">{{ '%.1f'|format(b.avg_model*100) }}%</div>
          </div>
        {% endfor %}
      </div>
    </div>
  {% endif %}

  <div class="section">
    <div class="section-title">
      <span>Recent predictions</span>
      <span class="hint">{{ predictions|length }} shown</span>
    </div>
    {% if predictions|length == 0 %}
      <div class="empty">
        No predictions logged yet.<br>
        Open the <a href="/">main page</a> to start logging today's matches.
      </div>
    {% else %}
      <div class="pred-grid">
        {% for p in predictions %}
          <div class="pred {{ p.status }}">
            <div class="pred-head">
              <div class="pred-name">{{ p.model_pick_name }}</div>
              <div class="pred-status {{ p.status }}">{{ p.status }}</div>
            </div>
            <div class="pred-meta">
              vs {{ p.p2_name if p.model_pick == 'p1' else p.p1_name }}
              · {{ p.tournament or '—' }} · {{ p.event_date or '—' }}
            </div>
            <div class="pred-stats">
              <div class="kv"><div class="k">Model</div><div class="v">{{ '%.1f'|format(p.model_pick_prob*100) }}%</div></div>
              <div class="kv">
                <div class="k">Book</div>
                <div class="v {% if not p.book_pick_odds %}muted{% endif %}">
                  {% if p.book_pick_odds %}{{ '%.2f'|format(p.book_pick_odds) }}{% else %}—{% endif %}
                </div>
              </div>
              <div class="kv">
                <div class="k">Edge</div>
                {% if p.book_pick_odds %}
                  {% set edge = p.model_pick_prob - (1/p.book_pick_odds) %}
                  <div class="v" style="color:{{ 'var(--hit)' if edge > 0 else 'var(--neg)' }}">
                    {{ '%+.1f'|format(edge*100) }}%
                  </div>
                {% else %}
                  <div class="v muted">—</div>
                {% endif %}
              </div>
              <div class="kv">
                <div class="k">Result</div>
                <div class="v">
                  {% if p.status == 'won' %}✓{% elif p.status == 'lost' %}✗{% elif p.status == 'void' %}—{% else %}·{% endif %}
                </div>
              </div>
            </div>
          </div>
        {% endfor %}
      </div>
    {% endif %}
  </div>
</div>
</body>
</html>
"""


# =================== Tracker template ===================
TRACKER_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tracker</title>
<style>
  :root{--bg0:#0a0d1a;--bg2:#1a1f36;--fg:#e8eaf2;--dim:#8892b0;--dim2:#b8c0dc;
    --line:rgba(255,255,255,0.08);--line2:rgba(255,255,255,0.04);
    --p1:#7fd4ff;--p2:#a78bfa;--hit:#6ee7b7;--hitbg:rgba(52,211,153,0.13);
    --miss:#94a3b8;--missbg:rgba(148,163,184,0.07);--neg:#f87171;
    --negbg:rgba(248,113,113,0.11);}
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
  .nav{display:flex;gap:8px;margin:12px 0 16px;flex-wrap:wrap}
  .nav a{background:rgba(255,255,255,0.05);border:1px solid var(--line);
    color:var(--dim2);padding:9px 14px;border-radius:9px;font-size:12px;
    text-decoration:none}
  .nav a:hover{border-color:rgba(127,212,255,0.4);color:var(--p1)}
  .nav form{display:inline}
  .nav button{background:linear-gradient(90deg,var(--p1),var(--p2));
    border:none;color:#0a0d1a;padding:9px 14px;border-radius:9px;
    font-size:12px;font-weight:800;cursor:pointer}
  .stats{background:linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.012));
    border:1px solid var(--line);border-radius:16px;
    padding:14px 16px;margin-bottom:16px}
  .stat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
  .stat{text-align:center}
  .stat .label{font-size:9.5px;color:var(--dim);letter-spacing:0.5px;
    text-transform:uppercase;font-weight:700}
  .stat .value{font-size:18px;color:var(--fg);font-weight:800;margin-top:3px;
    font-variant-numeric:tabular-nums}
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
  .pick{background:linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.012));
    border:1px solid var(--line);border-radius:14px;
    padding:12px 14px;margin-bottom:10px}
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
    .picks-grid { display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px; align-items: start; }
    .picks-grid .pick { margin-bottom: 0; }
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>📋 Singles Tracker</h1>
  <p class="sub">Only picks with edge · auto-settled · refreshes every 15 min</p>

  <div class="nav">
    <a href="/">← Matches</a>
    <a href="/predictions">📈 Predictions</a>
    <a href="/results">📊 Results</a>
    <form method="post" action="/settle_now">
      <button type="submit">↻ Settle now</button>
    </form>
  </div>

  {% if settle_summary %}
    <div style="background:rgba(127,212,255,0.08);border:1px solid rgba(127,212,255,0.25);
                border-radius:10px;padding:10px 14px;margin-bottom:14px;
                font-size:12px;color:var(--dim2)">
      Picks: <b style="color:var(--hit)">{{ settle_summary.settled }}</b> settled,
      <b style="color:var(--dim)">{{ settle_summary.voided }}</b> voided.
      Predictions: <b style="color:var(--hit)">{{ pred_summary.settled }}</b> settled,
      <b style="color:var(--dim)">{{ pred_summary.voided }}</b> voided.
      · {{ '%.1f'|format(settle_summary.elapsed + pred_summary.elapsed) }}s
    </div>
  {% endif %}

  <div class="stats">
    <div class="stat-grid">
      <div class="stat"><div class="label">Open</div><div class="value">{{ stats.open }}</div></div>
      <div class="stat"><div class="label">Settled</div><div class="value">{{ stats.settled }}</div></div>
      <div class="stat"><div class="label">Strike Rate</div>
        <div class="value {% if stats.strike and stats.strike > 0.5 %}hit{% endif %}">
          {% if stats.strike is not none %}{{ '%.1f'|format(stats.strike*100) }}%{% else %}—{% endif %}
        </div></div>
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
      No tracked picks yet. Open the main page to trigger auto-tracking.
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
            <form method="post" action="/settle/{{ p.id }}/won"><button class="btn-won" type="submit">✓ Won</button></form>
            <form method="post" action="/settle/{{ p.id }}/lost"><button class="btn-lost" type="submit">✗ Lost</button></form>
            <form method="post" action="/settle/{{ p.id }}/void"><button class="btn-void" type="submit">Void</button></form>
          </div>
        {% endif %}
      </div>
    {% endfor %}
  </div>
</div>
</body>
</html>
"""


# =================== Results template ===================
RESULTS_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results</title>
<style>
  :root{--bg0:#0a0d1a;--bg2:#1a1f36;--fg:#e8eaf2;--dim:#8892b0;--dim2:#b8c0dc;
    --line:rgba(255,255,255,0.08);--line2:rgba(255,255,255,0.04);
    --p1:#7fd4ff;--p2:#a78bfa;--hit:#6ee7b7;--neg:#f87171;--win:#6ee7b7;}
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
  .nav{display:flex;gap:8px;margin:12px 0 16px;flex-wrap:wrap}
  .nav a{background:rgba(255,255,255,0.05);border:1px solid var(--line);
    color:var(--dim2);padding:9px 14px;border-radius:9px;font-size:12px;
    text-decoration:none}
  .nav a:hover{border-color:rgba(127,212,255,0.4);color:var(--p1)}
  .nav a.active{background:linear-gradient(90deg,var(--p1),var(--p2));
    color:#0a0d1a;font-weight:800;border-color:transparent}
  .toolbar{display:flex;gap:8px;margin:12px 0;flex-wrap:wrap}
  .toolbar input[type=text]{flex:1;min-width:200px;
    background:rgba(255,255,255,0.05);border:1px solid var(--line);
    color:var(--fg);padding:9px 12px;border-radius:9px;
    font-size:13px;font-family:inherit}
  .toolbar input[type=date]{background:rgba(255,255,255,0.05);
    border:1px solid var(--line);color:var(--fg);padding:9px 12px;
    border-radius:9px;font-size:13px;font-family:inherit;color-scheme:dark}
  .toolbar a{background:rgba(110,231,183,0.12);border:1px solid rgba(110,231,183,0.35);
    color:var(--hit);padding:9px 14px;border-radius:9px;font-size:12px;
    font-weight:700;text-decoration:none;display:flex;align-items:center}
  .info-bar{background:rgba(167,139,250,0.08);
    border:1px solid rgba(167,139,250,0.25);border-radius:10px;
    padding:10px 14px;margin-bottom:16px;font-size:11.5px;
    color:var(--dim2);display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .info-bar b{color:var(--fg);font-weight:700}
  .dates-bar{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap}
  .dates-bar a{font-size:11px;padding:5px 10px;border-radius:6px;
    text-decoration:none;border:1px solid var(--line);color:var(--dim2);
    font-variant-numeric:tabular-nums}
  .dates-bar a.active{background:linear-gradient(90deg,var(--p1),var(--p2));
    color:#0a0d1a;font-weight:800;border-color:transparent}
  .results-grid{display:block}
  .result{background:linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.012));
    border:1px solid var(--line);border-radius:14px;
    padding:12px 14px;margin-bottom:10px}
  .result-head{display:flex;justify-content:space-between;align-items:baseline;
    margin-bottom:6px;gap:10px}
  .result-tour{font-size:10px;letter-spacing:1.2px;text-transform:uppercase;
    color:var(--p1);font-weight:700;overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap}
  .result-date{font-size:10px;color:var(--dim);
    font-variant-numeric:tabular-nums}
  .result-row{display:grid;grid-template-columns:1fr auto 1fr;gap:10px;
    align-items:center;margin-top:4px}
  .result-side{min-width:0}.result-side.right{text-align:right}
  .r-name{font-size:14px;font-weight:600;overflow:hidden;
    text-overflow:ellipsis;white-space:nowrap}
  .r-name.winner{color:var(--win)}
  .r-name.loser{color:var(--dim2);font-weight:500}
  .r-vs{font-size:10px;color:#3a4160;font-weight:800;letter-spacing:1px;
    padding:0 6px}
  .r-score{font-size:10px;color:var(--dim2);margin-top:4px;text-align:center;
    font-variant-numeric:tabular-nums;background:rgba(255,255,255,0.03);
    padding:4px 8px;border-radius:5px;display:inline-block;
    grid-column:1/-1;justify-self:center}
  .empty{background:rgba(255,255,255,0.03);border:1px dashed var(--line);
    border-radius:14px;padding:30px 22px;text-align:center;color:var(--dim2);
    font-size:13px;line-height:1.7}
  a{color:var(--p1)}
  @media (min-width: 900px) {
    body { padding: 32px 24px 80px; }
    .wrap { max-width: 1280px; }
    .nav, .toolbar, .info-bar, .dates-bar { max-width: 900px; }
    .results-grid { display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px; align-items: start; }
    .results-grid .result { margin-bottom: 0; }
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>📊 Results Archive</h1>
  <p class="sub">Finished matches, logged automatically as picks settle · {{ count }} on {{ selected_date_str }}</p>

  <div class="nav">
    <a href="/">← Matches</a>
    <a href="/tracker">📋 Tracker</a>
    <a href="/predictions">📈 Predictions</a>
    <a href="/results" class="active">📊 Results</a>
  </div>

  <div class="toolbar">
    <input id="q" type="text" placeholder="Search player…"
           value="{{ player_filter or '' }}"
           onchange="applyPlayer(this.value)">
    <input type="date" value="{{ selected_date_str }}"
           onchange="changeDate(this.value)">
    {% if not is_today %}
      <a href="/results">↺ Today</a>
    {% endif %}
  </div>

  {% if dates %}
    <div class="dates-bar">
      {% for d in dates[:10] %}
        <a href="/results?date={{ d }}"
           class="{% if d == selected_date_str %}active{% endif %}">{{ d }}</a>
      {% endfor %}
    </div>
  {% endif %}

  <div class="info-bar">
    <b>{{ count }}</b> result{{ 's' if count != 1 else '' }} on {{ selected_date_str }}
    {% if player_filter %} · filtered by "<b>{{ player_filter }}</b>"{% endif %}
  </div>

  {% if results|length == 0 %}
    <div class="empty">
      No results logged for {{ selected_date_str }} yet.<br>
      Results are saved when a tracked pick settles.
      <br><br><a href="/">Go to Matches →</a>
    </div>
  {% endif %}

  <div class="results-grid">
    {% for r in results %}
      <div class="result">
        <div class="result-head">
          <div class="result-tour">{{ r.tournament or 'Match' }}{% if r.round %} · {{ r.round }}{% endif %}{% if r.surface %} · {{ r.surface|title }}{% endif %}</div>
          <div class="result-date">{{ r.event_date or '—' }}</div>
        </div>
        <div class="result-row">
          <div class="result-side">
            <div class="r-name {{ 'winner' if r.winner_name == r.p1_name else 'loser' }}">{{ r.p1_name or '?' }}</div>
          </div>
          <div class="r-vs">VS</div>
          <div class="result-side right">
            <div class="r-name {{ 'winner' if r.winner_name == r.p2_name else 'loser' }}">{{ r.p2_name or '?' }}</div>
          </div>
          {% if r.score %}
            <div class="r-score">{{ r.score }}</div>
          {% endif %}
        </div>
      </div>
    {% endfor %}
  </div>
</div>
<script>
  function changeDate(v){if(!v)return;const u=new URL(location.href);u.searchParams.set('date',v);location=u.toString();}
  function applyPlayer(v){const u=new URL(location.href);if(v)u.searchParams.set('player',v);else u.searchParams.delete('player');location=u.toString();}
  const q=document.getElementById('q');
  if(q)q.addEventListener('keydown',e=>{if(e.key==='Enter')applyPlayer(e.target.value);});
</script>
</body>
</html>
"""


# =================== Odds dashboard template ===================
ODDS_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Odds Dashboard</title>
<style>
  :root{--bg0:#0a0d1a;--bg2:#1a1f36;--fg:#e8eaf2;--dim:#8892b0;--dim2:#b8c0dc;
    --line:rgba(255,255,255,0.08);--line2:rgba(255,255,255,0.04);
    --p1:#7fd4ff;--p2:#a78bfa;--hit:#6ee7b7;--hitbg:rgba(52,211,153,0.13);
    --neg:#f87171;--bm:#fbbf24;--bmbg:rgba(251,191,36,0.14);--muted:#7a85a8;}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,"Segoe UI",Roboto,sans-serif;
    background:radial-gradient(circle at top,var(--bg2) 0%,var(--bg0) 60%);
    color:var(--fg);min-height:100vh;padding:22px 12px 60px;-webkit-font-smoothing:antialiased}
  .wrap{max-width:760px;margin:0 auto}
  h1{font-size:22px;letter-spacing:-0.4px;background:linear-gradient(90deg,var(--p1),var(--p2));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .sub{color:var(--dim);font-size:12.5px;margin-top:4px}
  .nav,.toolbar,.chips{display:flex;gap:8px;margin:12px 0 0;flex-wrap:wrap}
  .nav a,.toolbar a,.chips a{background:rgba(255,255,255,0.05);border:1px solid var(--line);
    color:var(--dim2);padding:8px 13px;border-radius:9px;font-size:12px;text-decoration:none}
  .chips a.on,.nav a.on{border-color:rgba(127,212,255,0.5);color:var(--p1)}
  .toolbar input{background:rgba(255,255,255,0.05);border:1px solid var(--line);color:var(--fg);
    padding:8px 12px;border-radius:9px;font-size:13px;color-scheme:dark;font-family:inherit}
  .tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:14px 0}
  .tile{background:rgba(255,255,255,0.04);border:1px solid var(--line);border-radius:12px;padding:10px 8px;text-align:center}
  .tile b{display:block;font-size:19px;color:var(--fg)}
  .tile span{font-size:10.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.4px}
  .note{background:var(--bmbg);border:1px solid rgba(251,191,36,.25);color:var(--bm);
    padding:9px 12px;border-radius:10px;font-size:12px;margin-bottom:12px}
  .card{background:linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.012));
    border:1px solid var(--line);border-radius:14px;margin-bottom:12px;overflow:hidden}
  .card.value{border-color:rgba(110,231,183,0.45)}
  .head{padding:12px 14px 8px;display:flex;justify-content:space-between;gap:8px;font-size:11.5px;color:var(--dim)}
  .head b{color:var(--dim2)}
  .vbadge{background:var(--hitbg);color:var(--hit);border-radius:6px;padding:1px 7px;font-weight:700}
  table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
  th{color:var(--muted);font-weight:500;font-size:10.5px;text-align:right;padding:4px 8px;text-transform:uppercase}
  th:first-child,td:first-child{text-align:left;padding-left:14px}
  td{padding:7px 8px;text-align:right;border-top:1px solid var(--line2)}
  td.name{max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--fg)}
  .pos{color:var(--hit);font-weight:700}.neg{color:var(--neg)}.dim{color:var(--dim)}
  tr.val td{background:var(--hitbg)}
  .foot{padding:9px 14px 12px;font-size:12px;color:var(--dim2);border-top:1px solid var(--line2);
    display:flex;flex-wrap:wrap;gap:4px 12px;justify-content:space-between}
  .foot b{color:var(--p1)}
  .empty{background:rgba(255,255,255,0.03);border:1px dashed var(--line);border-radius:14px;
    padding:22px;text-align:center;color:var(--dim);font-size:13px;line-height:1.6}
  @media(max-width:480px){.tiles{grid-template-columns:repeat(2,1fr)}th.hide,td.hide{display:none}}
</style>
</head>
<body><div class="wrap">
  <h1>💰 Odds Dashboard</h1>
  <p class="sub">Matches with real bookmaker odds · {{ weekday_name }}</p>

  <div class="nav">
    <a href="/">← Matches</a><a href="/tracker">📋 Tracker</a>
    <a href="/predictions">📈 Predictions</a><a href="/results">📊 Results</a>
  </div>
  <div class="toolbar">
    <input type="date" value="{{ selected_date_str }}" onchange="go({date:this.value})">
    <a href="{{ qs(deep=1) }}">🔎 Check odds for more matches</a>
    <a href="{{ qs(refresh=1) }}">↻</a>
  </div>
  <div class="chips">
    <a class="{{ 'on' if not value_only }}" href="{{ qs(value=0) }}">All with odds</a>
    <a class="{{ 'on' if value_only }}" href="{{ qs(value=1) }}">◆ Value only</a>
    <a class="{{ 'on' if sort_by=='edge' }}" href="{{ qs(sort='edge') }}">Edge</a>
    <a class="{{ 'on' if sort_by=='ev' }}" href="{{ qs(sort='ev') }}">EV</a>
    <a class="{{ 'on' if sort_by=='prob' }}" href="{{ qs(sort='prob') }}">Confidence</a>
    <a class="{{ 'on' if sort_by=='time' }}" href="{{ qs(sort='time') }}">Time</a>
  </div>

  <div class="tiles">
    <div class="tile"><b>{{ summary.n }}</b><span>with odds</span></div>
    <div class="tile"><b style="color:var(--hit)">{{ summary.value }}</b><span>value picks</span></div>
    <div class="tile"><b>{{ '%.1f'|format(summary.margin*100) }}%</b><span>avg margin</span></div>
    <div class="tile"><b>{{ '%.1f'|format(summary.books) }}</b><span>avg books</span></div>
  </div>

  {% if deep_info %}<div class="note">{{ deep_info }}</div>{% endif %}
  <div class="note" style="background:rgba(127,212,255,.08);border-color:rgba(127,212,255,.25);color:var(--p1)">
    Edge = model − margin-free market probability. EV = model × odds − 1.
    ◆ value = model ≥ {{ '%.0f'|format(min_prob*100) }}%, edge {{ '%.0f'|format(min_edge*100) }}–{{ '%.0f'|format(max_edge*100) }}%, EV &gt; 0.
    Edges are only meaningful if the model really beats the market — judge that on the Predictions page, not from one day.
  </div>

  {% if rows|length == 0 %}
    <div class="empty">No matches with bookmaker odds{% if value_only %} and a value pick{% endif %} for this day yet.<br>
      Try “Check odds for more matches” (fetches up to {{ deep_max }} more).</div>
  {% endif %}

  {% for r in rows %}
  {% set m = r.m %}
  <div class="card {{ 'value' if r.value }}">
    <div class="head">
      <span><b>{{ m.match.tournament }}</b>{% if m.match.round %} · {{ m.match.round }}{% endif %} · {{ m.match.surface|title }}</span>
      <span>{% if r.value %}<span class="vbadge">◆ value</span> {% endif %}{{ m.match.time or '' }}</span>
    </div>
    <table>
      <tr><th></th><th>Model</th><th>Market</th><th>Odds</th><th class="hide">Best</th><th>Edge</th><th>EV</th></tr>
      {% for sd in r.sides %}
      <tr class="{{ 'val' if sd.value }}">
        <td class="name">{{ sd.name }} <span class="dim">#{{ sd.rank or 'NR' }}</span></td>
        <td>{{ '%.1f'|format(sd.p*100) }}%</td>
        <td class="dim">{{ '%.1f'|format(sd.mk*100) }}%</td>
        <td>{{ '%.2f'|format(sd.odds) }}</td>
        <td class="hide dim">{{ '%.2f'|format(sd.best_odds) }} <span style="font-size:10px">{{ sd.best_book }}</span></td>
        <td class="{{ 'pos' if sd.edge>0.03 else ('neg' if sd.edge<-0.03 else 'dim') }}">{{ '%+.1f'|format(sd.edge*100) }}%</td>
        <td class="{{ 'pos' if sd.ev>0 else 'neg' }}">{{ '%+.1f'|format(sd.ev*100) }}%</td>
      </tr>
      {% endfor %}
    </table>
    {% set top = m.pred.score.top %}
    <div class="foot">
      <span>🎯 <b>{{ (m.p1.name if top.side=='p1' else m.p2.name) }}</b> {{ top.score }}
        <span class="dim">({{ top.games }})</span> {{ '%.0f'|format(top.prob*100) }}%</span>
      <span class="dim">{{ r.bo.bookmaker }} · {{ r.books }} books · margin {{ '%.1f'|format(r.margin*100) }}%</span>
    </div>
  </div>
  {% endfor %}
</div>
<script>
  function go(o){const u=new URL(location.href);for(const k in o){u.searchParams.set(k,o[k]);}location=u.toString();}
</script>
</body></html>
"""


# =================== Routes ===================
def build_matches_with_markets(ranked_only=True, force=False,
                                target_date=None, sort_by="prob"):
    matches = load_matches(ranked_only=ranked_only, force=force,
                           target_date=target_date, sort_by=sort_by)
    for m in matches:
        p = m["pred"]
        p["fair_mw1"] = fair_odds(p["prob"])
        p["fair_mw2"] = fair_odds(1 - p["prob"])
    return matches


@app.route("/")
def index():
    ranked_only = request.args.get("filter", "ranked") != "all"
    force = request.args.get("refresh") == "1"
    sort_by = request.args.get("sort", "prob")
    if sort_by not in ("prob", "time"):
        sort_by = "prob"

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
            target_date=selected_date, sort_by=sort_by)
    except Exception as e:
        return f"<pre style='color:#f87171;padding:24px'>Error: {e}</pre>", 500

    logged_count = 0
    if is_today:
        try:
            logged_count = _log_predictions(matches, selected_date_str)
            if logged_count:
                print(f"[predictions] logged {logged_count} new predictions")
        except Exception as e:
            print(f"[predictions] log error: {e}")
        try:
            _auto_track_picks(matches)
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
        sort_by=sort_by,
        MODEL_SUPPORT_MIN_PROB=MODEL_SUPPORT_MIN_PROB,
        AUTO_TRACK_MIN_EDGE_MW=AUTO_TRACK_MIN_EDGE_MW,
        logged_count=logged_count,
    )


DEEP_FETCH_MAX = 60


def _odds_rows(matches):
    """Matches that have real bookmaker odds, with edge/EV per side."""
    rows = []
    for m in matches:
        bo = m["match"].get("bookmaker_odds")
        if not bo or not bo.get("p1_odds") or not bo.get("p2_odds"):
            continue
        o1, o2 = bo["p1_odds"], bo["p2_odds"]
        q1, q2 = 1.0 / o1, 1.0 / o2
        over = q1 + q2
        p1 = m["pred"]["prob"]
        books = bo.get("all_books") or {bo["bookmaker"]: [o1, o2]}
        sides = []
        for idx, (key, prob, mk, odds) in enumerate((
                ("p1", p1, q1 / over, o1), ("p2", 1 - p1, q2 / over, o2))):
            best_book, best_pair = max(books.items(), key=lambda kv: kv[1][idx])
            edge = prob - mk
            ev = prob * odds - 1.0
            value = (prob >= MODEL_SUPPORT_MIN_PROB
                     and AUTO_TRACK_MIN_EDGE_MW <= edge <= AUTO_TRACK_MAX_EDGE_MW
                     and ev > 0)
            sides.append({"key": key, "name": m[key]["name"],
                          "rank": m[key].get("ranking"), "p": prob, "mk": mk,
                          "odds": odds, "best_odds": best_pair[idx],
                          "best_book": best_book, "edge": edge, "ev": ev,
                          "value": value})
        rows.append({"m": m, "bo": bo, "margin": over - 1.0, "books": len(books),
                     "sides": sides, "value": any(x["value"] for x in sides),
                     "best_edge": max(x["edge"] for x in sides),
                     "best_ev": max(x["ev"] for x in sides),
                     "conf": max(p1, 1 - p1)})
    return rows


@app.route("/odds")
def odds_dashboard():
    ranked_only = request.args.get("filter", "ranked") != "all"
    force = request.args.get("refresh") == "1"
    sort_by = request.args.get("sort", "edge")
    if sort_by not in ("edge", "ev", "prob", "time"):
        sort_by = "edge"
    value_only = request.args.get("value") == "1"

    today = datetime.now(NAIROBI_TZ).date()
    date_str = request.args.get("date", "").strip()
    try:
        selected_date = (datetime.strptime(date_str, "%Y-%m-%d").date()
                         if date_str else today)
    except ValueError:
        selected_date = today

    try:
        matches = build_matches_with_markets(
            ranked_only=ranked_only, force=force,
            target_date=selected_date, sort_by="prob")
    except Exception as e:
        return f"<pre style='color:#f87171;padding:24px'>Error: {e}</pre>", 500

    deep_info = None
    if request.args.get("deep") == "1":
        todo = [x for x in matches
                if x["match"]["te_match_id"] and not x["match"]["bookmaker_odds"]]
        todo.sort(key=lambda x: max(x["pred"]["prob"], 1 - x["pred"]["prob"]),
                  reverse=True)
        todo = todo[:DEEP_FETCH_MAX]
        if todo and _ODDS_FETCH_LOCK.acquire(blocking=False):
            try:
                _fetch_odds_for(todo)
            finally:
                _ODDS_FETCH_LOCK.release()
            got = sum(1 for x in todo if x["match"]["bookmaker_odds"])
            deep_info = f"Checked {len(todo)} more matches · {got} had bookmaker odds."
        elif todo:
            deep_info = "An odds fetch is already running — reload in a few seconds."
        else:
            deep_info = "Every match with an odds page has already been checked."

    if selected_date == today:
        try:
            _auto_track_picks(matches)
        except Exception as e:
            print(f"[auto-track] error: {e}")

    rows = _odds_rows(matches)
    summary = {
        "n": len(rows), "value": sum(1 for r in rows if r["value"]),
        "margin": (sum(r["margin"] for r in rows) / len(rows)) if rows else 0.0,
        "books": (sum(r["books"] for r in rows) / len(rows)) if rows else 0.0,
    }
    if value_only:
        rows = [r for r in rows if r["value"]]
    if sort_by == "edge":
        rows.sort(key=lambda r: r["best_edge"], reverse=True)
    elif sort_by == "ev":
        rows.sort(key=lambda r: r["best_ev"], reverse=True)
    elif sort_by == "prob":
        rows.sort(key=lambda r: r["conf"], reverse=True)
    else:
        rows.sort(key=lambda r: (r["m"]["match"]["time"] == "",
                                 r["m"]["match"]["time"]))

    def qs(**kw):
        args = {"date": selected_date.isoformat(), "sort": sort_by,
                "value": "1" if value_only else "0",
                "filter": "ranked" if ranked_only else "all"}
        args.update({k: v for k, v in kw.items()})
        return "/odds?" + "&".join(f"{k}={v}" for k, v in args.items())

    return render_template_string(
        ODDS_PAGE, rows=rows, summary=summary, deep_info=deep_info,
        selected_date_str=selected_date.isoformat(),
        weekday_name=selected_date.strftime("%A, %d %B %Y"),
        sort_by=sort_by, value_only=value_only, qs=qs,
        min_prob=MODEL_SUPPORT_MIN_PROB, min_edge=AUTO_TRACK_MIN_EDGE_MW,
        max_edge=AUTO_TRACK_MAX_EDGE_MW, deep_max=DEEP_FETCH_MAX,
    )


@app.route("/predictions")
def predictions_page():
    days_arg = request.args.get("days", "30")
    try:
        days_back = int(days_arg)
    except ValueError:
        days_back = 30
    if days_back == 0:
        days_back = None

    stats = predictions.stats(days_back=days_back)
    rows = predictions.list_predictions(days_back=days_back, limit=200)

    return render_template_string(
        PREDICTIONS_PAGE,
        stats=stats,
        predictions=rows,
        days_back=days_back if days_back else 0,
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
        pred_summary={"settled": 0, "voided": 0, "elapsed": 0.0},
    )


@app.route("/results")
def results_page():
    today = datetime.now(NAIROBI_TZ).date()
    date_str = request.args.get("date", "").strip()
    try:
        selected_date = (datetime.strptime(date_str, "%Y-%m-%d").date()
                        if date_str else today)
    except ValueError:
        selected_date = today
    is_today = selected_date == today
    selected_date_str = selected_date.isoformat()
    player_filter = request.args.get("player", "").strip()

    rows = results.list_results(
        event_date=selected_date_str,
        player_name=player_filter or None,
        limit=500,
    )
    dates = results.available_dates()

    return render_template_string(
        RESULTS_PAGE,
        results=rows, count=len(rows),
        selected_date_str=selected_date_str,
        is_today=is_today, player_filter=player_filter, dates=dates,
    )


@app.route("/settle_now", methods=["POST"])
def settle_now_route():
    s1 = settler.settle_open_picks(verbose=True)
    s2 = predictions.settle_open_predictions(verbose=True)
    picks = tracker.list_picks()
    stats = tracker.stats()
    return render_template_string(
        TRACKER_PAGE,
        picks=picks, stats=stats,
        active_filter=None,
        settle_summary=s1, pred_summary=s2,
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
        out["now_nairobi"] = datetime.now(NAIROBI_TZ).strftime("%H:%M %Z")
        out["tracker_stats"] = tracker.stats()
        out["predictions_stats"] = predictions.stats(days_back=None)
    except Exception as e:
        out["error"] = str(e)
    return ("<pre style='color:#e8eaf2;background:#0a0d1a;padding:20px;"
            "white-space:pre-wrap;word-break:break-word;font-size:12px'>"
            f"{_json.dumps(out, indent=2, default=str)}</pre>")


if __name__ == "__main__":
    ensure_schema()
    tracker.init_db()
    results.init_db()
    predictions.init_db()

    settle_thread = threading.Thread(target=_auto_settle_loop, daemon=True)
    settle_thread.start()
    threading.Thread(target=_recent_sync_loop, daemon=True).start()
    print("[app] predictions auto-logging enabled · /predictions page active")
    print(f"[app] timezone: Berlin {datetime.now(BERLIN_TZ).strftime('%H:%M %Z')} "
          f"→ Nairobi {datetime.now(NAIROBI_TZ).strftime('%H:%M %Z')}")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
