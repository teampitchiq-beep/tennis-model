import os
import sqlite3
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "predictions.db")

# Log EVERY prediction (confidence >= 0.50). Logging only the confident ones
# hides the low-confidence band and makes calibration impossible to measure.
# Filtering for display/betting happens elsewhere (app.MODEL_SUPPORT_MIN_PROB).
MIN_SIGNAL = 0.50

# Auto-void predictions older than this many days
STALE_DAYS = 7


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT,
            te_match_id TEXT,
            event_date TEXT,
            tournament TEXT,
            round TEXT,
            surface TEXT,
            match_time TEXT,
            p1_name TEXT,
            p2_name TEXT,
            p1_ranking INTEGER,
            p2_ranking INTEGER,
            model_prob_p1 REAL,
            model_prob_p2 REAL,
            model_pick TEXT,
            model_pick_name TEXT,
            model_pick_prob REAL,
            book_odds_p1 REAL,
            book_odds_p2 REAL,
            book_pick_name TEXT,
            book_pick_odds REAL,
            elo_p1 REAL, elo_p2 REAL,
            form_p1 REAL, form_p2 REAL,
            rank_p1 REAL, rank_p2 REAL,
            status TEXT DEFAULT 'open',
            winner_name TEXT,
            loser_name TEXT,
            score TEXT,
            created_at TEXT,
            settled_at TEXT,
            UNIQUE(match_id, event_date)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(event_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_status ON predictions(status)")
    conn.commit()
    conn.close()


def add_prediction(**kwargs):
    cols = [
        "match_id", "te_match_id", "event_date", "tournament", "round",
        "surface", "match_time", "p1_name", "p2_name", "p1_ranking", "p2_ranking",
        "model_prob_p1", "model_prob_p2", "model_pick", "model_pick_name",
        "model_pick_prob", "book_odds_p1", "book_odds_p2", "book_pick_name", "book_pick_odds",
        "elo_p1", "elo_p2", "form_p1", "form_p2", "rank_p1", "rank_p2",
        "created_at",
    ]
    vals = [kwargs.get(c) for c in cols[:-1]]
    vals.append(datetime.now(timezone.utc).isoformat())

    conn = _conn()
    try:
        cur = conn.execute(
            f"INSERT OR IGNORE INTO predictions ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})", vals)
        conn.commit()
        return cur.lastrowid if cur.rowcount > 0 else None
    except Exception as e:
        print(f"[predictions] insert failed: {e}")
        return None
    finally:
        conn.close()


def list_predictions(event_date=None, status=None, player=None, days_back=None, limit=500):
    conn = _conn()
    q = "SELECT * FROM predictions WHERE 1=1"
    params = []
    if event_date:
        q += " AND event_date = ?"
        params.append(event_date)
    if days_back:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days_back)).isoformat()
        q += " AND event_date >= ?"
        params.append(cutoff)
    if status:
        q += " AND status = ?"
        params.append(status)
    if player:
        like = f"%{player}%"
        q += " AND (p1_name LIKE ? OR p2_name LIKE ?)"
        params.extend([like, like])
    q += " ORDER BY event_date DESC, id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_open_predictions(limit=200):
    conn = _conn()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE status = 'open' "
        "ORDER BY event_date DESC, id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def settle_prediction(pred_id, status, winner_name="", loser_name="", score=""):
    if status not in ("won", "lost", "void"): return False
    conn = _conn()
    conn.execute("""
        UPDATE predictions SET status = ?, winner_name = ?, loser_name = ?, score = ?, settled_at = ?
        WHERE id = ?
    """, (status, winner_name, loser_name, score, datetime.now(timezone.utc).isoformat(), pred_id))
    conn.commit()
    conn.close()
    return True


def settle_open_predictions(verbose=True):
    import settler
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time as _t
    
    preds = list_open_predictions(limit=150)
    if not preds:
        return {"checked": 0, "settled": 0, "skipped": 0, "errors": 0, "voided": 0, "elapsed": 0.0}

    t0 = _t.time()
    settled = skipped = errors = voided = 0

    def process(p):
        ts = p.get("created_at")
        if ts:
            try:
                created = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if created.tzinfo is None: created = created.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - created).total_seconds() / 86400 > STALE_DAYS:
                    return (p["id"], "void", "stale", None)
            except Exception: pass

        te_id = p.get("te_match_id")
        if not te_id: return (p["id"], "skip", "no te_match_id", None)

        soup = settler._fetch_te_match(te_id)
        if soup is None: return (p["id"], "error", "fetch failed", None)

        result = settler._parse_result(soup)
        if not result: return (p["id"], "skip", "not finished", None)

        winner, loser, score = result["winner"], result["loser"], result["score"]
        
        if settler._name_match(p["model_pick_name"], winner):
            return (p["id"], "won", winner, {"winner": winner, "loser": loser, "score": score})
        if settler._name_match(p["model_pick_name"], loser):
            return (p["id"], "lost", loser, {"winner": winner, "loser": loser, "score": score})
        return (p["id"], "skip", "no name match", None)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process, p): p for p in preds}
        for f in as_completed(futures):
            p = futures[f]
            try: pid, status, detail, meta = f.result()
            except Exception as e: pid, status, detail, meta = p["id"], "error", str(e), None

            if status == "won":
                settle_prediction(pid, "won", meta["winner"], meta["loser"], meta["score"])
                settled += 1
                if verbose: print(f"[pred-settle] {p['model_pick_name']} WON")
            elif status == "lost":
                settle_prediction(pid, "lost", meta["winner"], meta["loser"], meta["score"])
                settled += 1
                if verbose: print(f"[pred-settle] {p['model_pick_name']} LOST")
            elif status == "void":
                settle_prediction(pid, "void")
                voided += 1
            elif status == "error":
                errors += 1
            else: skipped += 1

    elapsed = _t.time() - t0
    if verbose:
        print(f"[pred-settle] done in {elapsed:.1f}s · settled={settled} voided={voided} skipped={skipped} errors={errors}")

    return {"checked": len(preds), "settled": settled, "skipped": skipped, "errors": errors, "voided": voided, "elapsed": elapsed}


def stats(days_back=None):
    rows = list_predictions(days_back=days_back, status=None, limit=5000)
    settled = [r for r in rows if r["status"] in ("won", "lost")]
    open_count = sum(1 for r in rows if r["status"] == "open")

    won = sum(1 for r in settled if r["status"] == "won")
    n = len(settled)

    strike = won / n if n else None
    avg_model = (sum(r["model_pick_prob"] for r in settled) / n) if n else None
    gap = (strike - avg_model) if (strike is not None and avg_model) else None

    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.01)]
    by_conf = []
    for lo, hi in buckets:
        b = [r for r in settled if lo <= r["model_pick_prob"] < hi]
        if not b: continue
        b_won = sum(1 for r in b if r["status"] == "won")
        b_strike = b_won / len(b)
        b_model = sum(r["model_pick_prob"] for r in b) / len(b)
        by_conf.append({"label": f"{lo*100:.0f}–{hi*100:.0f}%", "n": len(b), "won": b_won, "strike": b_strike, "avg_model": b_model, "gap": b_strike - b_model})

    with_odds = []
    for r in settled:
        if not r.get("book_pick_odds"): continue
        # compare with the margin-free market probability
        q_pick = 1.0 / r["book_pick_odds"]
        q_other = None
        if r.get("book_odds_p1") and r.get("book_odds_p2"):
            q_other = 1.0 / (r["book_odds_p2"] if r["model_pick"] == "p1" else r["book_odds_p1"])
        mkt = q_pick / (q_pick + q_other) if q_other else q_pick
        edge = r["model_pick_prob"] - mkt
        with_odds.append((r, edge))

    by_edge = []
    edge_buckets = [(-1.0, 0.0), (0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.10), (0.10, 1.0)]
    for lo, hi in edge_buckets:
        b = [(r, e) for r, e in with_odds if lo <= e < hi]
        if not b: continue
        b_won = sum(1 for r, _ in b if r["status"] == "won")
        b_strike = b_won / len(b)
        b_odds = sum(r["book_pick_odds"] for r, _ in b) / len(b)
        b_be = 1.0 / b_odds
        by_edge.append({"label": (f"{lo*100:.0f}% to {hi*100:.0f}%" if lo >= 0 else "< 0%"), "n": len(b), "won": b_won, "strike": b_strike, "avg_odds": b_odds, "breakeven": b_be, "edge_vs_book": b_strike - b_be})

    by_surface = []
    for s in ("hard", "clay", "grass"):
        b = [r for r in settled if (r["surface"] or "").lower() == s]
        if not b: continue
        b_won = sum(1 for r in b if r["status"] == "won")
        by_surface.append({"label": s.title(), "n": len(b), "won": b_won, "strike": b_won / len(b), "avg_model": sum(r["model_pick_prob"] for r in b) / len(b)})

    roi = None
    total_pl = 0.0
    if with_odds:
        for r, _ in with_odds:
            total_pl += (r["book_pick_odds"] - 1.0) if r["status"] == "won" else -1.0
        roi = total_pl / len(with_odds)

    return {
        "total": len(rows), "settled": n, "open": open_count, "won": won, "lost": n - won,
        "strike": strike, "avg_model": avg_model, "gap": gap, "with_odds_n": len(with_odds),
        "roi": roi, "total_pl": total_pl, "by_conf": by_conf, "by_edge": by_edge, "by_surface": by_surface,
    }
