import os
import sqlite3
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "results.db")


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT,
            te_match_id TEXT,
            event_date TEXT,
            tournament TEXT,
            round TEXT,
            surface TEXT,
            p1_name TEXT,
            p2_name TEXT,
            winner_name TEXT,
            loser_name TEXT,
            score TEXT,
            source TEXT,
            created_at TEXT,
            UNIQUE(te_match_id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_results_date ON results(event_date)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_results_player ON results(winner_name)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_results_loser ON results(loser_name)")
    conn.commit()
    conn.close()


def add_result(te_match_id, event_date, tournament="", round_="",
               surface="", p1_name="", p2_name="",
               winner_name="", loser_name="", score="", source="settler"):
    """Insert or update a match result. Returns the row id or None."""
    if not te_match_id or not winner_name:
        return None
    conn = _conn()
    try:
        cur = conn.execute("""
            INSERT INTO results (
                te_match_id, event_date, tournament, round, surface,
                p1_name, p2_name, winner_name, loser_name, score,
                source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(te_match_id) DO UPDATE SET
                winner_name = excluded.winner_name,
                loser_name = excluded.loser_name,
                score = excluded.score,
                source = excluded.source
        """, (
            str(te_match_id), event_date, tournament, round_, surface,
            p1_name, p2_name, winner_name, loser_name, score,
            source, datetime.now(timezone.utc).isoformat()
        ))
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        print(f"[results] insert failed: {e}")
        return None
    finally:
        conn.close()


def list_results(event_date=None, player_name=None, limit=500):
    conn = _conn()
    q = "SELECT * FROM results WHERE 1=1"
    params = []
    if event_date:
        q += " AND event_date = ?"
        params.append(event_date)
    if player_name:
        like = f"%{player_name}%"
        q += " AND (winner_name LIKE ? OR loser_name LIKE ?)"
        params.append(like)
        params.append(like)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def available_dates():
    """Return list of distinct event_date values, newest first."""
    conn = _conn()
    rows = conn.execute(
        "SELECT DISTINCT event_date FROM results "
        "WHERE event_date IS NOT NULL ORDER BY event_date DESC"
    ).fetchall()
    conn.close()
    return [r["event_date"] for r in rows]


def stats_for_date(event_date):
    """Return count of results for a given date."""
    conn = _conn()
    row = conn.execute(
        "SELECT COUNT(*) as n FROM results WHERE event_date = ?",
        (event_date,)).fetchone()
    conn.close()
    return row["n"] if row else 0


def h2h(player_a, player_b, limit=20):
    """Return all results where both players appear."""
    conn = _conn()
    like_a = f"%{player_a}%"
    like_b = f"%{player_b}%"
    rows = conn.execute("""
        SELECT * FROM results
        WHERE (winner_name LIKE ? AND loser_name LIKE ?)
           OR (winner_name LIKE ? AND loser_name LIKE ?)
        ORDER BY event_date DESC LIMIT ?
    """, (like_a, like_b, like_b, like_a, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
