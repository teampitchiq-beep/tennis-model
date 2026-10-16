import os
import sqlite3
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "picks.db")


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT,
            te_match_id TEXT,
            market TEXT DEFAULT 'mw',
            line REAL,
            pick_side TEXT,
            pick_name TEXT,
            opponent_name TEXT,
            tournament TEXT,
            surface TEXT,
            match_time TEXT,
            model_prob REAL,
            bookmaker TEXT,
            book_odds REAL,
            edge REAL,
            status TEXT DEFAULT 'open',
            created_at TEXT,
            settled_at TEXT
        )
    """)
    conn.commit()

    # ---- Migrations for older DBs ----
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(picks)").fetchall()]
    migrations = []
    if "te_match_id" not in cols:
        migrations.append("ALTER TABLE picks ADD COLUMN te_match_id TEXT")
    if "market" not in cols:
        migrations.append("ALTER TABLE picks ADD COLUMN market TEXT DEFAULT 'mw'")
    if "line" not in cols:
        migrations.append("ALTER TABLE picks ADD COLUMN line REAL")

    for sql in migrations:
        conn.execute(sql)
        print(f"[tracker] migrated: {sql}")
    if migrations:
        conn.commit()

    # ---- Unique index for auto-track dedup ----
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS picks_unique_idx "
            "ON picks(match_id, market, pick_side)"
        )
        conn.commit()
    except Exception as e:
        print(f"[tracker] unique index: {e}")

    conn.close()


def pick_exists(match_id, market, pick_side):
    conn = _conn()
    row = conn.execute(
        "SELECT id FROM picks WHERE match_id = ? AND market = ? "
        "AND pick_side = ?",
        (match_id, market, pick_side)).fetchone()
    conn.close()
    return row is not None


def add_pick(**kwargs):
    """
    Insert a pick.
    Returns the new id, or None if a duplicate already exists.
    """
    match_id = kwargs.get("match_id", "")
    market = kwargs.get("market", "mw")
    pick_side = kwargs.get("pick_side", "")

    # Explicit duplicate check (in addition to the unique index)
    if pick_exists(match_id, market, pick_side):
        return None

    cols = ["match_id", "te_match_id", "market", "line", "pick_side",
            "pick_name", "opponent_name", "tournament", "surface",
            "match_time", "model_prob", "bookmaker", "book_odds", "edge",
            "created_at"]
    vals = [kwargs.get(c) for c in cols[:-1]]
    vals.append(datetime.now(timezone.utc).isoformat())

    conn = _conn()
    try:
        cur = conn.execute(
            f"INSERT INTO picks ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            vals)
        conn.commit()
        pick_id = cur.lastrowid
        return pick_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def list_picks(status=None):
    conn = _conn()
    if status:
        rows = conn.execute(
            "SELECT * FROM picks WHERE status = ? ORDER BY id DESC",
            (status,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM picks ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def settle_pick(pick_id, status):
    if status not in ("won", "lost", "void"):
        return False
    conn = _conn()
    conn.execute(
        "UPDATE picks SET status = ?, settled_at = ? WHERE id = ?",
        (status, datetime.now(timezone.utc).isoformat(), pick_id))
    conn.commit()
    conn.close()
    return True


def delete_pick(pick_id):
    conn = _conn()
    conn.execute("DELETE FROM picks WHERE id = ?", (pick_id,))
    conn.commit()
    conn.close()


def stats():
    conn = _conn()
    rows = conn.execute("SELECT * FROM picks").fetchall()
    conn.close()

    total = len(rows)
    open_n = sum(1 for r in rows if r["status"] == "open")
    won = sum(1 for r in rows if r["status"] == "won")
    lost = sum(1 for r in rows if r["status"] == "lost")
    void = sum(1 for r in rows if r["status"] == "void")
    settled = won + lost

    strike = (won / settled) if settled > 0 else None

    settled_rows = [r for r in rows if r["status"] in ("won", "lost")]
    if settled_rows:
        avg_model = sum(r["model_prob"] for r in settled_rows) / len(settled_rows)
        avg_odds = sum(r["book_odds"] for r in settled_rows) / len(settled_rows)
        avg_edge = sum(r["edge"] for r in settled_rows) / len(settled_rows)
        breakeven = 1.0 / avg_odds if avg_odds > 0 else 0
        edge_vs_book = (strike - breakeven) if strike is not None else None

        total_pl = 0.0
        for r in settled_rows:
            if r["status"] == "won":
                total_pl += (r["book_odds"] - 1.0)
            else:
                total_pl -= 1.0
        roi = total_pl / len(settled_rows)
    else:
        avg_model = avg_odds = avg_edge = breakeven = edge_vs_book = None
        total_pl = 0.0
        roi = None

    return {
        "total": total,
        "open": open_n,
        "won": won,
        "lost": lost,
        "void": void,
        "settled": settled,
        "strike": strike,
        "avg_model_prob": avg_model,
        "avg_odds": avg_odds,
        "avg_edge": avg_edge,
        "breakeven": breakeven,
        "edge_vs_book": edge_vs_book,
        "total_pl": total_pl,
        "roi": roi,
    }
