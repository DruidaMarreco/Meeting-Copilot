"""
SQLite storage for meeting sessions and utterances.

Schema:
  sessions  — one row per meeting (id, title, started_at, ended_at)
  utterances — one row per transcript chunk (id, session_id, text, start_time, end_time, speaker)
"""

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "meetings.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT,
                started_at  TEXT NOT NULL,
                ended_at    TEXT,
                summary     TEXT
            );

            CREATE TABLE IF NOT EXISTS utterances (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL REFERENCES sessions(id),
                text        TEXT NOT NULL,
                start_time  REAL NOT NULL,
                end_time    REAL NOT NULL,
                speaker     TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_utterances_session
                ON utterances(session_id, start_time);
        """)
        # Migrate: add summary column to existing databases
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "summary" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN summary TEXT")


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Sessions ─────────────────────────────────────────────────────────────────


def create_session(title: str | None = None) -> tuple[str, str]:
    session_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    resolved_title = title or f"Meeting {now[:10]}"
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, started_at) VALUES (?, ?, ?)",
            (session_id, resolved_title, now),
        )
    return session_id, resolved_title


def end_session(session_id: str):
    now = datetime.now(UTC).isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (now, session_id),
        )


def get_session(session_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None


def list_sessions(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def delete_session(session_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM utterances WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def save_summary(session_id: str, summary: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET summary = ? WHERE id = ?",
            (summary, session_id),
        )


# ── Utterances ────────────────────────────────────────────────────────────────


def save_utterance(
    session_id: str,
    text: str,
    start_time: float,
    end_time: float,
    speaker: str | None = None,
) -> str:
    utterance_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO utterances
               (id, session_id, text, start_time, end_time, speaker, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (utterance_id, session_id, text, start_time, end_time, speaker, now),
        )
    return utterance_id


def get_utterances(session_id: str, limit: int = 500) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM utterances
               WHERE session_id = ?
               ORDER BY start_time ASC
               LIMIT ?""",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_utterances(session_id: str, last_n_seconds: float = 300) -> list[dict]:
    """Return utterances from the last N seconds of the meeting."""
    with get_conn() as conn:
        # Get max end_time for this session
        row = conn.execute(
            "SELECT MAX(end_time) as max_t FROM utterances WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        max_t = row["max_t"] or 0.0
        cutoff = max(0.0, max_t - last_n_seconds)

        rows = conn.execute(
            """SELECT * FROM utterances
               WHERE session_id = ? AND end_time >= ?
               ORDER BY start_time ASC""",
            (session_id, cutoff),
        ).fetchall()
        return [dict(r) for r in rows]
