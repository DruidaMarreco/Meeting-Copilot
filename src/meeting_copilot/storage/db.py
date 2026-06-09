"""
SQLite storage for meeting sessions, utterances, and Q&A answers.

Schema:
  sessions      — one row per meeting (id, title, started_at, ended_at, summary)
  utterances    — one row per transcript chunk (id, session_id, text, start_time, end_time, speaker)
  answers       — one row per PTT Q&A exchange (id, session_id, question, answer, created_at)
  session_tags  — many-to-many tags for sessions (session_id, tag)
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

            CREATE TABLE IF NOT EXISTS answers (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL REFERENCES sessions(id),
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_tags (
                session_id  TEXT NOT NULL REFERENCES sessions(id),
                tag         TEXT NOT NULL,
                PRIMARY KEY (session_id, tag)
            );

            CREATE INDEX IF NOT EXISTS idx_utterances_session
                ON utterances(session_id, start_time);
            CREATE INDEX IF NOT EXISTS idx_answers_session
                ON answers(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_session_tags_tag
                ON session_tags(tag);
        """)
        # Migrate existing databases
        session_cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "summary" not in session_cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN summary TEXT")
        if "action_items" not in session_cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN action_items TEXT")
        if "notes" not in session_cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN notes TEXT")
        if "is_starred" not in session_cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN is_starred INTEGER DEFAULT 0")


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
        if row is None:
            return None
        result = dict(row)
        tag_rows = conn.execute(
            "SELECT tag FROM session_tags WHERE session_id = ? ORDER BY tag ASC",
            (session_id,),
        ).fetchall()
        result["tags"] = [r["tag"] for r in tag_rows]
        return result


def list_sessions(limit: int = 20, offset: int = 0, tag: str | None = None) -> list[dict]:
    with get_conn() as conn:
        if tag:
            rows = conn.execute(
                """SELECT s.* FROM sessions s
                   JOIN session_tags t ON t.session_id = s.id
                   WHERE t.tag = ?
                   ORDER BY s.started_at DESC LIMIT ? OFFSET ?""",
                (tag.strip().lower(), limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        sessions = [dict(r) for r in rows]
        for s in sessions:
            tag_rows = conn.execute(
                "SELECT tag FROM session_tags WHERE session_id = ? ORDER BY tag ASC",
                (s["id"],),
            ).fetchall()
            s["tags"] = [r["tag"] for r in tag_rows]
        return sessions


def count_sessions(tag: str | None = None) -> int:
    with get_conn() as conn:
        if tag:
            row = conn.execute(
                """SELECT COUNT(DISTINCT s.id) AS cnt FROM sessions s
                   JOIN session_tags t ON t.session_id = s.id
                   WHERE t.tag = ?""",
                (tag.strip().lower(),),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM sessions").fetchone()
        return int(row["cnt"])


def delete_session(session_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM utterances WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM answers WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM session_tags WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def save_notes(session_id: str, notes: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET notes = ? WHERE id = ?",
            (notes, session_id),
        )


def save_action_items(session_id: str, action_items: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET action_items = ? WHERE id = ?",
            (action_items, session_id),
        )


def save_summary(session_id: str, summary: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET summary = ? WHERE id = ?",
            (summary, session_id),
        )


def update_session_title(session_id: str, title: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (title.strip(), session_id),
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


# ── Answers ───────────────────────────────────────────────────────────────────


def save_answer(session_id: str, question: str, answer: str) -> str:
    answer_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO answers (id, session_id, question, answer, created_at) VALUES (?, ?, ?, ?, ?)",
            (answer_id, session_id, question, answer, now),
        )
    return answer_id


def get_answers(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM answers WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_session_stats(session_id: str) -> dict | None:
    """
    Return aggregate stats for a session:
      duration_seconds, utterance_count, word_count, answer_count.
    Returns None if the session doesn't exist.
    """
    with get_conn() as conn:
        session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if session is None:
            return None
        session = dict(session)

        row = conn.execute(
            """SELECT COUNT(*) AS utterance_count,
                      COALESCE(SUM(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1), 0) AS word_count
               FROM utterances WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        utterance_count = row["utterance_count"]
        word_count = int(row["word_count"])

        answer_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM answers WHERE session_id = ?", (session_id,)
        ).fetchone()["cnt"]

    # Duration: from started_at to ended_at (or None if still active)
    duration: float | None = None
    if session.get("ended_at") and session.get("started_at"):
        from datetime import datetime  # noqa: PLC0415

        try:
            started = datetime.fromisoformat(session["started_at"])
            ended = datetime.fromisoformat(session["ended_at"])
            duration = (ended - started).total_seconds()
        except Exception:
            duration = None

    return {
        "session_id": session_id,
        "title": session["title"],
        "duration_seconds": duration,
        "utterance_count": utterance_count,
        "word_count": word_count,
        "answer_count": answer_count,
    }


def search_all_sessions(query: str, limit: int = 50) -> list[dict]:
    """
    Search utterances across all sessions.
    Returns a list of dicts: {session_id, title, started_at, utterances: [...]}.
    """
    pattern = f"%{query}%"
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT u.*, s.title AS session_title, s.started_at AS session_started_at
               FROM utterances u
               JOIN sessions s ON s.id = u.session_id
               WHERE u.text LIKE ? COLLATE NOCASE
               ORDER BY s.started_at DESC, u.start_time ASC
               LIMIT ?""",
            (pattern, limit),
        ).fetchall()

    grouped: dict[str, dict] = {}
    for r in rows:
        r = dict(r)
        sid = r["session_id"]
        if sid not in grouped:
            grouped[sid] = {
                "session_id": sid,
                "title": r.pop("session_title"),
                "started_at": r.pop("session_started_at"),
                "utterances": [],
            }
        else:
            r.pop("session_title")
            r.pop("session_started_at")
        grouped[sid]["utterances"].append(r)

    return list(grouped.values())


def search_utterances(session_id: str, query: str, limit: int = 100) -> list[dict]:
    """Return utterances whose text contains the query string (case-insensitive)."""
    pattern = f"%{query}%"
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM utterances
               WHERE session_id = ? AND text LIKE ? COLLATE NOCASE
               ORDER BY start_time ASC
               LIMIT ?""",
            (session_id, pattern, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Tags ─────────────────────────────────────────────────────────────────────


def add_tag(session_id: str, tag: str):
    tag = tag.strip().lower()
    if not tag:
        raise ValueError("Tag cannot be empty")
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO session_tags (session_id, tag) VALUES (?, ?)",
            (session_id, tag),
        )


def remove_tag(session_id: str, tag: str):
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM session_tags WHERE session_id = ? AND tag = ?",
            (session_id, tag.strip().lower()),
        )


def get_tags(session_id: str) -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT tag FROM session_tags WHERE session_id = ? ORDER BY tag ASC",
            (session_id,),
        ).fetchall()
        return [r["tag"] for r in rows]


def list_all_tags() -> list[str]:
    """Return all distinct tags across all sessions, sorted alphabetically."""
    with get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT tag FROM session_tags ORDER BY tag ASC").fetchall()
        return [r["tag"] for r in rows]


# ── Stars ─────────────────────────────────────────────────────────────────────


def star_session(session_id: str):
    """Mark a session as starred/favorite."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET is_starred = 1 WHERE id = ?",
            (session_id,),
        )


def unstar_session(session_id: str):
    """Unmark a session as starred."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET is_starred = 0 WHERE id = ?",
            (session_id,),
        )


def is_starred(session_id: str) -> bool:
    """Check if a session is starred."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT is_starred FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return bool(row["is_starred"]) if row else False


def get_starred_sessions(limit: int = 20, offset: int = 0) -> list[dict]:
    """Return starred sessions, newest first."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM sessions WHERE is_starred = 1
               ORDER BY started_at DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()
        sessions = [dict(r) for r in rows]
        for s in sessions:
            tag_rows = conn.execute(
                "SELECT tag FROM session_tags WHERE session_id = ? ORDER BY tag ASC",
                (s["id"],),
            ).fetchall()
            s["tags"] = [r["tag"] for r in tag_rows]
        return sessions


def count_starred_sessions() -> int:
    """Count total starred sessions."""
    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM sessions WHERE is_starred = 1").fetchone()
        return int(row["cnt"])


# ── Insights ──────────────────────────────────────────────────────────────────


def get_insights() -> dict:
    """
    Return aggregate insights across all sessions:
    - Total sessions, words, utterances, answers, action items
    - Average session length, words per session
    - Meeting activity by day of week
    - Most active week, most productive week by action items
    """
    with get_conn() as conn:
        # Total sessions
        session_count = conn.execute("SELECT COUNT(*) AS cnt FROM sessions").fetchone()["cnt"]

        # Total words, utterances, answers, action items
        utterance_row = conn.execute("""SELECT COUNT(*) AS utterance_count,
                      COALESCE(SUM(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1), 0) AS word_count
               FROM utterances""").fetchone()
        utterance_count = utterance_row["utterance_count"]
        total_words = int(utterance_row["word_count"])

        answer_count = conn.execute("SELECT COUNT(*) AS cnt FROM answers").fetchone()["cnt"]

        sessions_with_items = conn.execute(
            "SELECT COUNT(*) AS cnt FROM sessions WHERE action_items IS NOT NULL AND action_items != ''"
        ).fetchone()["cnt"]

        # Average session length (for ended sessions)
        duration_row = conn.execute(
            """SELECT AVG(CAST((julianday(ended_at) - julianday(started_at)) * 86400 AS FLOAT)) AS avg_seconds
               FROM sessions WHERE ended_at IS NOT NULL"""
        ).fetchone()
        avg_duration_seconds = duration_row["avg_seconds"] or 0

        # Activity by day of week (0=Sunday, 6=Saturday)
        activity_by_dow = conn.execute(
            """SELECT strftime('%w', started_at) AS dow, COUNT(*) AS count
               FROM sessions
               GROUP BY dow
               ORDER BY dow ASC"""
        ).fetchall()
        dow_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        busiest_by_day = {dow_names[int(r["dow"])]: r["count"] for r in activity_by_dow}

        # Activity by week (ISO week number)
        weekly_activity = conn.execute(
            """SELECT strftime('%Y-W%W', started_at) AS week, COUNT(*) AS count
               FROM sessions
               GROUP BY week
               ORDER BY week DESC
               LIMIT 12"""
        ).fetchall()
        weekly = [{"week": r["week"], "meetings": r["count"]} for r in weekly_activity]

        # Most common tags
        tag_counts = conn.execute("""SELECT tag, COUNT(*) AS count
               FROM session_tags
               GROUP BY tag
               ORDER BY count DESC
               LIMIT 5""").fetchall()
        top_tags = [{"tag": r["tag"], "count": r["count"]} for r in tag_counts]

    return {
        "total_sessions": session_count,
        "total_utterances": utterance_count,
        "total_words": total_words,
        "total_answers": answer_count,
        "sessions_with_action_items": sessions_with_items,
        "average_session_duration_seconds": round(avg_duration_seconds, 1),
        "activity_by_day_of_week": busiest_by_day,
        "weekly_activity": weekly,
        "top_tags": top_tags,
    }


def get_recent_insights(days: int = 7) -> dict:
    """Return insights for the last N days (meetings, words, answers)."""
    with get_conn() as conn:
        cutoff = datetime.now(UTC) - __import__("datetime").timedelta(days=days)
        cutoff_iso = cutoff.isoformat()

        session_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM sessions WHERE started_at >= ?", (cutoff_iso,)
        ).fetchone()["cnt"]

        utterance_row = conn.execute(
            """SELECT COUNT(*) AS utterance_count,
                      COALESCE(SUM(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1), 0) AS word_count
               FROM utterances
               WHERE created_at >= ?""",
            (cutoff_iso,),
        ).fetchone()
        utterance_count = utterance_row["utterance_count"]
        total_words = int(utterance_row["word_count"])

        answer_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM answers WHERE created_at >= ?", (cutoff_iso,)
        ).fetchone()["cnt"]

    return {
        "days": days,
        "sessions": session_count,
        "utterances": utterance_count,
        "words": total_words,
        "answers": answer_count,
    }


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
