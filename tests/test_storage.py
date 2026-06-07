"""
Tests for storage.db — session and utterance CRUD.
Uses a temp database so tests never touch data/meetings.db.
"""

import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temp file for every test."""
    import storage.db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    yield


# ── Sessions ──────────────────────────────────────────────────────────────────

def test_create_session_returns_id():
    from storage.db import create_session
    sid = create_session("Stand-up")
    assert isinstance(sid, str) and len(sid) == 36  # UUID


def test_get_session_fields():
    from storage.db import create_session, get_session
    sid = create_session("Weekly review")
    s = get_session(sid)
    assert s["title"] == "Weekly review"
    assert s["ended_at"] is None
    assert s["started_at"] is not None


def test_end_session_sets_ended_at():
    from storage.db import create_session, end_session, get_session
    sid = create_session()
    end_session(sid)
    s = get_session(sid)
    assert s["ended_at"] is not None


def test_list_sessions_contains_all():
    from storage.db import create_session, list_sessions
    for i in range(3):
        create_session(f"Meeting {i}")
    sessions = list_sessions()
    assert len(sessions) >= 3
    # All created sessions must appear (order may vary if timestamps collide)
    titles = {s["title"] for s in sessions}
    assert {"Meeting 0", "Meeting 1", "Meeting 2"}.issubset(titles)


def test_get_nonexistent_session_returns_none():
    from storage.db import get_session
    assert get_session("00000000-0000-0000-0000-000000000000") is None


# ── Utterances ────────────────────────────────────────────────────────────────

def test_save_and_get_utterances():
    from storage.db import create_session, save_utterance, get_utterances
    sid = create_session()
    save_utterance(sid, "Hello world", 0.0, 1.5)
    save_utterance(sid, "Second line", 2.0, 3.0)
    utts = get_utterances(sid)
    assert len(utts) == 2
    assert utts[0]["text"] == "Hello world"
    assert utts[1]["text"] == "Second line"


def test_utterances_ordered_by_start_time():
    from storage.db import create_session, save_utterance, get_utterances
    sid = create_session()
    save_utterance(sid, "Late", 10.0, 11.0)
    save_utterance(sid, "Early", 0.0, 1.0)
    utts = get_utterances(sid)
    assert utts[0]["text"] == "Early"
    assert utts[1]["text"] == "Late"


def test_get_recent_utterances_window():
    from storage.db import create_session, save_utterance, get_recent_utterances
    sid = create_session()
    save_utterance(sid, "Old", 0.0, 1.0)
    save_utterance(sid, "Recent", 400.0, 401.0)
    # last_n_seconds=300 — "Old" at t=0 should be excluded when max_t=401
    recent = get_recent_utterances(sid, last_n_seconds=300)
    texts = [u["text"] for u in recent]
    assert "Recent" in texts
    assert "Old" not in texts


def test_utterances_isolated_per_session():
    from storage.db import create_session, save_utterance, get_utterances
    s1 = create_session()
    s2 = create_session()
    save_utterance(s1, "Session 1 text", 0.0, 1.0)
    assert get_utterances(s2) == []
