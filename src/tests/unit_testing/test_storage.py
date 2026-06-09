"""
Tests for storage.db — session and utterance CRUD.
Uses a temp database so tests never touch data/meetings.db.
"""

import pytest


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temp file for every test."""
    import meeting_copilot.storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    yield


# ── Sessions ──────────────────────────────────────────────────────────────────


def test_create_session_returns_id():
    from meeting_copilot.storage.db import create_session

    sid, _title = create_session("Stand-up")
    assert isinstance(sid, str) and len(sid) == 36  # UUID


def test_get_session_fields():
    from meeting_copilot.storage.db import create_session, get_session

    sid, _ = create_session("Weekly review")
    s = get_session(sid)
    assert s is not None
    assert s["title"] == "Weekly review"
    assert s["ended_at"] is None
    assert s["started_at"] is not None


def test_end_session_sets_ended_at():
    from meeting_copilot.storage.db import create_session, end_session, get_session

    sid, _ = create_session()
    end_session(sid)
    s = get_session(sid)
    assert s is not None
    assert s["ended_at"] is not None


def test_list_sessions_contains_all():
    from meeting_copilot.storage.db import create_session, list_sessions

    for i in range(3):
        create_session(f"Meeting {i}")
    sessions = list_sessions()
    assert len(sessions) >= 3
    # All created sessions must appear (order may vary if timestamps collide)
    titles = {s["title"] for s in sessions}
    assert {"Meeting 0", "Meeting 1", "Meeting 2"}.issubset(titles)


def test_list_sessions_pagination():
    from meeting_copilot.storage.db import count_sessions, create_session, list_sessions

    for i in range(5):
        create_session(f"Page {i}")
    total = count_sessions()
    assert total >= 5

    page1 = list_sessions(limit=3, offset=0)
    page2 = list_sessions(limit=3, offset=3)
    assert len(page1) == 3
    ids1 = {s["id"] for s in page1}
    ids2 = {s["id"] for s in page2}
    assert not ids1 & ids2  # no overlap


def test_save_and_get_notes():
    from meeting_copilot.storage.db import create_session, get_session, save_notes

    sid, _ = create_session("Notes test")
    save_notes(sid, "Remember to follow up with Alice.")
    s = get_session(sid)
    assert s is not None
    assert s["notes"] == "Remember to follow up with Alice."


def test_notes_defaults_none():
    from meeting_copilot.storage.db import create_session, get_session

    sid, _ = create_session()
    s = get_session(sid)
    assert s is not None
    assert s["notes"] is None


def test_get_nonexistent_session_returns_none():
    from meeting_copilot.storage.db import get_session

    assert get_session("00000000-0000-0000-0000-000000000000") is None


# ── Utterances ────────────────────────────────────────────────────────────────


def test_save_and_get_utterances():
    from meeting_copilot.storage.db import create_session, get_utterances, save_utterance

    sid, _ = create_session()
    save_utterance(sid, "Hello world", 0.0, 1.5)
    save_utterance(sid, "Second line", 2.0, 3.0)
    utts = get_utterances(sid)
    assert len(utts) == 2
    assert utts[0]["text"] == "Hello world"
    assert utts[1]["text"] == "Second line"


def test_utterances_ordered_by_start_time():
    from meeting_copilot.storage.db import create_session, get_utterances, save_utterance

    sid, _ = create_session()
    save_utterance(sid, "Late", 10.0, 11.0)
    save_utterance(sid, "Early", 0.0, 1.0)
    utts = get_utterances(sid)
    assert utts[0]["text"] == "Early"
    assert utts[1]["text"] == "Late"


def test_get_recent_utterances_window():
    from meeting_copilot.storage.db import create_session, get_recent_utterances, save_utterance

    sid, _ = create_session()
    save_utterance(sid, "Old", 0.0, 1.0)
    save_utterance(sid, "Recent", 400.0, 401.0)
    # last_n_seconds=300 — "Old" at t=0 should be excluded when max_t=401
    recent = get_recent_utterances(sid, last_n_seconds=300)
    texts = [u["text"] for u in recent]
    assert "Recent" in texts
    assert "Old" not in texts


def test_utterances_isolated_per_session():
    from meeting_copilot.storage.db import create_session, get_utterances, save_utterance

    s1, _ = create_session()
    s2, _ = create_session()
    save_utterance(s1, "Session 1 text", 0.0, 1.0)
    assert get_utterances(s2) == []


# ── Rename ────────────────────────────────────────────────────────────────────


def test_update_session_title():
    from meeting_copilot.storage.db import create_session, get_session, update_session_title

    sid, _ = create_session("Original")
    update_session_title(sid, "Renamed")
    s = get_session(sid)
    assert s is not None
    assert s["title"] == "Renamed"


# ── Answers ───────────────────────────────────────────────────────────────────


def test_save_and_get_answers():
    from meeting_copilot.storage.db import create_session, get_answers, save_answer

    sid, _ = create_session()
    aid = save_answer(sid, "What was decided?", "Option A was chosen.")
    answers = get_answers(sid)
    assert len(answers) == 1
    assert answers[0]["id"] == aid
    assert answers[0]["question"] == "What was decided?"
    assert answers[0]["answer"] == "Option A was chosen."


def test_answers_ordered_by_created_at():
    from meeting_copilot.storage.db import create_session, get_answers, save_answer

    sid, _ = create_session()
    save_answer(sid, "Q1", "A1")
    save_answer(sid, "Q2", "A2")
    answers = get_answers(sid)
    assert answers[0]["question"] == "Q1"
    assert answers[1]["question"] == "Q2"


def test_delete_session_removes_answers():
    from meeting_copilot.storage.db import create_session, delete_session, get_answers, save_answer

    sid, _ = create_session()
    save_answer(sid, "Q?", "A.")
    delete_session(sid)
    assert get_answers(sid) == []


# ── Action Items ──────────────────────────────────────────────────────────────


def test_save_and_get_action_items():
    from meeting_copilot.storage.db import create_session, get_session, save_action_items

    sid, _ = create_session()
    save_action_items(sid, "- [ ] Send report (owner: Bob)")
    s = get_session(sid)
    assert s is not None
    assert s["action_items"] == "- [ ] Send report (owner: Bob)"


def test_action_items_defaults_none():
    from meeting_copilot.storage.db import create_session, get_session

    sid, _ = create_session()
    s = get_session(sid)
    assert s is not None
    assert s["action_items"] is None


# ── Search ────────────────────────────────────────────────────────────────────


def test_search_utterances_returns_matches():
    from meeting_copilot.storage.db import create_session, save_utterance, search_utterances

    sid, _ = create_session()
    save_utterance(sid, "We need to review the contract.", 0.0, 2.0)
    save_utterance(sid, "The budget was approved.", 3.0, 4.0)

    results = search_utterances(sid, "contract")
    assert len(results) == 1
    assert "contract" in results[0]["text"].lower()


def test_search_utterances_case_insensitive():
    from meeting_copilot.storage.db import create_session, save_utterance, search_utterances

    sid, _ = create_session()
    save_utterance(sid, "Alice will handle onboarding.", 0.0, 2.0)

    assert len(search_utterances(sid, "ALICE")) == 1


def test_search_utterances_empty_result():
    from meeting_copilot.storage.db import create_session, save_utterance, search_utterances

    sid, _ = create_session()
    save_utterance(sid, "Nothing here.", 0.0, 1.0)

    assert search_utterances(sid, "xylophone") == []


# ── Session stats ─────────────────────────────────────────────────────────────


def test_get_session_stats_empty():
    from meeting_copilot.storage.db import create_session, get_session_stats

    sid, _ = create_session("Empty meeting")
    stats = get_session_stats(sid)
    assert stats is not None
    assert stats["utterance_count"] == 0
    assert stats["word_count"] == 0
    assert stats["answer_count"] == 0
    assert stats["duration_seconds"] is None


def test_get_session_stats_counts():
    from meeting_copilot.storage.db import (
        create_session,
        end_session,
        get_session_stats,
        save_answer,
        save_utterance,
    )

    sid, _ = create_session("Stats session")
    save_utterance(sid, "Hello world", 0.0, 2.0)
    save_utterance(sid, "Testing one two", 2.0, 4.0)
    save_answer(sid, "Q?", "A.")
    end_session(sid)

    stats = get_session_stats(sid)
    assert stats is not None
    assert stats["utterance_count"] == 2
    assert stats["word_count"] == 5
    assert stats["answer_count"] == 1
    assert stats["duration_seconds"] is not None
    assert stats["duration_seconds"] >= 0


def test_get_session_stats_unknown_returns_none():
    from meeting_copilot.storage.db import get_session_stats

    assert get_session_stats("00000000-0000-0000-0000-000000000000") is None


def test_search_all_sessions_groups_by_session():
    from meeting_copilot.storage.db import (
        create_session,
        save_utterance,
        search_all_sessions,
    )

    s1, _ = create_session("Alpha")
    s2, _ = create_session("Beta")
    save_utterance(s1, "We need to ship the product.", 0.0, 2.0)
    save_utterance(s2, "The product demo is tomorrow.", 0.0, 2.0)
    save_utterance(s2, "Nothing else today.", 3.0, 4.0)

    results = search_all_sessions("product")
    assert len(results) == 2
    session_ids = {r["session_id"] for r in results}
    assert s1 in session_ids and s2 in session_ids

    s2_result = next(r for r in results if r["session_id"] == s2)
    assert len(s2_result["utterances"]) == 1  # only "product demo" line


def test_search_all_sessions_empty():
    from meeting_copilot.storage.db import search_all_sessions

    assert search_all_sessions("zzznotfound") == []


def test_search_utterances_isolated_per_session():
    from meeting_copilot.storage.db import create_session, save_utterance, search_utterances

    s1, _ = create_session()
    s2, _ = create_session()
    save_utterance(s1, "Budget discussion.", 0.0, 1.0)

    assert search_utterances(s2, "budget") == []


# ── Tags ──────────────────────────────────────────────────────────────────────


def test_add_and_get_tags():
    from meeting_copilot.storage.db import add_tag, create_session, get_tags

    sid, _ = create_session()
    add_tag(sid, "engineering")
    add_tag(sid, "Q3")
    tags = get_tags(sid)
    assert "engineering" in tags
    assert "q3" in tags  # stored lowercase


def test_add_tag_idempotent():
    from meeting_copilot.storage.db import add_tag, create_session, get_tags

    sid, _ = create_session()
    add_tag(sid, "duplicate")
    add_tag(sid, "duplicate")
    assert get_tags(sid).count("duplicate") == 1


def test_remove_tag():
    from meeting_copilot.storage.db import add_tag, create_session, get_tags, remove_tag

    sid, _ = create_session()
    add_tag(sid, "remove-me")
    remove_tag(sid, "remove-me")
    assert "remove-me" not in get_tags(sid)


def test_get_tags_empty():
    from meeting_copilot.storage.db import create_session, get_tags

    sid, _ = create_session()
    assert get_tags(sid) == []


def test_list_all_tags():
    from meeting_copilot.storage.db import add_tag, create_session, list_all_tags

    s1, _ = create_session()
    s2, _ = create_session()
    add_tag(s1, "alpha")
    add_tag(s2, "beta")
    add_tag(s1, "beta")  # same tag, different session
    tags = list_all_tags()
    assert "alpha" in tags
    assert "beta" in tags
    assert tags.count("beta") == 1  # distinct


def test_delete_session_removes_tags():
    from meeting_copilot.storage.db import add_tag, create_session, delete_session, get_tags

    sid, _ = create_session()
    add_tag(sid, "cleanup")
    delete_session(sid)
    assert get_tags(sid) == []


def test_get_session_includes_tags():
    from meeting_copilot.storage.db import add_tag, create_session, get_session

    sid, _ = create_session()
    add_tag(sid, "sprint")
    s = get_session(sid)
    assert s is not None
    assert "sprint" in s["tags"]


def test_list_sessions_includes_tags():
    from meeting_copilot.storage.db import add_tag, create_session, list_sessions

    sid, _ = create_session("Tagged meeting")
    add_tag(sid, "frontend")
    sessions = list_sessions()
    match = next((s for s in sessions if s["id"] == sid), None)
    assert match is not None
    assert "frontend" in match["tags"]


def test_list_sessions_filter_by_tag():
    from meeting_copilot.storage.db import add_tag, count_sessions, create_session, list_sessions

    s1, _ = create_session("Alpha meeting")
    s2, _ = create_session("Beta meeting")
    s3, _ = create_session("Gamma meeting")
    add_tag(s1, "finance")
    add_tag(s3, "finance")

    results = list_sessions(tag="finance")
    ids = {s["id"] for s in results}
    assert s1 in ids and s3 in ids
    assert s2 not in ids

    total = count_sessions(tag="finance")
    assert total == 2
