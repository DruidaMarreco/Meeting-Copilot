"""
API smoke tests using FastAPI's TestClient.

Heavy dependencies (WhisperModel, AudioCapture, Ollama) are mocked so
these tests run without GPU, audio hardware, or a running Ollama server.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_runtime_settings():
    """Restore runtime_settings to defaults after each test."""
    import meeting_copilot.runtime_settings as rs

    original = rs.get()
    yield
    rs._state.update(original)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with DB and all hardware deps mocked out."""
    import meeting_copilot.storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()

    # Ensure api.main is imported before patch() tries to resolve the target
    import meeting_copilot.api.main

    mock_capture = MagicMock()
    mock_capture.return_value.start.return_value = []
    mock_engine = MagicMock()

    with (
        patch("meeting_copilot.api.main.AudioCapture", mock_capture),
        patch("meeting_copilot.api.main.TranscriptionEngine", mock_engine),
    ):
        yield TestClient(meeting_copilot.api.main.app)


# ── Health ────────────────────────────────────────────────────────────────────


def test_health_endpoint(client):
    with patch("meeting_copilot.api.main.check_ollama", return_value=False):
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


# ── Meeting lifecycle ─────────────────────────────────────────────────────────


def test_start_meeting(client):
    r = client.post("/meeting/start", json={"title": "Test meeting"})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert data["title"] == "Test meeting"


def test_start_meeting_default_title(client):
    r = client.post("/meeting/start", json={})
    assert r.status_code == 200
    assert "session_id" in r.json()


def test_end_meeting(client):
    sid = client.post("/meeting/start", json={}).json()["session_id"]
    r = client.post(f"/meeting/{sid}/end")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_end_nonexistent_meeting_is_idempotent(client):
    r = client.post("/meeting/00000000-0000-0000-0000-000000000000/end")
    assert r.status_code == 200


# ── Transcript ────────────────────────────────────────────────────────────────


def test_get_transcript_empty(client):
    sid = client.post("/meeting/start", json={}).json()["session_id"]
    r = client.get(f"/meeting/{sid}/transcript")
    assert r.status_code == 200
    assert r.json()["utterances"] == []


def test_get_transcript_unknown_session(client):
    r = client.get("/meeting/00000000-0000-0000-0000-000000000000/transcript")
    assert r.status_code == 404


# ── Meeting list ──────────────────────────────────────────────────────────────


def test_list_meetings(client):
    client.post("/meeting/start", json={"title": "Alpha"})
    client.post("/meeting/start", json={"title": "Beta"})
    r = client.get("/meeting/list")
    assert r.status_code == 200
    data = r.json()
    titles = [s["title"] for s in data["sessions"]]
    assert "Alpha" in titles
    assert "Beta" in titles
    assert "total" in data


def test_list_meetings_pagination(client):
    for i in range(5):
        client.post("/meeting/start", json={"title": f"Meeting {i}"})

    r1 = client.get("/meeting/list?limit=3&offset=0")
    assert r1.status_code == 200
    d1 = r1.json()
    assert len(d1["sessions"]) == 3
    assert d1["total"] == 5

    r2 = client.get("/meeting/list?limit=3&offset=3")
    assert r2.status_code == 200
    d2 = r2.json()
    assert len(d2["sessions"]) == 2

    # No overlap between pages
    ids1 = {s["id"] for s in d1["sessions"]}
    ids2 = {s["id"] for s in d2["sessions"]}
    assert not ids1 & ids2


# ── PTT ───────────────────────────────────────────────────────────────────────


def test_ptt_returns_question_and_answer_id(client):
    """PTT endpoint transcribes audio and kicks off streaming; returns question + answer_id."""
    import io

    sid = client.post("/meeting/start", json={}).json()["session_id"]

    fake_segment = MagicMock()
    fake_segment.text = "What is the budget?"
    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([fake_segment], None)

    with (
        patch("meeting_copilot.api.main._get_ptt_model", return_value=fake_model),
        patch("meeting_copilot.api.main._stream_answer_to_ws"),  # don't actually call ollama
    ):
        r = client.post(
            f"/meeting/{sid}/ptt",
            files={"audio": ("q.wav", io.BytesIO(b"fake"), "audio/wav")},
        )

    assert r.status_code == 200
    data = r.json()
    assert data["question"] == "What is the budget?"
    assert "answer_id" in data


def test_ptt_unknown_session_returns_404(client):
    import io

    r = client.post(
        "/meeting/00000000-0000-0000-0000-000000000000/ptt",
        files={"audio": ("q.wav", io.BytesIO(b"fake"), "audio/wav")},
    )
    assert r.status_code == 404


# ── Export ────────────────────────────────────────────────────────────────────


def test_export_txt(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={"title": "Export Test"}).json()["session_id"]
    db_module.save_utterance(sid, "Hello world", 0.0, 2.0)
    client.post(f"/meeting/{sid}/end")

    r = client.get(f"/meeting/{sid}/transcript/export?format=txt")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "Hello world" in r.text
    assert "Export Test" in r.text
    assert r.headers["content-disposition"].endswith('.txt"')


def test_export_txt_includes_qa_and_action_items(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={"title": "Rich Export"}).json()["session_id"]
    db_module.save_utterance(sid, "We discussed the roadmap.", 0.0, 2.0)
    db_module.save_answer(sid, "What was decided?", "Q3 launch confirmed.")
    db_module.save_action_items(sid, "- [ ] Write spec (owner: Alice)")
    client.post(f"/meeting/{sid}/end")

    r = client.get(f"/meeting/{sid}/transcript/export?format=txt")
    assert r.status_code == 200
    assert "Q3 launch confirmed." in r.text
    assert "Write spec" in r.text
    assert "Q: What was decided?" in r.text


def test_export_md(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={"title": "MD Test"}).json()["session_id"]
    db_module.save_utterance(sid, "Markdown content", 5.0, 7.0)

    r = client.get(f"/meeting/{sid}/transcript/export?format=md")
    assert r.status_code == 200
    assert "# MD Test" in r.text
    assert "Markdown content" in r.text
    assert "[00:05]" in r.text


def test_export_md_includes_qa_and_action_items(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={"title": "MD Rich"}).json()["session_id"]
    db_module.save_utterance(sid, "Planning session.", 0.0, 2.0)
    db_module.save_answer(sid, "Budget?", "Approved at $50k.")
    db_module.save_action_items(sid, "- [ ] Hire engineer (owner: Bob)")
    client.post(f"/meeting/{sid}/end")

    r = client.get(f"/meeting/{sid}/transcript/export?format=md")
    assert r.status_code == 200
    assert "## Q&A" in r.text
    assert "Approved at $50k." in r.text
    assert "## Action Items" in r.text
    assert "Hire engineer" in r.text


def test_export_json(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={"title": "JSON Export"}).json()["session_id"]
    db_module.save_utterance(sid, "Hello JSON world", 0.0, 2.0)
    db_module.save_answer(sid, "Q?", "A.")
    client.post(f"/meeting/{sid}/end")

    r = client.get(f"/meeting/{sid}/transcript/export?format=json")
    assert r.status_code == 200
    assert "application/json" in r.headers["content-type"]
    data = r.json()
    assert data["session"]["title"] == "JSON Export"
    assert len(data["utterances"]) == 1
    assert data["utterances"][0]["text"] == "Hello JSON world"
    assert len(data["answers"]) == 1


def test_export_unknown_session(client):
    r = client.get("/meeting/00000000-0000-0000-0000-000000000000/transcript/export")
    assert r.status_code == 404


# ── Notes ─────────────────────────────────────────────────────────────────────


def test_get_notes_empty(client):
    sid = client.post("/meeting/start", json={}).json()["session_id"]
    r = client.get(f"/meeting/{sid}/notes")
    assert r.status_code == 200
    assert r.json()["notes"] == ""


def test_patch_notes_and_retrieve(client):
    sid = client.post("/meeting/start", json={}).json()["session_id"]
    r = client.patch(f"/meeting/{sid}/notes", json={"notes": "Follow up with Bob"})
    assert r.status_code == 200
    assert r.json()["notes"] == "Follow up with Bob"

    r2 = client.get(f"/meeting/{sid}/notes")
    assert r2.json()["notes"] == "Follow up with Bob"


def test_notes_unknown_session_returns_404(client):
    r = client.get("/meeting/00000000-0000-0000-0000-000000000000/notes")
    assert r.status_code == 404


def test_patch_notes_unknown_session_returns_404(client):
    r = client.patch(
        "/meeting/00000000-0000-0000-0000-000000000000/notes",
        json={"notes": "x"},
    )
    assert r.status_code == 404


# ── Delete ────────────────────────────────────────────────────────────────────


def test_delete_meeting(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={"title": "To Delete"}).json()["session_id"]
    db_module.save_utterance(sid, "Some text", 0.0, 1.0)
    client.post(f"/meeting/{sid}/end")

    r = client.delete(f"/meeting/{sid}")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # Session and utterances must be gone
    assert db_module.get_session(sid) is None
    assert db_module.get_utterances(sid) == []


def test_delete_active_meeting_stops_it(client):
    sid = client.post("/meeting/start", json={}).json()["session_id"]
    r = client.delete(f"/meeting/{sid}")
    assert r.status_code == 200
    assert r.json()["ok"] is True


# ── Summary ───────────────────────────────────────────────────────────────────


def test_summary_returns_summary_id(client):
    """Posting to /summary for a valid session starts streaming and returns a summary_id."""
    sid = client.post("/meeting/start", json={}).json()["session_id"]

    with patch("meeting_copilot.api.main._stream_summary_to_ws"):  # don't call ollama
        r = client.post(f"/meeting/{sid}/summary")

    assert r.status_code == 200
    data = r.json()
    assert data["cached"] is False
    assert "summary_id" in data


def test_summary_returns_cached(client):
    """If a summary was previously stored, /summary returns it without streaming."""
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={}).json()["session_id"]
    db_module.save_summary(sid, "This is the cached summary.")

    r = client.post(f"/meeting/{sid}/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["cached"] is True
    assert data["text"] == "This is the cached summary."


def test_summary_regenerate_ignores_cache(client):
    """/summary/regenerate always spawns a new stream even when cache exists."""
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={}).json()["session_id"]
    db_module.save_summary(sid, "Old summary")

    with patch("meeting_copilot.api.main._stream_summary_to_ws"):
        r = client.post(f"/meeting/{sid}/summary/regenerate")

    assert r.status_code == 200
    data = r.json()
    assert data["cached"] is False
    assert "summary_id" in data


def test_summary_unknown_session_returns_404(client):
    r = client.post("/meeting/00000000-0000-0000-0000-000000000000/summary")
    assert r.status_code == 404


# ── Rename ────────────────────────────────────────────────────────────────────


def test_rename_meeting(client):
    sid = client.post("/meeting/start", json={"title": "Old Name"}).json()["session_id"]
    r = client.patch(f"/meeting/{sid}/title", json={"title": "New Name"})
    assert r.status_code == 200
    assert r.json()["title"] == "New Name"

    import meeting_copilot.storage.db as db_module

    session = db_module.get_session(sid)
    assert session is not None
    assert session["title"] == "New Name"


def test_rename_empty_title_returns_422(client):
    sid = client.post("/meeting/start", json={}).json()["session_id"]
    r = client.patch(f"/meeting/{sid}/title", json={"title": "   "})
    assert r.status_code == 422


def test_rename_unknown_session_returns_404(client):
    r = client.patch("/meeting/00000000-0000-0000-0000-000000000000/title", json={"title": "X"})
    assert r.status_code == 404


# ── Answers ───────────────────────────────────────────────────────────────────


def test_get_answers_empty(client):
    sid = client.post("/meeting/start", json={}).json()["session_id"]
    r = client.get(f"/meeting/{sid}/answers")
    assert r.status_code == 200
    assert r.json()["answers"] == []


def test_get_answers_persisted(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={}).json()["session_id"]
    db_module.save_answer(sid, "What was decided?", "The team chose option A.")

    r = client.get(f"/meeting/{sid}/answers")
    assert r.status_code == 200
    ans = r.json()["answers"]
    assert len(ans) == 1
    assert ans[0]["question"] == "What was decided?"
    assert ans[0]["answer"] == "The team chose option A."


def test_get_answers_unknown_session_returns_404(client):
    r = client.get("/meeting/00000000-0000-0000-0000-000000000000/answers")
    assert r.status_code == 404


def test_delete_also_removes_answers(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={}).json()["session_id"]
    db_module.save_answer(sid, "Q?", "A.")
    client.post(f"/meeting/{sid}/end")

    client.delete(f"/meeting/{sid}")
    assert db_module.get_answers(sid) == []


# ── Action Items ──────────────────────────────────────────────────────────────


def test_action_items_returns_action_items_id(client):
    sid = client.post("/meeting/start", json={}).json()["session_id"]

    with patch("meeting_copilot.api.main._stream_action_items_to_ws"):
        r = client.post(f"/meeting/{sid}/action-items")

    assert r.status_code == 200
    data = r.json()
    assert data["cached"] is False
    assert "action_items_id" in data


def test_action_items_returns_cached(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={}).json()["session_id"]
    db_module.save_action_items(sid, "- [ ] Follow up on budget (owner: Alice)")

    r = client.post(f"/meeting/{sid}/action-items")
    assert r.status_code == 200
    data = r.json()
    assert data["cached"] is True
    assert "budget" in data["text"]


def test_action_items_regenerate_ignores_cache(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={}).json()["session_id"]
    db_module.save_action_items(sid, "Old action items")

    with patch("meeting_copilot.api.main._stream_action_items_to_ws"):
        r = client.post(f"/meeting/{sid}/action-items/regenerate")

    assert r.status_code == 200
    data = r.json()
    assert data["cached"] is False
    assert "action_items_id" in data


def test_action_items_unknown_session_returns_404(client):
    r = client.post("/meeting/00000000-0000-0000-0000-000000000000/action-items")
    assert r.status_code == 404


# ── Transcript Search ─────────────────────────────────────────────────────────


def test_search_transcript_returns_matches(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={}).json()["session_id"]
    db_module.save_utterance(sid, "The budget is fifty thousand.", 0.0, 2.0)
    db_module.save_utterance(sid, "We need to hire two engineers.", 3.0, 5.0)

    r = client.get(f"/meeting/{sid}/transcript/search?q=budget")
    assert r.status_code == 200
    data = r.json()
    assert len(data["utterances"]) == 1
    assert "budget" in data["utterances"][0]["text"].lower()


def test_search_transcript_case_insensitive(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={}).json()["session_id"]
    db_module.save_utterance(sid, "Alice will follow up.", 0.0, 2.0)

    r = client.get(f"/meeting/{sid}/transcript/search?q=ALICE")
    assert r.status_code == 200
    assert len(r.json()["utterances"]) == 1


def test_search_transcript_no_matches(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={}).json()["session_id"]
    db_module.save_utterance(sid, "Nothing relevant here.", 0.0, 2.0)

    r = client.get(f"/meeting/{sid}/transcript/search?q=xylophone")
    assert r.status_code == 200
    assert r.json()["utterances"] == []


def test_search_transcript_unknown_session_returns_404(client):
    r = client.get("/meeting/00000000-0000-0000-0000-000000000000/transcript/search?q=test")
    assert r.status_code == 404


# ── Cross-session search ──────────────────────────────────────────────────────


def test_search_meetings_returns_matching_sessions(client):
    import meeting_copilot.storage.db as db_module

    s1 = client.post("/meeting/start", json={"title": "Sprint Review"}).json()["session_id"]
    s2 = client.post("/meeting/start", json={"title": "Daily Standup"}).json()["session_id"]
    db_module.save_utterance(s1, "We shipped the new feature.", 0.0, 2.0)
    db_module.save_utterance(s2, "Bob is blocked on the feature branch.", 0.0, 2.0)
    db_module.save_utterance(s2, "Nothing else to report.", 3.0, 4.0)

    r = client.get("/meeting/search?q=feature")
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "feature"
    session_ids = {s["session_id"] for s in data["sessions"]}
    assert s1 in session_ids
    assert s2 in session_ids

    # s2 has 1 matching utterance (only "feature branch" line matches)
    s2_result = next(s for s in data["sessions"] if s["session_id"] == s2)
    assert len(s2_result["utterances"]) == 1


def test_search_meetings_no_results(client):
    r = client.get("/meeting/search?q=xylophone")
    assert r.status_code == 200
    assert r.json()["sessions"] == []


def test_search_meetings_missing_query_returns_422(client):
    r = client.get("/meeting/search")
    assert r.status_code == 422


# ── Settings ──────────────────────────────────────────────────────────────────


def test_get_settings_returns_defaults(client):
    r = client.get("/settings")
    assert r.status_code == 200
    data = r.json()
    assert "ollama_model" in data
    assert "whisper_language" in data
    assert "whisper_model_size" in data


def test_patch_settings_ollama_model(client):
    r = client.patch("/settings", json={"ollama_model": "mistral"})
    assert r.status_code == 200
    assert r.json()["ollama_model"] == "mistral"

    # Verify GET reflects the change
    assert client.get("/settings").json()["ollama_model"] == "mistral"


def test_patch_settings_whisper_language(client):
    r = client.patch("/settings", json={"whisper_language": "pt"})
    assert r.status_code == 200
    assert r.json()["whisper_language"] == "pt"


def test_patch_settings_empty_body_is_noop(client):
    before = client.get("/settings").json()
    r = client.patch("/settings", json={})
    assert r.status_code == 200
    assert r.json() == before


def test_patch_settings_readonly_key_returns_422(client):
    r = client.patch("/settings", json={"whisper_model_size": "medium"})
    assert r.status_code == 422


def test_list_ollama_models_returns_list(client):
    """When Ollama is unavailable (no server in CI), endpoint returns empty list gracefully."""
    r = client.get("/settings/models")
    assert r.status_code == 200
    assert isinstance(r.json()["models"], list)


# ── Meeting stats ─────────────────────────────────────────────────────────────


def test_stats_empty_session(client):
    sid = client.post("/meeting/start", json={"title": "Stats Test"}).json()["session_id"]
    r = client.get(f"/meeting/{sid}/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["utterance_count"] == 0
    assert data["word_count"] == 0
    assert data["answer_count"] == 0
    assert data["duration_seconds"] is None  # session not yet ended


def test_stats_counts_utterances_and_words(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={"title": "Count Test"}).json()["session_id"]
    db_module.save_utterance(sid, "Hello world", 0.0, 2.0)  # 2 words
    db_module.save_utterance(sid, "One two three", 2.0, 4.0)  # 3 words
    db_module.save_answer(sid, "Q?", "A.")

    r = client.get(f"/meeting/{sid}/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["utterance_count"] == 2
    assert data["word_count"] == 5
    assert data["answer_count"] == 1


def test_stats_includes_duration_when_ended(client):
    import meeting_copilot.storage.db as db_module

    sid = client.post("/meeting/start", json={"title": "Duration Test"}).json()["session_id"]
    client.post(f"/meeting/{sid}/end")
    db_module.save_utterance(sid, "Done", 0.0, 1.0)

    r = client.get(f"/meeting/{sid}/stats")
    assert r.status_code == 200
    assert r.json()["duration_seconds"] is not None
    assert r.json()["duration_seconds"] >= 0


def test_stats_unknown_session_returns_404(client):
    r = client.get("/meeting/00000000-0000-0000-0000-000000000000/stats")
    assert r.status_code == 404


# ── Tags ──────────────────────────────────────────────────────────────────────


def test_add_tag_to_meeting(client):
    sid = client.post("/meeting/start", json={"title": "Tag Test"}).json()["session_id"]
    r = client.post(f"/meeting/{sid}/tags", json={"tag": "Engineering"})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["tag"] == "engineering"  # lowercased
    assert "engineering" in data["tags"]


def test_add_tag_idempotent_via_api(client):
    sid = client.post("/meeting/start", json={}).json()["session_id"]
    client.post(f"/meeting/{sid}/tags", json={"tag": "dup"})
    client.post(f"/meeting/{sid}/tags", json={"tag": "dup"})
    r = client.post(f"/meeting/{sid}/tags", json={"tag": "dup"})
    assert r.json()["tags"].count("dup") == 1


def test_remove_tag_from_meeting(client):
    sid = client.post("/meeting/start", json={}).json()["session_id"]
    client.post(f"/meeting/{sid}/tags", json={"tag": "remove-me"})
    r = client.delete(f"/meeting/{sid}/tags/remove-me")
    assert r.status_code == 200
    assert "remove-me" not in r.json()["tags"]


def test_add_tag_to_unknown_session_returns_404(client):
    r = client.post("/meeting/00000000-0000-0000-0000-000000000000/tags", json={"tag": "x"})
    assert r.status_code == 404


def test_remove_tag_from_unknown_session_returns_404(client):
    r = client.delete("/meeting/00000000-0000-0000-0000-000000000000/tags/x")
    assert r.status_code == 404


def test_add_empty_tag_returns_422(client):
    sid = client.post("/meeting/start", json={}).json()["session_id"]
    r = client.post(f"/meeting/{sid}/tags", json={"tag": "  "})
    assert r.status_code == 422


def test_get_all_tags_empty(client):
    r = client.get("/meeting/tags")
    assert r.status_code == 200
    assert isinstance(r.json()["tags"], list)


def test_get_all_tags_returns_added_tags(client):
    s1 = client.post("/meeting/start", json={}).json()["session_id"]
    s2 = client.post("/meeting/start", json={}).json()["session_id"]
    client.post(f"/meeting/{s1}/tags", json={"tag": "alpha"})
    client.post(f"/meeting/{s2}/tags", json={"tag": "beta"})
    r = client.get("/meeting/tags")
    tags = r.json()["tags"]
    assert "alpha" in tags and "beta" in tags


def test_list_meetings_filtered_by_tag(client):
    s1 = client.post("/meeting/start", json={"title": "Session A"}).json()["session_id"]
    s2 = client.post("/meeting/start", json={"title": "Session B"}).json()["session_id"]
    client.post(f"/meeting/{s1}/tags", json={"tag": "finance"})

    r = client.get("/meeting/list?tag=finance")
    assert r.status_code == 200
    data = r.json()
    ids = {s["id"] for s in data["sessions"]}
    assert s1 in ids
    assert s2 not in ids
    assert data["total"] == 1


def test_session_includes_tags_in_list(client):
    sid = client.post("/meeting/start", json={"title": "With Tags"}).json()["session_id"]
    client.post(f"/meeting/{sid}/tags", json={"tag": "product"})
    r = client.get("/meeting/list")
    sessions = r.json()["sessions"]
    match = next((s for s in sessions if s["id"] == sid), None)
    assert match is not None
    assert "product" in match["tags"]


# ── Auto-title ────────────────────────────────────────────────────────────────


def test_auto_generate_title_endpoint(client):
    """Endpoint returns 200 and queues generation (fire-and-forget, not tested here)."""
    sid = client.post("/meeting/start", json={}).json()["session_id"]
    with patch("meeting_copilot.api.main.llm_generate_title", return_value="Sprint Planning Q3"):
        with (
            patch("meeting_copilot.api.main.loop")
            if False
            else __import__("contextlib").nullcontext()
        ):
            r = client.post(f"/meeting/{sid}/title/generate")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_auto_generate_title_unknown_session(client):
    r = client.post("/meeting/00000000-0000-0000-0000-000000000000/title/generate")
    assert r.status_code == 404
