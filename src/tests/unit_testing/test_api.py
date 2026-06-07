"""
API smoke tests using FastAPI's TestClient.

Heavy dependencies (WhisperModel, AudioCapture, Ollama) are mocked so
these tests run without GPU, audio hardware, or a running Ollama server.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with DB and all hardware deps mocked out."""
    import storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()

    # Ensure api.main is imported before patch() tries to resolve the target
    import api.main

    mock_capture = MagicMock()
    mock_capture.return_value.start.return_value = []
    mock_engine = MagicMock()

    with (
        patch("api.main.AudioCapture", mock_capture),
        patch("api.main.TranscriptionEngine", mock_engine),
    ):
        yield TestClient(api.main.app)


# ── Health ────────────────────────────────────────────────────────────────────


def test_health_endpoint(client):
    with patch("api.main.check_ollama", return_value=False):
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
    titles = [s["title"] for s in r.json()["sessions"]]
    assert "Alpha" in titles
    assert "Beta" in titles


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
        patch("api.main._get_ptt_model", return_value=fake_model),
        patch("api.main._stream_answer_to_ws"),  # don't actually call ollama
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
