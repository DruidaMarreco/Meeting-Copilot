"""
FastAPI backend — meeting lifecycle + WebSocket transcript stream + PTT Q&A.

Endpoints:
  POST   /meeting/start              → {session_id}
  POST   /meeting/{id}/end           → {ok}
  GET    /meeting/{id}/transcript    → [{utterance}, ...]
  POST   /meeting/{id}/ptt           → {answer} (multipart: audio file)
  GET    /meeting/list               → [{session}, ...]
  WS     /ws/{id}                    → live transcript events
  GET    /health                     → {ok, ollama_available}
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api import ws as websocket_manager
from audio.capture import AudioCapture
from config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL_SIZE
from llm.query import answer as llm_answer
from llm.query import check_ollama
from storage import db, init_db, save_utterance, vector_store
from transcription.engine import TranscriptChunk, TranscriptionEngine

# ── State ─────────────────────────────────────────────────────────────────────

_active_sessions: dict[str, dict] = {}  # session_id -> {capture, engine}
_ptt_model = None  # lazily loaded, reused across requests


def _get_ptt_model():
    global _ptt_model
    if _ptt_model is None:
        from faster_whisper import WhisperModel

        _ptt_model = WhisperModel(
            WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
    return _ptt_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
    # Cleanup: stop all active captures on shutdown
    for session_id, state in list(_active_sessions.items()):
        _stop_session(session_id, state)


app = FastAPI(title="Meeting Copilot", version="0.1.0", lifespan=lifespan)

# Serve the frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Models ────────────────────────────────────────────────────────────────────


class StartRequest(BaseModel):
    title: str | None = None


class StartResponse(BaseModel):
    session_id: str
    title: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_transcript_callback(session_id: str):
    def on_transcript(chunk: TranscriptChunk):
        # Persist
        utt_id = save_utterance(
            session_id=session_id,
            text=chunk.text,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
        )
        vector_store.add_utterance(
            session_id,
            utt_id,
            chunk.text,
            {
                "start": chunk.start_time,
                "end": chunk.end_time,
            },
        )
        # Broadcast to WebSocket clients
        websocket_manager.broadcast_sync(
            session_id,
            {
                "type": "utterance",
                "id": utt_id,
                "text": chunk.text,
                "start": chunk.start_time,
                "end": chunk.end_time,
            },
        )

    return on_transcript


def _stop_session(session_id: str, state: dict):
    try:
        state["capture"].stop()
    except Exception:
        pass
    try:
        state["engine"].stop()
    except Exception:
        pass
    db.end_session(session_id)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/")
async def serve_ui():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Meeting Copilot API — open /docs for endpoints"}


@app.get("/health")
async def health():
    return {"ok": True, "ollama_available": check_ollama()}


@app.post("/meeting/start", response_model=StartResponse)
async def start_meeting(req: StartRequest = StartRequest()):
    session_id, title = db.create_session(req.title)

    engine = TranscriptionEngine(on_transcript=_make_transcript_callback(session_id))
    capture = AudioCapture(callback=engine.feed)

    engine.start()
    capture.start()

    _active_sessions[session_id] = {"capture": capture, "engine": engine}

    return StartResponse(session_id=session_id, title=title)


@app.post("/meeting/{session_id}/end")
async def end_meeting(session_id: str):
    state = _active_sessions.pop(session_id, None)
    if state:
        _stop_session(session_id, state)
    else:
        db.end_session(session_id)  # idempotent
    return {"ok": True, "session_id": session_id}


@app.get("/meeting/{session_id}/transcript")
async def get_transcript(session_id: str):
    utterances = db.get_utterances(session_id)
    if not utterances and not db.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "utterances": utterances}


@app.post("/meeting/{session_id}/ptt")
async def push_to_talk(session_id: str, audio: UploadFile = File(...)):
    """
    Receives an audio file (WAV/MP3/WebM) from the PTT button.
    Transcribes the question, answers from the current meeting context.
    """
    if session_id not in _active_sessions and not db.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Save temp file for whisper
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        segments, _ = _get_ptt_model().transcribe(tmp_path, language="en")
        question = " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp_path)

    if not question:
        return {"question": "", "answer": "I couldn't hear that — please try again."}

    # Answer from meeting context
    response = llm_answer(session_id=session_id, question=question)

    # Broadcast to UI
    websocket_manager.broadcast_sync(
        session_id,
        {
            "type": "answer",
            "question": question,
            "text": response,
        },
    )

    return {"question": question, "answer": response}


@app.get("/meeting/list")
async def list_meetings():
    return {"sessions": db.list_sessions()}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket_manager.connect(session_id, websocket)
    # Send existing transcript on connect
    utterances = db.get_utterances(session_id)
    for u in utterances:
        await websocket.send_json(
            {
                "type": "utterance",
                "id": u["id"],
                "text": u["text"],
                "start": u["start_time"],
                "end": u["end_time"],
            }
        )
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        websocket_manager.disconnect(session_id, websocket)
