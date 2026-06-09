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

import asyncio
import os
import re
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from meeting_copilot.api import ws as websocket_manager
from meeting_copilot.audio.capture import AudioCapture
from meeting_copilot.config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL_SIZE
from meeting_copilot.llm.query import answer as llm_answer
from meeting_copilot.llm.query import check_ollama
from meeting_copilot.llm.query import extract_action_items as llm_action_items
from meeting_copilot.llm.query import summarize as llm_summarize
from meeting_copilot.storage import (
    db,
    init_db,
    save_action_items,
    save_answer,
    save_summary,
    save_utterance,
    search_utterances,
    vector_store,
)
from meeting_copilot.storage import delete_session as db_delete_session
from meeting_copilot.transcription.engine import TranscriptChunk, TranscriptionEngine

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
    websocket_manager.set_event_loop(asyncio.get_running_loop())
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


def _stream_answer_to_ws(session_id: str, question: str, answer_id: str) -> None:
    """
    Runs in a thread pool. Iterates the LLM stream, broadcasts each token via
    WebSocket, then persists the completed answer to the database.
    """
    websocket_manager.broadcast_sync(
        session_id,
        {"type": "answer_start", "question": question, "id": answer_id},
    )
    tokens: list[str] = []
    try:
        gen = llm_answer(session_id=session_id, question=question, stream=True)
        for token in gen:
            if token:
                tokens.append(token)
                websocket_manager.broadcast_sync(
                    session_id,
                    {"type": "answer_token", "id": answer_id, "token": token},
                )
    finally:
        full_answer = "".join(tokens)
        if full_answer:
            save_answer(session_id, question, full_answer)
        websocket_manager.broadcast_sync(
            session_id,
            {"type": "answer_end", "id": answer_id},
        )


@app.post("/meeting/{session_id}/ptt")
async def push_to_talk(session_id: str, audio: UploadFile = File(...)):
    """
    Receives an audio file (WAV/MP3/WebM) from the PTT button.
    Transcribes the question synchronously, then streams the LLM answer
    token-by-token to all WebSocket clients in the background.
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
        return {"question": "", "error": "Could not transcribe audio — please try again."}

    # Stream answer tokens to WebSocket clients in a worker thread
    answer_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _stream_answer_to_ws, session_id, question, answer_id)

    return {"question": question, "answer_id": answer_id}


@app.get("/meeting/list")
async def list_meetings():
    return {"sessions": db.list_sessions()}


class RenameRequest(BaseModel):
    title: str


@app.patch("/meeting/{session_id}/title")
async def rename_meeting(session_id: str, req: RenameRequest):
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Title cannot be empty")
    if not db.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    db.update_session_title(session_id, title)
    return {"ok": True, "title": title}


@app.get("/meeting/{session_id}/answers")
async def get_answers(session_id: str):
    if not db.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "answers": db.get_answers(session_id)}


# ── Export ────────────────────────────────────────────────────────────────────


def _fmt_time(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


def _safe_filename(title: str) -> str:
    return re.sub(r"[^\w\-]", "-", title).strip("-") or "transcript"


def _export_txt(session: dict, utterances: list[dict]) -> str:
    lines = [f"Meeting: {session['title']}", f"Date: {session['started_at'][:19]}", ""]
    for u in utterances:
        lines.append(f"[{_fmt_time(u['start_time'])}] {u['text']}")
    return "\n".join(lines) + "\n"


def _export_md(session: dict, utterances: list[dict]) -> str:
    lines = [
        f"# {session['title']}",
        "",
        f"**Date:** {session['started_at'][:19]}",
        "",
        "---",
        "",
    ]
    for u in utterances:
        lines.append(f"**[{_fmt_time(u['start_time'])}]** {u['text']}")
        lines.append("")
    return "\n".join(lines)


@app.get("/meeting/{session_id}/transcript/export")
async def export_transcript(
    session_id: str, format: str = Query(default="txt", pattern="^(txt|md)$")
):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    utterances = db.get_utterances(session_id)
    name = _safe_filename(session["title"])
    if format == "md":
        body = _export_md(session, utterances)
        return Response(
            content=body,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{name}.md"'},
        )
    body = _export_txt(session, utterances)
    return Response(
        content=body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}.txt"'},
    )


# ── Summary ───────────────────────────────────────────────────────────────────


def _stream_summary_to_ws(session_id: str, answer_id: str) -> None:
    """Generate meeting summary in a thread pool, stream tokens to WebSocket."""
    websocket_manager.broadcast_sync(
        session_id,
        {"type": "summary_start", "id": answer_id},
    )
    tokens: list[str] = []
    try:
        for token in llm_summarize(session_id=session_id, stream=True):
            if token:
                tokens.append(token)
                websocket_manager.broadcast_sync(
                    session_id,
                    {"type": "summary_token", "id": answer_id, "token": token},
                )
    finally:
        full_text = "".join(tokens)
        if full_text:
            save_summary(session_id, full_text)
        websocket_manager.broadcast_sync(
            session_id,
            {"type": "summary_end", "id": answer_id, "text": full_text},
        )


@app.post("/meeting/{session_id}/summary")
async def generate_summary(session_id: str):
    """
    Trigger a summary of the full meeting transcript.
    Streams tokens via WebSocket (summary_start / summary_token / summary_end).
    Also persists the finished summary to the database.
    Returns {summary_id} immediately.
    """
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Return cached summary if available and not regenerating
    if session.get("summary"):
        return {
            "summary_id": None,
            "cached": True,
            "text": session["summary"],
        }

    summary_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _stream_summary_to_ws, session_id, summary_id)
    return {"summary_id": summary_id, "cached": False}


@app.post("/meeting/{session_id}/summary/regenerate")
async def regenerate_summary(session_id: str):
    """Force-regenerate the summary, ignoring any cached version."""
    if not db.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    summary_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _stream_summary_to_ws, session_id, summary_id)
    return {"summary_id": summary_id, "cached": False}


# ── Delete ────────────────────────────────────────────────────────────────────


# ── Action Items ──────────────────────────────────────────────────────────────


def _stream_action_items_to_ws(session_id: str, action_items_id: str) -> None:
    """Extract action items in a thread pool and stream tokens to WebSocket."""
    websocket_manager.broadcast_sync(
        session_id,
        {"type": "action_items_start", "id": action_items_id},
    )
    tokens: list[str] = []
    try:
        for token in llm_action_items(session_id=session_id, stream=True):
            if token:
                tokens.append(token)
                websocket_manager.broadcast_sync(
                    session_id,
                    {"type": "action_items_token", "id": action_items_id, "token": token},
                )
    finally:
        full_text = "".join(tokens)
        if full_text:
            save_action_items(session_id, full_text)
        websocket_manager.broadcast_sync(
            session_id,
            {"type": "action_items_end", "id": action_items_id, "text": full_text},
        )


@app.post("/meeting/{session_id}/action-items")
async def get_action_items(session_id: str):
    """
    Return cached action items if available, otherwise start extraction.
    Streams tokens via WebSocket (action_items_start / token / end).
    """
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("action_items"):
        return {"action_items_id": None, "cached": True, "text": session["action_items"]}

    action_items_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _stream_action_items_to_ws, session_id, action_items_id)
    return {"action_items_id": action_items_id, "cached": False}


@app.post("/meeting/{session_id}/action-items/regenerate")
async def regenerate_action_items(session_id: str):
    """Force-regenerate action items, ignoring any cached version."""
    if not db.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    action_items_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _stream_action_items_to_ws, session_id, action_items_id)
    return {"action_items_id": action_items_id, "cached": False}


# ── Search ────────────────────────────────────────────────────────────────────


@app.get("/meeting/{session_id}/transcript/search")
async def search_transcript(session_id: str, q: str = Query(..., min_length=1)):
    """Search utterances in a session by keyword (case-insensitive substring match)."""
    if not db.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    results = search_utterances(session_id, q)
    return {"session_id": session_id, "query": q, "utterances": results}


@app.delete("/meeting/{session_id}")
async def delete_meeting(session_id: str):
    state = _active_sessions.pop(session_id, None)
    if state:
        _stop_session(session_id, state)
    db_delete_session(session_id)
    vector_store.delete_session(session_id)
    return {"ok": True}


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
