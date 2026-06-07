"""
WebSocket endpoint for live transcript streaming.

Clients connect to /ws/{session_id} and receive JSON messages:
  {"type": "utterance",    "text": "...", "start": 1.2, "end": 3.4}
  {"type": "answer_start", "question": "...", "id": "..."}
  {"type": "answer_token", "id": "...", "token": "..."}
  {"type": "answer_end",   "id": "..."}
  {"type": "error",        "message": "..."}

broadcast_sync() is called from worker threads (transcription engine,
streaming answer task) via run_coroutine_threadsafe against the main
event loop captured at server startup.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict

from fastapi import WebSocket

# session_id -> set of WebSocket connections
_connections: dict[str, set[WebSocket]] = defaultdict(set)

# Main event loop — set by set_event_loop() during server lifespan startup.
# Allows broadcast_sync() to schedule work from worker threads.
_loop: asyncio.AbstractEventLoop | None = None


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


async def connect(session_id: str, ws: WebSocket):
    await ws.accept()
    _connections[session_id].add(ws)


def disconnect(session_id: str, ws: WebSocket):
    _connections[session_id].discard(ws)


async def broadcast(session_id: str, message: dict):
    """Send a JSON message to all clients watching this session."""
    dead = set()
    for ws in _connections.get(session_id, set()):
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            dead.add(ws)
    for ws in dead:
        _connections[session_id].discard(ws)


def broadcast_sync(session_id: str, message: dict):
    """
    Thread-safe broadcast from a sync context (transcription thread, streaming task).

    Schedules work on the main event loop captured at startup. Safe to call from
    any worker thread while the server is running; silently no-ops in tests and
    after shutdown.
    """
    loop = _loop
    if loop is not None and loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast(session_id, message), loop)
