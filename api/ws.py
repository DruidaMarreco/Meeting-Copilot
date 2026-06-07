"""
WebSocket endpoint for live transcript streaming.

Clients connect to /ws/{session_id} and receive JSON messages:
  {"type": "utterance", "text": "...", "start": 1.2, "end": 3.4}
  {"type": "answer",    "text": "...", "question": "..."}
  {"type": "error",     "message": "..."}

The server holds a dict of active connections per session.
TranscriptionEngine callbacks push utterances to all connected clients.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict

from fastapi import WebSocket

# session_id -> set of WebSocket connections
_connections: dict[str, set[WebSocket]] = defaultdict(set)


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
    Thread-safe broadcast from a sync context (e.g., transcription callback).
    Uses asyncio.run_coroutine_threadsafe against the running event loop.
    """
    try:
        loop = asyncio.get_running_loop()
        asyncio.run_coroutine_threadsafe(broadcast(session_id, message), loop)
    except RuntimeError:
        pass  # No running event loop — in a test or post-shutdown
