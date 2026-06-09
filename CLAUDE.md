# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dev dependencies
uv sync --extra dev

# Install all dependencies including heavy audio/ML deps
uv sync --extra full

# Run all tests
uv run pytest src/tests/

# Run a single test
uv run pytest src/tests/unit_testing/test_storage.py::test_create_session_returns_id

# Lint
uv run ruff check .

# Format
uv run black .

# Type check
uv run ty check src

# Start the API server
uv run uvicorn meeting_copilot.api.main:app --reload

# Start with custom port / host
uv run uvicorn meeting_copilot.api.main:app --host 0.0.0.0 --port 8000
```

CI order: ruff → black → ty → pytest. Lint/typecheck failures block tests.

## Architecture

```
src/meeting_copilot/
├── api/
│   ├── main.py          # FastAPI app, all HTTP + WebSocket endpoints
│   └── ws.py            # WebSocketManager: broadcast, heartbeat
├── audio/
│   └── capture.py       # AudioCapture (WASAPI loopback on Windows, mic elsewhere)
├── llm/
│   └── query.py         # answer(), summarize(), extract_action_items() via Ollama
├── storage/
│   ├── db.py            # SQLite CRUD (sessions, utterances, answers)
│   └── vector_store.py  # ChromaDB wrapper for semantic utterance search
├── transcription/
│   └── engine.py        # TranscriptionEngine wrapping faster-whisper
└── runtime_settings.py  # Thread-safe mutable config (ollama_model, whisper_language)
```

**Data flow (live session):**
`AudioCapture` → audio chunks → `TranscriptionEngine` → utterances → saved to SQLite + ChromaDB → broadcast over WebSocket → frontend renders transcript

**WebSocket broadcast pattern:**
Worker threads call `asyncio.run_coroutine_threadsafe(manager.broadcast(...), loop)` where `loop` is the event loop captured at startup. The `ws.py` `WebSocketManager` holds all active connections and fans out messages.

**Heavy deps are optional/lazy:** `faster-whisper`, `chromadb`, `sounddevice`, and `pyaudio` are imported only inside functions that need them. The `--extra dev` install is enough to run tests; `--extra full` is needed to actually transcribe audio.

## Storage

`init_db()` is idempotent — it uses `PRAGMA table_info` + `ALTER TABLE` to add new columns to existing databases without losing data. Always follow this pattern when adding columns.

Sessions table key columns: `id` (UUID), `title`, `started_at`, `ended_at`, `summary`, `notes`, `action_items`.

## Testing patterns

- **DB isolation:** `isolated_db` fixture in `test_storage.py` monkeypatches `db_module.DB_PATH` to a `tmp_path` file and calls `init_db()`.
- **API isolation:** `test_api.py` uses `TestClient` with a `test_app` fixture that overrides `DB_PATH` and calls `init_db()`.
- **Settings isolation:** `reset_runtime_settings` autouse fixture saves and restores `runtime_settings._state` around each test.
- **Heavy deps:** `AudioCapture` and `TranscriptionEngine` are mocked in API tests — never import them directly in tests.

## Key constraints

- WASAPI loopback (system audio capture) is Windows-only; the app falls back to microphone on other platforms.
- Ollama must be running locally for LLM features; `check_ollama()` in `llm/query.py` probes availability.
- `runtime_settings.py` distinguishes patchable keys (`ollama_model`, `whisper_language`) from read-only ones (`whisper_model_size`) — the API returns 422 if a client tries to patch a read-only key.
- The frontend (`frontend/index.html`) is a single self-contained HTML file served as a static file by FastAPI.
