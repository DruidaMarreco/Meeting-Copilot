# Meeting Copilot

Offline-first meeting transcription + Q&A. Runs entirely on your machine.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Ollama and pull a model
# https://ollama.ai
ollama pull llama3

# 3. Start the server
uvicorn meeting_copilot.api.main:app --reload --port 8000

# 4. Open the UI
# Open frontend/index.html in your browser (or serve via the API)
```

## M1 Gate Test

Before anything else, verify loopback capture works during a real call:

```bash
python -m meeting_copilot.audio.capture
```

You should see your microphone AND the other participants' audio being captured. If only mic shows — check `audio/capture.py` loopback device enumeration.

## Project Structure

```
meeting-copilot/
├── api/
│   ├── main.py          # FastAPI app, meeting lifecycle endpoints
│   └── ws.py            # WebSocket for live transcript streaming
├── audio/
│   ├── capture.py       # WASAPI loopback + mic capture
│   └── __main__.py      # Standalone test runner
├── transcription/
│   └── engine.py        # faster-whisper wrapper, streaming chunks
├── storage/
│   ├── db.py            # SQLite: sessions, utterances
│   └── vector_store.py  # Chroma: per-session embeddings
├── llm/
│   └── query.py         # Ollama integration, grounded Q&A
├── frontend/
│   └── index.html       # Single-file UI: live transcript + PTT
├── requirements.txt
└── PROJECT_BRIEF.md
```

## Requirements

- Windows 10/11 (WASAPI loopback)
- Python 3.11+
- Ollama running locally
- ~4GB RAM for faster-whisper small model
