# Meeting Copilot — Project Brief

**One sentence:** A local-first, privacy-preserving meeting copilot that transcribes your calls in real time and answers questions grounded only in what was actually said.

---

## Problem

Fireflies.ai requires cloud upload of your meetings. You lose control, privacy, and offline access. There's no good offline-first alternative.

## Solution

A desktop app that:
1. Captures mic + system audio (i.e., other participants) via WASAPI loopback
2. Transcribes in real time using faster-whisper running locally
3. Lets you push-to-talk during a meeting to ask questions — answered by a local LLM (Ollama) using only the current meeting's transcript
4. Stores everything locally in SQLite + Chroma
5. Shows a live overlay with transcript and answers

Cloud is optional sync, not a dependency. Phase 2 evolves this into a broader personal AI assistant.

---

## Architecture

```
[Mic + System Audio]
        ↓
  WASAPI Loopback (audio/capture.py)
        ↓
  faster-whisper (transcription/engine.py)
        ↓
  SQLite (storage/db.py) + Chroma (storage/vector_store.py)
        ↓
  FastAPI Backend (api/main.py)
        ↓
  Frontend Overlay (frontend/index.html)
        ↑
  PTT button → Ollama Q&A (llm/query.py)
```

---

## Milestones

### M1: Transcript Spine (Gate: works in a real call)
- Desktop captures mic + system audio
- Streaming transcription with faster-whisper
- Per-session SQLite storage
- Live transcript in UI
- **No AI yet** — just accurate capture + display

### M2: Grounded Q&A
- PTT button → transcribe question → retrieve from current meeting
- Ollama answers with "say NOT MENTIONED if absent" rule
- Answer overlay in UI

### Phase 2+ (parked, not for POC)
- Cloud sync / multi-device
- Agent daemon
- Mobile thin client
- Voice-first OS features

---

## Key Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Platform | Desktop first | Phone can't run Whisper/Ollama |
| Audio capture | WASAPI loopback | Captures other participants, not just mic |
| Transcription | faster-whisper small/medium | Better accuracy than base Whisper |
| Question detection | None (PTT = question) | Heuristics misfire; PTT is explicit signal |
| Sync | Skip in POC | Single device for now |
| Storage | SQLite + Chroma | Simple, local, no dependencies |

---

## Go/No-Go Gate for M1

Run `python -m audio.capture` during a real Zoom/Meet call. If WASAPI loopback picks up the other participants — proceed. If not, debug loopback device enumeration before anything else.

---

## Stack

- Python 3.11+
- FastAPI + uvicorn
- faster-whisper
- Ollama (llama3 or equivalent)
- SQLite (built-in) + chromadb
- sounddevice / pyaudiowpatch (WASAPI loopback)
- Vanilla JS frontend (single file, no build step)
