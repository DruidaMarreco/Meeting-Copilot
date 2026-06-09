"""
Mutable runtime settings — start from env/config defaults, patchable via the API.

The LLM and transcription layers read from here so that changes take effect on
the next request without a server restart.  Whisper device and compute_type are
excluded from runtime patching because they require re-loading the model.
"""

from __future__ import annotations

import threading

from meeting_copilot.config import (
    OLLAMA_MODEL,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_SIZE,
)

_lock = threading.Lock()

_state: dict[str, str] = {
    "ollama_model": OLLAMA_MODEL,
    "whisper_model_size": WHISPER_MODEL_SIZE,
    "whisper_language": WHISPER_LANGUAGE,
}

_PATCHABLE = {"ollama_model", "whisper_language"}
_ALL_KEYS = set(_state)


def get() -> dict[str, str]:
    with _lock:
        return dict(_state)


def patch(updates: dict[str, str]) -> dict[str, str]:
    """Apply updates for patchable keys; return the full new state."""
    unknown = set(updates) - _ALL_KEYS
    if unknown:
        raise ValueError(f"Unknown settings: {unknown}")
    readonly = set(updates) - _PATCHABLE
    if readonly:
        raise ValueError(f"Read-only settings (require restart): {readonly}")
    with _lock:
        _state.update(updates)
        return dict(_state)


def get_ollama_model() -> str:
    with _lock:
        return _state["ollama_model"]


def get_whisper_language() -> str:
    with _lock:
        return _state["whisper_language"]
