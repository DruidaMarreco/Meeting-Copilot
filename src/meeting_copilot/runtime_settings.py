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

_DEFAULT_QA_PROMPT = """You are a meeting assistant. You can ONLY answer questions based on the meeting transcript provided below.

Rules:
- If the answer is in the transcript, answer concisely and directly.
- If the answer is NOT in the transcript, respond with exactly: "NOT MENTIONED in this meeting."
- Do not use any outside knowledge.
- Do not speculate or infer beyond what is explicitly stated.
- Keep answers under 3 sentences unless the question requires a list.

Meeting transcript context:
{context}"""

_DEFAULT_SUMMARY_PROMPT = """You are a meeting assistant. Summarize the following meeting transcript.

Rules:
- Write a concise executive summary (3-5 sentences).
- Then list the key decisions, action items, and topics discussed as bullet points.
- Use only information from the transcript — no speculation.
- If the transcript is empty or too short, say "Not enough content to summarize."

Meeting transcript:
{transcript}"""

_DEFAULT_ACTION_ITEMS_PROMPT = """You are a meeting assistant. Extract all action items from the following meeting transcript.

Rules:
- List ONLY concrete tasks, commitments, or follow-ups that were explicitly mentioned.
- Format each item as: "- [ ] <action> (owner: <name or 'unassigned'>, due: <date or 'unspecified'>)"
- If no action items were mentioned, respond with exactly: "No action items found."
- Do not invent tasks that were not stated. Use only what is in the transcript.

Meeting transcript:
{transcript}"""

_lock = threading.Lock()

_state: dict[str, str] = {
    "ollama_model": OLLAMA_MODEL,
    "whisper_model_size": WHISPER_MODEL_SIZE,
    "whisper_language": WHISPER_LANGUAGE,
    "system_prompt_qa": _DEFAULT_QA_PROMPT,
    "system_prompt_summary": _DEFAULT_SUMMARY_PROMPT,
    "system_prompt_action_items": _DEFAULT_ACTION_ITEMS_PROMPT,
}

_PATCHABLE = {
    "ollama_model",
    "whisper_language",
    "system_prompt_qa",
    "system_prompt_summary",
    "system_prompt_action_items",
}
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


def get_system_prompt_qa() -> str:
    with _lock:
        return _state.get("system_prompt_qa", _DEFAULT_QA_PROMPT)


def get_system_prompt_summary() -> str:
    with _lock:
        return _state.get("system_prompt_summary", _DEFAULT_SUMMARY_PROMPT)


def get_system_prompt_action_items() -> str:
    with _lock:
        return _state.get("system_prompt_action_items", _DEFAULT_ACTION_ITEMS_PROMPT)
