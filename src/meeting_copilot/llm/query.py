"""
Grounded Q&A using Ollama.

The key constraint: the LLM may ONLY answer from the provided context.
If the answer isn't in the transcript, it must say so — no hallucination.

Pipeline:
  1. Retrieve semantically relevant utterances from Chroma
  2. Also grab the last ~5 minutes as recency context
  3. Deduplicate + sort by time
  4. Build prompt with strict grounding instruction
  5. Stream Ollama response
"""

from __future__ import annotations

from collections.abc import Generator

import ollama

from meeting_copilot.config import OLLAMA_MODEL
from meeting_copilot.storage import db, vector_store

DEFAULT_MODEL = OLLAMA_MODEL
CONTEXT_WINDOW_SECONDS = 300  # last 5 minutes always included
TOP_K_SEMANTIC = 5


SYSTEM_PROMPT = """You are a meeting assistant. You can ONLY answer questions based on the meeting transcript provided below.

Rules:
- If the answer is in the transcript, answer concisely and directly.
- If the answer is NOT in the transcript, respond with exactly: "NOT MENTIONED in this meeting."
- Do not use any outside knowledge.
- Do not speculate or infer beyond what is explicitly stated.
- Keep answers under 3 sentences unless the question requires a list.

Meeting transcript context:
{context}"""


def _build_context(session_id: str, question: str) -> str:
    semantic_hits = vector_store.search(session_id, question, n_results=TOP_K_SEMANTIC)
    semantic_texts = {h["text"] for h in semantic_hits}

    recent = db.get_recent_utterances(session_id, last_n_seconds=CONTEXT_WINDOW_SECONDS)
    recent_texts = [u["text"] for u in recent]

    all_texts = list(recent_texts)
    for text in semantic_texts:
        if text not in set(recent_texts):
            all_texts.append(f"[Earlier in meeting] {text}")

    if not all_texts:
        return "(No transcript available yet)"

    return "\n".join(f"- {t}" for t in all_texts)


def _extract_content(chunk_or_response) -> str:
    """
    Extract the text content from an ollama response or stream chunk.

    ollama SDK >=0.3 returns typed objects (ChatResponse / chunk with .message);
    older versions returned plain dicts. Handle both shapes gracefully.
    """
    msg = getattr(chunk_or_response, "message", None)
    if msg is not None:
        return getattr(msg, "content", "") or ""
    # Fallback: dict-style (SDK <0.3)
    return chunk_or_response.get("message", {}).get("content", "")


def answer(
    session_id: str,
    question: str,
    model: str = DEFAULT_MODEL,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """
    Answer a question grounded in the current meeting's transcript.

    Returns a full string or a generator of text chunks when stream=True.
    """
    context = _build_context(session_id, question)
    system = SYSTEM_PROMPT.format(context=context)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    if stream:

        def _stream_gen():
            for chunk in ollama.chat(model=model, messages=messages, stream=True):
                yield _extract_content(chunk)

        return _stream_gen()
    else:
        return _extract_content(ollama.chat(model=model, messages=messages))


SUMMARY_PROMPT = """You are a meeting assistant. Summarize the following meeting transcript.

Rules:
- Write a concise executive summary (3-5 sentences).
- Then list the key decisions, action items, and topics discussed as bullet points.
- Use only information from the transcript — no speculation.
- If the transcript is empty or too short, say "Not enough content to summarize."

Meeting transcript:
{transcript}"""


def summarize(
    session_id: str,
    model: str = DEFAULT_MODEL,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """Generate a structured summary of the full meeting transcript."""
    utterances = db.get_utterances(session_id)
    if not utterances:
        transcript = "(No transcript available)"
    else:
        lines = [f"[{u['start_time']:.0f}s] {u['text']}" for u in utterances]
        transcript = "\n".join(lines)

    prompt = SUMMARY_PROMPT.format(transcript=transcript)
    messages = [{"role": "user", "content": prompt}]

    if stream:

        def _stream_gen():
            for chunk in ollama.chat(model=model, messages=messages, stream=True):
                yield _extract_content(chunk)

        return _stream_gen()
    else:
        return _extract_content(ollama.chat(model=model, messages=messages))


def check_ollama(model: str = DEFAULT_MODEL) -> bool:
    """Return True if Ollama is running and the requested model is available."""
    try:
        response = ollama.list()
        # SDK >=0.3: ListResponse with .models (list of Model objects, .model attr)
        # SDK <0.3:  dict with "models" key (list of dicts with "name" key)
        if hasattr(response, "models"):
            names = [
                getattr(m, "model", None) or getattr(m, "name", "") or "" for m in response.models
            ]
        else:
            names = [m.get("model", m.get("name", "")) for m in response.get("models", [])]
        return any(model in n for n in names)
    except Exception:
        return False
