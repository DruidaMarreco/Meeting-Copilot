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
import ollama
from typing import Optional, Generator
from storage import db, vector_store
from config import OLLAMA_MODEL


DEFAULT_MODEL = OLLAMA_MODEL
CONTEXT_WINDOW_SECONDS = 300   # last 5 minutes always included
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
    # Semantic retrieval
    semantic_hits = vector_store.search(session_id, question, n_results=TOP_K_SEMANTIC)
    semantic_texts = {h["text"] for h in semantic_hits}

    # Recent context (last 5 minutes)
    recent = db.get_recent_utterances(session_id, last_n_seconds=CONTEXT_WINDOW_SECONDS)
    recent_texts = [u["text"] for u in recent]

    # Combine: recent window first, then any semantic hits not already included
    all_texts = list(recent_texts)
    for text in semantic_texts:
        if text not in set(recent_texts):
            all_texts.append(f"[Earlier in meeting] {text}")

    if not all_texts:
        return "(No transcript available yet)"

    return "\n".join(f"- {t}" for t in all_texts)


def answer(
    session_id: str,
    question: str,
    model: str = DEFAULT_MODEL,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """
    Answer a question grounded in the current meeting's transcript.

    Args:
        session_id: Active meeting session
        question: The question (from PTT transcription)
        model: Ollama model name
        stream: If True, returns a generator of text chunks

    Returns:
        Full answer string, or generator if stream=True
    """
    context = _build_context(session_id, question)
    system = SYSTEM_PROMPT.format(context=context)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    if stream:
        def _stream_gen():
            response = ollama.chat(model=model, messages=messages, stream=True)
            for chunk in response:
                yield chunk["message"]["content"]
        return _stream_gen()
    else:
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]


def check_ollama(model: str = DEFAULT_MODEL) -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        models = ollama.list()
        available = [m["name"] for m in models.get("models", [])]
        return any(model in name for name in available)
    except Exception:
        return False
