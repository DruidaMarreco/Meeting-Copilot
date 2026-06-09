"""
Chroma vector store for meeting-scoped semantic retrieval.

Each meeting session gets its own Chroma collection.
Utterances are embedded and stored; at query time we retrieve
the top-k most semantically relevant chunks.
"""

from __future__ import annotations

from pathlib import Path

CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma"


def _chromadb():
    """Lazy-import chromadb so the server starts without it installed."""
    try:
        import chromadb as _chroma  # noqa: PLC0415

        return _chroma
    except ImportError as exc:
        raise RuntimeError("chromadb is not installed. Run: uv sync --extra full") from exc


def _client():
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return _chromadb().PersistentClient(path=str(CHROMA_PATH))


def _collection_name(session_id: str) -> str:
    # Chroma collection names: alphanumeric + hyphens, 3-63 chars
    return f"session-{session_id[:8]}"


def add_utterance(session_id: str, utterance_id: str, text: str, metadata: dict | None = None):
    """Embed and store an utterance in the session's collection."""
    client = _client()
    col = client.get_or_create_collection(name=_collection_name(session_id))
    col.add(
        ids=[utterance_id],
        documents=[text],
        metadatas=[metadata or {}],
    )


def search(session_id: str, query: str, n_results: int = 5) -> list[dict]:
    """
    Semantic search within a single session's collection.
    Returns list of {text, metadata, distance} dicts.
    """
    client = _client()
    try:
        col = client.get_collection(name=_collection_name(session_id))
    except Exception:
        return []

    count = col.count()
    if count == 0:
        return []

    results = col.query(
        query_texts=[query],
        n_results=min(n_results, count),
    )

    hits = []
    for i, doc in enumerate(results["documents"][0]):
        hits.append(
            {
                "text": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return hits


def delete_session(session_id: str):
    """Remove a session's collection entirely."""
    client = _client()
    try:
        client.delete_collection(_collection_name(session_id))
    except Exception:
        pass
