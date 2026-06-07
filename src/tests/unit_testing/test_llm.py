"""
Tests for llm.query — covers both the legacy dict-style and the modern
typed-object ollama SDK response shapes.
"""

from types import SimpleNamespace
from unittest.mock import patch

# ── _extract_content ──────────────────────────────────────────────────────────


def test_extract_content_object_style():
    """SDK >=0.3: response.message.content"""
    from llm.query import _extract_content

    response = SimpleNamespace(message=SimpleNamespace(content="hello"))
    assert _extract_content(response) == "hello"


def test_extract_content_dict_style():
    """SDK <0.3: response["message"]["content"]"""
    from llm.query import _extract_content

    response = {"message": {"content": "world"}}
    assert _extract_content(response) == "world"


def test_extract_content_empty_object():
    from llm.query import _extract_content

    response = SimpleNamespace(message=SimpleNamespace(content=None))
    assert _extract_content(response) == ""


# ── check_ollama ──────────────────────────────────────────────────────────────


def _make_model_obj(name: str):
    """Simulate a Model object as returned by SDK >=0.3."""
    return SimpleNamespace(model=name, name=name)


def test_check_ollama_object_style_found():
    from llm.query import check_ollama

    response = SimpleNamespace(models=[_make_model_obj("llama3:latest")])
    with patch("llm.query.ollama.list", return_value=response):
        assert check_ollama("llama3") is True


def test_check_ollama_object_style_not_found():
    from llm.query import check_ollama

    response = SimpleNamespace(models=[_make_model_obj("mistral:latest")])
    with patch("llm.query.ollama.list", return_value=response):
        assert check_ollama("llama3") is False


def test_check_ollama_dict_style_found():
    from llm.query import check_ollama

    response = {"models": [{"name": "llama3:latest"}]}
    with patch("llm.query.ollama.list", return_value=response):
        assert check_ollama("llama3") is True


def test_check_ollama_returns_false_on_exception():
    from llm.query import check_ollama

    with patch("llm.query.ollama.list", side_effect=ConnectionError):
        assert check_ollama("llama3") is False


# ── answer ────────────────────────────────────────────────────────────────────


def test_answer_non_stream(tmp_path, monkeypatch):
    import storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()

    response_obj = SimpleNamespace(message=SimpleNamespace(content="The budget is $50k."))
    with (
        patch("llm.query.vector_store.search", return_value=[]),
        patch("llm.query.ollama.chat", return_value=response_obj),
    ):
        from llm.query import answer

        result = answer(sid, "What is the budget?")

    assert result == "The budget is $50k."


def test_answer_stream(tmp_path, monkeypatch):
    import storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()

    chunks = [
        SimpleNamespace(message=SimpleNamespace(content="The ")),
        SimpleNamespace(message=SimpleNamespace(content="answer ")),
        SimpleNamespace(message=SimpleNamespace(content="is 42.")),
    ]
    with (
        patch("llm.query.vector_store.search", return_value=[]),
        patch("llm.query.ollama.chat", return_value=iter(chunks)),
    ):
        from llm.query import answer

        result = "".join(answer(sid, "What is the answer?", stream=True))

    assert result == "The answer is 42."


# ── summarize ─────────────────────────────────────────────────────────────────


def test_summarize_non_stream(tmp_path, monkeypatch):
    import storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()
    db_module.save_utterance(sid, "We agreed on a Q3 launch.", 0.0, 3.0)

    response_obj = SimpleNamespace(message=SimpleNamespace(content="Key decision: Q3 launch."))
    with patch("llm.query.ollama.chat", return_value=response_obj):
        from llm.query import summarize

        result = summarize(sid)

    assert "Q3" in result


def test_summarize_stream(tmp_path, monkeypatch):
    import storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()

    chunks = [
        SimpleNamespace(message=SimpleNamespace(content="Summary: ")),
        SimpleNamespace(message=SimpleNamespace(content="short meeting.")),
    ]
    with patch("llm.query.ollama.chat", return_value=iter(chunks)):
        from llm.query import summarize

        result = "".join(summarize(sid, stream=True))

    assert result == "Summary: short meeting."


def test_summarize_empty_session(tmp_path, monkeypatch):
    """Summarizing a session with no utterances should still call ollama (with empty transcript note)."""
    import storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()

    response_obj = SimpleNamespace(
        message=SimpleNamespace(content="Not enough content to summarize.")
    )
    with patch("llm.query.ollama.chat", return_value=response_obj) as mock_chat:
        from llm.query import summarize

        result = summarize(sid)

    assert mock_chat.called
    assert "Not enough" in result
