"""
Tests for llm.query — covers both the legacy dict-style and the modern
typed-object ollama SDK response shapes.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ── _extract_content ──────────────────────────────────────────────────────────


def test_extract_content_object_style():
    """SDK >=0.3: response.message.content"""
    from meeting_copilot.llm.query import _extract_content

    response = SimpleNamespace(message=SimpleNamespace(content="hello"))
    assert _extract_content(response) == "hello"


def test_extract_content_dict_style():
    """SDK <0.3: response["message"]["content"]"""
    from meeting_copilot.llm.query import _extract_content

    response = {"message": {"content": "world"}}
    assert _extract_content(response) == "world"


def test_extract_content_empty_object():
    from meeting_copilot.llm.query import _extract_content

    response = SimpleNamespace(message=SimpleNamespace(content=None))
    assert _extract_content(response) == ""


# ── check_ollama ──────────────────────────────────────────────────────────────


def _make_model_obj(name: str):
    """Simulate a Model object as returned by SDK >=0.3."""
    return SimpleNamespace(model=name, name=name)


def _mock_ollama(**kwargs):
    """Return a MagicMock that looks like the ollama module with given attr overrides."""
    m = MagicMock()
    for k, v in kwargs.items():
        setattr(m, k, v)
    return m


def test_check_ollama_object_style_found():
    from meeting_copilot.llm.query import check_ollama

    response = SimpleNamespace(models=[_make_model_obj("llama3:latest")])
    with patch(
        "meeting_copilot.llm.query._ollama", return_value=_mock_ollama(list=lambda: response)
    ):
        assert check_ollama("llama3") is True


def test_check_ollama_object_style_not_found():
    from meeting_copilot.llm.query import check_ollama

    response = SimpleNamespace(models=[_make_model_obj("mistral:latest")])
    with patch(
        "meeting_copilot.llm.query._ollama", return_value=_mock_ollama(list=lambda: response)
    ):
        assert check_ollama("llama3") is False


def test_check_ollama_dict_style_found():
    from meeting_copilot.llm.query import check_ollama

    response = {"models": [{"name": "llama3:latest"}]}
    with patch(
        "meeting_copilot.llm.query._ollama", return_value=_mock_ollama(list=lambda: response)
    ):
        assert check_ollama("llama3") is True


def test_check_ollama_returns_false_on_exception():
    from meeting_copilot.llm.query import check_ollama

    def _raise():
        raise ConnectionError

    with patch("meeting_copilot.llm.query._ollama", return_value=_mock_ollama(list=_raise)):
        assert check_ollama("llama3") is False


# ── answer ────────────────────────────────────────────────────────────────────


def test_answer_non_stream(tmp_path, monkeypatch):
    import meeting_copilot.storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()

    response_obj = SimpleNamespace(message=SimpleNamespace(content="The budget is $50k."))
    mock_mod = _mock_ollama(chat=MagicMock(return_value=response_obj))
    with (
        patch("meeting_copilot.llm.query.vector_store.search", return_value=[]),
        patch("meeting_copilot.llm.query._ollama", return_value=mock_mod),
    ):
        from meeting_copilot.llm.query import answer

        result = answer(sid, "What is the budget?")

    assert result == "The budget is $50k."


def test_answer_stream(tmp_path, monkeypatch):
    import meeting_copilot.storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()

    chunks = [
        SimpleNamespace(message=SimpleNamespace(content="The ")),
        SimpleNamespace(message=SimpleNamespace(content="answer ")),
        SimpleNamespace(message=SimpleNamespace(content="is 42.")),
    ]
    mock_mod = _mock_ollama(chat=MagicMock(return_value=iter(chunks)))
    with (
        patch("meeting_copilot.llm.query.vector_store.search", return_value=[]),
        patch("meeting_copilot.llm.query._ollama", return_value=mock_mod),
    ):
        from meeting_copilot.llm.query import answer

        result = "".join(answer(sid, "What is the answer?", stream=True))

    assert result == "The answer is 42."


# ── summarize ─────────────────────────────────────────────────────────────────


def test_summarize_non_stream(tmp_path, monkeypatch):
    import meeting_copilot.storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()
    db_module.save_utterance(sid, "We agreed on a Q3 launch.", 0.0, 3.0)

    response_obj = SimpleNamespace(message=SimpleNamespace(content="Key decision: Q3 launch."))
    mock_mod = _mock_ollama(chat=MagicMock(return_value=response_obj))
    with patch("meeting_copilot.llm.query._ollama", return_value=mock_mod):
        from meeting_copilot.llm.query import summarize

        result = summarize(sid)

    assert "Q3" in result


def test_summarize_stream(tmp_path, monkeypatch):
    import meeting_copilot.storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()

    chunks = [
        SimpleNamespace(message=SimpleNamespace(content="Summary: ")),
        SimpleNamespace(message=SimpleNamespace(content="short meeting.")),
    ]
    mock_mod = _mock_ollama(chat=MagicMock(return_value=iter(chunks)))
    with patch("meeting_copilot.llm.query._ollama", return_value=mock_mod):
        from meeting_copilot.llm.query import summarize

        result = "".join(summarize(sid, stream=True))

    assert result == "Summary: short meeting."


def test_extract_action_items_non_stream(tmp_path, monkeypatch):
    import meeting_copilot.storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()
    db_module.save_utterance(sid, "Alice will send the report by Friday.", 0.0, 3.0)

    response_obj = SimpleNamespace(
        message=SimpleNamespace(content="- [ ] Send report (owner: Alice, due: Friday)")
    )
    mock_mod = _mock_ollama(chat=MagicMock(return_value=response_obj))
    with patch("meeting_copilot.llm.query._ollama", return_value=mock_mod):
        from meeting_copilot.llm.query import extract_action_items

        result = extract_action_items(sid)

    assert "Alice" in result


def test_extract_action_items_stream(tmp_path, monkeypatch):
    import meeting_copilot.storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()

    chunks = [
        SimpleNamespace(message=SimpleNamespace(content="- [ ] ")),
        SimpleNamespace(message=SimpleNamespace(content="Follow up.")),
    ]
    mock_mod = _mock_ollama(chat=MagicMock(return_value=iter(chunks)))
    with patch("meeting_copilot.llm.query._ollama", return_value=mock_mod):
        from meeting_copilot.llm.query import extract_action_items

        result = "".join(extract_action_items(sid, stream=True))

    assert result == "- [ ] Follow up."


def test_generate_title_returns_string(tmp_path, monkeypatch):
    import meeting_copilot.storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()
    db_module.save_utterance(sid, "We discussed the Q3 product roadmap.", 0.0, 3.0)

    response_obj = SimpleNamespace(message=SimpleNamespace(content="Q3 Product Roadmap Planning"))
    mock_mod = _mock_ollama(chat=MagicMock(return_value=response_obj))
    with patch("meeting_copilot.llm.query._ollama", return_value=mock_mod):
        from meeting_copilot.llm.query import generate_title

        result = generate_title(sid)

    assert result == "Q3 Product Roadmap Planning"


def test_generate_title_empty_session_returns_fallback(tmp_path, monkeypatch):
    import meeting_copilot.storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()

    # No utterances → should return fallback without calling ollama
    from meeting_copilot.llm.query import generate_title

    result = generate_title(sid)
    assert result == "Untitled Meeting"


def test_generate_title_strips_quotes(tmp_path, monkeypatch):
    import meeting_copilot.storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()
    db_module.save_utterance(sid, "Budget review meeting.", 0.0, 2.0)

    response_obj = SimpleNamespace(message=SimpleNamespace(content='"Budget Review"'))
    mock_mod = _mock_ollama(chat=MagicMock(return_value=response_obj))
    with patch("meeting_copilot.llm.query._ollama", return_value=mock_mod):
        from meeting_copilot.llm.query import generate_title

        result = generate_title(sid)

    assert result == "Budget Review"


def test_summarize_empty_session(tmp_path, monkeypatch):
    """Summarizing a session with no utterances should still call ollama (with empty transcript note)."""
    import meeting_copilot.storage.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    db_module.init_db()
    sid, _ = db_module.create_session()

    response_obj = SimpleNamespace(
        message=SimpleNamespace(content="Not enough content to summarize.")
    )
    mock_chat = MagicMock(return_value=response_obj)
    mock_mod = _mock_ollama(chat=mock_chat)
    with patch("meeting_copilot.llm.query._ollama", return_value=mock_mod):
        from meeting_copilot.llm.query import summarize

        result = summarize(sid)

    assert mock_chat.called
    assert "Not enough" in result
