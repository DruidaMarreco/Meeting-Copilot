"""Tests for runtime_settings — thread-safe mutable config."""

import pytest

import meeting_copilot.runtime_settings as rs


@pytest.fixture(autouse=True)
def reset_settings():
    """Restore default state after each test."""
    original = rs.get()
    yield
    rs._state.update(original)


def test_get_returns_all_keys():
    s = rs.get()
    assert "ollama_model" in s
    assert "whisper_language" in s
    assert "whisper_model_size" in s


def test_patch_updates_patchable_key():
    rs.patch({"ollama_model": "mistral"})
    assert rs.get_ollama_model() == "mistral"


def test_patch_updates_whisper_language():
    rs.patch({"whisper_language": "fr"})
    assert rs.get_whisper_language() == "fr"


def test_patch_readonly_key_raises():
    with pytest.raises(ValueError, match="Read-only"):
        rs.patch({"whisper_model_size": "medium"})


def test_patch_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown"):
        rs.patch({"no_such_key": "x"})


def test_patch_returns_full_state():
    result = rs.patch({"ollama_model": "llama3.1"})
    assert result["ollama_model"] == "llama3.1"
    assert "whisper_language" in result


def test_get_returns_copy():
    """Mutating the returned dict must not affect internal state."""
    s = rs.get()
    s["ollama_model"] = "tampered"
    assert rs.get_ollama_model() != "tampered"
