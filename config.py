"""
Central configuration — read from environment variables, fall back to defaults.

Override any value by setting the env var before starting the server:
  WHISPER_MODEL_SIZE=medium uvicorn api.main:app
  OLLAMA_MODEL=mistral uvicorn api.main:app
"""

import os

# ── Whisper ───────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "en")

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

# ── Server ────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
