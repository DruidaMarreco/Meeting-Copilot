"""
Stub heavy, platform-specific dependencies before any test module is imported.

This lets the test suite run without GPU, audio hardware, a running Ollama
server, or installing chromadb/faster-whisper/pyaudiowpatch.
"""

import sys
from unittest.mock import MagicMock


def _stub(*names: str) -> None:
    for name in names:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()


_stub(
    "chromadb",
    "pyaudiowpatch",
    "faster_whisper",
    "ollama",
    "sounddevice",
    "numpy",
)
