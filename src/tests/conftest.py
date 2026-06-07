"""
Stub heavy, platform-specific dependencies before any test module is imported.

Applies to all test suites (unit_testing, integration_testing, mass_testing).
This lets the suites run without GPU, audio hardware, a running Ollama server,
or the full chromadb/faster-whisper install.
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
