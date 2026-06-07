"""
Integration test fixtures.

These tests run against singular real examples (not mass datasets) and
are intended for hot-fix validation — confirming a specific bug is fixed
against the concrete input that triggered it.

Fixture pattern:
  - Place example audio/transcript files under integration_testing/examples/
  - Reference them via the `examples_dir` fixture below
  - Integration tests may require Ollama to be running locally
"""

from pathlib import Path
import pytest


EXAMPLES_DIR = Path(__file__).parent / "examples"


@pytest.fixture
def examples_dir() -> Path:
    """Path to singular example inputs for integration tests."""
    return EXAMPLES_DIR
