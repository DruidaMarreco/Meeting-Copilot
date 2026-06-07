"""
Mass testing fixtures.

These tests run against ground truth datasets to measure model performance
at scale — transcription accuracy, Q&A correctness, latency regressions, etc.

Fixture pattern:
  - Datasets live outside the repo (or under mass_testing/datasets/ if small)
  - Reference them via the DATASET_PATH env var or the `dataset_path` fixture
  - Mass tests are gated behind the `metrics` CI workflow (manual trigger only)

Environment variables:
  DATASET_PATH   — filesystem path to the ground truth dataset directory
  DATASET_FORMAT — "json" | "csv" (default: "json")
"""

import os
from pathlib import Path
import pytest


@pytest.fixture
def dataset_path() -> Path:
    raw = os.getenv("DATASET_PATH")
    if not raw:
        pytest.skip("DATASET_PATH not set — skipping mass test")
    p = Path(raw)
    if not p.exists():
        pytest.skip(f"Dataset path not found: {p}")
    return p
