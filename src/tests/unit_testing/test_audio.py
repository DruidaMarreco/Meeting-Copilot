"""
Unit tests for audio capture helpers.
All pyaudiowpatch / hardware calls are mocked via conftest.py stubs.
"""

import struct

import numpy as np

# ── _resample ─────────────────────────────────────────────────────────────────


def test_resample_noop_when_rates_equal():
    from audio.capture import _resample

    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    out = _resample(audio, 16000, 16000)
    np.testing.assert_array_equal(out, audio)


def test_resample_downsample_length():
    from audio.capture import _resample

    audio = np.ones(44100, dtype=np.float32)
    out = _resample(audio, 44100, 16000)
    assert len(out) == round(44100 * 16000 / 44100)  # == 16000


def test_resample_upsample_length():
    from audio.capture import _resample

    audio = np.ones(16000, dtype=np.float32)
    out = _resample(audio, 16000, 44100)
    assert len(out) == round(16000 * 44100 / 16000)  # == 44100


def test_resample_output_dtype():
    from audio.capture import _resample

    audio = np.ones(100, dtype=np.float32)
    out = _resample(audio, 48000, 16000)
    assert out.dtype == np.float32


# ── _make_callback ────────────────────────────────────────────────────────────


def _make_pcm_bytes(samples: list[int], n_channels: int = 1) -> bytes:
    """Pack int16 samples into bytes (interleaved if stereo)."""
    return struct.pack(f"<{len(samples)}h", *samples)


def test_callback_mono_no_resample(tmp_path):
    """Mono 16 kHz stream: samples pass through unchanged."""
    from unittest.mock import MagicMock

    from audio.capture import AudioCapture

    cap = AudioCapture.__new__(AudioCapture)
    cap._queue = __import__("queue").Queue()

    cap._p = MagicMock()

    received = []
    cap.callback = received.append

    cb = cap._make_callback(n_channels=1, native_rate=16000)

    # 4 mono samples: 0, 16384, -16384, 32767
    raw = _make_pcm_bytes([0, 16384, -16384, 32767])
    cb(raw, 4, None, None)

    chunk = cap._queue.get_nowait()
    expected = np.array([0.0, 0.5, -0.5, 32767 / 32768.0], dtype=np.float32)
    np.testing.assert_allclose(chunk, expected, atol=1e-4)


def test_callback_stereo_mixed_to_mono():
    """Stereo stream: L/R channels are averaged to mono."""
    from audio.capture import AudioCapture

    cap = AudioCapture.__new__(AudioCapture)
    cap._queue = __import__("queue").Queue()
    cap.callback = lambda _: None

    cb = cap._make_callback(n_channels=2, native_rate=16000)

    # 2 stereo frames: [L=16384, R=0] [L=0, R=16384]
    raw = _make_pcm_bytes([16384, 0, 0, 16384])
    cb(raw, 2, None, None)

    chunk = cap._queue.get_nowait()
    # each frame averages to 0.25
    assert chunk.shape == (2,)
    np.testing.assert_allclose(chunk, [0.25, 0.25], atol=1e-4)


def test_callback_resamples_loopback_rate():
    """44100 Hz loopback stream is resampled to 16000 Hz."""
    from audio.capture import AudioCapture

    cap = AudioCapture.__new__(AudioCapture)
    cap._queue = __import__("queue").Queue()
    cap.callback = lambda _: None

    cb = cap._make_callback(n_channels=1, native_rate=44100)

    raw = _make_pcm_bytes([0] * 441)  # 441 samples @ 44100 Hz ≈ 10 ms
    cb(raw, 441, None, None)

    chunk = cap._queue.get_nowait()
    expected_len = round(441 * 16000 / 44100)
    assert len(chunk) == expected_len
