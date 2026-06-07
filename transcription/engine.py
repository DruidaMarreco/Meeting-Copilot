"""
Transcription engine using faster-whisper.

Accumulates audio chunks into a rolling buffer, flushes when silence
is detected (or buffer exceeds max_seconds), and yields TranscriptChunk
objects via a callback.

Model sizes: tiny / base / small / medium / large-v3
Recommended for meetings: small (fast, decent accuracy) or medium (slower, better)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

from config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_LANGUAGE, WHISPER_MODEL_SIZE

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01  # RMS below this = silence
SILENCE_DURATION = 1.0  # seconds of silence before flush
MAX_BUFFER_SECONDS = 30  # force flush after this long regardless
MIN_CHUNK_SECONDS = 0.5  # don't flush tiny snippets


@dataclass
class TranscriptChunk:
    text: str
    start_time: float  # seconds since meeting start
    end_time: float
    speaker: str | None = None  # reserved for future diarization
    confidence: float = 1.0


class TranscriptionEngine:
    """
    Feed audio (float32, mono, 16kHz numpy arrays) via `feed()`.
    Receives TranscriptChunk objects via `on_transcript` callback.
    """

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        on_transcript: Callable[[TranscriptChunk], None] | None = None,
        language: str = WHISPER_LANGUAGE,
    ):
        print(f"[transcription] Loading faster-whisper {model_size} ({device}/{compute_type})…")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.on_transcript = on_transcript
        self.language = language

        self._buffer: list[np.ndarray] = []
        self._buffer_seconds = 0.0
        self._last_sound_time = time.monotonic()
        self._meeting_start = time.monotonic()
        self._lock = threading.Lock()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._running = False
        print("[transcription] Engine ready")

    def start(self):
        self._running = True
        self._meeting_start = time.monotonic()
        self._flush_thread.start()

    def stop(self):
        self._running = False
        self._flush(force=True)

    def feed(self, audio: np.ndarray):
        """Accept a chunk of float32 mono 16kHz audio."""
        with self._lock:
            self._buffer.append(audio)
            self._buffer_seconds += len(audio) / SAMPLE_RATE
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms > SILENCE_THRESHOLD:
                self._last_sound_time = time.monotonic()

    def _flush_loop(self):
        while self._running:
            time.sleep(0.2)
            elapsed_silence = time.monotonic() - self._last_sound_time
            with self._lock:
                buf_secs = self._buffer_seconds

            if buf_secs < MIN_CHUNK_SECONDS:
                continue

            if elapsed_silence >= SILENCE_DURATION or buf_secs >= MAX_BUFFER_SECONDS:
                self._flush()

    def _flush(self, force: bool = False):
        with self._lock:
            if not self._buffer:
                return
            audio = np.concatenate(self._buffer)
            buf_secs = self._buffer_seconds
            self._buffer = []
            self._buffer_seconds = 0.0

        if buf_secs < MIN_CHUNK_SECONDS and not force:
            return

        start_offset = time.monotonic() - self._meeting_start - buf_secs

        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            chunk = TranscriptChunk(
                text=text,
                start_time=start_offset + seg.start,
                end_time=start_offset + seg.end,
                confidence=getattr(seg, "avg_logprob", 1.0),
            )
            if self.on_transcript:
                self.on_transcript(chunk)
