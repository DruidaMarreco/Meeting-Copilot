"""
Audio capture: mic + WASAPI system-audio loopback.

On Windows, pyaudiowpatch exposes loopback devices for each output device.
This lets us capture what's being played back (other participants in a call)
without screen-recording or virtual cables.

Usage:
    python -m audio  # standalone test
    or import and use AudioCapture in your pipeline
"""

import queue
import threading
from collections.abc import Callable

import numpy as np
import pyaudiowpatch as pyaudio

SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono float32
CHUNK_FRAMES = 1024
FORMAT = pyaudio.paInt16


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear-interpolation resample — good enough for speech, zero extra deps."""
    if from_rate == to_rate:
        return audio
    new_len = round(len(audio) * to_rate / from_rate)
    return np.interp(
        np.linspace(0, len(audio) - 1, new_len),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def list_devices() -> list[dict]:
    """Return all audio devices with their indices and names."""
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        devices.append(
            {
                "index": i,
                "name": info["name"],
                "max_input_channels": info["maxInputChannels"],
                "max_output_channels": info["maxOutputChannels"],
                "is_loopback": info.get("isLoopbackDevice", False),
            }
        )
    p.terminate()
    return devices


def find_loopback_device(p: pyaudio.PyAudio) -> dict | None:
    """
    Find the default WASAPI loopback device.
    Falls back to the first loopback device found.
    """
    try:
        default_output = p.get_default_wasapi_loopback()
        if default_output:
            return default_output
    except Exception:
        pass

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("isLoopbackDevice") and info["maxInputChannels"] > 0:
            return info

    return None


class AudioCapture:
    """
    Captures audio from mic and/or system loopback.
    Delivers float32 mono 16 kHz chunks to `callback`.
    """

    def __init__(
        self,
        callback: Callable[[np.ndarray], None],
        use_loopback: bool = True,
        use_mic: bool = True,
    ):
        self.callback = callback
        self.use_loopback = use_loopback
        self.use_mic = use_mic
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._threads: list[threading.Thread] = []
        self._p = pyaudio.PyAudio()

    def _make_callback(self, n_channels: int, native_rate: int):
        """
        Return a per-stream PortAudio callback that normalises to
        float32 mono 16 kHz before enqueuing.

        A single shared _stream_callback cannot do this because
        np.frombuffer always yields a 1-D array — we need to know
        the channel count at callback creation time to reshape correctly.
        """

        def _cb(in_data, frame_count, time_info, status):
            if in_data:
                pcm = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                if n_channels > 1:
                    # Deinterleave [L R L R …] → (frames, channels) → mean → mono
                    pcm = pcm.reshape(-1, n_channels).mean(axis=1)
                if native_rate != SAMPLE_RATE:
                    pcm = _resample(pcm, native_rate, SAMPLE_RATE)
                self._queue.put(pcm)
            return (None, pyaudio.paContinue)

        return _cb

    def _process_queue(self):
        while self._running or not self._queue.empty():
            try:
                chunk = self._queue.get(timeout=0.1)
                self.callback(chunk)
            except queue.Empty:
                continue

    def start(self):
        self._running = True
        streams = []

        if self.use_loopback:
            loopback_dev = find_loopback_device(self._p)
            if loopback_dev is None:
                print("[capture] WARNING: no loopback device found, system audio will be missing")
            else:
                n_ch = min(int(loopback_dev["maxInputChannels"]), 2)
                native_rate = int(loopback_dev["defaultSampleRate"])
                print(
                    f"[capture] Loopback: {loopback_dev['name']} "
                    f"({n_ch}ch @ {native_rate} Hz → resampled to {SAMPLE_RATE} Hz)"
                )
                stream = self._p.open(
                    format=FORMAT,
                    channels=n_ch,
                    rate=native_rate,
                    input=True,
                    input_device_index=int(loopback_dev["index"]),
                    frames_per_buffer=CHUNK_FRAMES,
                    stream_callback=self._make_callback(n_ch, native_rate),
                )
                streams.append(stream)

        if self.use_mic:
            mic_dev = self._p.get_default_input_device_info()
            print(f"[capture] Mic: {mic_dev['name']} (mono @ {SAMPLE_RATE} Hz)")
            stream = self._p.open(
                format=FORMAT,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=int(mic_dev["index"]),
                frames_per_buffer=CHUNK_FRAMES,
                stream_callback=self._make_callback(1, SAMPLE_RATE),
            )
            streams.append(stream)

        for s in streams:
            s.start_stream()

        processor = threading.Thread(target=self._process_queue, daemon=True)
        processor.start()
        self._threads.append(processor)

        print(f"[capture] Started — {len(streams)} stream(s)")
        return streams

    def stop(self):
        self._running = False
        self._p.terminate()
        for t in self._threads:
            t.join(timeout=2)
        print("[capture] Stopped")


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=== Audio Device List ===")
    for d in list_devices():
        flag = " [LOOPBACK]" if d["is_loopback"] else ""
        print(
            f"  [{d['index']}] {d['name']}{flag}  in={d['max_input_channels']} out={d['max_output_channels']}"
        )

    print("\n=== Capturing for 10 seconds (speak + play audio) ===")
    chunks_received = []

    def on_chunk(audio: np.ndarray):
        rms = float(np.sqrt(np.mean(audio**2)))
        chunks_received.append(rms)
        bar = "█" * int(rms * 200)
        print(f"\r  level: {bar:<40} {rms:.4f}", end="", flush=True)

    cap = AudioCapture(callback=on_chunk)
    cap.start()
    time.sleep(10)
    cap.stop()

    print(f"\n\nReceived {len(chunks_received)} chunks.")
    print("✓ M1 gate test — check levels above for both mic and loopback.")
