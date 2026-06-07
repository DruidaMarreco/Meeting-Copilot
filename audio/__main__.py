"""python -m audio — runs the standalone capture test."""
import time
import numpy as np
from audio.capture import AudioCapture, list_devices

print("=== Audio Device List ===")
for d in list_devices():
    flag = " [LOOPBACK]" if d["is_loopback"] else ""
    print(f"  [{d['index']}] {d['name']}{flag}  in={d['max_input_channels']} out={d['max_output_channels']}")

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
print("✓ M1 gate test complete — check levels above for both mic and loopback.")
