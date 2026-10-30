#!/usr/bin/env python3
"""
Audio Diagnostics & Signal Verification Utility
Lists input devices and monitors the one you choose with a real-time VU meter.

Usage:
  python test_audio.py              # prompt for a device
  python test_audio.py mic          # default microphone
  python test_audio.py blackhole    # BlackHole loopback
  python test_audio.py 2            # PortAudio device index
"""

import sys
import time
import numpy as np
import sounddevice as sd

def print_vu_meter(amplitude: float, peak: float, label: str = "Audio Level", bar_len: int = 40):
    """Renders a real-time responsive visual VU meter bar in the terminal."""
    fill_len = int(bar_len * min(amplitude * 10, 1.0))
    peak_pos = int(bar_len * min(peak * 10, 1.0))

    meter = ["="] * fill_len + [" "] * (bar_len - fill_len)
    if 0 <= peak_pos < bar_len:
        meter[peak_pos] = "|"

    bar = "".join(meter)
    sys.stdout.write(f"\r{label}: [{bar}] {amplitude:.4f} (Peak: {peak:.4f})")
    sys.stdout.flush()

def list_devices():
    """Lists all available input audio devices on macOS."""
    print("\n[SCAN] Scanning Host Audio Devices:")
    devices = sd.query_devices()
    input_devices = []
    blackhole_idx = None

    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append((idx, dev))
            is_blackhole = "BlackHole" in dev['name']
            if is_blackhole:
                blackhole_idx = idx
            star = " * [Virtual Loopback]" if is_blackhole else ""
            default_marker = " (Default Input)" if idx == sd.default.device[0] else ""
            print(f"  [{idx:2d}] {dev['name']}{star}{default_marker}")
            print(f"       Channels: {dev['max_input_channels']} | Sample Rate: {int(dev['default_samplerate'])} Hz")

    if blackhole_idx is not None:
        print(f"\n[OK] BlackHole Loopback Driver found at device index [{blackhole_idx}]")
    else:
        print("\n[INFO] BlackHole 2ch loopback not detected.")
        print("   (To capture both sides of calls, install BlackHole: brew install blackhole-2ch)")

    return input_devices, blackhole_idx

def default_input_index(input_devices):
    """Prefer the system default microphone, then the first input."""
    valid = {idx for idx, _dev in input_devices}
    default = sd.default.device[0]
    if default in valid:
        return default
    if input_devices:
        return input_devices[0][0]
    return None

def resolve_selection(token, input_devices, blackhole_idx):
    """Map a menu or CLI token to an input device index.

    Returns the index, or None when the token should be asked again.
    """
    token = token.strip().lower()
    valid = {idx for idx, _dev in input_devices}
    if token in ("", "mic", "microphone", "default"):
        return default_input_index(input_devices)
    if token in ("b", "bh", "blackhole", "loopback"):
        if blackhole_idx is None:
            print("[INFO] BlackHole is not installed. Choose another device.")
            return None
        return blackhole_idx
    if token in ("q", "quit", "exit"):
        raise SystemExit(0)
    try:
        idx = int(token)
    except ValueError:
        print(f"[INFO] Unknown choice: {token}")
        return None
    if idx not in valid:
        print(f"[INFO] Device [{idx}] is not an input. Choose an index from the list.")
        return None
    return idx

def choose_device(input_devices, blackhole_idx):
    """Use a CLI argument when given, otherwise prompt in a terminal."""
    if not input_devices:
        print("\n[ERROR] No input devices found.")
        raise SystemExit(1)

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        selected = resolve_selection(arg, input_devices, blackhole_idx)
        if selected is None:
            raise SystemExit(1)
        return selected

    if not sys.stdin.isatty():
        selected = default_input_index(input_devices)
        print(f"\n[INFO] No terminal prompt available. Monitoring the default microphone [{selected}].")
        return selected

    default_idx = default_input_index(input_devices)
    print("\n[CHOOSE] Which input should the meter listen to?")
    print(f"         Press Enter for the default microphone [{default_idx}].")
    if blackhole_idx is not None:
        print(f"         Type blackhole for the loopback driver [{blackhole_idx}].")
    print("         Or type a device index from the list above. q quits.")
    while True:
        try:
            token = input("Selection: ")
        except EOFError:
            print()
            return default_idx
        selected = resolve_selection(token, input_devices, blackhole_idx)
        if selected is not None:
            return selected

def run_diagnostics(target_device: int = None, duration_seconds: int = 10):
    """Captures audio stream and displays real-time signal monitor."""
    devices = sd.query_devices()
    if target_device is None:
        target_device = sd.default.device[0]
        if target_device is None or target_device < 0:
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    target_device = idx
                    break

    dev_info = sd.query_devices(target_device)
    samplerate = int(dev_info.get('default_samplerate', 16000))

    print("\n" + "=" * 60)
    print(f"[AUDIO] Monitoring Audio Device [{target_device}]: {dev_info['name']}")
    print(f"    Frequency: {samplerate} Hz | Blocksize: 250ms | Duration: {duration_seconds}s")
    print("    Speak into your mic or play sound on your computer...")
    print("=" * 60 + "\n")

    peak_amplitude = 0.0
    start_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        nonlocal peak_amplitude
        if status:
            sys.stderr.write(f"\n[Status]: {status}\n")
        rms = float(np.sqrt(np.mean(indata**2)))
        peak = float(np.max(np.abs(indata)))
        if peak > peak_amplitude:
            peak_amplitude = peak
        print_vu_meter(rms, peak_amplitude)

    try:
        with sd.InputStream(
            device=target_device,
            channels=1,
            samplerate=samplerate,
            callback=audio_callback,
            blocksize=int(samplerate * 0.25)
        ):
            while time.time() - start_time < duration_seconds:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n[STOP] Monitor stopped by user.")
    except Exception as e:
        print(f"\n\n[ERROR] Audio Stream Error: {e}")
        return

    print("\n\n" + "=" * 60)
    if peak_amplitude < 0.005:
        print("[WARN] Warning: Very low or zero audio detected!")
        print("   If using a physical mic, verify macOS Terminal permissions:")
        print("   System Settings -> Privacy & Security -> Microphone -> Terminal [ON]")
    else:
        print(f"[PASSED] Verification Complete! Peak Signal Amplitude: {peak_amplitude:.4f}")
        print("   Audio pipeline is functional and ready for live transcription.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        raise SystemExit(0)
    inputs, bh_idx = list_devices()
    selected_dev = choose_device(inputs, bh_idx)
    run_diagnostics(target_device=selected_dev, duration_seconds=8)
