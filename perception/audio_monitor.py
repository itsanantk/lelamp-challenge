"""Lightweight ambient-audio monitor for interruption awareness: don't
fire attention-seeking behavior while the user is talking or the TV's on
nearby.

This is energy-based (RMS level over a rolling window), not speech
recognition or audio scene classification. Deliberately -- the question
the lamp actually needs answered is "is something audio-wise happening
that I shouldn't talk over," not "what is being said" or "is this
specifically a TV vs. a person." A sustained RMS above the ambient floor
covers talking and TV/media audio the same way, which is honestly as far
as interruption awareness needs to go for a desk lamp. Runs on its own
background thread via sounddevice's callback API so it doesn't cost
anything in the main video loop.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd

import config


class AudioActivityMonitor:
    def __init__(self, sustain_s: float = 0.6, threshold: float | None = None):
        self._sustain_s = sustain_s
        self._threshold = threshold if threshold is not None else config.AUDIO_GATE_RMS_THRESHOLD
        self._active_until = 0.0
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status) -> None:
        rms = float(np.sqrt(np.mean(np.square(indata, dtype=np.float64))))
        if rms > self._threshold:
            with self._lock:
                self._active_until = time.monotonic() + self._sustain_s

    def start(self) -> None:
        try:
            self._stream = sd.InputStream(
                samplerate=config.VOICE_SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=int(config.VOICE_SAMPLE_RATE * 0.1), callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            print(f"[audio-gate] couldn't open microphone -- interruption awareness disabled ({e})")
            self._stream = None

    def is_active(self) -> bool:
        """True if there's been sustained audio above the ambient floor
        in roughly the last `sustain_s` seconds -- someone talking, TV
        on, etc. False (harmlessly) if the mic never opened."""
        if self._stream is None:
            return False
        with self._lock:
            return time.monotonic() < self._active_until

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
