"""Unit tests for audio_output.py's shared playback lock. Real sd.play()/
sd.wait() need actual audio hardware (live-verified, same boundary as
conversation/voice.py's own mic I/O) -- these fake both to prove the
locking behavior itself: two concurrent play_and_wait() calls must
serialize, never overlap, which is the whole point of routing
lamp/sim_backend.py's sound cues and conversation/voice.py's TTS through
one shared lock instead of each calling sounddevice directly.
Run with: python -m pytest tests/ -v
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import audio_output


def test_play_and_wait_serializes_two_concurrent_calls(monkeypatch):
    # Real bug, reproduced live: an attention-seek chirp (lamp.play_sound,
    # its own thread) landing while a reminder's voice.speak() was already
    # playing cut the speech off mid-sentence -- unguarded sd.play() stops
    # whatever's currently playing rather than queuing behind it. This
    # proves the shared lock actually prevents that overlap rather than
    # just existing.
    in_progress = threading.Event()
    release_first = threading.Event()
    overlap_detected = threading.Event()
    order = []

    def fake_play(samples, samplerate):
        if in_progress.is_set():
            overlap_detected.set()
        in_progress.set()
        order.append(("play", samplerate))

    def fake_wait():
        # First call waits here for the test to explicitly release it,
        # simulating a real playback still in progress. Second call has
        # nothing to wait on -- it only reaches fake_play at all once the
        # lock is free, by which point the first call has already finished.
        release_first.wait(timeout=1.0)
        in_progress.clear()
        order.append("wait")

    monkeypatch.setattr(audio_output.sd, "play", fake_play)
    monkeypatch.setattr(audio_output.sd, "wait", fake_wait)

    t1 = threading.Thread(target=audio_output.play_and_wait, args=(np.zeros(4), 16000))
    t1.start()
    time.sleep(0.05)  # let t1 acquire the lock and block inside fake_wait

    t2 = threading.Thread(target=audio_output.play_and_wait, args=(np.zeros(4), 22050))
    t2.start()
    time.sleep(0.05)  # t2 must still be blocked on the lock, not inside fake_play yet

    assert order == [("play", 16000)]

    release_first.set()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)

    assert not overlap_detected.is_set()
    assert order == [("play", 16000), "wait", ("play", 22050), "wait"]
