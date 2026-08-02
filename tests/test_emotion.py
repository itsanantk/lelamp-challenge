"""Unit tests for conversation/emotion.py's heuristic tone classification.
Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from conversation import emotion


def test_loud_fast_variable_pitch_classifies_as_energetic():
    assert emotion._classify(rms=0.017, pitch_var=25.0, rate=4.0) == "energetic"


def test_loud_fast_flat_pitch_classifies_as_tense():
    assert emotion._classify(rms=0.017, pitch_var=5.0, rate=4.0) == "tense"


def test_quiet_overrides_everything_else():
    assert emotion._classify(rms=0.001, pitch_var=25.0, rate=4.0) == "quiet"


def test_moderate_speech_classifies_as_calm():
    assert emotion._classify(rms=0.01, pitch_var=5.0, rate=2.0) == "calm"


def test_loud_but_not_fast_still_registers_as_tense():
    # A short, clipped, frustrated sentence doesn't reliably hit "fast" the
    # way a longer excited one does -- the old AND-gate required loud AND
    # fast simultaneously, so a clearly raised voice at a normal pace
    # silently fell through to "calm". Scored as an average instead: a
    # strong signal in just one dimension is now enough to tip out of calm.
    assert emotion._classify(rms=0.02, pitch_var=5.0, rate=1.5) == "tense"


def test_fast_but_not_loud_still_registers_as_energetic():
    assert emotion._classify(rms=0.008, pitch_var=25.0, rate=5.0) == "energetic"


def test_analyze_uses_peak_rms_not_a_flat_whole_clip_average():
    # A brief loud burst in an otherwise-quiet clip: the old flat-average
    # rms would dilute this below _LOUD_RMS on every real utterance
    # (that was the actual bug -- "energetic"/"tense" were unreachable).
    # Peak-based measurement should reflect the burst instead.
    samplerate = 16000
    quiet = np.zeros(int(2.0 * samplerate), dtype="float32")
    burst = (0.05 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.3, int(0.3 * samplerate)))).astype("float32")
    audio = np.concatenate([quiet, burst])
    tone = emotion.analyze(audio, sample_rate=samplerate)
    flat_avg = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    assert tone.rms > flat_avg * 2


if __name__ == "__main__":
    test_loud_fast_variable_pitch_classifies_as_energetic()
    test_loud_fast_flat_pitch_classifies_as_tense()
    test_quiet_overrides_everything_else()
    test_moderate_speech_classifies_as_calm()
    test_loud_but_not_fast_still_registers_as_tense()
    test_fast_but_not_loud_still_registers_as_energetic()
    test_analyze_uses_peak_rms_not_a_flat_whole_clip_average()
    print("ALL EMOTION TESTS PASSED")
