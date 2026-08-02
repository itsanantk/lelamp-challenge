"""Unit tests for conversation/emotion.py's heuristic tone classification.
Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from conversation import emotion


def test_loud_fast_variable_pitch_classifies_as_energetic():
    assert emotion._classify(rms=0.017, pitch_var=20.0, rate=4.0) == "energetic"


def test_loud_fast_flat_pitch_classifies_as_tense():
    assert emotion._classify(rms=0.017, pitch_var=5.0, rate=4.0) == "tense"


def test_quiet_overrides_everything_else():
    assert emotion._classify(rms=0.001, pitch_var=20.0, rate=4.0) == "quiet"


def test_moderate_speech_classifies_as_calm():
    assert emotion._classify(rms=0.01, pitch_var=5.0, rate=2.0) == "calm"


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
    test_analyze_uses_peak_rms_not_a_flat_whole_clip_average()
    print("ALL EMOTION TESTS PASSED")
