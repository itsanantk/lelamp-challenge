"""Unit tests for conversation/emotion.py's heuristic tone classification.
Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from conversation import emotion


def test_fast_variable_pitch_below_the_angry_floor_classifies_as_energetic():
    # The pitch-variability split (energetic vs. tense) only applies below
    # _ANGRY_RMS -- at or above it, loudness alone decides "tense" outright
    # regardless of pitch (see that constant's own comment, and
    # test_loud_classifies_as_tense_regardless_of_pitch below). This tests
    # the blended, sub-threshold case specifically: aroused via loudness
    # *and* rate together, not loud enough alone to skip the blend.
    just_under_angry = emotion._ANGRY_RMS - 0.001
    assert emotion._classify(rms=just_under_angry, pitch_var=25.0, rate=4.0) == "energetic"


def test_loud_classifies_as_tense_regardless_of_pitch():
    # Live bug report: a loud/raised tone (rms=0.0978, pitch_var=39.6,
    # well above _VARIABLE_PITCH_HZ) still came out "energetic" instead of
    # "tense" -- a raised/angry voice shouldn't get relabeled just because
    # its pitch happened to vary. At/above _ANGRY_RMS, pitch is ignored
    # entirely; "tense" either way.
    assert emotion._classify(rms=emotion._ANGRY_RMS, pitch_var=25.0, rate=4.0) == "tense"
    assert emotion._classify(rms=emotion._ANGRY_RMS, pitch_var=5.0, rate=4.0) == "tense"


def test_quiet_overrides_everything_else():
    assert emotion._classify(rms=emotion._QUIET_RMS - 0.001, pitch_var=25.0, rate=4.0) == "quiet"


def test_moderate_speech_classifies_as_calm():
    # Just above the quiet floor, not maxed against the loud one -- this
    # is exactly the "ordinary talking" case that used to misclassify.
    quiet_ish = emotion._QUIET_RMS + 0.005
    assert emotion._classify(rms=quiet_ish, pitch_var=5.0, rate=2.0) == "calm"


def test_loud_but_not_fast_still_registers_as_tense():
    # A short, clipped, frustrated sentence doesn't reliably hit "fast" the
    # way a longer excited one does -- the old AND-gate required loud AND
    # fast simultaneously, so a clearly raised voice at a normal pace
    # silently fell through to "calm". Scored as an average instead: a
    # strong signal in just one dimension is now enough to tip out of calm.
    assert emotion._classify(rms=emotion._LOUD_RMS, pitch_var=5.0, rate=1.5) == "tense"


def test_fast_but_not_loud_still_registers_as_energetic():
    quiet_ish = emotion._QUIET_RMS + 0.005
    assert emotion._classify(rms=quiet_ish, pitch_var=25.0, rate=5.0) == "energetic"


def test_loud_but_slow_still_registers_as_not_calm():
    # The actual live bug report: rms=0.081 rate=0.61Hz ("but if I yell at
    # it...") classified as "calm" -- a slow rate score, averaged in,
    # dragged an already-loud utterance's arousal back under
    # _AROUSAL_THRESHOLD. _ANGRY_RMS exists specifically so loudness this
    # high doesn't get diluted by rate at all, same as _YELL_RMS one tier
    # up. "tense," not "energetic" -- see test_loud_classifies_as_tense_
    # regardless_of_pitch for why pitch no longer matters here either.
    assert emotion._classify(rms=0.081, pitch_var=80.0, rate=0.61) == "tense"
    assert emotion._classify(rms=emotion._ANGRY_RMS, pitch_var=5.0, rate=0.1) == "tense"


def test_just_under_the_angry_floor_can_still_be_calm_with_a_slow_rate():
    # The blended path (arousal averaging loudness and rate) is only
    # bypassed at/above _ANGRY_RMS -- below it, a slow, moderate-volume
    # utterance should still read as calm, same as before this fix.
    just_under = emotion._ANGRY_RMS - 0.001
    assert emotion._classify(rms=just_under, pitch_var=5.0, rate=0.5) == "calm"


def test_very_loud_classifies_as_yelling_regardless_of_pitch_or_rate():
    # Loudness alone decides "yelling" -- unlike tense/energetic, it isn't
    # blended with rate/pitch, since a genuine yell is unambiguous on raw
    # loudness alone and blending would let a fast-but-quiet sentence
    # dilute a real yell's classification. Keyed off the real threshold,
    # not a hardcoded RMS -- a fixed literal here silently stopped meaning
    # "very loud" the last time VOICE_YELL_RMS_THRESHOLD was retuned.
    above_threshold = emotion.config.VOICE_YELL_RMS_THRESHOLD + 0.01
    assert emotion._classify(rms=above_threshold, pitch_var=5.0, rate=1.0) == "yelling"
    assert emotion._classify(rms=above_threshold, pitch_var=25.0, rate=4.0) == "yelling"


def test_yelling_threshold_is_a_hard_floor_above_tense():
    # Just under the yell floor still classifies via the normal blend
    # (tense here); just at/over it is yelling outright.
    just_under = emotion.config.VOICE_YELL_RMS_THRESHOLD - 0.001
    assert emotion._classify(rms=just_under, pitch_var=5.0, rate=4.0) == "tense"
    assert emotion._classify(rms=emotion.config.VOICE_YELL_RMS_THRESHOLD, pitch_var=5.0, rate=1.0) == "yelling"


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
    test_fast_variable_pitch_below_the_angry_floor_classifies_as_energetic()
    test_loud_classifies_as_tense_regardless_of_pitch()
    test_quiet_overrides_everything_else()
    test_moderate_speech_classifies_as_calm()
    test_loud_but_not_fast_still_registers_as_tense()
    test_fast_but_not_loud_still_registers_as_energetic()
    test_loud_but_slow_still_registers_as_not_calm()
    test_just_under_the_angry_floor_can_still_be_calm_with_a_slow_rate()
    test_very_loud_classifies_as_yelling_regardless_of_pitch_or_rate()
    test_yelling_threshold_is_a_hard_floor_above_tense()
    test_analyze_uses_peak_rms_not_a_flat_whole_clip_average()
    print("ALL EMOTION TESTS PASSED")
