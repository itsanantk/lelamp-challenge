"""Bonus: a coarse, heuristic read on vocal tone from the raw recording --
not a trained emotion classifier, three cheap prosodic features (loudness,
pitch level/variability, speaking rate) mapped to a handful of buckets.

Good enough to color how the lamp reacts (a brighter pulse for energetic,
a calmer glow for quiet/subdued, a startled flinch if you flat-out yell)
-- nowhere near good enough to be
presented as clinically meaningful, and it will absolutely get fooled by
things like a monotone loud voice or a quiet excited whisper. Real
emotion recognition from speech needs a model trained on labeled prosody
data (or a multimodal call on the audio itself, which costs a real API
round trip per utterance); this is the version buildable with signal
processing and no new heavy dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from conversation import voice

# Tuned by ear against a handful of test recordings, not a labeled
# dataset -- treat these as starting points, not calibrated thresholds.
# _LOUD_RMS was originally 0.02 against a flat whole-utterance average,
# which real speech on this mic essentially never reaches (see
# VOICE_GATE_RMS_THRESHOLD's own calibration note in config.py) -- every
# utterance fell through to "quiet"/"calm" and "energetic"/"tense" were
# effectively dead code. Now measured as peak RMS (same fix as the
# wake-word gate) with a correspondingly higher bar, since peak reads
# louder than a flat average for the same clip.
_QUIET_RMS = 0.005
_LOUD_RMS = 0.016
# config.VOICE_YELL_RMS_THRESHOLD, not a private constant here -- the same
# "was that a yell" line is also checked live, mid-recording, off raw
# audio in conversation/voice.py (see its own comment on that constant).
# Blending loudness with rate/pitch the way _classify() does for
# tense-vs-energetic would let a fast, animated but not-actually-loud
# sentence read as a yell; a yell is unambiguous on raw loudness alone, so
# it's checked first here and skips the blend entirely.
_YELL_RMS = config.VOICE_YELL_RMS_THRESHOLD
_CALM_RATE_HZ = 1.0   # rate floor for scoring -- a short, clipped, tense
_FAST_RATE_HZ = 3.5   # sentence doesn't reliably hit "fast" the way a long excited one does
# 15.0 was calibrated against std-dev; IQR runs larger for the same
# underlying spread (~1.35x for roughly bell-shaped data), so the switch
# to IQR alone would silently make "variable pitch" fire more often than
# before. Scaled proportionally as a starting estimate, not re-derived
# from real data -- same caveat as every other threshold in this file.
_VARIABLE_PITCH_HZ = 20.0
_AROUSAL_THRESHOLD = 0.5  # scored average of loud/fast needed to leave "calm"


@dataclass
class VoiceTone:
    label: str  # "yelling" | "energetic" | "tense" | "quiet" | "calm"
    rms: float
    pitch_hz: float | None
    pitch_variability_hz: float | None
    speaking_rate_hz: float


def _estimate_pitch_hz(frame: np.ndarray, sample_rate: int, fmin: float = 70.0, fmax: float = 400.0) -> float | None:
    """Autocorrelation F0 estimate for one short frame. None if the frame
    doesn't look clearly voiced (too quiet, or no strong periodicity)."""
    frame = frame - np.mean(frame)
    if np.sqrt(np.mean(np.square(frame))) < _QUIET_RMS:
        return None
    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2:]
    min_lag, max_lag = int(sample_rate / fmax), int(sample_rate / fmin)
    if max_lag >= len(corr) or corr[0] <= 0:
        return None
    segment = corr[min_lag:max_lag]
    if len(segment) == 0 or np.max(segment) <= 0:
        return None
    peak_lag = min_lag + int(np.argmax(segment))
    if corr[peak_lag] < 0.3 * corr[0]:
        return None  # not periodic enough to trust as voiced pitch
    return sample_rate / peak_lag


def analyze(audio: np.ndarray, sample_rate: int = config.VOICE_SAMPLE_RATE) -> VoiceTone:
    rms = voice._peak_rms(audio, sample_rate) if audio.size else 0.0

    frame_len = int(sample_rate * 0.04)  # 40ms analysis frames
    hop = max(frame_len // 2, 1)
    pitches = [
        p for start in range(0, max(len(audio) - frame_len, 1), hop)
        if (p := _estimate_pitch_hz(audio[start:start + frame_len], sample_rate)) is not None
    ]
    pitch_hz = float(np.mean(pitches)) if pitches else None
    # Interquartile range instead of std-dev -- autocorrelation pitch
    # estimation occasionally jumps to half/double the true pitch on a
    # single frame (an octave error), which badly skews a plain std-dev;
    # IQR barely notices a handful of outlier frames.
    pitch_var = float(np.percentile(pitches, 75) - np.percentile(pitches, 25)) if len(pitches) > 2 else None

    # Rough speaking-rate proxy: how often the smoothed amplitude envelope
    # crosses above/below its own mean -- choppier/faster speech crosses
    # more often per second than slow, smooth speech.
    envelope = np.abs(audio)
    win = max(int(sample_rate * 0.02), 1)
    smoothed = np.convolve(envelope, np.ones(win) / win, mode="same")
    voiced = smoothed > (smoothed.mean() * 0.5)
    crossings = int(np.sum(np.abs(np.diff(voiced.astype(int)))))
    duration_s = max(len(audio) / sample_rate, 1e-3)
    speaking_rate = crossings / duration_s

    label = _classify(rms, pitch_var, speaking_rate)
    return VoiceTone(label, rms, pitch_hz, pitch_var, speaking_rate)


def _normalize(value: float, lo: float, hi: float) -> float:
    return min(max((value - lo) / (hi - lo), 0.0), 1.0)


def _classify(rms: float, pitch_var: float | None, rate: float) -> str:
    if rms >= _YELL_RMS:
        return "yelling"
    if rms < _QUIET_RMS:
        return "quiet"

    # Scored as an average instead of requiring loud AND fast at once --
    # real tension doesn't reliably max out both dimensions simultaneously.
    # A short, clipped, moderately raised sentence ("why can't you help
    # me") can carry real frustration while only strongly tripping one of
    # the two signals; the old AND-gate silently called that "calm".
    loud_score = _normalize(rms, _QUIET_RMS, _LOUD_RMS)
    fast_score = _normalize(rate, _CALM_RATE_HZ, _FAST_RATE_HZ)
    arousal = (loud_score + fast_score) / 2
    if arousal < _AROUSAL_THRESHOLD:
        return "calm"

    variable_pitch = pitch_var is not None and pitch_var > _VARIABLE_PITCH_HZ
    return "energetic" if variable_pitch else "tense"
