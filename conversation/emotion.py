"""Bonus: a coarse, heuristic read on vocal tone from the raw recording --
not a trained emotion classifier, three cheap prosodic features (loudness,
pitch level/variability, speaking rate) mapped to a handful of buckets.

Good enough to color how the lamp reacts (a brighter pulse for energetic,
a calmer glow for quiet/subdued) -- nowhere near good enough to be
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
_FAST_RATE_HZ = 3.5
_VARIABLE_PITCH_HZ = 15.0

BGR_BY_LABEL = {
    "energetic": (60, 200, 255),    # bright warm gold
    "tense": (60, 60, 220),          # saturated red-ish
    "quiet": (200, 160, 120),        # soft muted blue
    "calm": (180, 220, 180),         # gentle green-white
}


@dataclass
class VoiceTone:
    label: str  # "energetic" | "tense" | "quiet" | "calm"
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
    pitch_var = float(np.std(pitches)) if len(pitches) > 2 else None

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


def _classify(rms: float, pitch_var: float | None, rate: float) -> str:
    if rms < _QUIET_RMS:
        return "quiet"
    variable_pitch = pitch_var is not None and pitch_var > _VARIABLE_PITCH_HZ
    if rms > _LOUD_RMS and rate > _FAST_RATE_HZ and variable_pitch:
        return "energetic"
    if rms > _LOUD_RMS and rate > _FAST_RATE_HZ and not variable_pitch:
        return "tense"  # loud and fast but flat/monotone
    return "calm"
