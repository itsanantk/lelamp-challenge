"""Hand-wave detection: a real hand, not a proxy.

The tracked-object wave (behavior/object_watch.py, before this existed)
inferred a "wave" from the tracked phone's own bearing oscillating back
and forth -- reusable, since ObjectWatcher already computed a re-aim
signal every tick, but it could only ever notice a wave if you happened
to be waving the tracked object itself. Detecting an actual hand needs
its own model: MediaPipe's HandLandmarker, run separately from (and only
while) the person is ENGAGED -- see config.py's own note on why this is
gated rather than run every frame like FaceLandmarker.

Same reversal-counting idea as the old object_watch detector, just fed by
an actual wrist position instead of a bounding box. _WaveTracker holds
that logic on its own (no MediaPipe import), same split as
conversation/voice.py's _check_wake_chunk vs wait_for_wake_word -- lets it
be unit tested without loading a real hand-landmark model.
"""
from __future__ import annotations

import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import config

_WRIST_LANDMARK = 0  # HandLandmarker's 21-point hand model, landmark 0 = wrist


class _WaveTracker:
    """Feed it a wrist x-position (0..1, normalized frame width) each
    time a hand is seen; returns a bearing to react toward the tick
    several quick direction reversals land inside config.WATCH_WAVE_WINDOW_S.
    reset() on any tick the hand isn't visible -- a wave has to be one
    continuous gesture, not reversals stitched together across a gap."""

    def __init__(self):
        self._last_x: float | None = None
        self._signs: list[tuple[float, float]] = []
        self._waved = False

    def feed(self, x: float, now: float) -> float | None:
        wave_bearing = None
        if self._last_x is not None and not self._waved:
            delta = x - self._last_x
            if abs(delta) > config.HAND_WAVE_MIN_DELTA:
                sign = 1.0 if delta > 0 else -1.0
                self._signs.append((now, sign))
                self._signs = [(t, s) for t, s in self._signs if now - t <= config.WATCH_WAVE_WINDOW_S]
                reversals = sum(1 for (_, s0), (_, s1) in zip(self._signs, self._signs[1:]) if s0 != s1)
                if reversals >= config.WATCH_WAVE_MIN_REVERSALS:
                    self._waved = True
                    wave_bearing = (0.5 - x) * config.CAMERA_HFOV_DEG
        self._last_x = x
        return wave_bearing

    def reset(self) -> None:
        self._last_x = None
        self._signs = []
        self._waved = False


class HandWaveDetector:
    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.HAND_LANDMARKER_MODEL)),
            num_hands=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_scan_t = 0.0
        self._tracker = _WaveTracker()

    def update(self, frame: np.ndarray, now: float | None = None) -> float | None:
        """Call every loop tick; internally gated to
        config.HAND_SCAN_INTERVAL_S (ticks in between are a cheap no-op,
        returning None). Returns the bearing (degrees, same convention as
        vision_memory.py/engagement.py) to react toward on the tick a wave
        is newly detected; fires once per continuously-visible hand."""
        now = now if now is not None else time.monotonic()
        if now - self._last_scan_t < config.HAND_SCAN_INTERVAL_S:
            return None
        self._last_scan_t = now

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect(mp_image)
        if not result.hand_landmarks:
            self._tracker.reset()
            return None
        x = result.hand_landmarks[0][_WRIST_LANDMARK].x
        return self._tracker.feed(x, now)

    def close(self) -> None:
        self.landmarker.close()
