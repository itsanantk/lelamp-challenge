"""Ambient light sensing: samples the camera frame's average luminance so
the lamp can brighten itself a little in a dark room and dim a little in
a bright one -- the same instinct a real desk lamp's owner has (more
light when it's dim, no need to blast at full brightness when the room's
already lit). Deliberately just a *nudge*: this never touches the mood
dial the user actually set (lamp/color.py, SimulatedLamp.get_mood/
set_mood) the same way tracking/tone/recall don't -- see
behavior/state_machine.py's ambient_brightness_nudge field for where it's
actually applied (only to the idle/resting look, not every state -- see
that field's own comment for why).
"""
from __future__ import annotations

import time

import cv2
import numpy as np

import config


class AmbientLightSensor:
    def __init__(self):
        # -inf, not 0.0 -- update() is often first called with now=0.0
        # (monotonic clocks, or a test), and gating "now - last < interval"
        # against a last of 0.0 would silently reject that very first
        # sample instead of taking it immediately.
        self._last_sample_t = float("-inf")
        self._smoothed_luma = config.AMBIENT_LIGHT_MIDPOINT
        self._first_sample = True

    def update(self, frame: np.ndarray, now: float | None = None) -> None:
        """Call every loop tick; internally gated to
        config.AMBIENT_LIGHT_SAMPLE_INTERVAL_S -- room lighting doesn't
        change fast, sampling every frame would be wasted work."""
        now = now if now is not None else time.monotonic()
        if now - self._last_sample_t < config.AMBIENT_LIGHT_SAMPLE_INTERVAL_S:
            return
        self._last_sample_t = now
        # Downsample first -- this only needs a rough room-brightness
        # read, not per-pixel precision, and a full 1920x1080 grayscale
        # convert+mean every sample is wasted work at this cadence.
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # A high percentile, not the plain mean -- a flashlight (or any
        # small, genuinely bright thing) pointed at the camera only lights
        # up a fraction of the frame; the rest stays at normal room
        # brightness, or even darkens as the webcam's own auto-exposure
        # compensates for the bright spot. Averaged over the whole frame,
        # that swing gets diluted down to barely registering. The
        # brightest ~20% of pixels, though, clearly reflects it -- this
        # still reads as an unlit room's overall level in a normal room
        # (nothing pathologically bright anywhere), but responds to a
        # concentrated bright source the mean was blind to. No real
        # calibration data behind 80 specifically (same caveat as every
        # other threshold in this module) -- lower catches a smaller/more
        # distant light source but drifts closer to the mean's original
        # blind spot; higher stays cleaner in a normal room but needs a
        # bigger fraction of the frame lit up before it responds at all.
        luma = float(np.percentile(gray, 80)) / 255.0
        if self._first_sample:
            self._smoothed_luma = luma
            self._first_sample = False
        else:
            a = config.AMBIENT_LIGHT_SMOOTHING
            self._smoothed_luma = a * luma + (1 - a) * self._smoothed_luma

    def brightness_nudge(self) -> float:
        """A (db)-style brightness delta -- positive in a dark room
        (brighten to help), negative in a bright room (dim, since less
        additional light is needed), ~0 around a "normal room" midpoint.
        Clamped to AMBIENT_LIGHT_MAX_NUDGE so this reads as gently
        compensating, never as swinging the lamp between off and full
        blast on its own."""
        delta = config.AMBIENT_LIGHT_MIDPOINT - self._smoothed_luma  # positive when darker than normal
        nudge = delta * config.AMBIENT_LIGHT_RESPONSE
        return float(np.clip(nudge, -config.AMBIENT_LIGHT_MAX_NUDGE, config.AMBIENT_LIGHT_MAX_NUDGE))
