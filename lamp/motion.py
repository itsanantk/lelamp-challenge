"""Joint-space motion with easing, anticipation, and overshoot.

Linear interpolation between poses looks robotic. Borrowing a bit of the
classic animation playbook (wind-up, overshoot, settle) is what makes the
arm read as alive instead of a servo sweeping between two setpoints. Pure
math, no rendering or I/O, so it's easy to test on its own.
"""
from __future__ import annotations

import random

import numpy as np

# easeOutBack: overshoots the target slightly then settles back, like a
# spring with a bit of momentum. c1/c3 are the standard easing constants.
_C1 = 1.70158


def ease_out_back(t: float, strength: float = 1.0) -> float:
    # strength varies the overshoot per-move (see Trajectory.__init__) --
    # identical strength every time read as a servo repeating a preset
    # motion rather than something with a bit of physical variance.
    t = np.clip(t, 0.0, 1.0)
    c1 = _C1 * strength
    c3 = c1 + 1.0
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease_in_out_cubic(t: float) -> float:
    t = np.clip(t, 0.0, 1.0)
    return 4 * t ** 3 if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2


class Trajectory:
    """A single joint-space move from `start` to `target` over `duration`
    seconds, with an optional anticipation dip and overshoot settle.
    """

    def __init__(
        self,
        start: np.ndarray,
        target: np.ndarray,
        duration: float,
        anticipation: bool = True,
        overshoot: bool = True,
    ):
        self.start = np.array(start, dtype=float)
        self.target = np.array(target, dtype=float)
        # Same move every time (same duration, same wind-up, same
        # overshoot) reads as a servo replaying a preset, not a body with
        # a bit of natural physical variance -- jitter each by a modest
        # random amount per trajectory instead.
        self.duration = max(duration, 1e-3) * random.uniform(0.9, 1.12)
        self.anticipation = anticipation
        self.overshoot = overshoot
        self.elapsed = 0.0
        self._overshoot_strength = random.uniform(0.7, 1.3)

        delta = self.target - self.start
        # Anticipation: a brief "wind-up" opposite the direction of travel,
        # proportional to how far we're about to move (capped so tiny moves
        # don't wind up dramatically).
        anticipation_scale = random.uniform(0.75, 1.25)
        self.anticipation_pose = self.start - np.clip(delta * 0.08 * anticipation_scale, -6.0, 6.0)

        # 'Wavey' organic ripple layered on top of the eased path during
        # the main move -- a pure eased interpolation, even with
        # anticipation/overshoot, still moves every joint through the
        # exact same normalized time, which reads as one rigid rod, not a
        # limb with a bit of give in it. A per-joint phase offset makes
        # the ripple travel down the chain (base leads, head trails) like
        # a whip/tail instead of the whole arm shaking in unison. The
        # envelope (see step()) is exactly 0 at both ends of the main
        # phase, so it never disturbs the wind-up entry or the exact
        # final landing on target.
        #
        # Deliberately only for flourish moves (anticipation or overshoot
        # already True), never a calm/continuous one -- ObjectWatcher's
        # _reaim already sets both False specifically because repeated
        # short retargets read as a startle, not tracking (see its
        # docstring); layering an *independent* random wobble onto each
        # of those short back-to-back trajectories was exactly what read
        # as jitter tracking a moving phone, not liveliness. Slower and
        # gentler than the first pass too -- reads as a wavey undulation
        # now, not a shake.
        n = len(self.start)
        self._wave_freq = random.uniform(0.8, 1.3)
        self._wave_amp_deg = random.uniform(0.8, 1.6) if (anticipation or overshoot) else 0.0
        wave_base_phase = random.uniform(0.0, 2 * np.pi)
        wave_lag_per_joint = random.uniform(0.35, 0.65)
        self._wave_phase = wave_base_phase + np.arange(n) * wave_lag_per_joint

    @property
    def done(self) -> bool:
        return self.elapsed >= self.duration

    def step(self, dt: float) -> np.ndarray:
        self.elapsed = min(self.elapsed + dt, self.duration)
        t = self.elapsed / self.duration

        if self.anticipation and t < 0.15:
            # Wind up during the first 15% of the move.
            local_t = ease_in_out_cubic(t / 0.15)
            return self.start + (self.anticipation_pose - self.start) * local_t

        # Remap the remaining 85% of the timeline to the main ease curve.
        main_t = (t - 0.15) / 0.85 if self.anticipation else t
        main_t = np.clip(main_t, 0.0, 1.0)
        eased = ease_out_back(main_t, self._overshoot_strength) if self.overshoot else ease_in_out_cubic(main_t)

        origin = self.anticipation_pose if self.anticipation else self.start
        pose = origin + (self.target - origin) * eased

        # Envelope peaks mid-move and is exactly 0 at main_t=0 and
        # main_t=1 -- ease_out_back(1, ...) == 1 exactly too, so the final
        # landing on self.target is untouched regardless of amplitude.
        wave_envelope = np.sin(np.pi * main_t)
        wave = self._wave_amp_deg * wave_envelope * np.sin(
            2 * np.pi * self._wave_freq * self.elapsed + self._wave_phase)
        return pose + wave

    def retarget(self, new_target: np.ndarray, current: np.ndarray, duration: float | None = None):
        """Replace the target mid-flight without a visible snap, starting a
        fresh trajectory from wherever the arm currently is."""
        self.__init__(
            current,
            new_target,
            duration if duration is not None else self.duration,
            anticipation=self.anticipation,
            overshoot=self.overshoot,
        )


def lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))
