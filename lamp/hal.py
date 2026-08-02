"""Hardware abstraction layer for the lamp's actuators.

behavior/state_machine.py only ever talks to a LampActuator, never to a
specific backend. Right now the only implementation is SimulatedLamp
(sim_backend.py) since I don't have physical LeLamp hardware. A RealLamp
backend driving actual servos/LEDs/speaker over serial would implement
this same interface and nothing upstream would need to change — see
real_backend.py for the stub and what a real version would need.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LampActuator(ABC):
    """Everything the behavior layer is allowed to ask the physical (or
    simulated) lamp to do."""

    @abstractmethod
    def set_target_pose(
        self,
        angles: np.ndarray,
        duration: float = 0.6,
        anticipation: bool = True,
        overshoot: bool = True,
    ) -> None:
        """Command the 6 joints toward `angles` (degrees) over `duration`
        seconds. Non-blocking: the actual motion is advanced by `update()`."""

    @abstractmethod
    def set_light(self, rgb: tuple[int, int, int], transition_s: float = 0.4) -> None:
        """Fade the lamp's light to `rgb` over `transition_s` seconds."""

    @abstractmethod
    def play_sound(self, event: str) -> None:
        """Trigger a named expressive sound cue (e.g. 'engage', 'attention_seek').
        Must not block the caller."""

    @abstractmethod
    def update(self, dt: float) -> None:
        """Advance internal animation/light state by `dt` seconds. Called
        once per render tick from the main loop."""

    @abstractmethod
    def get_current_pose(self) -> np.ndarray:
        """Current 6 joint angles (degrees), reflecting in-flight motion."""

    def close(self) -> None:
        pass
