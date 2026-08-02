"""Stub for driving an actual LeLamp robot instead of the simulator.

I don't have the hardware to actually build this, but sketching out what
it'd need:

  * Servo bus driver (Feetech STS3215 over serial TTL is the usual choice
    for SO-100/SO-101-style arms) mapping the 6 joints in
    kinematics.JOINT_NAMES to servo IDs plus a degrees-to-raw-position
    calibration per servo (zero offset, direction, travel limits).
  * A trajectory thread writing interpolated setpoints to the bus at the
    servo controller's max update rate (50-100Hz for this class of servo)
    instead of relying on onboard interpolation, so Trajectory.step()
    stays the one place that defines how motion looks.
  * set_light driving an addressable LED (WS2812 in the shade, probably)
    through a small microcontroller bridge — USB serial isn't great for
    tight LED timing.
  * play_sound writing to a real speaker (I2S amp + small speaker) instead
    of winsound.

None of behavior/state_machine.py would change to swap this in for
SimulatedLamp, which is the whole point of having the HAL.
"""
from __future__ import annotations

import numpy as np

from .hal import LampActuator


class RealLamp(LampActuator):
    def __init__(self, serial_port: str, baud: int = 1_000_000):
        raise NotImplementedError(
            "RealLamp needs actual LeLamp hardware (servo bus + LED + "
            "speaker) I don't have. See the module docstring for what a "
            "real implementation would need."
        )

    def set_target_pose(self, angles, duration=0.6, anticipation=True, overshoot=True) -> None:
        raise NotImplementedError

    def set_light(self, rgb, transition_s=0.4) -> None:
        raise NotImplementedError

    def play_sound(self, event: str) -> None:
        raise NotImplementedError

    def update(self, dt: float) -> None:
        raise NotImplementedError

    def get_current_pose(self) -> np.ndarray:
        raise NotImplementedError
