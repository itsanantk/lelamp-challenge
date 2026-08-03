"""Idle look-around: when nothing else has the lamp's attention for a
while, sweep base_yaw across a few waypoints to actually discover objects
outside the current view.

This matters because object detection is restricted to a crop that
follows wherever the lamp is currently aimed (see perception/
vision_memory.py's pan_zoom) -- without this, a lamp resting at bearing 0
can never notice something placed off to the side until some *other*
reason happens to turn it that way. This is what makes noticing something
new actually proactive instead of purely reactive to whatever wanders
into the current crop.

Only activates in the deepest idle case: FSM state IDLE specifically (not
DISENGAGED, which already has its own attention-seeking timeline aimed at
the *person*; not ENGAGED/ATTENTION_SEEKING, which already own the lamp)
and nothing already being tracked. Preemption is silent -- on losing
eligibility mid-sweep this just clears its own flag and stops touching
the lamp, rather than snapping to neutral, since whatever preempted it
(the FSM re-engaging, ObjectWatcher acquiring something the sweep just
turned up) is about to set its own correct pose on the very same tick and
a competing neutral-pose command would just be a race against that.
"""
from __future__ import annotations

import random

import config
from behavior.state_machine import State
from lamp import kinematics
from lamp.hal import LampActuator

IDLE_SCAN_ALERTNESS = 0.35   # a gentle glance, not a full curious lean-in
IDLE_SCAN_PITCH_DEG = -20.0  # roughly desk height, same ballpark as object-watch


class IdleScanner:
    def __init__(self, lamp: LampActuator):
        self.lamp = lamp
        self.active = False
        self._idle_t = 0.0
        self._waypoint_t = 0.0
        self._waypoint_idx = 0
        self._waypoints: list[float] = []

    def update(self, fsm_state: State, object_watch_active: bool, dt: float) -> None:
        eligible = fsm_state == State.IDLE and not object_watch_active
        if not eligible:
            self.active = False  # relinquish silently -- see module docstring
            self._idle_t = 0.0
            return

        if not self.active:
            self._idle_t += dt
            if self._idle_t >= config.IDLE_SCAN_DELAY_S:
                self._start()
            return

        self._waypoint_t += dt
        if self._waypoint_t >= config.IDLE_SCAN_HOLD_S:
            self._advance()

    def _start(self) -> None:
        self.active = True
        self._idle_t = 0.0
        span = kinematics.JOINT_LIMITS[0]
        # A handful of bearings spanning the reachable range, shuffled --
        # a fixed left-to-right sweep every time reads as mechanical
        # scanning, not curious glancing around.
        self._waypoints = [span[0] * 0.7, span[0] * 0.3, 0.0, span[1] * 0.3, span[1] * 0.7]
        random.shuffle(self._waypoints)
        self._waypoint_idx = 0
        self._waypoint_t = 0.0
        self._look_at_current_waypoint()

    def _look_at_current_waypoint(self) -> None:
        bearing = self._waypoints[self._waypoint_idx]
        pose = kinematics.pose_for_look_at(bearing, IDLE_SCAN_PITCH_DEG, alertness=IDLE_SCAN_ALERTNESS)
        self.lamp.set_target_pose(pose, duration=1.2, anticipation=False, overshoot=False)

    def _advance(self) -> None:
        self._waypoint_idx += 1
        self._waypoint_t = 0.0
        if self._waypoint_idx >= len(self._waypoints):
            self._finish()
            return
        self._look_at_current_waypoint()

    def _finish(self) -> None:
        # A natural, un-preempted sweep completion -- unlike the
        # preemption case above, nothing else is about to set a pose this
        # tick, so this is the one place that actively settles back.
        self.active = False
        self.lamp.set_target_pose(kinematics.NEUTRAL_POSE, duration=0.8, overshoot=False)
