"""Watches for a tracked object (default: your phone) and has the lamp
follow it while it's visible, so setting the phone down -- or picking it
back up and moving it around -- reads as "the lamp noticed and is
tracking it" rather than a silent database write.

Tracks continuously, not as a one-shot glance: as long as the object
keeps getting detected, the lamp keeps re-aiming at its current bearing
(throttled by WATCH_REAIM_DEG so frame-to-frame bbox noise doesn't
jitter the arm). It only settles back once the object hasn't been seen
for WATCH_LOST_GRACE_S -- long enough to ride out a run of missed
detections (bad angle, brief occlusion) without the gesture flickering
on and off every time a single scan misses.

Doesn't touch memory writes at all -- VisionMemory already logs every
sighting on its own scan cadence, and get_latest_by_class already returns
the most recent one, so wherever the phone was last seen is automatically
"where you put it" once it stops moving. This module is purely about the
lamp's reaction while that's happening.

Only ATTENTION_SEEKING hard-blocks this -- that's a short, deliberately
timed animation (BehaviorFSM._tick_attention_seek) that would look
broken if interrupted mid-gesture. ENGAGED does *not* block it: checked
BehaviorFSM directly rather than assuming, and it only calls
set_target_pose *once*, on the transition into ENGAGED -- it doesn't
re-assert the pose every tick the way this class does. So there's no
real fight for control: the FSM's "look at the person" gesture plays
once, and this class is then free to track the object without anything
pulling the pose back on its own. If the object then gets lost while
still ENGAGED, settling to NEUTRAL_POSE/IDLE_COLOR would incorrectly
undo the FSM's engaged look, since the FSM never re-applies it -- so the
lost-tracking settle returns to the engaged look-at-user pose instead
of neutral when that's the state it happens in.
"""
from __future__ import annotations

import config
from behavior.state_machine import ENGAGED_COLOR, IDLE_COLOR, LOOK_UP_PITCH_DEG, State
from lamp import kinematics
from lamp.hal import LampActuator

WATCH_COLOR = (170, 220, 120)  # BGR, soft curious green-cyan
WATCH_PITCH_DEG = -30.0        # looking down at desk level


class ObjectWatcher:
    def __init__(self, lamp: LampActuator, tracked_classes: list[str] | None = None):
        self.lamp = lamp
        self.tracked_classes = tracked_classes or config.TRACKED_CLASSES
        self.active = False
        self._current_bearing = 0.0
        self._lost_t = 0.0
        self._time_since_reaim = 0.0

    def update(self, detections, fsm_state: State, dt: float,
               user_bearing_deg: float | None = None) -> None:
        fsm_owns_lamp = fsm_state == State.ATTENTION_SEEKING
        target = None if fsm_owns_lamp else next(
            (d for d in (detections or []) if d.object_class in self.tracked_classes), None)

        if target is not None:
            self._lost_t = 0.0
            if not self.active:
                self._acquire(target.bearing_deg)
            elif abs(target.bearing_deg - self._current_bearing) > config.WATCH_REAIM_DEG:
                self._reaim(target.bearing_deg)
            else:
                self._time_since_reaim += dt
        else:
            self._lost_t += dt
            self._time_since_reaim += dt
            if self.active and self._lost_t > config.WATCH_LOST_GRACE_S:
                if fsm_owns_lamp:
                    # ATTENTION_SEEKING owns the lamp right now. It owns
                    # pose/light -- settling back here would stomp on
                    # whatever it just set and never get corrected, since
                    # the FSM only re-actuates on its *own* transitions.
                    # Just clear our flag.
                    self.active = False
                elif fsm_state == State.ENGAGED:
                    self._return_to_engaged_look(user_bearing_deg)
                else:
                    self._stop_watch()

    def should_scan_fast(self) -> bool:
        """True while the object has actually been moving recently, or
        for a few seconds after it's lost -- catches a placement or a
        follow in progress without staying in fast-scan mode forever once
        it's just sitting there.

        The two signals apply in mutually exclusive situations, which
        matters: _lost_t resets to 0 on every successful detection, so
        "_lost_t < 3.0" is trivially true on *every* tick the object is
        currently visible, not just briefly after losing it. ORing that
        unconditionally against _time_since_reaim would make fast-scan
        permanent for anything continuously in frame, moving or not --
        exactly the "stationary phone pins the loop at the fast interval
        forever" bug this was meant to fix. So: while currently detected,
        go by whether it's actually moving; while not currently detected,
        go by how recently it was lost."""
        if self._lost_t > 0:
            return self._lost_t < 3.0
        return self._time_since_reaim < config.WATCH_FAST_SCAN_SETTLE_S

    def _acquire(self, bearing_deg: float) -> None:
        self.active = True
        self._current_bearing = bearing_deg
        self._time_since_reaim = 0.0
        pose = kinematics.pose_for_look_at(bearing_deg, WATCH_PITCH_DEG, alertness=0.65)
        self.lamp.set_target_pose(pose, duration=0.35, anticipation=True, overshoot=True)
        self.lamp.set_light(WATCH_COLOR, transition_s=0.25)

    def _reaim(self, bearing_deg: float) -> None:
        # Following an object already in view should read as calm
        # tracking, not a repeated startle -- no anticipation/overshoot
        # flourish here, that's reserved for the initial acquire.
        self._current_bearing = bearing_deg
        self._time_since_reaim = 0.0
        pose = kinematics.pose_for_look_at(bearing_deg, WATCH_PITCH_DEG, alertness=0.65)
        self.lamp.set_target_pose(pose, duration=0.4, anticipation=False, overshoot=False)

    def _stop_watch(self) -> None:
        self.active = False
        self.lamp.set_target_pose(kinematics.NEUTRAL_POSE, duration=0.6, overshoot=False)
        self.lamp.set_light(IDLE_COLOR, transition_s=0.6)

    def _return_to_engaged_look(self, user_bearing_deg: float | None) -> None:
        """Losing the object while still ENGAGED shouldn't settle to
        neutral/idle -- the person is still there, and the FSM won't
        re-assert its own "looking at you" pose to correct it (see the
        module docstring: it only sets that pose once, on entry). Mirrors
        BehaviorFSM._enter_engaged's own pose/light exactly so the lamp
        looks like it was never distracted, not like it's resetting."""
        self.active = False
        if user_bearing_deg is not None:
            pose = kinematics.pose_for_look_at(user_bearing_deg, LOOK_UP_PITCH_DEG, alertness=1.0)
            self.lamp.set_target_pose(pose, duration=0.5, anticipation=False, overshoot=False)
        self.lamp.set_light(ENGAGED_COLOR, transition_s=0.4)
