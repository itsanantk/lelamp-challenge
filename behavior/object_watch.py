"""Watches for a tracked object (default: your phone) and has the lamp
follow it while it's visible, so setting the phone down -- or picking it
back up and moving it around -- reads as "the lamp noticed and is
tracking it" rather than a silent database write.

Tracks continuously, not as a one-shot glance: as long as the object
keeps getting detected *and moving*, the lamp keeps re-aiming at its
current bearing (throttled by WATCH_REAIM_DEG so frame-to-frame bbox
noise doesn't jitter the arm). Two separate ways tracking ends: the
object goes missing for WATCH_LOST_GRACE_S (rides out a bad angle or
brief occlusion without flickering on and off), or it's still visible
but has stopped moving for WATCH_STATIONARY_TIMEOUT_S -- continuing to
fixate on something that's just sitting there reads as staring, not
attentiveness, so it settles back to a normal resting look either way.
A small subsequent bump doesn't immediately restart the whole acquire
flourish after a stationary give-up (see WATCH_REACQUIRE_MARGIN) -- only
a real, deliberate move does.

Breathing is explicitly suppressed on the shared lamp while actively
tracking (see lamp.hal.LampActuator.set_breathing) -- the idle wobble
stacking on top of frequent short re-aims is what read as jitter
following a moving phone, not liveliness.

Also detects a "wave": several quick direction reversals of the tracked
object's bearing in a short window. Reusing the same reaim signal this
class already computes rather than any new perception -- a real gesture
recognizer is out of scope, but a phone (or hand holding one) waved back
and forth *is* directly visible as oscillating bearing.

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
pulling the pose back on its own. If tracking then ends while still
ENGAGED, settling to NEUTRAL_POSE/IDLE_COLOR would incorrectly undo the
FSM's engaged look, since the FSM never re-applies it -- so settling
returns to the engaged look-at-user pose instead of neutral when that's
the state it happens in.
"""
from __future__ import annotations

import config
from behavior.state_machine import ENGAGED_COLOR, IDLE_COLOR, LOOK_UP_PITCH_DEG, State
from lamp import kinematics
from lamp.hal import LampActuator

WATCH_COLOR = (170, 220, 120)  # BGR, soft curious green-cyan
WATCH_PITCH_DEG = -30.0        # looking down at desk level
WAVE_TWIST_DEG = 25.0          # head_twist flick for the wave-back -- kinematics.py calls this
                                 # joint out as the "blink/twitch accents" DOF, a natural fit


class ObjectWatcher:
    def __init__(self, lamp: LampActuator, tracked_classes: list[str] | None = None):
        self.lamp = lamp
        self.tracked_classes = tracked_classes or config.TRACKED_CLASSES
        self.active = False
        self._current_bearing = 0.0
        self._lost_t = 0.0
        self._time_since_reaim = 0.0
        self._gave_up_bearing: float | None = None  # set when a stationary object is let go of --
                                                       # see WATCH_REACQUIRE_MARGIN
        self._reaim_signs: list[tuple[float, float]] = []  # (clock, direction sign) history for wave detection
        self._waved_back = False  # debounce -- at most one wave-back per acquisition
        self._clock = 0.0

    def update(self, detections, fsm_state: State, dt: float,
               user_bearing_deg: float | None = None) -> None:
        self._clock += dt
        fsm_owns_lamp = fsm_state == State.ATTENTION_SEEKING
        target = None if fsm_owns_lamp else next(
            (d for d in (detections or []) if d.object_class in self.tracked_classes), None)

        if target is not None:
            self._lost_t = 0.0
            if not self.active:
                # After giving up on a stationary object, require a real
                # move (not just reaim-sized noise) to start watching
                # again -- otherwise a tiny bump on an otherwise-still
                # phone restarts the whole acquire flourish every few
                # seconds, which reads as twitchy rather than attentive.
                if self._gave_up_bearing is not None and \
                        abs(target.bearing_deg - self._gave_up_bearing) < \
                        config.WATCH_REAIM_DEG * config.WATCH_REACQUIRE_MARGIN:
                    self._time_since_reaim += dt
                else:
                    self._acquire(target.bearing_deg)
            elif abs(target.bearing_deg - self._current_bearing) > config.WATCH_REAIM_DEG:
                self._reaim(target.bearing_deg)
            elif self._time_since_reaim > config.WATCH_STATIONARY_TIMEOUT_S:
                self._give_up_stationary(fsm_owns_lamp, fsm_state, user_bearing_deg)
            else:
                self._time_since_reaim += dt
        else:
            self._lost_t += dt
            self._time_since_reaim += dt
            if self.active and self._lost_t > config.WATCH_LOST_GRACE_S:
                self._gave_up_bearing = None  # a genuine loss, not a stationary give-up --
                                                # the reacquire margin doesn't apply next time
                self._settle(fsm_owns_lamp, fsm_state, user_bearing_deg)

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
        self._gave_up_bearing = None
        self._reaim_signs = []
        self._waved_back = False
        self.lamp.set_breathing(False)
        pose = kinematics.pose_for_look_at(bearing_deg, WATCH_PITCH_DEG, alertness=0.65)
        self.lamp.set_target_pose(pose, duration=0.35, anticipation=True, overshoot=True)
        self.lamp.set_light(WATCH_COLOR, transition_s=0.25)

    def _reaim(self, bearing_deg: float) -> None:
        # Following an object already in view should read as calm
        # tracking, not a repeated startle -- no anticipation/overshoot
        # flourish here, that's reserved for the initial acquire.
        delta = bearing_deg - self._current_bearing
        self._current_bearing = bearing_deg
        self._time_since_reaim = 0.0
        pose = kinematics.pose_for_look_at(bearing_deg, WATCH_PITCH_DEG, alertness=0.65)
        self.lamp.set_target_pose(pose, duration=0.4, anticipation=False, overshoot=False)
        self._check_for_wave(delta)

    def _check_for_wave(self, delta: float) -> None:
        if self._waved_back:
            return  # already waved back this acquisition
        sign = 1.0 if delta > 0 else -1.0
        self._reaim_signs.append((self._clock, sign))
        self._reaim_signs = [(t, s) for t, s in self._reaim_signs
                              if self._clock - t <= config.WATCH_WAVE_WINDOW_S]
        reversals = sum(1 for (_, s0), (_, s1) in zip(self._reaim_signs, self._reaim_signs[1:]) if s0 != s1)
        if reversals >= config.WATCH_WAVE_MIN_REVERSALS:
            self._wave_back(sign)

    def _wave_back(self, last_sign: float) -> None:
        self._waved_back = True
        pose = kinematics.pose_for_look_at(self._current_bearing, WATCH_PITCH_DEG, alertness=0.65)
        pose[4] = last_sign * WAVE_TWIST_DEG        # wrist_roll: a quick head-cock
        pose[5] = -last_sign * WAVE_TWIST_DEG * 0.6  # head_twist: with a little spin, like a nod hello
        self.lamp.set_target_pose(pose, duration=0.3, anticipation=False, overshoot=True)
        self.lamp.play_sound("wave_back")

    def _give_up_stationary(self, fsm_owns_lamp: bool, fsm_state: State, user_bearing_deg: float | None) -> None:
        # Still visible, just not moving anymore -- remember where so a
        # small bump doesn't immediately restart the acquire flourish
        # (see WATCH_REACQUIRE_MARGIN), then settle the same way losing
        # it entirely would.
        self._gave_up_bearing = self._current_bearing
        self._settle(fsm_owns_lamp, fsm_state, user_bearing_deg)

    def _settle(self, fsm_owns_lamp: bool, fsm_state: State, user_bearing_deg: float | None) -> None:
        if fsm_owns_lamp:
            # ATTENTION_SEEKING owns the lamp right now. It owns
            # pose/light -- settling back here would stomp on whatever it
            # just set and never get corrected, since the FSM only
            # re-actuates on its *own* transitions. Just clear our flag.
            self.active = False
            self.lamp.set_breathing(True)
        elif fsm_state == State.ENGAGED:
            self._return_to_engaged_look(user_bearing_deg)
        else:
            self._stop_watch()

    def _stop_watch(self) -> None:
        self.active = False
        self.lamp.set_breathing(True)
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
        self.lamp.set_breathing(True)
        if user_bearing_deg is not None:
            pose = kinematics.pose_for_look_at(user_bearing_deg, LOOK_UP_PITCH_DEG, alertness=1.0)
            self.lamp.set_target_pose(pose, duration=0.5, anticipation=False, overshoot=False)
        self.lamp.set_light(ENGAGED_COLOR, transition_s=0.4)
