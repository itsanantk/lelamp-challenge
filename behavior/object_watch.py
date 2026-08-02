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

Wave-back is driven from outside this class now (perception/hand_wave.py,
via main.py) -- an actual hand-wave detector, not this module's own
oscillating-bearing proxy. play_wave_back() below is the reaction itself
(pose + sound), kept here since it's the same "head-cock-with-spin"
gesture and the only other place duration/color for it would need to
agree; whoever calls it (this class or the hand-wave path) supplies the
bearing to point it at.

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

import numpy as np

import config
from behavior.state_machine import ENGAGED_NUDGE, LOOK_UP_PITCH_DEG, State
from lamp import color, kinematics
from lamp.hal import LampActuator

# Cooler + a touch brighter than whatever mood is currently showing --
# curious, alert, but a *nudge*, not the independent aqua/green-cyan
# absolute color this used to be (see the module docstring on why that
# read as inconsistent: "warm light -> tracks phone -> aqua -> back to
# white" instead of staying visually related to the chosen mood).
WATCH_NUDGE = (-0.12, 0.10)
WATCH_PITCH_DEG = -30.0        # looking down at desk level
# pose_for_look_at blends the aim pose with NEUTRAL_POSE by this factor --
# NEUTRAL_POSE's base_yaw is 0, so anything below 1.0 *damps* the actual
# turn: at the old 0.65, a phone at 30 deg only got a 19.5 deg turn, which
# is why tracking looked like it wasn't clearly pointed at the object.
# 0.92, not 1.0 -- still keeps a hint of the "settled" softness the same
# function gives every other alertness<1 pose, but the arm should
# basically always land on the real bearing while actively tracking.
WATCH_ALERTNESS = 0.92
WAVE_TWIST_DEG = 25.0          # head_twist flick for the wave-back -- kinematics.py calls this
                                 # joint out as the "blink/twitch accents" DOF, a natural fit
# How much the head visibly turns with the arm while tracking (wrist_roll
# cock + head_twist), scaled by how far a re-aim just moved -- previously
# only base_yaw/shoulder/elbow ever changed while tracking, so the head
# itself looked static even as the arm swung, which read as flat/2D.
TRACK_HEAD_TWIST_PER_DEG = 0.4
TRACK_HEAD_TWIST_MAX_DEG = 12.0
TRACK_WRIST_ROLL_PER_DEG = 0.2
TRACK_WRIST_ROLL_MAX_DEG = 8.0


def play_wave_back(lamp: LampActuator, bearing_deg: float) -> None:
    """Plays the lamp's wave-back reaction pointed at bearing_deg -- a
    quick head-cock with a little spin, like a nod hello. Doesn't depend
    on ObjectWatcher's own tracking state, so a hand-wave detected off to
    the side of (or instead of) whatever object is currently being
    tracked can still trigger it."""
    pose = kinematics.pose_for_look_at(bearing_deg, WATCH_PITCH_DEG, alertness=WATCH_ALERTNESS)
    pose[4] = WAVE_TWIST_DEG           # wrist_roll: a quick head-cock
    pose[5] = -WAVE_TWIST_DEG * 0.6    # head_twist: with a little spin
    lamp.set_target_pose(pose, duration=0.3, anticipation=False, overshoot=True)
    lamp.play_sound("wave_back")


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

    def update(self, detections, fsm_state: State, dt: float,
               user_bearing_deg: float | None = None) -> None:
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

    def _watch_color(self) -> tuple[int, int, int]:
        return color.from_dials(*color.nudge(*self.lamp.get_mood(), *WATCH_NUDGE))

    def _acquire(self, bearing_deg: float) -> None:
        self.active = True
        self._current_bearing = bearing_deg
        self._time_since_reaim = 0.0
        self._gave_up_bearing = None
        self.lamp.set_breathing(False)
        pose = kinematics.pose_for_look_at(bearing_deg, WATCH_PITCH_DEG, alertness=WATCH_ALERTNESS)
        # A small head_twist toward the object on first sighting -- see
        # _reaim for why the head otherwise looked static while tracking.
        pose[5] = float(np.clip(bearing_deg * TRACK_HEAD_TWIST_PER_DEG * 0.3, -TRACK_HEAD_TWIST_MAX_DEG,
                                 TRACK_HEAD_TWIST_MAX_DEG))
        self.lamp.set_target_pose(pose, duration=0.35, anticipation=True, overshoot=True)
        self.lamp.set_light(self._watch_color(), transition_s=0.25)

    def _reaim(self, bearing_deg: float) -> None:
        # Following an object already in view should read as calm
        # tracking, not a repeated startle -- no anticipation/overshoot
        # flourish here, that's reserved for the initial acquire.
        delta = bearing_deg - self._current_bearing
        self._current_bearing = bearing_deg
        self._time_since_reaim = 0.0
        pose = kinematics.pose_for_look_at(bearing_deg, WATCH_PITCH_DEG, alertness=WATCH_ALERTNESS)
        # wrist_roll/head_twist scaled by how far this re-aim just moved --
        # previously only base_yaw/shoulder/elbow ever changed while
        # tracking, so the head looked static even as the arm swung,
        # reading as flat. A twitch proportional to the turn itself (not a
        # fixed amount) means small tracking corrections stay subtle and
        # only a real, larger move visibly turns the head.
        pose[4] = float(np.clip(delta * TRACK_WRIST_ROLL_PER_DEG, -TRACK_WRIST_ROLL_MAX_DEG, TRACK_WRIST_ROLL_MAX_DEG))
        pose[5] = float(np.clip(delta * TRACK_HEAD_TWIST_PER_DEG, -TRACK_HEAD_TWIST_MAX_DEG, TRACK_HEAD_TWIST_MAX_DEG))
        self.lamp.set_target_pose(pose, duration=0.4, anticipation=False, overshoot=False)

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
        self.lamp.set_light(color.from_dials(*self.lamp.get_mood()), transition_s=0.6)

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
        self.lamp.set_light(color.from_dials(*color.nudge(*self.lamp.get_mood(), *ENGAGED_NUDGE)), transition_s=0.4)
