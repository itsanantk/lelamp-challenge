"""Behavior FSM: turns an engagement signal into lamp actuation.

    IDLE  --looks at lamp-->  ENGAGED
    ENGAGED  --looks away-->  DISENGAGED (grace hold, then cools down)
    DISENGAGED  --stays away long enough-->  ATTENTION_SEEKING (fires once)
                --re-engages any time-->  ENGAGED
    ATTENTION_SEEKING  --exhausts attempts-->  IDLE (gives up, a little dejected)

Only calls the LampActuator interface (lamp/hal.py), so this works
unchanged against the simulator or a real-hardware backend later. No
camera/model code in here at all, which is what lets me unit test the
transitions directly (tests/test_state_machine.py) instead of only being
able to eyeball it live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

import config
from lamp import kinematics
from lamp.hal import LampActuator
from lamp.motion import lerp_color

IDLE_COLOR = (40, 110, 200)       # BGR, dim warm amber
ENGAGED_COLOR = (255, 220, 140)    # BGR, bright cool alert
ATTENTION_PULSE_COLOR = (0, 165, 255)     # BGR, warm orange pulse -- first attempt
ATTENTION_PULSE_URGENT_COLOR = (0, 80, 255)  # BGR, more saturated red-orange -- last attempt
GIVEN_UP_COLOR = (75, 85, 95)  # BGR, flat desaturated blue-gray -- deliberately cooler and
                                 # duller than the warm IDLE_COLOR, so the brief dip on giving
                                 # up reads as a little deflated. Only held for the dip itself
                                 # (see _tick_give_up) -- it then fades back to IDLE_COLOR, not
                                 # a permanent mood.

LOOK_UP_PITCH_DEG = -15.0  # desk-level lamp looking slightly up at a seated person
ATTENTION_TILT_DEG = 20.0  # wrist_roll head-cock angle -- kinematics.py calls this joint
                            # out by name as the "expressive cock the head" DOF, but nothing
                            # used it before this -- it's the single clearest way to read as
                            # a curious, confused puppy instead of a robotic alert-pulse
WATCH_HOPEFUL_ALERTNESS = 0.2  # low-alertness lean used *between* nudges within one attempt --
                                 # still oriented toward the person, not a full reset to neutral;
                                 # it hasn't given up yet, just catching its breath
# A small, brief dip -- shoulder/elbow settle a little lower and the head tilts down a
# touch, like a short sigh -- NOT a collapse. Held only for GIVE_UP_DIP_S before recovering
# to kinematics.NEUTRAL_POSE (see _tick_give_up): the reaction should read as "aww, ok" and
# then a normal resting lamp, not a lamp that fell over and stayed down.
GIVEN_UP_DIP_POSE = np.array([0.0, -38.0, 50.0, -22.0, 0.0, 0.0])
GIVE_UP_DIP_S = 0.6      # how long the dip itself is held before recovering
GIVE_UP_RECOVER_S = 1.1  # how long the recovery-to-neutral transition takes


class State(Enum):
    IDLE = auto()
    ENGAGED = auto()
    DISENGAGED = auto()
    ATTENTION_SEEKING = auto()


@dataclass
class BehaviorFSM:
    lamp: LampActuator
    state: State = State.IDLE
    # These three default from config.py but are plain instance fields, not
    # module constants read directly -- so behavior/adaptation.py (bonus:
    # self-learning) can retune them per-user at runtime without this class
    # knowing adaptation exists. Everything else about the FSM stays exactly
    # as fixed/testable as before.
    attention_seek_delay_s: float = config.ATTENTION_SEEK_DELAY_S
    attention_seek_max_attempts: int = config.ATTENTION_SEEK_MAX_ATTEMPTS
    attention_seek_cooldown_s: float = config.ATTENTION_SEEK_COOLDOWN_S
    _time_in_disengaged: float = 0.0
    _time_since_last_attempt: float = 1e9
    _attempts: int = 0
    _last_user_bearing: float = 0.0
    _attention_phase_timer: float = 0.0
    # 0 = not seeking, 1 = first nudge, 2 = puppy-eyes hold, 3 = second nudge, 4 = settle to a
    # hopeful watch (not yet given up -- see attention_seek_cooldown_s for why the gap to the
    # next attempt is now short instead of a long scheduled wait)
    _attention_phase: int = 0
    _seek_bearing_offset: float = 0.0  # set by _start_attention_seek, reused by later phases in _tick_attention_seek
    _seek_alertness: float = 0.5
    _seek_tilt_sign: float = 1.0  # alternates per attempt -- a puppy doesn't cock its head the same way either
    _give_up_phase: int = 0  # 0 = not giving up, 1 = brief dip, 2 = recovering to a normal resting look
    _give_up_timer: float = 0.0

    def update(self, engaged: bool, user_bearing_deg: float | None, dt: float,
               user_busy: bool = False) -> State:
        """user_busy: interruption awareness -- True if ambient audio
        suggests the user is talking or media is playing nearby (see
        perception/audio_monitor.py). Only gates *starting* a new
        attention-seek attempt; doesn't touch engagement itself, since
        noticing someone's looking at it isn't an interruption."""
        if user_bearing_deg is not None:
            self._last_user_bearing = user_bearing_deg

        if engaged and self.state != State.ENGAGED:
            self._enter_engaged()
        elif not engaged and self.state == State.ENGAGED:
            self._enter_disengaged()
        elif not engaged and self.state in (State.DISENGAGED, State.ATTENTION_SEEKING):
            self._tick_disengaged(dt, user_busy)

        if self.state == State.ATTENTION_SEEKING:
            self._tick_attention_seek(dt)
        elif self._give_up_phase != 0:
            self._tick_give_up(dt)

        return self.state

    # -- transitions ----------------------------------------------------

    def _enter_engaged(self) -> None:
        self.state = State.ENGAGED
        self._attempts = 0
        self._time_in_disengaged = 0.0
        # Re-engaging can interrupt an attention-seek gesture mid-animation
        # (phase 1 or 2, not yet back to 0) -- without this reset that
        # nonzero phase sticks forever, since _tick_attention_seek (the only
        # place that advances it back to 0) only runs while
        # state == ATTENTION_SEEKING. A stuck phase silently blocks every
        # future attention-seek attempt via the _attention_phase != 0 guard
        # in _tick_disengaged, i.e. one quick glance at the lamp mid-wiggle
        # permanently kills the behavior for the rest of the session.
        self._attention_phase = 0
        self._attention_phase_timer = 0.0
        self._give_up_phase = 0  # same reasoning -- don't leave a stale give-up dip/recover
        # pending to fire later after an unrelated future disengage
        target = kinematics.pose_for_look_at(self._last_user_bearing, LOOK_UP_PITCH_DEG, alertness=1.0)
        self.lamp.set_target_pose(target, duration=0.5, anticipation=True, overshoot=True)
        self.lamp.set_light(ENGAGED_COLOR, transition_s=0.3)
        self.lamp.play_sound("engage")

    def _enter_disengaged(self) -> None:
        self.state = State.DISENGAGED
        self._time_in_disengaged = 0.0
        self._time_since_last_attempt = 1e9
        # Settle slowly toward neutral rather than snapping -- a lamp that
        # instantly drops back to rest reads as mechanical, not wistful.
        self.lamp.set_target_pose(kinematics.NEUTRAL_POSE, duration=config.DISENGAGE_GRACE_S, overshoot=False)
        self.lamp.set_light(IDLE_COLOR, transition_s=config.DISENGAGE_GRACE_S)
        self.lamp.play_sound("disengage")

    def _tick_disengaged(self, dt: float, user_busy: bool = False) -> None:
        self._time_in_disengaged += dt
        self._time_since_last_attempt += dt

        if self._time_in_disengaged < config.DISENGAGE_GRACE_S:
            return  # still in the brief "hopeful" hold

        if self._attention_phase != 0:
            return  # nudge/hold/settle animation still playing -- let it
            # finish and hand control back before deciding anything else

        if self._attempts >= self.attention_seek_max_attempts:
            self._give_up()
            return

        if user_busy:
            return  # don't talk over someone -- wait for the room to go quiet

        if self._time_since_last_attempt >= self.attention_seek_cooldown_s and \
                self._time_in_disengaged >= self.attention_seek_delay_s:
            self._start_attention_seek()

    def _start_attention_seek(self) -> None:
        self.state = State.ATTENTION_SEEKING
        self._attempts += 1
        self._time_since_last_attempt = 0.0
        self._attention_phase = 1
        self._attention_phase_timer = 0.0
        self._seek_tilt_sign = -self._seek_tilt_sign

        # The exact same gesture on every attempt reads as a broken loop,
        # not persistence -- lean in a little further and pulse a touch
        # more saturated each try, capped at max_attempts so the last one
        # is the most noticeable. Kept modest on purpose (a small lean,
        # not a lunge) -- this should read as "did you notice me?"
        # curiosity, not an alarm demanding attention.
        intensity = min(self._attempts / self.attention_seek_max_attempts, 1.0)
        self._seek_bearing_offset = 8.0 + 5.0 * intensity
        self._seek_alertness = 0.4 + 0.2 * intensity
        color = lerp_color(ATTENTION_PULSE_COLOR, ATTENTION_PULSE_URGENT_COLOR, intensity)

        # Phase 1: the first nudge -- a small lean-in toward the person,
        # with the sound. A head-tilt is already starting here rather than
        # snapping in fully, so the whole gesture reads as one continuous
        # motion into the hold instead of two disconnected poses.
        nudge = kinematics.pose_for_look_at(self._last_user_bearing + self._seek_bearing_offset, -5.0,
                                             alertness=self._seek_alertness)
        nudge[4] = self._seek_tilt_sign * ATTENTION_TILT_DEG * 0.5  # wrist_roll: cock the head, partway
        self.lamp.set_target_pose(nudge, duration=0.35, anticipation=True, overshoot=True)
        self.lamp.set_light(color, transition_s=0.3)
        self.lamp.play_sound("attention_seek")

    def _tick_attention_seek(self, dt: float) -> None:
        self._attention_phase_timer += dt
        if self._attention_phase == 1 and self._attention_phase_timer >= 0.45:
            # Phase 2: tilt the head the rest of the way and just look --
            # the "puppy eyes" hold, the longest beat in the sequence on
            # purpose. No anticipation/overshoot here -- a flourish on a
            # hold would read as a twitch, not a wistful stare.
            self._attention_phase = 2
            self._attention_phase_timer = 0.0
            look = kinematics.pose_for_look_at(self._last_user_bearing + self._seek_bearing_offset, -5.0,
                                                alertness=self._seek_alertness)
            look[4] = self._seek_tilt_sign * ATTENTION_TILT_DEG
            self.lamp.set_target_pose(look, duration=0.4, anticipation=False, overshoot=False)
        elif self._attention_phase == 2 and self._attention_phase_timer >= 1.1:
            # Phase 3: the second nudge -- tries again, tilting back the
            # other way, like repositioning for another look rather than
            # repeating the exact same pose.
            self._attention_phase = 3
            self._attention_phase_timer = 0.0
            nudge2 = kinematics.pose_for_look_at(self._last_user_bearing + self._seek_bearing_offset * 0.7, -5.0,
                                                  alertness=self._seek_alertness * 0.9)
            nudge2[4] = -self._seek_tilt_sign * ATTENTION_TILT_DEG * 0.7
            self.lamp.set_target_pose(nudge2, duration=0.35, anticipation=True, overshoot=True)
        elif self._attention_phase == 3 and self._attention_phase_timer >= 0.5:
            # Phase 4: settle, but only to a quiet hopeful watch, not
            # fully neutral -- it hasn't given up between nudges, just
            # resting a beat.
            self._attention_phase = 4
            self._attention_phase_timer = 0.0
            watching = kinematics.pose_for_look_at(self._last_user_bearing, -5.0, alertness=WATCH_HOPEFUL_ALERTNESS)
            self.lamp.set_target_pose(watching, duration=0.6, anticipation=False, overshoot=False)
            self.lamp.set_light(IDLE_COLOR, transition_s=0.6)
        elif self._attention_phase == 4 and self._attention_phase_timer >= 0.4:
            self.state = State.DISENGAGED
            self._attention_phase = 0

    def _give_up(self) -> None:
        # Every attempt failed -- dip briefly and play a small sigh so
        # being ignored actually reads as a little dejected, then recover
        # to a normal resting lamp (_tick_give_up) rather than staying
        # slumped forever. No lingering mood beyond that either way:
        # re-engaging any time goes straight back to _enter_engaged's
        # normal energetic look, since that always fires unconditionally
        # on the ENGAGED transition regardless of what state came before.
        self.state = State.IDLE
        self._give_up_phase = 1
        self._give_up_timer = 0.0
        self.lamp.set_target_pose(GIVEN_UP_DIP_POSE, duration=0.5, overshoot=False)
        self.lamp.set_light(GIVEN_UP_COLOR, transition_s=0.5)
        self.lamp.play_sound("give_up")

    def _tick_give_up(self, dt: float) -> None:
        self._give_up_timer += dt
        if self._give_up_phase == 1 and self._give_up_timer >= GIVE_UP_DIP_S:
            # Recover to a normal resting look -- a brief dip reads as
            # "aww, ok"; staying slumped forever reads as broken, not sad.
            self._give_up_phase = 2
            self._give_up_timer = 0.0
            self.lamp.set_target_pose(kinematics.NEUTRAL_POSE, duration=GIVE_UP_RECOVER_S, overshoot=False)
            self.lamp.set_light(IDLE_COLOR, transition_s=GIVE_UP_RECOVER_S)
        elif self._give_up_phase == 2 and self._give_up_timer >= GIVE_UP_RECOVER_S:
            self._give_up_phase = 0

    def debug_info(self) -> dict:
        return {
            "state": self.state.name,
            "attempts": self._attempts,
            "time_in_disengaged": round(self._time_in_disengaged, 1),
        }
