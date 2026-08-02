"""Behavior FSM: turns an engagement signal into lamp actuation.

    IDLE  --looks at lamp-->  ENGAGED
    ENGAGED  --looks away-->  DISENGAGED (grace hold, then cools down)
    DISENGAGED  --stays away long enough-->  ATTENTION_SEEKING (fires once)
                --re-engages any time-->  ENGAGED
    ATTENTION_SEEKING  --exhausts attempts-->  IDLE (stops nagging)

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

LOOK_UP_PITCH_DEG = -15.0  # desk-level lamp looking slightly up at a seated person


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
    _attention_phase: int = 0  # 0 = not seeking, 1 = perk up, 2 = settle back

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
            return  # perk-up/settle-back animation still playing -- let it
            # finish and hand control back before deciding anything else

        if self._attempts >= self.attention_seek_max_attempts:
            self.state = State.IDLE  # stop nagging
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

        # The exact same gesture on every attempt reads as a broken loop,
        # not persistence -- lean in further and pulse brighter each try,
        # capped at max_attempts so the last one is the most noticeable.
        intensity = min(self._attempts / self.attention_seek_max_attempts, 1.0)
        bearing_offset = 14.0 + 10.0 * intensity
        alertness = 0.5 + 0.3 * intensity
        color = lerp_color(ATTENTION_PULSE_COLOR, ATTENTION_PULSE_URGENT_COLOR, intensity)

        curious = kinematics.pose_for_look_at(self._last_user_bearing + bearing_offset, -5.0, alertness=alertness)
        self.lamp.set_target_pose(curious, duration=0.35, anticipation=True, overshoot=True)
        self.lamp.set_light(color, transition_s=0.25)
        self.lamp.play_sound("attention_seek")

    def _tick_attention_seek(self, dt: float) -> None:
        self._attention_phase_timer += dt
        if self._attention_phase == 1 and self._attention_phase_timer >= 0.6:
            # Settle back down; still "waiting," not yet given up.
            self._attention_phase = 2
            self._attention_phase_timer = 0.0
            self.lamp.set_target_pose(kinematics.NEUTRAL_POSE, duration=0.5, overshoot=False)
            self.lamp.set_light(IDLE_COLOR, transition_s=0.5)
        elif self._attention_phase == 2 and self._attention_phase_timer >= 0.5:
            self.state = State.DISENGAGED
            self._attention_phase = 0

    def debug_info(self) -> dict:
        return {
            "state": self.state.name,
            "attempts": self._attempts,
            "time_in_disengaged": round(self._time_in_disengaged, 1),
        }
