"""Unit tests for the behavior FSM against a fake (non-rendering)
LampActuator. Run with: python -m pytest tests/ -v  (or just run this file).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from behavior.state_machine import BehaviorFSM, IDLE_COLOR, State


class FakeLamp:
    def __init__(self):
        self.calls = []
        self.pose = np.zeros(6)

    def set_target_pose(self, angles, duration=0.6, anticipation=True, overshoot=True):
        self.calls.append(("pose", tuple(np.round(angles, 1))))
        self.pose = np.array(angles)

    def set_light(self, rgb, transition_s=0.4):
        self.calls.append(("light", rgb))

    def play_sound(self, event):
        self.calls.append(("sound", event))

    def update(self, dt):
        pass

    def get_current_pose(self):
        return self.pose


def test_idle_to_engaged():
    fsm = BehaviorFSM(lamp=FakeLamp())
    assert fsm.state == State.IDLE
    fsm.update(engaged=True, user_bearing_deg=10.0, dt=0.03)
    assert fsm.state == State.ENGAGED
    assert ("sound", "engage") in fsm.lamp.calls


def test_engaged_to_disengaged_grace_then_attention_seek():
    fsm = BehaviorFSM(lamp=FakeLamp())
    fsm.update(engaged=True, user_bearing_deg=0.0, dt=0.03)
    fsm.update(engaged=False, user_bearing_deg=0.0, dt=0.03)
    assert fsm.state == State.DISENGAGED

    # step through the grace period without re-engaging
    for _ in range(int((config.DISENGAGE_GRACE_S + 0.1) / 0.05)):
        fsm.update(engaged=False, user_bearing_deg=0.0, dt=0.05)
    assert fsm.state == State.DISENGAGED  # grace passed, still waiting for attention-seek delay

    # step until attention-seek delay elapses
    t = 0.0
    while t < config.ATTENTION_SEEK_DELAY_S + 0.2 and fsm.state != State.ATTENTION_SEEKING:
        fsm.update(engaged=False, user_bearing_deg=0.0, dt=0.05)
        t += 0.05
    assert fsm.state == State.ATTENTION_SEEKING
    assert ("sound", "attention_seek") in fsm.lamp.calls


def test_reengagement_cancels_attention_seeking():
    fsm = BehaviorFSM(lamp=FakeLamp())
    fsm.update(engaged=True, user_bearing_deg=0.0, dt=0.03)
    fsm.update(engaged=False, user_bearing_deg=0.0, dt=0.03)
    fsm.update(engaged=True, user_bearing_deg=5.0, dt=0.03)
    assert fsm.state == State.ENGAGED
    assert fsm._attempts == 0


def test_user_busy_suppresses_attention_seeking():
    fsm = BehaviorFSM(lamp=FakeLamp())
    fsm.update(engaged=True, user_bearing_deg=0.0, dt=0.03)
    fsm.update(engaged=False, user_bearing_deg=0.0, dt=0.03)

    # step well past when it would normally start attention-seeking, but
    # with user_busy=True the whole time -- should never trigger
    t = 0.0
    while t < config.ATTENTION_SEEK_DELAY_S + config.DISENGAGE_GRACE_S + 2.0:
        fsm.update(engaged=False, user_bearing_deg=0.0, dt=0.05, user_busy=True)
        t += 0.05
    assert fsm.state == State.DISENGAGED
    assert fsm._attempts == 0

    # once the room goes quiet, it should proceed normally
    t = 0.0
    while t < config.ATTENTION_SEEK_DELAY_S + 0.5 and fsm.state != State.ATTENTION_SEEKING:
        fsm.update(engaged=False, user_bearing_deg=0.0, dt=0.05, user_busy=False)
        t += 0.05
    assert fsm.state == State.ATTENTION_SEEKING


def test_quick_reengagement_mid_gesture_does_not_kill_future_attention_seeking():
    """Re-engaging fast enough to interrupt the attention-seek gesture
    itself (before it finishes its perk-up/settle-back phases) used to
    leave _attention_phase stuck at a nonzero value forever, since only
    _tick_attention_seek ever advances it back to 0 and that only runs
    while state == ATTENTION_SEEKING. The stuck value then silently
    blocked every later attention-seek attempt via the
    _attention_phase != 0 guard in _tick_disengaged -- one quick glance at
    the lamp mid-wiggle would permanently kill the behavior for the rest
    of the session. Runs two full disengage cycles to prove the second one
    still fires."""
    fsm = BehaviorFSM(lamp=FakeLamp())
    fsm.update(engaged=True, user_bearing_deg=0.0, dt=0.03)

    for round_i in range(2):
        fsm.update(engaged=False, user_bearing_deg=0.0, dt=0.05)
        t = 0.0
        while t < config.ATTENTION_SEEK_DELAY_S + config.DISENGAGE_GRACE_S + 1.0 \
                and fsm.state != State.ATTENTION_SEEKING:
            fsm.update(engaged=False, user_bearing_deg=0.0, dt=0.05)
            t += 0.05
        assert fsm.state == State.ATTENTION_SEEKING, f"round {round_i}: never re-entered attention-seeking"

        # re-engage immediately -- interrupts the gesture well before its
        # ~1.1s perk-up/settle-back animation would finish on its own
        fsm.update(engaged=True, user_bearing_deg=0.0, dt=0.05)
        assert fsm.state == State.ENGAGED
        assert fsm._attention_phase == 0, f"round {round_i}: attention phase left stuck at {fsm._attention_phase}"


def test_gives_up_after_max_attempts():
    fsm = BehaviorFSM(lamp=FakeLamp())
    fsm.update(engaged=True, user_bearing_deg=0.0, dt=0.03)
    fsm.update(engaged=False, user_bearing_deg=0.0, dt=0.03)

    t = 0.0
    max_t = (config.DISENGAGE_GRACE_S + config.ATTENTION_SEEK_MAX_ATTEMPTS *
              (config.ATTENTION_SEEK_COOLDOWN_S + 1.5) + 5.0)
    while t < max_t and fsm.state != State.IDLE:
        fsm.update(engaged=False, user_bearing_deg=0.0, dt=0.05)
        t += 0.05
    assert fsm.state == State.IDLE
    assert fsm._attempts == config.ATTENTION_SEEK_MAX_ATTEMPTS


def test_attention_seek_escalates_across_attempts():
    """The same gesture on every attempt reads as a broken loop, not
    persistence -- later attempts should lean in further (bigger bearing
    offset) and pulse a more saturated color than the first."""
    fsm = BehaviorFSM(lamp=FakeLamp())
    fsm.update(engaged=True, user_bearing_deg=0.0, dt=0.03)
    fsm.update(engaged=False, user_bearing_deg=0.0, dt=0.03)

    light_colors = []
    t = 0.0
    max_t = (config.DISENGAGE_GRACE_S + config.ATTENTION_SEEK_MAX_ATTEMPTS *
              (config.ATTENTION_SEEK_COOLDOWN_S + 1.5) + 5.0)
    prev_state = fsm.state
    while t < max_t and fsm.state != State.IDLE:
        fsm.update(engaged=False, user_bearing_deg=0.0, dt=0.05)
        if prev_state != State.ATTENTION_SEEKING and fsm.state == State.ATTENTION_SEEKING:
            light_colors.append(next(c for kind, c in reversed(fsm.lamp.calls) if kind == "light"))
        prev_state = fsm.state
        t += 0.05

    assert len(light_colors) == config.ATTENTION_SEEK_MAX_ATTEMPTS
    assert light_colors[0] != light_colors[-1]


def test_give_up_dips_briefly_then_recovers_to_a_normal_resting_look():
    """The final attention-seek attempt used to cut off mid-animation (state
    flipped to IDLE while _attention_phase was still nonzero), leaving the
    lamp stuck holding the curious pose and orange light forever. Runs
    against the real SimulatedLamp, not FakeLamp, so it checks actual eased
    pose/light state instead of just which calls got made.

    Giving up now dips briefly (GIVEN_UP_DIP_POSE/GIVEN_UP_COLOR) so being
    ignored reads as a little dejected, then recovers to the normal
    NEUTRAL_POSE/IDLE_COLOR resting look -- not a pose it stays slumped in
    forever, which read as broken rather than sad."""
    from behavior.state_machine import GIVE_UP_DIP_S, GIVE_UP_RECOVER_S, GIVEN_UP_COLOR, GIVEN_UP_DIP_POSE
    from lamp import SimulatedLamp, kinematics

    lamp = SimulatedLamp(mute=True)
    fsm = BehaviorFSM(lamp=lamp)
    fsm.update(engaged=True, user_bearing_deg=0.0, dt=0.05)
    lamp.update(0.05)

    t, dt = 0.0, 0.05
    while t < 60.0 and fsm.state != State.IDLE:
        fsm.update(engaged=False, user_bearing_deg=0.0, dt=dt)
        lamp.update(dt)
        t += dt

    assert fsm.state == State.IDLE

    # _give_up()'s dip starts on this exact tick, so the loop above
    # stopping the instant state flips to IDLE cuts it off before it's
    # actually played out -- keep ticking, same as a real running app
    # would (main.py never stops calling lamp.update() just because the
    # FSM state changed). Check partway through the dip...
    for _ in range(int(GIVE_UP_DIP_S / dt) - 1):
        fsm.update(engaged=False, user_bearing_deg=0.0, dt=dt)
        lamp.update(dt)
    assert np.allclose(lamp.get_current_pose(), GIVEN_UP_DIP_POSE, atol=2.0), \
        f"lamp pose did not dip on giving up: {lamp.get_current_pose()}"
    assert all(abs(a - b) <= 2 for a, b in zip(lamp.get_current_light(), GIVEN_UP_COLOR)), \
        f"lamp light did not dip on giving up: {lamp.get_current_light()}"

    # ...then confirm it actually recovers, rather than staying dipped.
    for _ in range(int(GIVE_UP_RECOVER_S / dt) + 5):
        fsm.update(engaged=False, user_bearing_deg=0.0, dt=dt)
        lamp.update(dt)
    assert np.allclose(lamp.get_current_pose(), kinematics.NEUTRAL_POSE, atol=2.0), \
        f"lamp pose did not recover to neutral after the give-up dip: {lamp.get_current_pose()}"
    assert all(abs(a - b) <= 2 for a, b in zip(lamp.get_current_light(), IDLE_COLOR)), \
        f"lamp light did not recover to idle after the give-up dip: {lamp.get_current_light()}"
    lamp.close()


if __name__ == "__main__":
    test_idle_to_engaged()
    test_engaged_to_disengaged_grace_then_attention_seek()
    test_reengagement_cancels_attention_seeking()
    test_user_busy_suppresses_attention_seeking()
    test_quick_reengagement_mid_gesture_does_not_kill_future_attention_seeking()
    test_gives_up_after_max_attempts()
    test_give_up_settles_the_real_lamp_not_just_fsm_state()
    print("ALL STATE MACHINE TESTS PASSED")
