"""Unit + integration tests for behavior/adaptation.py (bonus: self-learning
attention-seek timing). Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
import behavior.adaptation as adaptation_mod
from behavior.adaptation import (
    AdaptationEngine, MIN_DELAY_S, MAX_DELAY_S, MIN_COOLDOWN_S, MAX_COOLDOWN_S,
    MAX_ATTEMPTS, ROLLING_WINDOW, RESPONSE_WINDOW_S,
)
from behavior.state_machine import BehaviorFSM, State


@pytest.fixture(autouse=True)
def _isolate_state_path(tmp_path, monkeypatch):
    # AdaptationEngine._record() calls self.save() on every recorded
    # sample, so any test that calls on_engaged/on_check_timeout on a real
    # engine writes to disk as a side effect -- redirect every test in
    # this file to a throwaway path so the real logs/adaptation_state.json
    # (a genuine cross-run artifact, not test scratch) never gets touched.
    monkeypatch.setattr(adaptation_mod, "STATE_PATH", tmp_path / "adaptation_state.json")


# -- isolated engine behavior, no FSM involved --------------------------

def test_response_within_window_counts_as_responded():
    engine = AdaptationEngine()
    engine.on_attention_seek_started(now=0.0)
    engine.on_engaged(now=5.0)
    assert engine.history[-1]["responded"] is True
    assert engine.summary()["samples"] == 1


def test_response_outside_window_via_timeout_counts_as_miss():
    engine = AdaptationEngine()
    engine.on_attention_seek_started(now=0.0)
    engine.on_check_timeout(now=RESPONSE_WINDOW_S + 5.0)
    assert engine.history[-1]["responded"] is False


def test_engaged_with_no_pending_attempt_is_noop():
    engine = AdaptationEngine()
    engine.on_engaged(now=5.0)
    assert engine.history == []


def test_check_timeout_before_window_elapses_is_noop():
    engine = AdaptationEngine()
    engine.on_attention_seek_started(now=0.0)
    engine.on_check_timeout(now=RESPONSE_WINDOW_S - 5.0)
    assert engine.history == []
    assert engine._pending_attempt_t == 0.0  # still pending, not resolved yet


def test_needs_min_samples_before_adapting():
    engine = AdaptationEngine()
    base_delay = engine.delay_s
    for i in range(2):  # MIN_SAMPLES_TO_ADAPT is 3 -- 2 shouldn't move anything
        engine.on_attention_seek_started(now=i * 30.0)
        engine.on_engaged(now=i * 30.0 + 1.0)
    assert engine.delay_s == base_delay


def test_high_response_rate_shortens_delay_and_cooldown():
    engine = AdaptationEngine()
    base_delay, base_cooldown = engine.delay_s, engine.cooldown_s
    for i in range(3):
        engine.on_attention_seek_started(now=i * 30.0)
        engine.on_engaged(now=i * 30.0 + 1.0)
    assert engine.delay_s < base_delay
    assert engine.cooldown_s < base_cooldown


def test_low_response_rate_lengthens_delay_and_cooldown():
    engine = AdaptationEngine()
    base_delay, base_cooldown = engine.delay_s, engine.cooldown_s
    for i in range(3):
        engine.on_attention_seek_started(now=i * 30.0)
        engine.on_check_timeout(now=i * 30.0 + RESPONSE_WINDOW_S + 5.0)
    assert engine.delay_s > base_delay
    assert engine.cooldown_s > base_cooldown


def test_max_attempts_only_adjusts_once_rolling_window_is_full():
    engine = AdaptationEngine()
    base_attempts = engine.max_attempts
    for i in range(ROLLING_WINDOW - 1):
        engine.on_attention_seek_started(now=i * 30.0)
        engine.on_engaged(now=i * 30.0 + 1.0)
    assert engine.max_attempts == base_attempts  # window not full yet

    engine.on_attention_seek_started(now=ROLLING_WINDOW * 30.0)
    engine.on_engaged(now=ROLLING_WINDOW * 30.0 + 1.0)
    assert engine.max_attempts == base_attempts + 1  # window just filled


def test_bounds_are_never_exceeded():
    engine = AdaptationEngine()
    for i in range(60):
        engine.on_attention_seek_started(now=i * 30.0)
        engine.on_engaged(now=i * 30.0 + 1.0)
    assert engine.delay_s >= MIN_DELAY_S
    assert engine.cooldown_s >= MIN_COOLDOWN_S
    assert engine.max_attempts <= MAX_ATTEMPTS

    engine2 = AdaptationEngine()
    for i in range(60):
        engine2.on_attention_seek_started(now=i * 30.0)
        engine2.on_check_timeout(now=i * 30.0 + RESPONSE_WINDOW_S + 5.0)
    assert engine2.delay_s <= MAX_DELAY_S
    assert engine2.cooldown_s <= MAX_COOLDOWN_S


def test_save_load_roundtrip():
    engine = AdaptationEngine()
    for i in range(4):
        engine.on_attention_seek_started(now=i * 30.0)
        engine.on_engaged(now=i * 30.0 + 1.0)
    engine.save()

    loaded = AdaptationEngine.load()
    assert loaded.delay_s == engine.delay_s
    assert loaded.max_attempts == engine.max_attempts
    assert loaded.cooldown_s == engine.cooldown_s
    assert len(loaded.history) == len(engine.history)


def test_load_with_no_state_file_falls_back_to_defaults():
    engine = AdaptationEngine.load()
    assert engine.delay_s == config.ATTENTION_SEEK_DELAY_S
    assert engine.history == []


# -- integration with the real FSM ---------------------------------------

class _FakeLamp:
    """Minimal no-op LampActuator -- this test cares about FSM state
    transitions and the adaptation engine's bookkeeping, not rendering."""

    def set_target_pose(self, *a, **k):
        pass

    def set_light(self, *a, **k):
        pass

    def play_sound(self, *a, **k):
        pass

    def update(self, dt):
        pass

    def get_current_pose(self):
        return None

    def get_mood(self):
        return (0.6, 0.55)


def test_multi_round_learning_with_a_consistently_responsive_user():
    """Drives the FSM through several disengage -> attention-seek ->
    quick-reengage cycles, using the exact glue logic main.py runs every
    frame (see the on_attention_seek_started/on_engaged/on_check_timeout/
    apply_to call sites in main.py's loop).

    This is the scenario that exposed the _attention_phase-stuck bug in
    state_machine.py: re-engaging immediately, right as attention-seeking
    starts, interrupts the gesture before it finishes its own phase
    animation. Before the fix in BehaviorFSM._enter_engaged (resetting
    _attention_phase back to 0), round 2 onward never reached
    ATTENTION_SEEKING again and this test hung at samples=1 forever.
    See test_state_machine.py::
    test_quick_reengagement_mid_gesture_does_not_kill_future_attention_seeking
    for the isolated FSM-only version of the same bug."""
    fsm = BehaviorFSM(lamp=_FakeLamp())
    engine = AdaptationEngine()
    engine.apply_to(fsm)

    dt = 0.05
    fake_now = [0.0]

    def tick(engaged):
        fsm.update(engaged=engaged, user_bearing_deg=0.0, dt=dt)
        fake_now[0] += dt

    tick(True)  # IDLE -> ENGAGED
    prev_state = fsm.state

    ROUNDS = 6
    for round_i in range(ROUNDS):
        tick(False)
        if prev_state != State.ATTENTION_SEEKING and fsm.state == State.ATTENTION_SEEKING:
            engine.on_attention_seek_started(now=fake_now[0])
        prev_state = fsm.state

        t = 0.0
        while t < 15.0 and fsm.state != State.ATTENTION_SEEKING:
            tick(False)
            if prev_state != State.ATTENTION_SEEKING and fsm.state == State.ATTENTION_SEEKING:
                engine.on_attention_seek_started(now=fake_now[0])
            engine.on_check_timeout(now=fake_now[0])
            engine.apply_to(fsm)
            prev_state = fsm.state
            t += dt
        assert fsm.state == State.ATTENTION_SEEKING, f"round {round_i}: never re-entered attention-seeking"

        tick(True)  # responsive user -- re-engages right away, mid-gesture
        if prev_state != State.ENGAGED and fsm.state == State.ENGAGED:
            engine.on_engaged(now=fake_now[0])
        prev_state = fsm.state

        tick(False)
        prev_state = fsm.state

    summary = engine.summary()
    assert summary["samples"] == ROUNDS
    assert summary["recent_response_rate"] == 1.0
    assert engine.delay_s < config.ATTENTION_SEEK_DELAY_S
    assert engine.cooldown_s < config.ATTENTION_SEEK_COOLDOWN_S


if __name__ == "__main__":
    test_response_within_window_counts_as_responded()
    test_response_outside_window_via_timeout_counts_as_miss()
    test_engaged_with_no_pending_attempt_is_noop()
    test_check_timeout_before_window_elapses_is_noop()
    test_needs_min_samples_before_adapting()
    test_high_response_rate_shortens_delay_and_cooldown()
    test_low_response_rate_lengthens_delay_and_cooldown()
    test_max_attempts_only_adjusts_once_rolling_window_is_full()
    test_bounds_are_never_exceeded()
    test_multi_round_learning_with_a_consistently_responsive_user()
    print("ALL ADAPTATION TESTS PASSED (except the two that need tmp_path/monkeypatch -- run via pytest)")
