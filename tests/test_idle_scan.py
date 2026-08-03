"""Unit tests for behavior/idle_scan.py. Run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from behavior.idle_scan import IdleScanner
from behavior.state_machine import State


class _FakeLamp:
    def __init__(self):
        self.calls = []

    def set_target_pose(self, angles, duration=0.6, anticipation=True, overshoot=True):
        self.calls.append(("pose", tuple(angles)))


def _tick(scanner, fsm_state=State.IDLE, object_watch_active=False, dt=0.5, n=1):
    for _ in range(n):
        scanner.update(fsm_state, object_watch_active, dt)


def test_stays_idle_and_does_nothing_before_the_delay_elapses():
    lamp = _FakeLamp()
    scanner = IdleScanner(lamp)
    _tick(scanner, n=int(config.IDLE_SCAN_DELAY_S / 0.5) - 2)
    assert not scanner.active
    assert lamp.calls == []


def test_starts_sweeping_once_idle_long_enough():
    lamp = _FakeLamp()
    scanner = IdleScanner(lamp)
    _tick(scanner, n=int(config.IDLE_SCAN_DELAY_S / 0.5) + 2)
    assert scanner.active
    assert any(c[0] == "pose" for c in lamp.calls)


def test_engagement_preempts_without_issuing_a_pose_command():
    # Preemption must be silent -- see idle_scan.py's docstring on why a
    # competing neutral-pose command here would race whatever the FSM/
    # ObjectWatcher is about to set on the very same tick.
    lamp = _FakeLamp()
    scanner = IdleScanner(lamp)
    _tick(scanner, n=int(config.IDLE_SCAN_DELAY_S / 0.5) + 2)
    assert scanner.active
    lamp.calls.clear()

    scanner.update(State.ENGAGED, False, 0.05)
    assert not scanner.active
    assert lamp.calls == []


def test_an_object_being_tracked_preempts_without_issuing_a_pose_command():
    lamp = _FakeLamp()
    scanner = IdleScanner(lamp)
    _tick(scanner, n=int(config.IDLE_SCAN_DELAY_S / 0.5) + 2)
    assert scanner.active
    lamp.calls.clear()

    scanner.update(State.IDLE, True, 0.05)  # object_watch_active=True
    assert not scanner.active
    assert lamp.calls == []


def test_idle_timer_resets_while_not_eligible():
    lamp = _FakeLamp()
    scanner = IdleScanner(lamp)
    _tick(scanner, n=int(config.IDLE_SCAN_DELAY_S / 0.5) - 2)
    assert not scanner.active

    scanner.update(State.ENGAGED, False, 0.05)  # briefly not idle
    lamp.calls.clear()

    # should need the *full* delay again, not just the couple of ticks left before
    _tick(scanner, n=int(config.IDLE_SCAN_DELAY_S / 0.5) - 2)
    assert not scanner.active


def test_completed_sweep_settles_back_to_neutral():
    lamp = _FakeLamp()
    scanner = IdleScanner(lamp)
    _tick(scanner, n=int(config.IDLE_SCAN_DELAY_S / 0.5) + 2)
    assert scanner.active

    # run past every waypoint's hold time -- 5 waypoints max
    t, dt = 0.0, 0.5
    while t < 5 * (config.IDLE_SCAN_HOLD_S + 1.0) and scanner.active:
        scanner.update(State.IDLE, False, dt)
        t += dt
    assert not scanner.active


if __name__ == "__main__":
    test_stays_idle_and_does_nothing_before_the_delay_elapses()
    test_starts_sweeping_once_idle_long_enough()
    test_engagement_preempts_without_issuing_a_pose_command()
    test_an_object_being_tracked_preempts_without_issuing_a_pose_command()
    test_idle_timer_resets_while_not_eligible()
    test_completed_sweep_settles_back_to_neutral()
    print("ALL IDLE SCAN TESTS PASSED")
