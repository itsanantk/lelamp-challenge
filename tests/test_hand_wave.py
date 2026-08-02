"""Unit tests for perception/hand_wave.py's _WaveTracker -- the pure
reversal-counting logic, split out from HandWaveDetector so it's testable
without loading a real MediaPipe hand-landmark model (same split as
conversation/voice.py's _check_wake_chunk vs wait_for_wake_word).
Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from perception.hand_wave import _WaveTracker


def test_waving_back_and_forth_returns_a_bearing():
    tracker = _WaveTracker()
    t, x, step = 0.0, 0.5, config.HAND_WAVE_MIN_DELTA + 0.03
    bearing = None
    for i in range(7):  # enough alternating moves to cross WATCH_WAVE_MIN_REVERSALS
        x += step if i % 2 == 0 else -step
        t += 0.1
        bearing = tracker.feed(x, t)
        if bearing is not None:
            break
    assert bearing is not None


def test_one_directional_movement_never_triggers_a_wave():
    tracker = _WaveTracker()
    t, x = 0.0, 0.5
    for _ in range(8):
        x += config.HAND_WAVE_MIN_DELTA + 0.03
        t += 0.1
        assert tracker.feed(x, t) is None


def test_sub_threshold_jitter_is_ignored():
    tracker = _WaveTracker()
    t, x = 0.0, 0.5
    for i in range(8):
        x += (config.HAND_WAVE_MIN_DELTA * 0.3) * (1 if i % 2 == 0 else -1)
        t += 0.1
        assert tracker.feed(x, t) is None


def test_wave_only_fires_once_until_reset():
    tracker = _WaveTracker()
    t, x, step = 0.0, 0.5, config.HAND_WAVE_MIN_DELTA + 0.03
    fired = 0
    for i in range(14):
        x += step if i % 2 == 0 else -step
        t += 0.1
        if tracker.feed(x, t) is not None:
            fired += 1
    assert fired == 1


def test_reset_allows_a_new_wave_to_fire_again():
    tracker = _WaveTracker()
    t, x, step = 0.0, 0.5, config.HAND_WAVE_MIN_DELTA + 0.03
    for i in range(7):
        x += step if i % 2 == 0 else -step
        t += 0.1
        tracker.feed(x, t)

    tracker.reset()  # hand left frame and came back

    bearing = None
    for i in range(7):
        x += step if i % 2 == 0 else -step
        t += 0.1
        bearing = tracker.feed(x, t)
        if bearing is not None:
            break
    assert bearing is not None


def test_reversals_outside_the_window_do_not_count():
    tracker = _WaveTracker()
    step = config.HAND_WAVE_MIN_DELTA + 0.03
    x, t = 0.5, 0.0
    # two reversals, then a long gap that pushes them outside the window,
    # then only one more -- should never reach WATCH_WAVE_MIN_REVERSALS
    for i in range(3):
        x += step if i % 2 == 0 else -step
        t += 0.1
        assert tracker.feed(x, t) is None
    t += config.WATCH_WAVE_WINDOW_S + 1.0
    x += step
    assert tracker.feed(x, t) is None
