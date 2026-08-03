"""Unit tests for perception/ambient_light.py. Run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from perception.ambient_light import AmbientLightSensor


def _frame(value: int) -> np.ndarray:
    return np.full((100, 200, 3), value, dtype=np.uint8)


def test_dark_room_produces_a_positive_nudge():
    sensor = AmbientLightSensor()
    sensor.update(_frame(10), now=0.0)
    assert sensor.brightness_nudge() > 0


def test_bright_room_produces_a_negative_nudge():
    sensor = AmbientLightSensor()
    sensor.update(_frame(250), now=0.0)
    assert sensor.brightness_nudge() < 0


def test_normal_room_produces_a_small_nudge():
    sensor = AmbientLightSensor()
    midpoint_value = int(config.AMBIENT_LIGHT_MIDPOINT * 255)
    sensor.update(_frame(midpoint_value), now=0.0)
    assert abs(sensor.brightness_nudge()) < 0.02


def test_nudge_is_capped():
    sensor = AmbientLightSensor()
    sensor.update(_frame(0), now=0.0)
    assert sensor.brightness_nudge() <= config.AMBIENT_LIGHT_MAX_NUDGE


def test_does_not_resample_before_the_interval_elapses():
    sensor = AmbientLightSensor()
    sensor.update(_frame(10), now=0.0)  # dark
    before = sensor.brightness_nudge()
    sensor.update(_frame(250), now=0.1)  # bright, but too soon to resample
    assert sensor.brightness_nudge() == before


def test_resamples_and_smooths_after_the_interval():
    sensor = AmbientLightSensor()
    sensor.update(_frame(10), now=0.0)  # dark
    dark_nudge = sensor.brightness_nudge()
    sensor.update(_frame(250), now=config.AMBIENT_LIGHT_SAMPLE_INTERVAL_S + 0.1)  # bright, past the interval
    bright_nudge = sensor.brightness_nudge()
    assert bright_nudge < dark_nudge


def test_a_small_bright_region_still_registers_despite_a_normal_frame_average():
    # The actual flashlight-at-camera scenario: a bright source lights up
    # only part of the frame while the rest stays at a normal room level
    # -- a plain frame-wide mean dilutes that down to barely moving.
    # Percentile-based reading should still catch it clearly.
    frame = _frame(int(config.AMBIENT_LIGHT_MIDPOINT * 255))  # normal room level everywhere
    frame[:50, :90] = 255  # a bright patch covering roughly a quarter of the frame
    plain_mean = float(np.mean(frame)) / 255.0

    sensor = AmbientLightSensor()
    sensor.update(frame, now=0.0)

    assert sensor._smoothed_luma > plain_mean + 0.1


if __name__ == "__main__":
    test_dark_room_produces_a_positive_nudge()
    test_bright_room_produces_a_negative_nudge()
    test_normal_room_produces_a_small_nudge()
    test_nudge_is_capped()
    test_does_not_resample_before_the_interval_elapses()
    test_resamples_and_smooths_after_the_interval()
    print("ALL AMBIENT LIGHT TESTS PASSED")
