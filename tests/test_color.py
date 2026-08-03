"""Unit tests for lamp/color.py. Run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lamp import color


def test_to_dials_round_trips_from_dials():
    # Excludes very low brightness on purpose: from_dials rounds to 8-bit
    # integer channels, and dividing by a near-zero brightness to recover
    # warmth amplifies that rounding noise -- real quantization loss, not
    # a bug in the inversion itself, and visually inconsequential (a
    # nearly-off light doesn't show its hue either way).
    for warmth in (0.0, 0.2, 0.5, 0.6, 0.85, 1.0):
        for brightness in (0.3, 0.55, 0.8, 1.0):
            bgr = color.from_dials(warmth, brightness)
            recovered_warmth, recovered_brightness = color.to_dials(bgr)
            assert abs(recovered_warmth - warmth) < 0.02, f"warmth {warmth} -> {recovered_warmth}"
            assert abs(recovered_brightness - brightness) < 0.02, f"brightness {brightness} -> {recovered_brightness}"


def test_to_dials_of_off_is_zero_brightness():
    assert color.to_dials((0, 0, 0)) == (0.0, 0.0)


def test_to_dials_of_a_nudged_color_reflects_the_nudge():
    base = (0.6, 0.55)
    nudged = color.nudge(*base, dw=-0.15, db=0.2)
    bgr = color.from_dials(*nudged)
    warmth, brightness = color.to_dials(bgr)
    assert warmth < base[0]
    assert brightness > base[1]
