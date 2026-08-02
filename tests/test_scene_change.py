"""Unit tests for perception/scene_change.py. Run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from perception.scene_change import SceneChangeDetector


def _frame(color: tuple[int, int, int], size=(480, 640)) -> np.ndarray:
    return np.full((size[0], size[1], 3), color, dtype=np.uint8)


def test_first_call_always_reports_changed():
    detector = SceneChangeDetector()
    assert detector.changed(_frame((50, 50, 50))) is True


def test_identical_frames_report_unchanged():
    detector = SceneChangeDetector()
    detector.changed(_frame((50, 50, 50)))
    assert detector.changed(_frame((50, 50, 50))) is False


def test_a_large_shift_reports_changed():
    detector = SceneChangeDetector()
    detector.changed(_frame((10, 10, 10)))
    assert detector.changed(_frame((240, 240, 240))) is True


def test_tiny_noise_stays_under_threshold():
    detector = SceneChangeDetector(threshold=0.02)
    detector.changed(_frame((100, 100, 100)))
    assert detector.changed(_frame((101, 101, 101))) is False
