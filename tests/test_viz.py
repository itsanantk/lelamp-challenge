"""Unit tests for viz.py's display-only mirroring. Run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import viz


class _Det:
    def __init__(self, object_class, confidence, bbox_xyxy):
        self.object_class = object_class
        self.confidence = confidence
        self.bbox_xyxy = bbox_xyxy


def test_mirror_frame_flips_pixels_left_right():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[:, 0] = (255, 0, 0)  # mark the left edge
    mirrored = viz.mirror_frame(frame)
    assert np.array_equal(mirrored[:, -1], frame[:, 0])
    assert np.array_equal(mirrored[:, 0], frame[:, -1])


def test_mirror_frame_preserves_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert viz.mirror_frame(frame).shape == frame.shape


def test_mirror_detections_flips_bbox_x_and_keeps_y():
    # a box against the left edge of a 640-wide frame...
    det = _Det("cell phone", 0.9, (0.0, 100.0, 50.0, 150.0))
    mirrored = viz.mirror_detections([det], frame_width=640)
    assert len(mirrored) == 1
    x1, y1, x2, y2 = mirrored[0].bbox_xyxy
    # ...should land against the right edge once mirrored, y unchanged
    assert x1 == 590.0 and x2 == 640.0
    assert y1 == 100.0 and y2 == 150.0
    assert mirrored[0].object_class == "cell phone"
    assert mirrored[0].confidence == 0.9


def test_mirror_detections_handles_empty_and_none():
    assert viz.mirror_detections([], frame_width=640) == []
    assert viz.mirror_detections(None, frame_width=640) == []


def test_mirrored_bbox_still_draws_without_error():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = _Det("cell phone", 0.9, (100.0, 100.0, 200.0, 200.0))
    mirrored = viz.mirror_detections([det], frame_width=640)
    out = viz.draw_detections(frame, mirrored)
    assert out.shape == frame.shape


def test_draw_detections_mutates_its_input_in_place():
    # Pinning this on purpose: draw_detections stopped defensively copying
    # (main.py always calls it right after draw_webcam_hud's own fresh
    # copy) -- this test exists so a future caller that needs the
    # pre-draw frame preserved fails loudly here instead of silently.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = _Det("cell phone", 0.9, (100.0, 100.0, 200.0, 200.0))
    out = viz.draw_detections(frame, [det])
    assert out is frame
    assert not np.array_equal(frame, np.zeros((480, 640, 3), dtype=np.uint8))


if __name__ == "__main__":
    test_mirror_frame_flips_pixels_left_right()
    test_mirror_frame_preserves_shape()
    test_mirror_detections_flips_bbox_x_and_keeps_y()
    test_mirror_detections_handles_empty_and_none()
    test_mirrored_bbox_still_draws_without_error()
    print("ALL VIZ TESTS PASSED")
