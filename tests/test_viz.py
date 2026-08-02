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


def test_pan_crop_centered_bearing_is_a_pure_zoom():
    # bearing 0.0 -- centered -- should crop symmetrically and preserve
    # output shape (a digital zoom in place, not a shift).
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    cropped, x0, scale = viz.pan_crop_frame(frame, 0.0, hfov_deg=60.0, zoom=1.25)
    assert cropped.shape == frame.shape
    crop_w = 200 / 1.25
    assert abs(x0 - (200 - crop_w) / 2) <= 1  # centered crop window
    assert abs(scale - 1.25) < 1e-6


def test_pan_crop_follows_a_rightward_bearing():
    # Positive bearing = user's right = right side of a mirrored frame
    # (same convention mirror_detections already uses) -- the crop window
    # should shift toward the right edge, not the left.
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    _, x0_center, _ = viz.pan_crop_frame(frame, 0.0, hfov_deg=60.0, zoom=1.25)
    _, x0_right, _ = viz.pan_crop_frame(frame, 20.0, hfov_deg=60.0, zoom=1.25)
    assert x0_right > x0_center


def test_pan_crop_unmirrored_uses_the_opposite_sign():
    # perception/vision_memory.py crops the *raw* (unmirrored) frame --
    # the raw-frame bearing convention is already flipped relative to the
    # mirrored display convention (see mirror_detections' docstring), so
    # a positive bearing should shift the unmirrored crop window left,
    # not right.
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    _, x0_center, _ = viz.pan_crop_frame(frame, 0.0, hfov_deg=60.0, zoom=1.25, mirrored=False)
    _, x0_right_bearing, _ = viz.pan_crop_frame(frame, 20.0, hfov_deg=60.0, zoom=1.25, mirrored=False)
    assert x0_right_bearing < x0_center


def test_pan_crop_clamps_to_frame_edges():
    # A bearing far past the frame's actual FOV shouldn't crop past the
    # frame boundary or produce a negative/out-of-range window.
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    cropped, x0, scale = viz.pan_crop_frame(frame, 200.0, hfov_deg=60.0, zoom=1.25)
    crop_w = round(200 / 1.25)
    assert 0 <= x0 <= 200 - crop_w
    assert cropped.shape == frame.shape


def test_remap_boxes_for_pan_keeps_boxes_on_the_object():
    # A box drawn against the raw (already-mirrored) frame should land in
    # the same *visual* place once the pan crop is applied -- i.e.
    # applying the crop's own x0/scale to the box coordinates, not left
    # pointing at the pre-crop position.
    det = _Det("cell phone", 0.9, (110.0, 40.0, 130.0, 60.0))
    remapped = viz.remap_boxes_for_pan([det], crop_x0=100, x_scale=2.0)
    x1, y1, x2, y2 = remapped[0].bbox_xyxy
    assert x1 == 20.0 and x2 == 60.0  # (110-100)*2, (130-100)*2
    assert y1 == 40.0 and y2 == 60.0  # vertical untouched -- pan is horizontal-only


def test_remap_boxes_for_pan_handles_empty_and_none():
    assert viz.remap_boxes_for_pan([], crop_x0=0, x_scale=1.0) == []
    assert viz.remap_boxes_for_pan(None, crop_x0=0, x_scale=1.0) == []


if __name__ == "__main__":
    test_mirror_frame_flips_pixels_left_right()
    test_mirror_frame_preserves_shape()
    test_mirror_detections_flips_bbox_x_and_keeps_y()
    test_mirror_detections_handles_empty_and_none()
    test_mirrored_bbox_still_draws_without_error()
    print("ALL VIZ TESTS PASSED")
