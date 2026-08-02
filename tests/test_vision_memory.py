"""Unit tests for perception/vision_memory.py's pan-crop restriction --
constructs VisionMemory without going through __init__ (that loads a real
YOLO model + does a warmup inference) and substitutes a fake model whose
predict() returns pre-built boxes in "already scanned" pixel space, so the
crop/bearing math can be checked without any real inference. Run with:
python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from perception.scene_change import SceneChangeDetector
from perception.vision_memory import VisionMemory


class _FakeBox:
    def __init__(self, cls_idx, conf, xyxy):
        self.cls = [cls_idx]
        self.conf = [conf]
        self.xyxy = [xyxy]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, names, boxes):
        self.names = names
        self._boxes = boxes
        self.last_predict_frame = None

    def predict(self, frame, **kwargs):
        self.last_predict_frame = frame
        return [_FakeResult(self._boxes)]


class _FakeStore:
    def __init__(self):
        self.observations = []

    def add_observation(self, **kwargs):
        self.observations.append(kwargs)
        return len(self.observations)

    def commit(self):
        pass


def _make_vision_memory(model: _FakeModel, store: _FakeStore) -> VisionMemory:
    """Bypasses __init__ (real YOLO construction + warmup predict) --
    same idea as the rest of the test suite's fakes (_FakeLamp etc.),
    just via object.__new__ since VisionMemory doesn't take its
    dependencies as constructor args the way those do."""
    vm = object.__new__(VisionMemory)
    vm.model = model
    vm.store = store
    vm._last_scan_t = 0.0
    vm._group_counter = 0
    vm.last_detections = []
    vm.last_scan_latency_ms = None
    vm.fast_mode = False
    vm._scene_change = SceneChangeDetector(threshold=config.SCENE_CHANGE_THRESHOLD)
    vm._skipped_since_scan = 0
    vm.scan_count = 0
    vm._force_scan = True  # bypass interval/scene-change gating for deterministic tests
    return vm


def test_uncropped_scan_computes_bearing_from_the_full_frame():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    # A box centered at x=100 (dead center of a 200-wide frame) -> bearing 0.
    model = _FakeModel({0: "cell phone"}, [_FakeBox(0, 0.9, (90.0, 40.0, 110.0, 60.0))])
    vm = _make_vision_memory(model, _FakeStore())

    detections = vm.maybe_scan(frame)

    assert len(detections) == 1
    assert abs(detections[0].bearing_deg) < 0.1
    assert model.last_predict_frame is frame  # no cropping happened


def test_cropped_scan_converts_the_box_back_to_the_same_global_bearing():
    # Round-trip check: place a detection at the pixel position a 10-degree
    # bearing maps to in a 200-wide, 60-degree-HFOV frame, precompute where
    # that lands inside a centered zoom=1.25 crop (once resized back up to
    # the full 200px width), and confirm maybe_scan's conversion recovers
    # ~10 degrees -- not some crop-relative value that would send the arm
    # to the wrong place, and not the original uncropped-formula answer
    # either (which would only coincidentally match here).
    w, h = 200, 100
    hfov = config.CAMERA_HFOV_DEG
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    target_bearing = 10.0
    cx_full = w * (0.5 - target_bearing / hfov)  # inverse of the raw-frame bearing formula
    zoom = 1.25
    crop_w = w / zoom
    x0 = (w - crop_w) / 2  # pan_bearing_deg=0.0 -> centered crop
    scale = w / crop_w
    cx_scanned = (cx_full - x0) * scale  # position within the resized-back-up scan frame

    box_half_w = 5.0
    model = _FakeModel({0: "bottle"}, [
        _FakeBox(0, 0.9, (cx_scanned - box_half_w, 40.0, cx_scanned + box_half_w, 60.0))
    ])
    vm = _make_vision_memory(model, _FakeStore())

    detections = vm.maybe_scan(frame, pan_bearing_deg=0.0, pan_zoom=zoom)

    assert len(detections) == 1
    assert abs(detections[0].bearing_deg - target_bearing) < 0.5
    # predict() must have actually received a cropped/resized frame, not
    # the original array untouched.
    assert model.last_predict_frame is not frame
    assert model.last_predict_frame.shape == frame.shape


def test_cropped_scan_stores_full_frame_bbox_coordinates():
    # Downstream code (viz.mirror_detections, draw_detections) assumes
    # bbox_xyxy is in full-raw-frame pixel space regardless of whether
    # cropping happened -- this must not leak crop-relative coordinates.
    w, h = 200, 100
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    model = _FakeModel({0: "bottle"}, [_FakeBox(0, 0.9, (90.0, 40.0, 110.0, 60.0))])
    vm = _make_vision_memory(model, _FakeStore())

    detections = vm.maybe_scan(frame, pan_bearing_deg=0.0, pan_zoom=1.25)

    x1, y1, x2, y2 = detections[0].bbox_xyxy
    assert 0 <= x1 < x2 <= w
    assert 0 <= y1 < y2 <= h


def test_pan_zoom_none_never_crops_regardless_of_pan_bearing():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    model = _FakeModel({0: "cell phone"}, [])
    vm = _make_vision_memory(model, _FakeStore())

    vm.maybe_scan(frame, pan_bearing_deg=25.0, pan_zoom=None)

    assert model.last_predict_frame is frame
