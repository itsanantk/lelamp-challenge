"""Memory formation: periodically run object detection over the webcam
feed and write what's seen (class, confidence, bearing) into the memory
store.

Runs on a wall-clock interval (config.YOLO_SCAN_INTERVAL_S), not every
frame. No CUDA on this machine, and YOLO11s at 320px runs ~28ms/scan in
isolation -- but in the live loop it's sharing the process with
MediaPipe's own CPU inference, and torch's default thread count (16 on
this 22-core machine) oversubscribes against that instead of leaving
room for it, which is most of why a scan in main.py costs more like
70-90ms than the isolated 28ms. Capping torch to a handful of threads
gets some of that back. Engagement detection needs to stay fast and
continuous since it gates the whole interaction; scene memory doesn't --
objects on a desk don't move between one glance and the next -- so
running detection every single frame would still be wasted CPU even with
threading fixed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

torch.set_num_threads(4)

from ultralytics import YOLO

import config
from memory.store import MemoryStore
from perception.scene_change import SceneChangeDetector


@dataclass
class Detection:
    object_class: str
    confidence: float
    bearing_deg: float
    bbox_xyxy: tuple[float, float, float, float]  # pixel coords in the source frame


class VisionMemory:
    def __init__(self, store: MemoryStore):
        self.model = YOLO(str(config.YOLO_MODEL))
        # first inference eats a one-time graph/JIT warmup cost (1.3-2.7s
        # locally) -- do it here at construction instead of on whatever
        # frame triggers the first real scan, or it shows up as a latency
        # spike in the middle of a session
        self.model.predict(np.zeros((config.YOLO_IMG_SIZE, config.YOLO_IMG_SIZE, 3), dtype=np.uint8),
                            imgsz=config.YOLO_IMG_SIZE, verbose=False)
        self.store = store
        self._last_scan_t = 0.0
        self._group_counter = 0
        self.last_detections: list[Detection] = []
        self.last_scan_latency_ms: float | None = None
        self.fast_mode = False
        self._scene_change = SceneChangeDetector(threshold=config.SCENE_CHANGE_THRESHOLD)
        self._skipped_since_scan = 0

    def maybe_scan(self, frame: np.ndarray, now: float | None = None) -> list[Detection] | None:
        """Call every loop tick. Only actually runs YOLO once the scan
        interval has elapsed; returns None on ticks where it didn't run.
        Interval shortens to YOLO_FAST_SCAN_INTERVAL_S while self.fast_mode
        is set (see ObjectWatcher), so a tracked object gets checked more
        often while it's actually being moved around."""
        now = now if now is not None else time.monotonic()
        interval = config.YOLO_FAST_SCAN_INTERVAL_S if self.fast_mode else config.YOLO_SCAN_INTERVAL_S
        if now - self._last_scan_t < interval:
            return None
        self._last_scan_t = now

        if self.fast_mode:
            # Actively tracking something -- keep the change detector's
            # reference frame fresh so it doesn't misread a big jump once
            # fast_mode ends, but don't gate on it here: a moving object
            # usually does register as change, but a phone held very
            # still is exactly the case fast_mode exists to stay
            # responsive for, so this cadence always scans regardless.
            self._scene_change.changed(frame)
        else:
            changed = self._scene_change.changed(frame)
            if not changed:
                self._skipped_since_scan += 1
                if self._skipped_since_scan < config.SCENE_CHANGE_MAX_SKIPS:
                    return None
            self._skipped_since_scan = 0

        t0 = time.perf_counter()
        # Run inference at the lower of the two confidence bars so a
        # tracked-class detection that scores between the two thresholds
        # isn't discarded by YOLO itself before the per-class check below
        # ever gets a chance to see it.
        scan_conf = min(config.YOLO_CONF_THRESHOLD, config.TRACKED_CLASS_CONF_THRESHOLD)
        # agnostic_nms=True: ultralytics' NMS suppression is per-class by
        # default, so one physical object can legitimately produce two
        # overlapping boxes with different labels (e.g. a phone scored as
        # both "cell phone" and "remote") and both survive, since they're
        # never compared against each other. Cross-class suppression keeps
        # only the higher-confidence label for a given region.
        results = self.model.predict(
            frame, imgsz=config.YOLO_IMG_SIZE, conf=scan_conf, agnostic_nms=True, verbose=False,
        )[0]
        self.last_scan_latency_ms = (time.perf_counter() - t0) * 1000

        h, w = frame.shape[:2]
        self._group_counter += 1
        group_id = self._group_counter
        detections: list[Detection] = []

        for box in results.boxes:
            cls_name = self.model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            threshold = (config.TRACKED_CLASS_CONF_THRESHOLD if cls_name in config.TRACKED_CLASSES
                         else config.YOLO_CONF_THRESHOLD)
            if conf < threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            # negated for the same reason as _face_bearing_deg in
            # engagement.py: camera faces the user, so raw image-right is
            # the user's own left. Positive bearing = user's right.
            bearing = (0.5 - (cx / w)) * config.CAMERA_HFOV_DEG
            detections.append(Detection(cls_name, conf, bearing, (x1, y1, x2, y2)))
            self.store.add_observation(
                object_class=cls_name, confidence=conf, bearing_deg=bearing,
                bbox_cx=cx / w, bbox_cy=cy / h, frame_group_id=group_id,
            )

        self.last_detections = detections
        return detections
