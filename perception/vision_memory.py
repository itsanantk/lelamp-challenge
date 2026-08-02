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
import viz
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
        self.scan_count = 0  # bumped once per *actual* YOLO call -- lets another thread
                               # (chat.py's recall) detect when a requested scan has landed
        self._force_scan = False

    def request_immediate_scan(self) -> None:
        """Bypasses both the interval and scene-change-skip gating for the
        very next maybe_scan() call. Used by recall (chat.py, possibly on
        a different thread) when it needs a guaranteed-current answer to
        "is this visible right now" rather than whatever the last
        opportunistic scan happened to catch -- main.py's own loop
        overwrites fast_mode from ObjectWatcher every tick, so setting
        that directly from another thread would just get clobbered before
        it ever took effect."""
        self._force_scan = True

    def maybe_scan(self, frame: np.ndarray, now: float | None = None,
                    pan_bearing_deg: float = 0.0, pan_zoom: float | None = None) -> list[Detection] | None:
        """Call every loop tick. Only actually runs YOLO once the scan
        interval has elapsed; returns None on ticks where it didn't run.
        Interval shortens to YOLO_FAST_SCAN_INTERVAL_S while self.fast_mode
        is set (see ObjectWatcher), so a tracked object gets checked more
        often while it's actually being moved around.

        pan_zoom, if given, restricts YOLO's input to a crop of `frame`
        centered on pan_bearing_deg (see viz.pan_crop_frame) instead of
        the full frame -- simulating a camera mounted on the lamp's own
        head that genuinely can't see what it isn't currently pointed at,
        not main.py just happening to also show a cropped viewport. An
        object near the crop's edge still gets detected (it's not cut out
        entirely at this zoom) and pulls the lamp to re-aim toward it via
        the normal tracking path, which re-centers it -- that's what
        actually lets the lamp "notice something on the right and turn
        to bring it into view" the way a single-camera device would, and
        is also why a wide-open zoom matters: too tight a crop and
        nothing off to the side ever gets seen in the first place to
        react to. Detected boxes are converted back to full-frame pixel
        coordinates before being stored/returned, so every caller and the
        stored bearing are identical either way -- the crop is purely an
        inference-time input restriction, not a new coordinate system
        leaking out. None (the default) scans the full frame, unchanged
        from before this existed."""
        now = now if now is not None else time.monotonic()
        interval = config.YOLO_FAST_SCAN_INTERVAL_S if self.fast_mode else config.YOLO_SCAN_INTERVAL_S
        if not self._force_scan and now - self._last_scan_t < interval:
            return None
        self._last_scan_t = now
        forced = self._force_scan
        self._force_scan = False

        if self.fast_mode or forced:
            # Actively tracking something (or a caller needs a guaranteed
            # fresh read right now) -- keep the change detector's
            # reference frame fresh so it doesn't misread a big jump once
            # fast_mode ends, but don't gate on it here: a moving object
            # usually does register as change, but a phone held very
            # still is exactly the case fast_mode/request_immediate_scan
            # exist to stay responsive for, so this cadence always scans
            # regardless.
            self._scene_change.changed(frame)
        else:
            changed = self._scene_change.changed(frame)
            if not changed:
                self._skipped_since_scan += 1
                if self._skipped_since_scan < config.SCENE_CHANGE_MAX_SKIPS:
                    return None
            self._skipped_since_scan = 0

        self.scan_count += 1
        t0 = time.perf_counter()
        # Run inference at the lower of the two confidence bars so a
        # tracked-class detection that scores between the two thresholds
        # isn't discarded by YOLO itself before the per-class check below
        # ever gets a chance to see it.
        scan_conf = min(config.YOLO_CONF_THRESHOLD, config.TRACKED_CLASS_CONF_THRESHOLD)

        scan_frame = frame
        crop_x0, crop_scale = 0, 1.0
        if pan_zoom is not None:
            scan_frame, crop_x0, crop_scale = viz.pan_crop_frame(
                frame, pan_bearing_deg, config.CAMERA_HFOV_DEG, zoom=pan_zoom, mirrored=False)

        # agnostic_nms=True: ultralytics' NMS suppression is per-class by
        # default, so one physical object can legitimately produce two
        # overlapping boxes with different labels (e.g. a phone scored as
        # both "cell phone" and "remote") and both survive, since they're
        # never compared against each other. Cross-class suppression keeps
        # only the higher-confidence label for a given region.
        results = self.model.predict(
            scan_frame, imgsz=config.YOLO_IMG_SIZE, conf=scan_conf, agnostic_nms=True, verbose=False,
        )[0]
        self.last_scan_latency_ms = (time.perf_counter() - t0) * 1000

        h, w = frame.shape[:2]  # full raw frame dims -- bearing/normalization math is
                                  # unaffected by cropping, see the box-coordinate conversion below
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
            if pan_zoom is not None:
                # scan_frame is a zoomed-in crop resized back up to the
                # same (w, h) as the full frame -- box coordinates come
                # back in that same resized space, so converting to true
                # full-frame pixels is just undoing the crop's own scale
                # and offset (y is untouched -- the pan crop is
                # horizontal-only). Once converted, everything below is
                # identical to the uncropped path: same bearing formula,
                # same stored coordinates, nothing downstream needs to
                # know cropping happened at all.
                x1 = crop_x0 + x1 / crop_scale
                x2 = crop_x0 + x2 / crop_scale
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
        if detections:
            # One commit per scan, not one per detected box -- a busy
            # scene can easily produce a dozen-plus boxes in a single
            # scan, and committing each individually forces a real disk
            # fsync per box (measured as the dominant per-frame cost once
            # the db had enough classes for a typical scan to hit several
            # at once -- see MemoryStore.add_observation's docstring).
            self.store.commit()

        self.last_detections = detections
        return detections
