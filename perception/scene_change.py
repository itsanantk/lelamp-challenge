"""Cheap frame-diff gate for VisionMemory.maybe_scan(): skip a YOLO scan
when the scene hasn't visibly changed since the last one it actually ran,
instead of paying full inference cost on a fixed interval regardless of
whether there's anything new to see. YOLO and MediaPipe fight for the
same CPU cores in this loop (see vision_memory.py's own docstring), so a
skipped scan gives that contention back to engagement detection -- not
primarily a battery/energy thing, a CPU-contention thing.
"""
from __future__ import annotations

import cv2
import numpy as np

_SIZE = (64, 48)  # downsample target -- change detection doesn't need real resolution


class SceneChangeDetector:
    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold
        self._prev_gray: np.ndarray | None = None

    def changed(self, frame_bgr: np.ndarray) -> bool:
        """Compares against whatever frame this was last called with, not
        every raw camera frame -- callers should only call this once per
        scan-interval tick (see maybe_scan), so the diff reflects real
        accumulated change since the last decision point instead of
        frame-to-frame sensor noise between consecutive 30fps frames."""
        small = cv2.resize(frame_bgr, _SIZE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._prev_gray is None:
            self._prev_gray = gray
            return True  # nothing to compare yet -- always scan the first time
        score = float(np.abs(gray - self._prev_gray).mean()) / 255.0
        self._prev_gray = gray
        return score > self.threshold
