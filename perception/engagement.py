"""Engagement detection: is the user looking at the lamp right now?

Head pose (yaw/pitch from MediaPipe's facial transformation matrix) is
the main signal, not iris tracking. Iris position degrades fast at
webcam resolution once the head turns even slightly off-axis, which is
most of what this system sees day to day at desk-lamp distances
(30-80cm, wide horizontal range). Head pose holds up across that whole
range. Iris centering (_iris_offset_ok) is a small secondary check for
"head mostly forward, eyes clearly looking elsewhere" -- it never
overrides head pose on its own.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import config

# Landmark indices for the iris rings and eye corners (MediaPipe's 478-point
# refined face mesh). Used only for the secondary "eyes off to the side"
# veto described above.
_LEFT_IRIS = [474, 475, 476, 477]
_RIGHT_IRIS = [469, 470, 471, 472]
_LEFT_EYE_CORNERS = (33, 133)
_RIGHT_EYE_CORNERS = (362, 263)


@dataclass
class EngagementReading:
    timestamp_ms: float
    face_found: bool
    engaged: bool
    yaw_deg: float | None
    pitch_deg: float | None
    ema_yaw_deg: float | None
    ema_pitch_deg: float | None
    user_bearing_deg: float | None  # horizontal position of face in-frame -> angle
    latency_ms: float


class EngagementClassifier:
    """Pure logic, no I/O: two-threshold + dwell-frame hysteresis over
    smoothed yaw/pitch. Kept separate from the camera/model code so it can
    be unit-tested and swept over threshold values independently."""

    def __init__(self):
        self.engaged = False
        self.ema_yaw: float | None = None
        self.ema_pitch: float | None = None
        self._enter_streak = 0
        self._exit_streak = 0

    def update(self, yaw: float | None, pitch: float | None) -> bool:
        if yaw is None or pitch is None:
            looking_now = False
            self.ema_yaw = self.ema_yaw  # hold last smoothed value
            self.ema_pitch = self.ema_pitch
        else:
            a = config.ENGAGE_EMA_ALPHA
            self.ema_yaw = yaw if self.ema_yaw is None else a * yaw + (1 - a) * self.ema_yaw
            self.ema_pitch = pitch if self.ema_pitch is None else a * pitch + (1 - a) * self.ema_pitch

            yaw_thresh = config.ENGAGE_ENTER_YAW_DEG if not self.engaged else config.ENGAGE_EXIT_YAW_DEG
            pitch_thresh = config.ENGAGE_ENTER_PITCH_DEG if not self.engaged else config.ENGAGE_EXIT_PITCH_DEG
            looking_now = abs(self.ema_yaw) < yaw_thresh and abs(self.ema_pitch) < pitch_thresh

        if looking_now:
            self._enter_streak += 1
            self._exit_streak = 0
        else:
            self._exit_streak += 1
            self._enter_streak = 0

        if not self.engaged and self._enter_streak >= config.ENGAGE_ENTER_DWELL_FRAMES:
            self.engaged = True
        elif self.engaged and self._exit_streak >= config.ENGAGE_EXIT_DWELL_FRAMES:
            self.engaged = False

        return self.engaged


def _head_pose_from_matrix(matrix: np.ndarray) -> tuple[float, float]:
    """Extract yaw/pitch (degrees) from MediaPipe's 4x4 facial transformation
    matrix. Camera convention: positive yaw = face turned to the viewer's
    right, positive pitch = face tilted down."""
    yaw = np.degrees(np.arctan2(-matrix[2, 0], np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2)))
    pitch = np.degrees(np.arctan2(matrix[2, 1], matrix[2, 2]))
    return float(yaw), float(pitch)


def _iris_offset_ok(landmarks, yaw_thresh_deg: float = 20.0) -> bool:
    """Secondary veto: with the head roughly forward, are the irises also
    roughly centered in the eye? Returns True (i.e. "no objection") whenever
    this check isn't meaningfully applicable, since it is a refinement, not
    a primary signal."""
    try:
        lx = np.mean([landmarks[i].x for i in _LEFT_IRIS])
        l0, l1 = landmarks[_LEFT_EYE_CORNERS[0]].x, landmarks[_LEFT_EYE_CORNERS[1]].x
        left_ratio = (lx - min(l0, l1)) / (abs(l1 - l0) + 1e-6)

        rx = np.mean([landmarks[i].x for i in _RIGHT_IRIS])
        r0, r1 = landmarks[_RIGHT_EYE_CORNERS[0]].x, landmarks[_RIGHT_EYE_CORNERS[1]].x
        right_ratio = (rx - min(r0, r1)) / (abs(r1 - r0) + 1e-6)

        avg_ratio = (left_ratio + right_ratio) / 2
        return 0.30 < avg_ratio < 0.70
    except (IndexError, ZeroDivisionError):
        return True


def _face_bearing_deg(landmarks) -> float:
    """Bearing relative to the *user's own* left/right, not the raw camera
    frame. The camera faces the user, so it's a face-to-face flip: an
    object toward the right edge of the raw image is actually on the
    user's left (same reason your video-call self-view usually gets
    mirrored). Negating cx here means positive bearing = user's right,
    matching what bearing_to_direction() reports out loud."""
    xs = [pt.x for pt in landmarks]
    cx = (min(xs) + max(xs)) / 2
    offset = (0.5 - cx) * 2  # -1 (user's left) .. +1 (user's right)
    return offset * (config.CAMERA_HFOV_DEG / 2)


class EngagementPipeline:
    """Owns the webcam and the FaceLandmarker model; produces one
    EngagementReading per frame."""

    def __init__(self, camera_index: int = config.CAMERA_INDEX):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.FACE_LANDMARKER_MODEL)),
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.classifier = EngagementClassifier()
        self._start_time = time.monotonic()

    def _timestamp_ms(self) -> int:
        return int((time.monotonic() - self._start_time) * 1000)

    def read(self) -> tuple[np.ndarray | None, EngagementReading | None]:
        ok, frame = self.cap.read()
        if not ok:
            return None, None

        t0 = time.perf_counter()
        ts_ms = self._timestamp_ms()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        yaw = pitch = bearing = None
        face_found = bool(result.face_landmarks)

        if face_found:
            landmarks = result.face_landmarks[0]
            bearing = _face_bearing_deg(landmarks)
            if result.facial_transformation_matrixes:
                matrix = np.array(result.facial_transformation_matrixes[0])
                yaw, pitch = _head_pose_from_matrix(matrix)
                if not _iris_offset_ok(landmarks) and abs(yaw) < config.ENGAGE_ENTER_YAW_DEG:
                    # Head is within the "looking at the lamp" cone but the
                    # eyes are clearly off to one side: push yaw outside the
                    # enter threshold so it correctly counts against
                    # engagement without a separate branch in the
                    # classifier. Guard spans the whole enter cone (not a
                    # narrow dead-zone) so this actually fires across the
                    # range the docstring claims it covers.
                    yaw = yaw + (25.0 if yaw >= 0 else -25.0)

        engaged = self.classifier.update(yaw, pitch)
        latency_ms = (time.perf_counter() - t0) * 1000

        reading = EngagementReading(
            timestamp_ms=ts_ms,
            face_found=face_found,
            engaged=engaged,
            yaw_deg=yaw,
            pitch_deg=pitch,
            ema_yaw_deg=self.classifier.ema_yaw,
            ema_pitch_deg=self.classifier.ema_pitch,
            user_bearing_deg=bearing,
            latency_ms=latency_ms,
        )
        return frame, reading

    def close(self) -> None:
        self.cap.release()
        self.landmarker.close()
