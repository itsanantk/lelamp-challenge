"""Bonus: multi-user interaction -- when more than one face is in frame,
figure out which one is actually talking, and point at them.

No audio-based speaker diarization here (that needs per-person
voiceprints and a much bigger build, and this system doesn't have a mic
array to localize sound direction from anyway). Instead: while the mic is
recording a question, this samples video frames over the same window and
tracks each detected face's mouth-openness across them. A talking mouth
moves; a quiet one mostly doesn't -- the face with the most mouth-openness
variance over the window is the one that was talking. Cheap, purely
visual, and matches what a lamp with one camera (not a mic array) can
actually do -- which is itself a reasonable thing to point at when asked
why this approach over real diarization.

get_frame, if given, hands this whatever main.py's own loop most recently
read (see perception/vision_memory.py's last_frame) instead of opening a
second cv2.VideoCapture on the same camera index -- many webcam drivers
only allow one open handle at a time, so the standalone (own-camera)
fallback below would silently degrade to unavailable the moment main.py's
EngagementPipeline already had the camera open, which is exactly the
--chat --voice case this is built for. Only chat.py run standalone (no
live camera loop of its own to ask) still needs the fallback.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import config

MAX_FACES = 4
# Inner-lip center landmarks, MediaPipe's 478-point face mesh.
_UPPER_LIP = 13
_LOWER_LIP = 14
_TRACK_MATCH_DEG = 15.0  # frame-to-frame face matching tolerance for the naive tracker below


@dataclass
class _FaceSample:
    bearing_deg: float
    mouth_open: float


class SpeakerDetector:
    def __init__(self, camera_index: int = config.CAMERA_INDEX,
                 get_frame: Callable[[], np.ndarray | None] | None = None):
        self._get_frame = get_frame
        self.cap = None
        if get_frame is None:
            # Standalone fallback -- owns its own camera handle. See the
            # module docstring for why the embedded (--chat) path avoids
            # this entirely instead.
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.available = get_frame is not None or self.cap.isOpened()
        self.landmarker = None
        if self.available:
            try:
                options = vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(config.FACE_LANDMARKER_MODEL)),
                    num_faces=MAX_FACES,
                    running_mode=vision.RunningMode.VIDEO,
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
            except Exception:
                self.available = False
        self._t0 = time.monotonic()
        self._last_frame_seen: np.ndarray | None = None

    def _timestamp_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def _read_frame(self) -> np.ndarray | None:
        if self._get_frame is not None:
            return self._get_frame()
        ok, frame = self.cap.read()
        return frame if ok else None

    def _sample_frame(self) -> list[_FaceSample]:
        frame = self._read_frame()
        if frame is None:
            return []
        if self._get_frame is not None:
            # main.py's loop only refreshes its shared frame once per
            # tick, slower than this method's own polling rate -- reusing
            # the exact same array object for detect_for_video() twice in
            # a row is harmless (MediaPipe VIDEO mode wants a
            # non-decreasing timestamp, not a genuinely new frame every
            # call), so just skip the redundant inference instead of
            # re-running it on identical pixels.
            if frame is self._last_frame_seen:
                return []
            self._last_frame_seen = frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms())
        samples = []
        for landmarks in result.face_landmarks:
            xs = [p.x for p in landmarks]
            cx = (min(xs) + max(xs)) / 2
            # same user-relative bearing convention as perception/engagement.py
            bearing = (0.5 - cx) * config.CAMERA_HFOV_DEG
            face_h = max(p.y for p in landmarks) - min(p.y for p in landmarks)
            mouth_open = abs(landmarks[_LOWER_LIP].y - landmarks[_UPPER_LIP].y) / max(face_h, 1e-4)
            samples.append(_FaceSample(bearing, mouth_open))
        return samples

    def detect_active_speaker(self, stop_event: threading.Event,
                               max_duration_s: float = 15.0) -> tuple[float | None, int]:
        """Samples frames until stop_event is set -- meant to fire the
        moment the mic's silence-terminated recording actually ends, so
        this always finishes at roughly the same time as the audio it's
        overlapping with, however long that turns out to be, instead of
        running to a separately-guessed fixed duration. max_duration_s is
        just a safety cap in case stop_event is never set. Returns
        (bearing of the most likely speaker, number of distinct faces
        seen). (None, 0) if the camera isn't available or nobody was in
        frame."""
        if not self.available:
            return None, 0

        tracks: list[list[_FaceSample]] = []
        t0 = time.perf_counter()
        while not stop_event.is_set() and time.perf_counter() - t0 < max_duration_s:
            for s in self._sample_frame():
                best_track, best_d = None, _TRACK_MATCH_DEG
                for track in tracks:
                    d = abs(track[-1].bearing_deg - s.bearing_deg)
                    if d < best_d:
                        best_d, best_track = d, track
                if best_track is not None:
                    best_track.append(s)
                else:
                    tracks.append([s])
            time.sleep(0.03)

        if not tracks:
            return None, 0
        if len(tracks) == 1:
            return float(np.mean([s.bearing_deg for s in tracks[0]])), 1

        def _mouth_variance(track: list[_FaceSample]) -> float:
            return float(np.var([s.mouth_open for s in track])) if len(track) > 1 else 0.0

        speaker_track = max(tracks, key=_mouth_variance)
        return float(np.mean([s.bearing_deg for s in speaker_track])), len(tracks)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self.landmarker is not None:
            self.landmarker.close()
