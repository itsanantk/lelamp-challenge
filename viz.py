"""Compositing and HUD overlay: turns the raw webcam frame + lamp render +
pipeline state into the single annotated frame used for both the on-screen
window and the auto-recorded demo video."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import cv2
import numpy as np


def enable_dpi_awareness() -> None:
    """Declares this process DPI-aware to Windows. Must be called once,
    as early in process startup as possible, before any window gets
    created.

    Without this, Windows treats the process as DPI-unaware and does two
    things behind its back: GetSystemMetrics reports a *virtualized*,
    scaled-down screen resolution instead of the real one (on this
    laptop's 4K panel at 300% scaling, that's 1280x800 instead of the
    real ~3840x2400), and every window the process creates gets bitmap-
    stretched by the OS to fill the real physical pixels. Both of those
    turned into real, user-visible bugs here: the virtualized resolution
    being smaller than the ~1360px-wide composite window is what caused
    it to clip on the right (see open_fitted_window's history below), and
    the fix for *that* -- shrinking the window with resizeWindow -- made
    the picture visibly blurrier, because now the image was being scaled
    down by resizeWindow *and then back up* by the OS's bitmap stretch,
    instead of just stretched once. Declaring DPI-awareness turns off the
    OS-side stretching entirely and reports the real screen size, which
    fixes both problems at the root instead of working around either one:
    GetSystemMetrics now returns real values the composite comfortably
    fits inside without needing to shrink, and cv2's window renders at
    native resolution with no automatic scaling in the loop at all."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except (AttributeError, OSError):
        pass  # not Windows, older Windows without shcore, or already set -- harmless either way


def open_fitted_window(name: str, content_w: int, content_h: int, margin: int = 80) -> None:
    """Creates a resizable window, shrinks it only if it genuinely
    doesn't fit the screen (real screen size, once enable_dpi_awareness()
    has been called -- see that docstring for why "real" matters here),
    and pins its top-left corner to a fixed on-screen spot.

    Resizing alone wasn't enough to fix placement: whatever position the
    OS/OpenCV chose by default for a fresh window put its *left* edge
    off-screen in testing (visible in a screenshot as the title bar and
    FSM label text truncated on the left), independent of whatever
    width-related sizing this does. moveWindow pins it to a small, known
    offset from the actual top-left corner so placement isn't left to
    chance. Windows-only; harmless no-op elsewhere (falls back to the
    normal WINDOW_AUTOSIZE behavior, default OS placement)."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    try:
        user32 = ctypes.windll.user32
        screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except (AttributeError, OSError):
        return  # not Windows, or the call failed -- leave it at native size

    max_w, max_h = max(screen_w - margin, 320), max(screen_h - margin, 240)
    scale = min(1.0, max_w / content_w, max_h / content_h)
    cv2.resizeWindow(name, int(content_w * scale), int(content_h * scale))
    cv2.moveWindow(name, margin // 4, margin // 4)

    # Also bring it to the foreground -- OpenCV windows don't request
    # focus on their own, so on a desktop with anything else already
    # open (a browser, an editor) the demo window opens silently behind
    # it. Looked up by exact title via FindWindowW rather than trying to
    # get an HWND out of cv2 directly, which isn't reliably exposed
    # across OpenCV builds.
    hwnd = user32.FindWindowW(None, name)
    if hwnd:
        user32.SetForegroundWindow(hwnd)


STATE_COLORS = {
    "IDLE": (150, 150, 150),
    "ENGAGED": (140, 220, 255),
    "DISENGAGED": (120, 160, 210),
    "ATTENTION_SEEKING": (0, 165, 255),
}


@dataclass
class _MirroredDetection:
    object_class: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


def mirror_frame(frame: np.ndarray) -> np.ndarray:
    """Horizontal flip for *display only* -- a raw, un-mirrored webcam
    frame shows you like a security camera would (your right hand is on
    the frame's left side, since the camera faces you), which reads as
    "backwards" to anyone used to a normal selfie-view mirror. All the
    actual bearing math (perception/engagement.py, vision_memory.py) stays
    on the original, un-flipped frame -- flipping *that* would undo the
    user-relative-bearing fix already in place there. This only touches
    what gets drawn on screen."""
    return cv2.flip(frame, 1)


def pan_crop_frame(frame: np.ndarray, pan_bearing_deg: float, hfov_deg: float,
                    zoom: float = 1.35, mirrored: bool = True) -> tuple[np.ndarray, int, float]:
    """Crops+rescales a frame to follow pan_bearing_deg -- replicates the
    lamp visually "panning" the camera toward wherever it's currently
    aimed. Digital crop only (never real camera PTZ). Horizontal-only --
    the lamp's own gaze is driven by base_yaw (a pan), there's no reason
    to also zoom vertically.

    Two callers, two frame conventions: main.py's display path calls this
    on the already-mirrored frame (mirrored=True, the default) purely as
    a display-layer transform -- the underlying frame used for detection/
    engagement is untouched by that call. perception/vision_memory.py
    calls this on the *raw* frame (mirrored=False) to restrict what YOLO
    actually gets to see to the lamp's own current "field of view" --
    simulating a camera mounted on the lamp's head that genuinely can't
    see what it isn't pointed at, not just a cosmetic viewport. The sign
    flips between the two because mirroring itself flips which edge of
    the frame a positive (user's-right) bearing sits on -- see
    mirror_detections()'s own docstring for the raw-frame convention.

    Returns (cropped_frame, crop_x0_px, x_scale) so a caller can remap
    detection boxes (or bearings) through the exact same transform instead
    of them drifting off the object as the view pans."""
    h, w = frame.shape[:2]
    crop_w = max(1, int(round(w / zoom)))
    center_frac = (0.5 + pan_bearing_deg / hfov_deg) if mirrored else (0.5 - pan_bearing_deg / hfov_deg)
    center_frac = float(np.clip(center_frac, 0.0, 1.0))
    x0 = int(round(center_frac * w - crop_w / 2))
    x0 = max(0, min(x0, w - crop_w))
    cropped = frame[:, x0:x0 + crop_w]
    # INTER_NEAREST, not INTER_LINEAR -- this runs on the full display
    # frame every tick (1920x1080), and the other panel resizes in this
    # file already learned that lesson (see compose_panels/make_text_panel):
    # linear interpolation at this resolution is expensive enough to show
    # up as real per-frame latency, and nobody's looking closely enough at
    # a live webcam feed for the softer edges to matter.
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)
    scale = w / crop_w
    return resized, x0, scale


def remap_boxes_for_pan(detections, crop_x0: int, x_scale: float) -> list[_MirroredDetection]:
    """Remaps already-mirrored detection boxes (mirror_detections) through
    the same crop+resize pan_crop_frame just applied, so boxes keep
    landing on the object instead of drifting as the view pans. Vertical
    coordinates are untouched -- the pan/zoom is horizontal-only."""
    out = []
    for d in detections or []:
        x1, y1, x2, y2 = d.bbox_xyxy
        out.append(_MirroredDetection(d.object_class, d.confidence,
                                       ((x1 - crop_x0) * x_scale, y1, (x2 - crop_x0) * x_scale, y2)))
    return out


def mirror_detections(detections, frame_width: int) -> list[_MirroredDetection]:
    """Mirrors detection boxes to match a mirror_frame()'d display frame,
    so YOLO boxes still land on the object instead of its un-mirrored
    former position. Detection logic and stored bearings are untouched --
    this is a draw-time-only transform, like mirror_frame."""
    out = []
    for d in detections or []:
        x1, y1, x2, y2 = d.bbox_xyxy
        out.append(_MirroredDetection(d.object_class, d.confidence,
                                       (frame_width - x2, y1, frame_width - x1, y2)))
    return out



# All the HUD/detection/text-panel drawing below was originally tuned
# against a 640x480 webcam frame and a 240px-wide side panel. Since the
# frame is now 1920x1080 (config.FRAME_WIDTH/HEIGHT) and the side panel
# is 3x wider (see main.py), fixed pixel sizes/positions would render
# proportionally tiny in the corner of a much bigger frame -- _UI_SCALE
# keeps every font size, line thickness, and margin proportional to the
# original tuning instead of just leaving them small.
_UI_SCALE = 1920 / 640.0


def draw_webcam_hud(frame: np.ndarray, reading, fsm_state: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    s = _UI_SCALE

    color = STATE_COLORS.get(fsm_state, (200, 200, 200))
    bar_h = int(36 * s)
    cv2.rectangle(out, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.putText(out, f"STATE: {fsm_state}", (int(10 * s), int(25 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65 * s, color, max(1, int(2 * s)), cv2.LINE_AA)

    if reading is not None:
        face_txt = "face: yes" if reading.face_found else "face: no"
        yaw = f"{reading.ema_yaw_deg:.1f}" if reading.ema_yaw_deg is not None else "-"
        pitch = f"{reading.ema_pitch_deg:.1f}" if reading.ema_pitch_deg is not None else "-"
        info = f"{face_txt}  yaw={yaw}  pitch={pitch}  lat={reading.latency_ms:.0f}ms"
        cv2.putText(out, info, (int(10 * s), h - int(14 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * s, (220, 220, 220), max(1, int(1 * s)), cv2.LINE_AA)

    return out


def draw_detections(frame: np.ndarray, detections) -> np.ndarray:
    # Mutates frame in place -- main.py always calls this right after
    # draw_webcam_hud, which already returned a fresh copy, so a second
    # defensive copy here would just be extra large-array overhead at
    # 1920x1080 for nothing.
    out = frame
    s = _UI_SCALE
    for d in detections or []:
        x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), (100, 255, 140), max(1, int(2 * s)), cv2.LINE_AA)
        label = f"{d.object_class} {d.confidence:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - int(6 * s))), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * s, (100, 255, 140), max(1, int(1 * s)), cv2.LINE_AA)
    return out


def compose_panels(webcam_frame: np.ndarray, lamp_panel: np.ndarray,
                    info_panel: np.ndarray | None = None) -> np.ndarray:
    """Row 1: webcam feed and lamp sim side by side (matching heights).
    Row 2 (optional): a full-width info/debug panel spanning beneath
    both, rather than a third narrow column squeezed in next to them --
    camera and lamp sim are the two things worth being big and sharp;
    FSM state, memory, and debug numbers are worth reading, not staring
    at. One combined frame means one VideoWriter can record the whole
    session."""
    target_h = webcam_frame.shape[0]

    def _resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
        if img.shape[0] == h:
            return img  # already matches -- skip a no-op resize of a large array
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h), interpolation=cv2.INTER_NEAREST)

    lamp_resized = _resize_to_h(lamp_panel, target_h)
    row1 = np.hstack([webcam_frame, lamp_resized])

    if info_panel is None:
        return row1

    row1_w = row1.shape[1]
    if info_panel.shape[1] != row1_w:
        # Safety net for whatever width the caller actually built the
        # panel at -- main.py computes the exact row1 width up front so
        # this is normally a no-op, not a stretch that'd distort the text.
        info_panel = cv2.resize(info_panel, (row1_w, info_panel.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.vstack([row1, info_panel])


def make_text_panel(width: int, height: int, lines: list[str], title: str = "",
                     columns: int = 1, scale: float | None = None) -> np.ndarray:
    """Renders lines top-to-bottom, wrapping into additional columns once
    one fills up, instead of a single column that either runs off the
    bottom or leaves most of a wide panel empty. scale defaults to
    _UI_SCALE (matches the webcam/detection text) but takes an override --
    a panel this wide at full _UI_SCALE would need very tall columns to
    fit typical debug-line counts, so the caller can size it down for a
    shorter, denser row instead."""
    s = scale if scale is not None else _UI_SCALE
    panel = np.full((height, width, 3), (25, 24, 22), dtype=np.uint8)
    pad = int(10 * s)
    title_y = int(24 * s)
    title_h = int(28 * s) if title else 0
    line_h = int(20 * s)
    y_top = title_y + title_h
    usable_h = max(height - y_top - pad, line_h)
    lines_per_col = max(1, usable_h // line_h)
    col_w = width // max(columns, 1)

    if title:
        cv2.putText(panel, title, (pad, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s,
                    (200, 200, 200), max(1, int(1 * s)), cv2.LINE_AA)

    col, row = 0, 0
    for line in lines:
        if row >= lines_per_col:
            col += 1
            row = 0
        if col >= columns:
            break  # out of room even with every column used
        x = pad + col * col_w
        y = y_top + row * line_h
        cv2.putText(panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s,
                    (180, 220, 180), max(1, int(1 * s)), cv2.LINE_AA)
        row += 1
    return panel
