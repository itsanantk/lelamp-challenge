"""Simulated LampActuator: renders the 6-DOF arm, light, and sound, so the
full loop works without physical LeLamp hardware.

Sound is synthesized (numpy chirps played via sounddevice, already a
project dependency for voice I/O) rather than winsound.Beep -- a flat
fixed-frequency square wave reads as an alarm/appliance beep no matter
what frequency you pick, since it has no pitch movement or envelope. A
short pitch-swept tone with a soft attack/decay reads as a rounder, more
alive "boop" instead, without imitating an actual dog bark. Runs on a
dedicated worker thread pulling from a queue so overlapping play_sound()
calls never block the render loop or cut each other off.
"""
from __future__ import annotations

import queue
import threading

import random

import cv2
import numpy as np
import sounddevice as sd

from . import color, kinematics
from .hal import LampActuator
from .motion import Trajectory, lerp_color

CANVAS_SIZE = 1080  # matches the new webcam height (config.FRAME_HEIGHT) exactly, so
                     # compose_panels' resize-to-match-height in viz.py is a no-op (scale
                     # 1.0) instead of upscaling a smaller render and blurring it
_SCALE = CANVAS_SIZE / 420.0  # every pixel size/position below was tuned against the
                                # original 420px canvas -- this keeps all of them
                                # proportional at the new size instead of leaving the
                                # drawing tiny in the corner of a much bigger canvas
BG_COLOR = (30, 28, 26)  # BGR, warm dark background
ARM_COLOR = (210, 210, 210)
JOINT_COLOR = (120, 200, 255)

SOUND_SAMPLE_RATE = 44100  # matches this machine's Bluetooth output's native rate --
                            # confirmed live that a mismatched rate plays inaudibly on it
_AMPLITUDE = 0.85  # live-tested on Bluetooth output: 0.75 was "barely audible" -- higher headroom here
_VIBRATO_HZ = 20.0    # wobble rate on top of the base sweep -- a perfectly steady
_VIBRATO_DEPTH_HZ = 22.0  # pitch reads as a clean synth tone, not a living voice

# (start_hz, end_hz, duration_ms) frequency segments per named event,
# played as ONE continuous phase-integrated sweep with no internal
# silence -- NOT verified reliably audible on Bluetooth output (see
# README/architecture doc), only confirmed on the built-in speakers.
# Live A/B testing found that ANY internal silence (a gap between
# segments, even a deliberate pre-roll) went completely silent on
# Bluetooth, while a gapless sweep came through, if faintly. Upward
# sweeps read as curious/excited, downward as a gentle "aww, ok".
EVENT_SOUNDS: dict[str, list[tuple[int, int, int]]] = {
    "engage": [(600, 950, 160), (950, 1350, 190)],
    "disengage": [(750, 520, 160), (520, 340, 190)],
    # A quick 4-segment up-down-up-up alarm sweep read as insistent
    # beeping, not the "did you notice me?" curiosity the motion itself
    # is going for -- two gentle rising "hm?" inflections instead, lower
    # and slower, like a soft questioning whimper.
    "attention_seek": [(520, 680, 220), (600, 520, 70), (520, 700, 260)],
    "recall_point": [(900, 1350, 150), (1350, 1750, 180)],
    "wake": [(1000, 1400, 160)],  # single quick upward sweep -- "heard you, go ahead"
    # Slow, quiet, downward -- every nudge failed. A small sigh, not a
    # sad trombone.
    "give_up": [(480, 340, 320), (340, 260, 380)],
    # Quick, bright, playful double-blip -- a cheerful little "hi!" back.
    "wave_back": [(700, 950, 90), (950, 750, 80), (750, 1050, 110)],
}


def _jitter(value: float, pct: float) -> float:
    return value * random.uniform(1 - pct, 1 + pct)


def _event_waveform(event: str) -> np.ndarray:
    # Same event playing back byte-identical every time is exactly what
    # reads as a machine, not a creature -- jitter pitch/duration/vibrato/
    # volume a bit per play, same idea as the motion jitter in motion.py.
    base_freq = np.concatenate([
        np.linspace(_jitter(f0, 0.05), _jitter(f1, 0.05),
                    max(1, int(SOUND_SAMPLE_RATE * _jitter(dur_ms, 0.12) / 1000.0)))
        for f0, f1, dur_ms in EVENT_SOUNDS[event]
    ])
    n = len(base_freq)
    t = np.arange(n) / SOUND_SAMPLE_RATE
    # Vibrato + a touch of second-harmonic: a dead-steady pitch and a pure
    # sine both read as a clean synth tone, not a living voice -- a small
    # wobble and a little harmonic texture is what pushes it toward
    # "creature" instead of "sci-fi slide".
    vibrato_hz = _jitter(_VIBRATO_HZ, 0.25)
    vibrato_depth = _jitter(_VIBRATO_DEPTH_HZ, 0.25)
    freq = base_freq + vibrato_depth * np.sin(2 * np.pi * vibrato_hz * t)
    phase = 2 * np.pi * np.cumsum(freq) / SOUND_SAMPLE_RATE
    wave = np.sin(phase) + 0.2 * np.sin(2 * phase)
    wave = wave / np.max(np.abs(wave))
    fade = max(1, int(n * 0.08))  # soft attack/decay -- avoids the instant-on/off click of a raw square wave
    envelope = np.ones(n)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    amplitude = _jitter(_AMPLITUDE, 0.08)
    return (wave * envelope * amplitude).astype(np.float32)


def _project(point: np.ndarray, origin: tuple[int, int], scale: float) -> tuple[int, int]:
    """Cheap isometric-style projection from lamp-space (x, y, z) to screen
    pixels. Good enough for a HUD panel; not a real camera model."""
    x, y, z = point
    sx = origin[0] + (x - y) * np.cos(np.radians(30)) * scale
    sy = origin[1] - z * scale - (x + y) * np.sin(np.radians(30)) * scale
    return int(sx), int(sy)


class _SoundWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._q: queue.Queue[str | None] = queue.Queue()
        self.start()

    def enqueue(self, event: str) -> None:
        if event in EVENT_SOUNDS:
            self._q.put(event)

    def stop(self) -> None:
        # Being a daemon thread means it'd otherwise get killed abruptly at
        # interpreter shutdown -- live-confirmed that killing it mid-blocked
        # sd.play()/PortAudio call segfaults the process on this machine.
        # Signal it to exit its own loop and join before the process does.
        self._q.put(None)
        self.join(timeout=2.0)

    def run(self) -> None:
        while True:
            event = self._q.get()
            if event is None:
                return
            try:
                sd.play(_event_waveform(event), samplerate=SOUND_SAMPLE_RATE)
                sd.wait()
            except Exception:
                pass  # no audio device available; fail silently


class SimulatedLamp(LampActuator):
    def __init__(self, mute: bool = False):
        # Reentrant because set_target_pose() calls get_current_pose() on
        # itself while held. Only guards against a genuine cross-thread
        # write/write race (main render loop's update() vs. a command
        # arriving from the voice conversation thread once both share one
        # lamp instance) -- pure reads (get_current_pose/get_current_light/
        # render) stay unlocked, since a stale-by-one-frame read here is a
        # cosmetic non-issue, not a correctness one.
        self._lock = threading.RLock()
        self._pose = kinematics.NEUTRAL_POSE.copy()
        self._trajectory: Trajectory | None = None
        # Persistent (warmth, brightness) baseline -- see lamp/color.py.
        # Only set_mood() (an explicit user choice, e.g. a voice preset)
        # changes this; everything else nudges a delta off it and reverts.
        self._mood_warmth = 0.6
        self._mood_brightness = 0.55
        self._light_from = color.from_dials(self._mood_warmth, self._mood_brightness)
        self._light_to = self._light_from
        self._light_t = 1.0
        self._light_duration = 0.4
        self._light_elapsed = 0.0
        self._sound = None if mute else _SoundWorker()
        self._t = 0.0  # running clock, for idle breathing
        # Two incommensurate frequencies summed beats a single sine for
        # idle restlessness -- a lone fixed frequency has an obvious,
        # short loop period; summing two slightly different ones
        # (randomized per lamp instance) makes the drift read as organic
        # instead of a metronome. This is the fast attentive micro-wobble
        # (small glances/sway), separate from the actual breathing cycle
        # below, which is deliberately much slower.
        self._breathe_freqs = [random.uniform(0.55, 0.85) for _ in range(2)]
        self._breathe_phases = [random.uniform(0.0, 2 * np.pi) for _ in range(2)]
        # A real breath is slow (~3.5-4.5s per cycle, not the ~1-2s of the
        # micro-wobble above) and shows up as the shoulder/chest rising
        # and falling, not a jitter -- shoulder_pitch carries most of it,
        # elbow_pitch a touch, so it reads as one coherent rise/fall.
        self._breath_freq_hz = random.uniform(0.22, 0.29)
        self._breath_phase = random.uniform(0.0, 2 * np.pi)
        self._breathing_enabled = True

    # -- LampActuator interface -------------------------------------------------

    def set_breathing(self, enabled: bool) -> None:
        with self._lock:
            self._breathing_enabled = enabled

    def set_mood(self, warmth: float, brightness: float) -> None:
        with self._lock:
            self._mood_warmth = float(np.clip(warmth, 0.0, 1.0))
            self._mood_brightness = float(np.clip(brightness, 0.0, 1.0))

    def get_mood(self) -> tuple[float, float]:
        return (self._mood_warmth, self._mood_brightness)

    def set_target_pose(self, angles, duration=0.6, anticipation=True, overshoot=True) -> None:
        with self._lock:
            target = kinematics.clamp_pose(np.array(angles, dtype=float))
            current = self.get_current_pose()
            self._trajectory = Trajectory(current, target, duration, anticipation, overshoot)

    def set_light(self, rgb_bgr, transition_s=0.4) -> None:
        with self._lock:
            self._light_from = self.get_current_light()
            self._light_to = rgb_bgr
            self._light_duration = max(transition_s, 1e-3)
            self._light_elapsed = 0.0

    def play_sound(self, event: str) -> None:
        if self._sound is not None:
            self._sound.enqueue(event)

    def update(self, dt: float) -> None:
        with self._lock:
            self._t += dt
            if self._trajectory is not None:
                self._pose = self._trajectory.step(dt)
            self._light_elapsed = min(self._light_elapsed + dt, self._light_duration)

    def get_current_pose(self) -> np.ndarray:
        idle_still = self._trajectory is None or self._trajectory.done
        if idle_still and self._breathing_enabled:
            # Never look "dead" at rest -- a fast restless micro-wobble
            # (small sway/glance) plus a genuinely slow breathing cycle,
            # not one wobble doing both jobs at once.
            f1, f2 = self._breathe_freqs
            p1, p2 = self._breathe_phases
            wobble_a = (np.sin(self._t * f1 + p1) + np.sin(self._t * f2 + p2)) / 2
            wobble_b = (np.sin(self._t * f1 + p1 + 1.0) + np.sin(self._t * f2 + p2 + 1.0)) / 2
            breath = np.sin(2 * np.pi * self._breath_freq_hz * self._t + self._breath_phase)

            breathe = np.zeros(kinematics.NUM_JOINTS)
            breathe[0] = 1.0 * wobble_a   # base_yaw -- gentle attentive sway
            breathe[1] = 1.6 * breath     # shoulder_pitch -- the actual breathing rise/fall
            breathe[2] = 1.0 * breath     # elbow_pitch -- follows the chest, smaller
            breathe[5] = 1.4 * wobble_b   # head_twist -- restless micro-glance
            return self._pose + breathe
        return self._pose

    def get_current_light(self) -> tuple[int, int, int]:
        t = self._light_elapsed / self._light_duration
        return lerp_color(self._light_from, self._light_to, t)

    def close(self) -> None:
        if self._sound is not None:
            self._sound.stop()

    # -- Rendering ----------------------------------------------------------

    def render(self) -> np.ndarray:
        canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), BG_COLOR, dtype=np.uint8)
        origin = (CANVAS_SIZE // 2, int(CANVAS_SIZE * 0.82))
        scale = 8.0 * _SCALE

        pose = self.get_current_pose()
        points = kinematics.forward_kinematics(pose)
        arm_pts = [_project(p, origin, scale) for p in points]

        cv2.circle(canvas, origin, int(26 * _SCALE), (50, 48, 45), -1)  # desk base

        line_w = max(1, int(6 * _SCALE))
        joint_r = max(1, int(6 * _SCALE))
        for a, b in zip(arm_pts[:-1], arm_pts[1:]):
            cv2.line(canvas, a, b, ARM_COLOR, line_w, cv2.LINE_AA)
        for p in arm_pts[:-1]:
            cv2.circle(canvas, p, joint_r, JOINT_COLOR, -1, cv2.LINE_AA)

        # The shade as a 2D "billboard" cone: apex (the sharp, thin back of
        # the shade) hidden behind a wide opening that faces the viewer
        # head-on at base_yaw=0, rotating toward edge-on as base_yaw swings
        # toward either limit -- the same way a cone (or a coin) actually
        # looks as you turn it: the opening narrows into an oval and the
        # apex becomes visible poking out past the rim on the side it's
        # turned toward.
        #
        # Deliberately NOT derived from the true 3D chain projected through
        # this isometric camera (an earlier version was) -- the camera's own
        # "facing the viewer" direction isn't the same as base_yaw=0 in this
        # projection (confirmed by the actual trig: aligning them exactly
        # would need base_yaw near 45 deg and the shoulder/elbow resting
        # geometry reworked to match, which risks every other pose already
        # tuned against it), so a "true" 3D shade never looked round when
        # centered on the user, which is the most common case. Faking the
        # rotation directly in 2D against base_yaw sidesteps that mismatch
        # entirely and just draws what was actually asked for.
        head = arm_pts[4]
        yaw_rad = np.radians(float(pose[0]))
        shade_r = int(26 * _SCALE)  # smaller overall -- 34 read as oversized next to the arm
        rx = max(int(shade_r * 0.05), int(shade_r * abs(np.cos(yaw_rad))))  # lower floor -- reads
                                                                              # noticeably thinner
                                                                              # once turned, not just
                                                                              # nearly-full-width always
        ry = shade_r
        # How far the apex pokes out past the narrowed rim -- positive
        # base_yaw is the user's right (established bearing convention),
        # which is the right side of this un-mirrored sim panel too, so the
        # apex offsets the same direction the lamp is actually turning.
        apex_dx = int(shade_r * 0.85 * np.sin(yaw_rad))
        apex = (head[0] + apex_dx, head[1])
        light_color = self.get_current_light()

        side_w = max(1, int(2 * _SCALE))
        outline_w = max(1, int(1 * _SCALE))
        if abs(apex_dx) > int(shade_r * 0.15):  # only visible once turned enough to matter
            cv2.line(canvas, (head[0], head[1] - ry), apex, ARM_COLOR, side_w, cv2.LINE_AA)
            cv2.line(canvas, (head[0], head[1] + ry), apex, ARM_COLOR, side_w, cv2.LINE_AA)
        cv2.ellipse(canvas, head, (rx, ry), 0, 0, 360, light_color, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, head, (rx, ry), 0, 0, 360, (255, 255, 255), outline_w, cv2.LINE_AA)
        # A small dark "seam" mark nudged by wrist_roll/head_twist -- the
        # billboard above is driven by base_yaw alone, so this is what
        # keeps those two joints visibly meaningful instead of inert.
        seam = (head[0] + int(rx * 0.5 * (float(pose[5]) / 60.0)),
                head[1] + int(ry * 0.5 * (float(pose[4]) / 45.0)))
        cv2.circle(canvas, seam, max(1, int(4 * _SCALE)), (35, 30, 28), -1, cv2.LINE_AA)

        cv2.putText(canvas, "SIMULATED LAMP", (int(12 * _SCALE), int(24 * _SCALE)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * _SCALE, (150, 150, 150),
                    max(1, int(1 * _SCALE)), cv2.LINE_AA)

        self._draw_mood_bars(canvas)
        return canvas

    def _draw_mood_bars(self, canvas: np.ndarray) -> None:
        """The two dials (lamp/color.py) as actual bars, not just a color
        you have to eyeball -- this is self._mood_warmth/_mood_brightness
        directly (the persistent baseline set_mood() controls), not the
        currently-displayed light, which can be transiently nudged away
        from it by tracking/tone/ambient -- the bars show the setting,
        not every momentary reaction on top of it."""
        s = _SCALE
        bar_w, bar_h = int(150 * s), int(14 * s)
        gap = int(28 * s)
        x0 = int(12 * s)
        y0 = CANVAS_SIZE - int(20 * s) - 2 * bar_h - gap
        font_scale = 0.42 * s
        text_w = max(1, int(1 * s))

        rows = [
            ("WARMTH", self._mood_warmth, (25, 130, 255)),     # fill = the warm endpoint's own color
            ("BRIGHT", self._mood_brightness, (225, 225, 225)),  # fill = a plain white -- brightness has no hue of its own
        ]
        for i, (label, value, fill_color) in enumerate(rows):
            y = y0 + i * (bar_h + gap)
            cv2.putText(canvas, f"{label} {int(round(value * 100))}%", (x0, y - int(4 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (190, 190, 190), text_w, cv2.LINE_AA)
            cv2.rectangle(canvas, (x0, y), (x0 + bar_w, y + bar_h), (55, 52, 48), -1)
            fill_w = int(bar_w * float(np.clip(value, 0.0, 1.0)))
            if fill_w > 0:
                cv2.rectangle(canvas, (x0, y), (x0 + fill_w, y + bar_h), fill_color, -1)
            cv2.rectangle(canvas, (x0, y), (x0 + bar_w, y + bar_h), (110, 108, 104), text_w)
