"""Forward kinematics for the simulated 6-DOF lamp arm.

Joint layout mirrors a Luxo-Jr-style desk lamp on a serial arm (the same
convention real hobby "lamp robots" like LeLamp use: a base rotation, two
arm links, a wrist, and a head that can tilt and twist independently):

    0. base_yaw       - rotates the whole arm around the vertical axis
    1. shoulder_pitch  - lifts the lower arm segment
    2. elbow_pitch      - lifts the upper arm segment relative to the lower one
    3. wrist_pitch      - tilts the head/shade up-down
    4. wrist_roll       - rolls the head/shade (expressive "cock the head")
    5. head_twist       - spins the shade itself (blink/twitch accents)

All angles are in degrees, measured from the neutral "resting" pose
(pointing straight up, as if idly standing on a desk).
"""
from __future__ import annotations

import numpy as np

JOINT_NAMES = [
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "wrist_pitch",
    "wrist_roll",
    "head_twist",
]

NUM_JOINTS = len(JOINT_NAMES)

# Link lengths in arbitrary "lamp units" (roughly cm if you imagine a desk lamp).
BASE_HEIGHT = 4.0
LOWER_ARM_LEN = 14.0
UPPER_ARM_LEN = 12.0
HEAD_LEN = 6.0

# A relaxed, idle "at rest" pose.
NEUTRAL_POSE = np.array([0.0, -35.0, 55.0, -15.0, 0.0, 0.0])

# Soft per-joint limits (min, max) in degrees, used to clamp any commanded pose.
JOINT_LIMITS = [
    (-90.0, 90.0),   # base_yaw
    (-80.0, 20.0),    # shoulder_pitch
    (-20.0, 130.0),   # elbow_pitch
    (-70.0, 70.0),    # wrist_pitch
    (-45.0, 45.0),    # wrist_roll
    (-60.0, 60.0),    # head_twist
]


def clamp_pose(angles: np.ndarray) -> np.ndarray:
    out = np.array(angles, dtype=float)
    for i, (lo, hi) in enumerate(JOINT_LIMITS):
        out[i] = np.clip(out[i], lo, hi)
    return out


def _rot_z(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _rot_x(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def _rot_y(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def _translate(x: float, y: float, z: float) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = [x, y, z]
    return t


def forward_kinematics(angles: np.ndarray) -> list[np.ndarray]:
    """Return the 3D joint positions (world frame) for a given pose.

    Output is a list of 5 points: [base, shoulder, elbow, wrist, head_tip],
    each an (x, y, z) numpy array. z is "up".
    """
    base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_roll, head_twist = angles

    T = _translate(0, 0, BASE_HEIGHT) @ _rot_z(base_yaw)
    base = T[:3, 3].copy()

    T = T @ _rot_y(shoulder_pitch) @ _translate(0, 0, LOWER_ARM_LEN)
    shoulder = T[:3, 3].copy()

    T = T @ _rot_y(elbow_pitch) @ _translate(0, 0, UPPER_ARM_LEN)
    elbow = T[:3, 3].copy()

    T = T @ _rot_y(wrist_pitch) @ _rot_x(wrist_roll)
    wrist = T[:3, 3].copy()

    T = T @ _rot_z(head_twist) @ _translate(0, 0, HEAD_LEN)
    head_tip = T[:3, 3].copy()

    return [base, shoulder, elbow, wrist, head_tip]


def pose_for_look_at(yaw_deg: float, pitch_deg: float, alertness: float = 1.0) -> np.ndarray:
    """Map a desired gaze direction (yaw/pitch, camera convention) onto the
    6 joint angles, blended with the neutral pose by `alertness` (0=neutral
    idle posture, 1=fully leaned in toward the target).

    This is a hand-authored (not IK-solved) mapping: shoulder+elbow raise the
    head up and forward as alertness increases, wrist_pitch tracks the
    vertical look angle, and base_yaw tracks the horizontal look angle.
    """
    alert_pose = np.array([
        yaw_deg,
        -55.0,
        95.0,
        np.clip(-pitch_deg * 0.6, -40.0, 40.0),
        0.0,
        0.0,
    ])
    blended = NEUTRAL_POSE * (1 - alertness) + alert_pose * alertness
    return clamp_pose(blended)
