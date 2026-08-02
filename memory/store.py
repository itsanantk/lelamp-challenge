"""Persistent scene memory: what the lamp has seen, where, and when.

bearing_deg is relative to the *user's own* left/right, not the raw
camera frame: negative is the user's left, positive the user's right, 0
dead center. The camera faces the user, so a naive "positive x in frame"
reading is actually mirrored from the user's perspective -- both
perception/engagement.py and perception/vision_memory.py correct for
that when they compute a bearing, so everything landing in this table is
already in the frame the user would actually check by looking around.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

import config

# Common synonyms -> COCO class names (the 80 classes YOLO11n was trained
# on). Kept small and desk/room-scene-focused rather than exhaustive.
CLASS_ALIASES = {
    "phone": "cell phone", "cellphone": "cell phone", "mobile": "cell phone",
    "mug": "cup", "computer": "laptop", "monitor": "tv", "screen": "tv",
    "controller": "remote", "control": "remote", "glasses": "glasses",
    "bag": "backpack", "purse": "handbag", "couch": "sofa",
}


def normalize_class(query: str) -> str:
    q = query.strip().lower()
    return CLASS_ALIASES.get(q, q)


def bearing_to_direction(bearing_deg: float) -> str:
    if bearing_deg <= -20:
        return "off to the left"
    if bearing_deg <= -6:
        return "slightly left of center"
    if bearing_deg < 6:
        return "roughly in the center"
    if bearing_deg < 20:
        return "slightly right of center"
    return "off to the right"


@dataclass
class Observation:
    id: int
    timestamp: float
    object_class: str
    confidence: float
    bearing_deg: float
    frame_group_id: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    object_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    bearing_deg REAL NOT NULL,
    bbox_cx REAL, bbox_cy REAL,
    frame_group_id INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_class ON observations(object_class);
CREATE INDEX IF NOT EXISTS idx_group ON observations(frame_group_id);
"""


class MemoryStore:
    def __init__(self, db_path=config.MEMORY_DB, fresh: bool = False):
        if fresh and db_path.exists():
            db_path.unlink()
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def add_observation(self, object_class: str, confidence: float, bearing_deg: float,
                         bbox_cx: float, bbox_cy: float, frame_group_id: int,
                         timestamp: float | None = None) -> int:
        ts = timestamp if timestamp is not None else time.time()
        cur = self.conn.execute(
            "INSERT INTO observations (timestamp, object_class, confidence, bearing_deg, "
            "bbox_cx, bbox_cy, frame_group_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts, object_class.lower(), confidence, bearing_deg, bbox_cx, bbox_cy, frame_group_id),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_latest_by_class(self, object_class: str) -> Observation | None:
        target = normalize_class(object_class)
        row = self.conn.execute(
            "SELECT id, timestamp, object_class, confidence, bearing_deg, frame_group_id "
            "FROM observations WHERE object_class = ? ORDER BY timestamp DESC LIMIT 1",
            (target,),
        ).fetchone()
        if row is None:
            # fall back to a loose substring match (e.g. "bottle" query vs "water bottle")
            row = self.conn.execute(
                "SELECT id, timestamp, object_class, confidence, bearing_deg, frame_group_id "
                "FROM observations WHERE object_class LIKE ? ORDER BY timestamp DESC LIMIT 1",
                (f"%{target}%",),
            ).fetchone()
        return Observation(*row) if row else None

    def get_cooccurring(self, frame_group_id: int, exclude_class: str | None = None) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT object_class FROM observations WHERE frame_group_id = ?",
            (frame_group_id,),
        ).fetchall()
        classes = [r[0] for r in rows]
        if exclude_class:
            classes = [c for c in classes if c != exclude_class.lower()]
        return classes

    def list_known_classes(self) -> list[str]:
        rows = self.conn.execute("SELECT DISTINCT object_class FROM observations").fetchall()
        return sorted(r[0] for r in rows)

    def list_recent(self, limit: int = 20) -> list[Observation]:
        rows = self.conn.execute(
            "SELECT id, timestamp, object_class, confidence, bearing_deg, frame_group_id "
            "FROM observations ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [Observation(*r) for r in rows]

    def close(self) -> None:
        self.conn.close()
