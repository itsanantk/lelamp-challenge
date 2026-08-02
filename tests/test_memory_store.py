"""Unit tests for memory/store.py. Run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.store import MemoryStore


def _store():
    return MemoryStore(db_path=":memory:")


def test_cooccurring_includes_each_objects_own_bearing():
    # Two objects can share a frame_group_id (same scan) while sitting on
    # opposite sides of a wide camera view -- the caller needs each one's
    # own bearing to avoid implying they're physically close together.
    store = _store()
    store.add_observation("cell phone", 0.9, bearing_deg=-40.0, bbox_cx=0.1, bbox_cy=0.5, frame_group_id=1)
    store.add_observation("bottle", 0.9, bearing_deg=45.0, bbox_cx=0.9, bbox_cy=0.5, frame_group_id=1)

    cooccurring = store.get_cooccurring(frame_group_id=1, exclude_class="cell phone")

    assert cooccurring == [("bottle", 45.0)]


def test_cooccurring_excludes_the_queried_class_itself():
    store = _store()
    store.add_observation("cell phone", 0.9, bearing_deg=0.0, bbox_cx=0.5, bbox_cy=0.5, frame_group_id=1)
    store.add_observation("book", 0.9, bearing_deg=10.0, bbox_cx=0.6, bbox_cy=0.5, frame_group_id=1)

    cooccurring = store.get_cooccurring(frame_group_id=1, exclude_class="cell phone")

    assert cooccurring == [("book", 10.0)]


def test_cooccurring_averages_multiple_instances_of_the_same_class():
    store = _store()
    store.add_observation("cup", 0.9, bearing_deg=-30.0, bbox_cx=0.1, bbox_cy=0.5, frame_group_id=1)
    store.add_observation("cup", 0.9, bearing_deg=-10.0, bbox_cx=0.3, bbox_cy=0.5, frame_group_id=1)
    store.add_observation("phone", 0.9, bearing_deg=0.0, bbox_cx=0.5, bbox_cy=0.5, frame_group_id=1)

    cooccurring = store.get_cooccurring(frame_group_id=1, exclude_class="phone")

    assert cooccurring == [("cup", -20.0)]
