"""Unit tests for memory/store.py. Run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.store import MemoryStore


def _store():
    return MemoryStore(db_path=":memory:")


def test_cooccurring_includes_each_objects_own_bearing_and_confidence():
    # Two objects can share a frame_group_id (same scan) while sitting on
    # opposite sides of a wide camera view -- the caller needs each one's
    # own bearing to avoid implying they're physically close together, and
    # its own confidence so a shaky detection can be flagged as such
    # instead of reported as plain fact (or invented by the LLM with no
    # real data behind it).
    store = _store()
    store.add_observation("cell phone", 0.9, bearing_deg=-40.0, bbox_cx=0.1, bbox_cy=0.5, frame_group_id=1)
    store.add_observation("bottle", 0.5, bearing_deg=45.0, bbox_cx=0.9, bbox_cy=0.5, frame_group_id=1)

    cooccurring = store.get_cooccurring(frame_group_id=1, exclude_class="cell phone")

    assert cooccurring == [("bottle", 45.0, 0.5)]


def test_cooccurring_excludes_the_queried_class_itself():
    store = _store()
    store.add_observation("cell phone", 0.9, bearing_deg=0.0, bbox_cx=0.5, bbox_cy=0.5, frame_group_id=1)
    store.add_observation("book", 0.9, bearing_deg=10.0, bbox_cx=0.6, bbox_cy=0.5, frame_group_id=1)

    cooccurring = store.get_cooccurring(frame_group_id=1, exclude_class="cell phone")

    assert cooccurring == [("book", 10.0, 0.9)]


def test_cooccurring_averages_multiple_instances_of_the_same_class():
    store = _store()
    store.add_observation("cup", 0.8, bearing_deg=-30.0, bbox_cx=0.1, bbox_cy=0.5, frame_group_id=1)
    store.add_observation("cup", 0.6, bearing_deg=-10.0, bbox_cx=0.3, bbox_cy=0.5, frame_group_id=1)
    store.add_observation("phone", 0.9, bearing_deg=0.0, bbox_cx=0.5, bbox_cy=0.5, frame_group_id=1)

    cooccurring = store.get_cooccurring(frame_group_id=1, exclude_class="phone")

    assert cooccurring == [("cup", -20.0, 0.7)]


def test_get_latest_by_class_matches_a_more_specific_query_against_a_shorter_stored_class():
    # COCO's classes are mostly single generic words ("bottle", "cup"), but
    # people ask about things more specifically ("water bottle"). The old
    # substring fallback only matched a short query against a longer stored
    # class -- it never matched this far more common opposite direction, so
    # "water bottle" silently found nothing even though "bottle" sightings
    # existed.
    store = _store()
    store.add_observation("bottle", 0.9, bearing_deg=20.0, bbox_cx=0.5, bbox_cy=0.5, frame_group_id=1)

    obs = store.get_latest_by_class("water bottle")

    assert obs is not None
    assert obs.object_class == "bottle"


def test_get_latest_by_class_still_matches_a_short_query_against_a_longer_stored_class():
    store = _store()
    store.add_observation("sports bottle", 0.9, bearing_deg=20.0, bbox_cx=0.5, bbox_cy=0.5, frame_group_id=1)

    obs = store.get_latest_by_class("bottle")

    assert obs is not None
    assert obs.object_class == "sports bottle"
