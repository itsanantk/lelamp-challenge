"""Unit tests for perception/speaker_id.py -- exercises _match_and_update
directly with synthetic embeddings so this doesn't need to load the real
resemblyzer model (identify() itself is a thin, untested wrapper around
that plus this logic). Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from perception.speaker_id import SpeakerIdentifier

_A = np.array([1.0, 0.0, 0.0])
_A_CLOSE = np.array([0.95, 0.05, 0.0])  # same "speaker," slight natural variation
_B = np.array([0.0, 1.0, 0.0])  # a clearly different "speaker"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "speakers.json"


def test_first_utterance_enrolls_a_new_speaker(db_path):
    ident = SpeakerIdentifier(db_path=db_path, similarity_threshold=0.8)
    sid, is_new = ident._match_and_update(_A)
    assert is_new
    assert sid in ident.speakers
    assert ident.speakers[sid]["seen_count"] == 1


def test_a_close_embedding_matches_the_existing_speaker(db_path):
    ident = SpeakerIdentifier(db_path=db_path, similarity_threshold=0.8)
    sid1, _ = ident._match_and_update(_A)
    sid2, is_new = ident._match_and_update(_A_CLOSE)
    assert not is_new
    assert sid2 == sid1
    assert ident.speakers[sid1]["seen_count"] == 2


def test_a_distant_embedding_enrolls_as_a_different_speaker(db_path):
    ident = SpeakerIdentifier(db_path=db_path, similarity_threshold=0.8)
    sid1, _ = ident._match_and_update(_A)
    sid2, is_new = ident._match_and_update(_B)
    assert is_new
    assert sid2 != sid1
    assert len(ident.speakers) == 2


def test_matched_speaker_embedding_updates_as_a_running_average(db_path):
    ident = SpeakerIdentifier(db_path=db_path, similarity_threshold=0.8)
    sid, _ = ident._match_and_update(_A)
    ident._match_and_update(_A_CLOSE)
    # running average of _A and _A_CLOSE, not a straight overwrite by either
    updated = ident.speakers[sid]["embedding"]
    assert not np.allclose(updated, _A)
    assert not np.allclose(updated, _A_CLOSE)


def test_persists_across_instances(db_path):
    ident1 = SpeakerIdentifier(db_path=db_path, similarity_threshold=0.8)
    sid, _ = ident1._match_and_update(_A)

    ident2 = SpeakerIdentifier(db_path=db_path, similarity_threshold=0.8)
    sid2, is_new = ident2._match_and_update(_A_CLOSE)
    assert not is_new
    assert sid2 == sid


def test_identify_returns_a_throwaway_id_for_audio_too_short():
    ident = SpeakerIdentifier(db_path=Path("unused.json"), similarity_threshold=0.8)
    short_audio = np.zeros(100, dtype="float32")  # far under SPEAKER_MIN_AUDIO_S at any real sample rate
    sid, is_new = ident.identify(short_audio, sample_rate=16000)
    assert is_new
    assert sid.startswith("unknown-")
    assert ident.speakers == {}  # never actually enrolled


def test_set_name_and_display_name(db_path):
    ident = SpeakerIdentifier(db_path=db_path, similarity_threshold=0.8)
    sid, _ = ident._match_and_update(_A)
    assert ident.display_name(sid) == f"speaker {sid.split('-')[1]}"

    ident.set_name(sid, "Anant")
    assert ident.display_name(sid) == "Anant"


def test_display_name_for_an_unknown_throwaway_id():
    ident = SpeakerIdentifier(db_path=Path("unused.json"), similarity_threshold=0.8)
    assert ident.display_name("unknown-abcd1234") == "someone"
