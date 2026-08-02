"""Unit tests for conversation/voice.py's pure logic (wake-word matching,
RMS gating). The actual mic I/O (record_until_silence's InputStream
plumbing) is hardware-dependent and live-verified separately, same as
perception/audio_monitor.py -- see docs/ARCHITECTURE.md.
Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from conversation import voice


def test_rms_of_silence_is_near_zero():
    silence = np.zeros(1000, dtype="float32")
    assert voice._rms(silence) < 1e-6


def test_rms_of_loud_signal_is_high():
    loud = np.ones(1000, dtype="float32") * 0.5
    assert voice._rms(loud) > 0.4


def test_peak_rms_catches_a_brief_loud_burst_that_flat_averaging_would_miss():
    # mostly-silent chunk with one loud 0.25s burst -- averaged over the
    # whole chunk that burst dilutes below the gate; peak-over-subframes
    # is what actually caught real speech in the field-tested transcript.
    samplerate = 16000
    quiet = np.zeros(int(10.0 * samplerate), dtype="float32")
    burst = np.full(int(0.15 * samplerate), 0.05, dtype="float32")
    audio = np.concatenate([quiet, burst])
    assert voice._rms(audio) < 0.008
    assert voice._peak_rms(audio, samplerate) > 0.03


def test_transcribe_of_empty_audio_short_circuits_without_loading_whisper():
    # If this touched preload()/whisper it'd either hang downloading a
    # model or throw in a clean test env -- the empty-audio guard must
    # return before any of that.
    assert voice.transcribe(np.zeros(0, dtype="float32")) == ""


def test_wait_for_wake_word_ignores_silent_chunks_and_transcribes_only_loud_ones(monkeypatch):
    transcribed = []
    chunks = iter([
        (0.0, "n/a"),                       # silent -- should be skipped, never transcribed
        (1.0, "just some background noise"),
        (1.0, "hey lamp what time is it"),
    ])
    state = {}

    def fake_record(duration_s, samplerate=None):
        rms_val, text = next(chunks)
        state["pending_text"] = text
        return np.full(10, rms_val, dtype="float32")

    def fake_transcribe(audio):
        transcribed.append(state["pending_text"])
        return state["pending_text"]

    monkeypatch.setattr(voice, "record", fake_record)
    monkeypatch.setattr(voice, "transcribe", fake_transcribe)
    monkeypatch.setattr(voice, "preload", lambda: None)

    result = voice.wait_for_wake_word("hey lamp", chunk_s=0.01)

    # the silent first chunk must never have reached transcribe()
    assert transcribed == ["just some background noise", "hey lamp what time is it"]
    assert result == "wake"


def test_wait_for_wake_word_matches_phrase_split_across_a_chunk_boundary(monkeypatch):
    """The wake phrase can get cut in half between one chunk's recording
    and the next -- concatenating each chunk's transcript with the
    previous one is what catches that instead of silently missing both
    halves. Neither chunk alone contains "lamp"."""
    chunks = iter(["turn on the hey", "lamp please"])

    def fake_record(duration_s, samplerate=None):
        return np.ones(10, dtype="float32")

    def fake_transcribe(audio):
        return next(chunks)

    monkeypatch.setattr(voice, "record", fake_record)
    monkeypatch.setattr(voice, "transcribe", fake_transcribe)
    monkeypatch.setattr(voice, "preload", lambda: None)

    result = voice.wait_for_wake_word("hey lamp", chunk_s=0.01)
    assert result == "wake"


def test_wait_for_wake_word_requires_the_distinctive_word_not_just_hey(monkeypatch):
    """"hey" alone is common enough (filler word, Whisper artifact) that
    it shouldn't trigger it by itself -- only "lamp" (the last, most
    distinctive word of the phrase) actually has to be heard."""
    calls = {"n": 0}
    chunks = iter(["hey there, how's it going", "ok hey lamp for real this time"])

    def fake_record(duration_s, samplerate=None):
        return np.ones(10, dtype="float32")

    def fake_transcribe(audio):
        calls["n"] += 1
        return next(chunks)

    monkeypatch.setattr(voice, "record", fake_record)
    monkeypatch.setattr(voice, "transcribe", fake_transcribe)
    monkeypatch.setattr(voice, "preload", lambda: None)

    result = voice.wait_for_wake_word("hey lamp", chunk_s=0.01)
    assert result == "wake"
    assert calls["n"] == 2, "should NOT have matched on the first ('hey' only) chunk"


def test_wait_for_wake_word_matches_case_insensitively(monkeypatch):
    calls = {"n": 0}

    def fake_record(duration_s, samplerate=None):
        return np.ones(10, dtype="float32")

    def fake_transcribe(audio):
        calls["n"] += 1
        return "HEY LAMP are you there"

    monkeypatch.setattr(voice, "record", fake_record)
    monkeypatch.setattr(voice, "transcribe", fake_transcribe)
    monkeypatch.setattr(voice, "preload", lambda: None)

    result = voice.wait_for_wake_word("hey lamp", chunk_s=0.01)
    assert calls["n"] == 1
    assert result == "wake"


def test_wait_for_wake_word_recognizes_quit_without_needing_the_wake_word_first(monkeypatch):
    """A wake-gated loop with no quit path of its own is a dead end short
    of Ctrl-C -- the user has to be able to leave voice mode without first
    successfully waking it up."""
    monkeypatch.setattr(voice, "record", lambda duration_s, samplerate=None: np.ones(10, dtype="float32"))
    monkeypatch.setattr(voice, "transcribe", lambda audio: "quit")
    monkeypatch.setattr(voice, "preload", lambda: None)

    result = voice.wait_for_wake_word("hey lamp", chunk_s=0.01)
    assert result == "quit"


if __name__ == "__main__":
    test_rms_of_silence_is_near_zero()
    test_rms_of_loud_signal_is_high()
    test_transcribe_of_empty_audio_short_circuits_without_loading_whisper()
    print("ALL VOICE LOGIC TESTS PASSED (monkeypatch tests need pytest)")
