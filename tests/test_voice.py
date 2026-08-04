"""Unit tests for conversation/voice.py's pure logic (wake-word matching,
RMS gating). The actual mic I/O (record_until_silence's and
wait_for_wake_word's InputStream plumbing) is hardware-dependent and
live-verified separately, same as perception/audio_monitor.py -- see
docs/ARCHITECTURE.md.
Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from conversation import voice


@pytest.fixture(autouse=True)
def _no_real_wake_whisper(monkeypatch):
    # _check_wake_chunk calls _load_wake_whisper() before transcribe() --
    # without this, every test below would trigger a real whisper.load_model
    # ("tiny.en") the first time it ran, even though voice.transcribe
    # itself is monkeypatched per-test. None here simulates "the smaller
    # model isn't available," which _check_wake_chunk already falls back
    # from cleanly (see its own comment).
    monkeypatch.setattr(voice, "_load_wake_whisper", lambda: None)


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


def test_is_speaking_reflects_the_speak_lock():
    # speak() itself needs real TTS hardware (live-verified, see module
    # docstring) -- is_speaking() only needs to reflect _speak_lock, which
    # is testable directly without going anywhere near an audio device.
    assert not voice.is_speaking()
    with voice._speak_lock:
        assert voice.is_speaking()
    assert not voice.is_speaking()


def test_is_listening_reflects_the_speech_active_event():
    # record_until_silence()'s actual speech-detection logic (when this
    # event gets set/cleared) needs real mic hardware -- live-verified,
    # see module docstring. is_listening() only needs to reflect
    # _speech_active, which is testable directly.
    assert not voice.is_listening()
    voice._speech_active.set()
    assert voice.is_listening()
    voice._speech_active.clear()
    assert not voice.is_listening()


def test_transcribe_of_empty_audio_short_circuits_without_loading_whisper():
    # If this touched preload()/whisper it'd either hang downloading a
    # model or throw in a clean test env -- the empty-audio guard must
    # return before any of that.
    assert voice.transcribe(np.zeros(0, dtype="float32")) == ""


# _check_wake_chunk is what wait_for_wake_word's InputStream plumbing
# calls per buffered chunk -- tested directly, same split as
# record_until_silence's hardware I/O vs its own testable RMS logic (see
# module docstring). wait_for_wake_word's actual mic capture is
# live-verified instead, not unit tested.

_SR = 16000


def test_check_wake_chunk_ignores_silent_audio_and_never_transcribes_it(monkeypatch):
    transcribed = []
    monkeypatch.setattr(voice, "transcribe", lambda audio, initial_prompt=None, model=None: transcribed.append("called") or "n/a")

    silent = np.zeros(10, dtype="float32")
    result, prev_text = voice._check_wake_chunk(silent, _SR, "hey lamp", "lamp", "")

    assert result is None
    assert transcribed == []  # silent chunk must never reach transcribe()


def test_check_wake_chunk_matches_the_wake_word_in_loud_audio(monkeypatch):
    monkeypatch.setattr(voice, "transcribe", lambda audio, initial_prompt=None, model=None: "hey lamp what time is it")

    loud = np.ones(10, dtype="float32")
    result, prev_text = voice._check_wake_chunk(loud, _SR, "hey lamp", "lamp", "")

    assert result == "wake"


def test_check_wake_chunk_matches_phrase_split_across_a_chunk_boundary(monkeypatch):
    """The wake phrase can get cut in half between one chunk and the
    next -- concatenating each chunk's transcript with the previous one
    is what catches that instead of silently missing both halves. Neither
    chunk alone contains "lamp"."""
    loud = np.ones(10, dtype="float32")

    monkeypatch.setattr(voice, "transcribe", lambda audio, initial_prompt=None, model=None: "turn on the hey")
    result, prev_text = voice._check_wake_chunk(loud, _SR, "hey lamp", "lamp", "")
    assert result is None

    monkeypatch.setattr(voice, "transcribe", lambda audio, initial_prompt=None, model=None: "lamp please")
    result, prev_text = voice._check_wake_chunk(loud, _SR, "hey lamp", "lamp", prev_text)
    assert result == "wake"


def test_check_wake_chunk_requires_the_distinctive_word_not_just_hey(monkeypatch):
    """"hey" alone is common enough (filler word, Whisper artifact) that
    it shouldn't trigger it by itself -- only "lamp" (the last, most
    distinctive word of the phrase) actually has to be heard."""
    loud = np.ones(10, dtype="float32")

    monkeypatch.setattr(voice, "transcribe", lambda audio, initial_prompt=None, model=None: "hey there, how's it going")
    result, prev_text = voice._check_wake_chunk(loud, _SR, "hey lamp", "lamp", "")
    assert result is None, "should NOT have matched on 'hey' alone"

    monkeypatch.setattr(voice, "transcribe", lambda audio, initial_prompt=None, model=None: "ok hey lamp for real this time")
    result, prev_text = voice._check_wake_chunk(loud, _SR, "hey lamp", "lamp", prev_text)
    assert result == "wake"


def test_check_wake_chunk_matches_case_insensitively(monkeypatch):
    monkeypatch.setattr(voice, "transcribe", lambda audio, initial_prompt=None, model=None: "HEY LAMP are you there")

    loud = np.ones(10, dtype="float32")
    result, prev_text = voice._check_wake_chunk(loud, _SR, "hey lamp", "lamp", "")

    assert result == "wake"


def test_check_wake_chunk_transcribes_with_the_wake_word_model_when_available(monkeypatch):
    # Latency fix: wake-word chunks decode on the smaller/faster
    # WAKE_WHISPER_MODEL, not the main (more accurate but slower) one --
    # confirms _check_wake_chunk actually passes _load_wake_whisper()'s
    # result through to transcribe() rather than always using the default.
    sentinel = object()
    monkeypatch.setattr(voice, "_load_wake_whisper", lambda: sentinel)
    seen_models = []
    monkeypatch.setattr(voice, "transcribe",
                         lambda audio, initial_prompt=None, model=None: seen_models.append(model) or "hey lamp")

    loud = np.ones(10, dtype="float32")
    voice._check_wake_chunk(loud, _SR, "hey lamp", "lamp", "")

    assert seen_models == [sentinel]


def test_check_wake_chunk_falls_back_to_the_main_model_if_the_wake_model_failed_to_load(monkeypatch):
    monkeypatch.setattr(voice, "_load_wake_whisper", lambda: None)
    monkeypatch.setattr(voice, "_whisper_model", "main-model-sentinel")
    seen_models = []
    monkeypatch.setattr(voice, "transcribe",
                         lambda audio, initial_prompt=None, model=None: seen_models.append(model) or "hey lamp")

    loud = np.ones(10, dtype="float32")
    voice._check_wake_chunk(loud, _SR, "hey lamp", "lamp", "")

    assert seen_models == ["main-model-sentinel"]


def test_check_wake_chunk_recognizes_quit_without_needing_the_wake_word_first(monkeypatch):
    """A wake-gated loop with no quit path of its own is a dead end short
    of Ctrl-C -- the user has to be able to leave voice mode without first
    successfully waking it up."""
    monkeypatch.setattr(voice, "transcribe", lambda audio, initial_prompt=None, model=None: "quit")

    loud = np.ones(10, dtype="float32")
    result, prev_text = voice._check_wake_chunk(loud, _SR, "hey lamp", "lamp", "")

    assert result == "quit"


if __name__ == "__main__":
    test_rms_of_silence_is_near_zero()
    test_rms_of_loud_signal_is_high()
    test_transcribe_of_empty_audio_short_circuits_without_loading_whisper()
    print("ALL VOICE LOGIC TESTS PASSED (monkeypatch tests need pytest)")
