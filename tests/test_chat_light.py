"""Unit tests for chat.py's voice-controlled lighting, and the
standalone/embedded split added when main.py started sharing its lamp
with chat.py's conversation loop instead of each owning a separate one.
Run with: python -m pytest tests/ -v
"""
import argparse
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chat
from behavior.state_machine import State


class _FakeLamp:
    def __init__(self, current_light=(100, 100, 100)):
        self.calls = []
        self._light = current_light
        self.closed = False

    def set_light(self, rgb, transition_s=0.4):
        self.calls.append(rgb)
        self._light = rgb

    def get_current_light(self):
        return self._light

    def set_target_pose(self, *args, **kwargs):
        pass

    def play_sound(self, event):
        pass

    def update(self, dt):
        pass

    def render(self):
        return None

    def close(self):
        self.closed = True


class _FakeFSM:
    def __init__(self, state):
        self.state = state


def test_preset_actions_use_their_exact_defined_color():
    lamp = _FakeLamp()
    chat._apply_light_command(lamp, "cozy", show_gui=False, seconds=0.0)
    assert lamp.calls[-1] == chat.LIGHT_PRESETS["cozy"]


def test_off_is_fully_dark():
    lamp = _FakeLamp()
    chat._apply_light_command(lamp, "off", show_gui=False, seconds=0.0)
    assert lamp.calls[-1] == (0, 0, 0)


def test_dim_scales_down_from_whatever_is_currently_showing():
    lamp = _FakeLamp(current_light=(100, 100, 100))
    chat._apply_light_command(lamp, "dim", show_gui=False, seconds=0.0)
    dimmed = lamp.calls[-1]
    assert all(c < 100 for c in dimmed)
    assert all(c > 0 for c in dimmed)


def test_repeated_dim_never_reaches_pure_black():
    # Repeated "dim" should settle near-off, not hit literal (0,0,0) --
    # that's what "off" is explicitly for, dim should stay a distinct action.
    lamp = _FakeLamp(current_light=(100, 100, 100))
    for _ in range(20):
        chat._apply_light_command(lamp, "dim", show_gui=False, seconds=0.0)
    assert all(c >= chat._DIM_FLOOR for c in lamp.calls[-1])
    assert lamp.calls[-1] != (0, 0, 0)


def test_on_brightens_the_current_hue_instead_of_a_flat_gray():
    # "on" used to be a fixed (180,180,180) preset -- asking for full
    # brightness on a warm color turned it flat gray instead of a
    # brighter version of the same hue.
    lamp = _FakeLamp(current_light=(20, 100, 50))
    target = chat._apply_light_command(lamp, "on", show_gui=False, seconds=0.0)
    assert max(target) == 255
    # ratios preserved (proportionally the same hue, just brighter)
    b, g, r = (20, 100, 50)
    assert round(target[1] / target[0], 1) == round(g / b, 1)
    assert round(target[2] / target[0], 1) == round(r / b, 1)


def test_on_from_off_restores_a_sensible_default_not_black_times_anything():
    # Scaling (0, 0, 0) up by any factor is still (0, 0, 0) -- "on" after
    # "off" needs an actual fallback color, not a no-op.
    lamp = _FakeLamp(current_light=(0, 0, 0))
    target = chat._apply_light_command(lamp, "on", show_gui=False, seconds=0.0)
    assert target == chat.ENGAGED_COLOR


def test_reactive_tones_are_exactly_tense_and_quiet():
    assert set(chat._REACTIVE_TONES) == {"tense", "quiet"}
    assert "calm" not in chat._REACTIVE_TONES
    assert "energetic" not in chat._REACTIVE_TONES


def test_render_loop_runs_even_without_a_gui_window():
    # show_gui=False should still tick the lamp's own animation forward
    # (update()), just skip the cv2 display calls -- confirms the render
    # loop itself doesn't silently no-op when there's no window.
    lamp = _FakeLamp()
    ticked = []
    lamp.update = lambda dt: ticked.append(dt)
    chat._apply_light_command(lamp, "cozy", show_gui=False, seconds=0.02)
    assert len(ticked) > 0


def test_embedded_mode_never_ticks_the_lamp_itself():
    # standalone=False means main.py's own loop is already calling
    # lamp.update() every frame -- chat's flourish just sets state and
    # waits, it must not *also* tick the lamp (that would double-advance
    # the same trajectory/light transition from two different threads).
    lamp = _FakeLamp()
    ticked = []
    lamp.update = lambda dt: ticked.append(dt)
    chat._apply_light_command(lamp, "cozy", show_gui=False, seconds=0.02, standalone=False)
    assert ticked == []


def test_voice_command_waits_out_an_in_progress_attention_seek():
    # ATTENTION_SEEKING owns pose/light for a short, precisely-timed
    # animation (state_machine._tick_attention_seek) -- the same conflict
    # object_watch.py already solved by deferring rather than fighting.
    # Once the FSM moves on, the command should still go through.
    fsm = _FakeFSM(State.ATTENTION_SEEKING)
    lamp = _FakeLamp()

    import threading
    def _clear_after_a_beat():
        import time
        time.sleep(0.05)
        fsm.state = State.DISENGAGED
    threading.Thread(target=_clear_after_a_beat).start()

    chat._apply_light_command(lamp, "cozy", show_gui=False, seconds=0.0, standalone=False, fsm=fsm)
    assert lamp.calls[-1] == chat.LIGHT_PRESETS["cozy"]


def test_wait_out_attention_seek_gives_up_after_its_timeout():
    # If the FSM never clears (shouldn't happen in practice -- the real
    # animation is bounded to ~1.1s -- but this must not hang forever),
    # the wait bails out rather than blocking the conversation thread
    # indefinitely.
    fsm = _FakeFSM(State.ATTENTION_SEEKING)
    chat._wait_out_attention_seek(fsm, timeout=0.05)  # returns, doesn't hang


def test_run_with_a_shared_lamp_and_store_never_closes_them(monkeypatch):
    # Embedded in main.py, the lamp/store are owned by main.py and must
    # survive after this conversation turn returns -- closing either out
    # from under main.py's own still-running loop would be a real bug.
    class _FakeAgent:
        def __init__(self, store):
            self.last_light_action = None
            self.last_recall = type("R", (), {"observation": None})()

        def ask(self, question):
            return "fake reply"

    monkeypatch.setattr(chat, "MemoryAgent", _FakeAgent)

    class _FakeStore:
        def __init__(self):
            self.closed = False

        def list_known_classes(self):
            return []

        def close(self):
            self.closed = True

    lamp = _FakeLamp()
    store = _FakeStore()
    args = argparse.Namespace(no_gui=True, voice=False, mute=True, ask="where is my phone",
                               multi_user=False, wake_word="hey lamp")

    chat.run(args, lamp=lamp, store=store)

    assert lamp.closed is False
    assert store.closed is False


def test_shutdown_event_already_set_never_touches_real_mic_io(monkeypatch):
    # If main.py's shutdown_event is already set the instant the
    # conversation thread starts (e.g. an --auto-quit-after fired
    # immediately), the top-of-loop check must exit before ever calling
    # wait_for_wake_word -- not rely on that call itself noticing, since
    # that's real InputStream hardware I/O this test has no business
    # touching.
    called = []
    monkeypatch.setattr(chat.voice, "preload", lambda: None)
    monkeypatch.setattr(chat.voice, "wait_for_wake_word", lambda *a, **k: called.append(1) or "wake")

    class _FakeAgent:
        def __init__(self, store):
            self.last_light_action = None
            self.last_recall = type("R", (), {"observation": None})()

        def ask(self, question):
            return "reply"

    monkeypatch.setattr(chat, "MemoryAgent", _FakeAgent)

    class _FakeStore:
        def list_known_classes(self):
            return []

        def close(self):
            pass

    shutdown = threading.Event()
    shutdown.set()
    args = argparse.Namespace(no_gui=True, voice=True, mute=True, ask=None,
                               multi_user=False, wake_word="hey lamp")

    chat.run(args, lamp=_FakeLamp(), store=_FakeStore(), shutdown_event=shutdown)

    assert called == []


def test_being_engaged_skips_the_wake_word_gate_entirely(monkeypatch):
    # Already being looked at is the invitation to talk -- requiring "hey
    # lamp" on top of that would be a redundant gate. Confirms the ENGAGED
    # branch never touches wait_for_wake_word at all, and listens with the
    # same patient follow-up timeout as an already-open conversation
    # (a quick glance shouldn't kick off a question 3s later).
    def _must_not_be_called(*a, **k):
        raise AssertionError("wait_for_wake_word must not be called while already engaged")

    monkeypatch.setattr(chat.voice, "preload", lambda: None)
    monkeypatch.setattr(chat.voice, "wait_for_wake_word", _must_not_be_called)

    calls = []

    def _fake_listen(speaker_detector, max_s=None, no_speech_timeout_s=None, shutdown_event=None):
        calls.append(max_s)
        return "quit", None, 0, np.zeros(1600, dtype="float32")

    monkeypatch.setattr(chat, "_listen", _fake_listen)

    class _FakeAgent:
        def __init__(self, store):
            self.last_light_action = None
            self.last_recall = type("R", (), {"observation": None})()

        def ask(self, question):
            return "reply"

    monkeypatch.setattr(chat, "MemoryAgent", _FakeAgent)

    class _FakeStore:
        def list_known_classes(self):
            return []

        def close(self):
            pass

    args = argparse.Namespace(no_gui=True, voice=True, mute=True, ask=None,
                               multi_user=False, wake_word="hey lamp")

    chat.run(args, lamp=_FakeLamp(), store=_FakeStore(), fsm=_FakeFSM(State.ENGAGED))

    assert calls == [chat.config.CONVERSATION_FOLLOWUP_TIMEOUT_S]


def test_relax_to_idle_does_not_fight_an_active_engaged_look():
    # If the follow-up window lapses while the person is still right
    # there (still ENGAGED), BehaviorFSM's own look-at-you pose/color is
    # already correct and won't be re-asserted on its own (it only fires
    # on the FSM's own transitions) -- relaxing to idle here would
    # incorrectly undo it.
    lamp = _FakeLamp()
    chat._relax_to_idle(lamp, fsm=_FakeFSM(State.ENGAGED))
    assert lamp.calls == []


def test_relax_to_idle_still_relaxes_when_not_engaged():
    lamp = _FakeLamp()
    chat._relax_to_idle(lamp, fsm=_FakeFSM(State.DISENGAGED))
    assert lamp.calls == [chat.IDLE_COLOR]


def test_calm_speech_never_triggers_a_tone_flash(monkeypatch):
    # Real bug, reproduced live: every voice turn used to flash *some*
    # tone color regardless of label, so plain calm conversation kept
    # tinting the lamp green for no reason. Exercises the real handle()
    # path (via run()), not a reimplementation of the gating logic.
    monkeypatch.setattr(chat.voice, "preload", lambda: None)
    monkeypatch.setattr(chat.voice, "speak", lambda text: None)
    monkeypatch.setattr(chat.voice, "wait_for_wake_word", lambda *a, **k: "wake")

    flashed = []
    monkeypatch.setattr(chat, "_flash_tone", lambda *a, **k: flashed.append(True))

    calls = {"n": 0}

    def _fake_listen(speaker_detector, max_s=None, no_speech_timeout_s=None, shutdown_event=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return "talking calmly", None, 0, np.zeros(1600, dtype="float32")
        return "quit", None, 0, np.zeros(1600, dtype="float32")

    monkeypatch.setattr(chat, "_listen", _fake_listen)
    monkeypatch.setattr(chat.emotion, "analyze",
                         lambda audio: type("T", (), {"label": "calm", "rms": 0.0,
                                                       "speaking_rate_hz": 0.0,
                                                       "pitch_variability_hz": None})())

    class _FakeAgent:
        def __init__(self, store):
            self.last_light_action = None
            self.last_recall = type("R", (), {"observation": None})()

        def ask(self, question):
            return "sounds good"

    monkeypatch.setattr(chat, "MemoryAgent", _FakeAgent)

    class _FakeStore:
        def list_known_classes(self):
            return []

        def close(self):
            pass

    args = argparse.Namespace(no_gui=True, voice=True, mute=True, ask=None,
                               multi_user=False, wake_word="hey lamp")

    chat.run(args, lamp=_FakeLamp(), store=_FakeStore())

    assert flashed == []


class _FakeDetection:
    def __init__(self, object_class, bearing_deg):
        self.object_class = object_class
        self.bearing_deg = bearing_deg


class _FakeVisionMemory:
    """scan_count only advances once request_immediate_scan() has been
    honored -- real enough to exercise _find_live_bearing's poll loop
    without loading an actual YOLO model."""

    def __init__(self, detections_after_scan):
        self.scan_count = 0
        self.last_detections = []
        self._detections_after_scan = detections_after_scan
        self._requested = False

    def request_immediate_scan(self):
        self._requested = True
        self.scan_count += 1
        self.last_detections = self._detections_after_scan


def test_find_live_bearing_returns_none_without_a_vision_memory():
    # Standalone chat.py has no live camera loop to ask -- must fall
    # through immediately, not hang waiting on something that'll never
    # update scan_count.
    assert chat._find_live_bearing(None, "cell phone") is None


def test_find_live_bearing_returns_the_live_bearing_when_currently_visible():
    vm = _FakeVisionMemory([_FakeDetection("cell phone", 12.5)])
    assert chat._find_live_bearing(vm, "cell phone") == 12.5
    assert vm._requested


def test_find_live_bearing_returns_none_when_not_currently_visible():
    vm = _FakeVisionMemory([_FakeDetection("cup", 5.0)])
    assert chat._find_live_bearing(vm, "cell phone") is None


def test_find_live_bearing_gives_up_after_its_timeout_if_the_scan_never_lands():
    class _StuckVisionMemory:
        def __init__(self):
            self.scan_count = 0
            self.last_detections = []

        def request_immediate_scan(self):
            pass  # scan_count never advances -- simulates main.py's loop not running

    # Must return (not hang forever) even if nothing ever ticks scan_count.
    assert chat._find_live_bearing(_StuckVisionMemory(), "cell phone", timeout_s=0.05) is None
