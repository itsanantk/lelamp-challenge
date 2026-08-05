"""Unit tests for chat.py's voice-controlled lighting, and the
standalone/embedded split added when main.py started sharing its lamp
with chat.py's conversation loop instead of each owning a separate one.
Run with: python -m pytest tests/ -v
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chat
from behavior.state_machine import State


@pytest.fixture(autouse=True)
def _unmute_after():
    # voice.set_muted is process-global (see its own docstring) -- a test
    # that mutes and forgets to clean up would silently break every later
    # test in the whole suite that touches voice.speak(), not just this
    # file's own tests.
    yield
    chat.voice.set_muted(False)


class _FakeLamp:
    def __init__(self, current_light=(100, 100, 100), mood=(0.6, 0.55), current_pose=None):
        self.calls = []
        self._light = current_light
        self._mood = mood
        self.closed = False
        self._pose = current_pose if current_pose is not None else np.zeros(6)
        self.pose_calls = []
        self.sounds = []

    def set_light(self, rgb, transition_s=0.4):
        self.calls.append(rgb)
        self._light = rgb

    def get_current_light(self):
        return self._light

    def set_mood(self, warmth, brightness):
        self._mood = (warmth, brightness)

    def get_mood(self):
        return self._mood

    def set_target_pose(self, angles, *args, **kwargs):
        self.pose_calls.append(np.array(angles))
        self._pose = np.array(angles)

    def get_current_pose(self):
        return self._pose

    def play_sound(self, event):
        self.sounds.append(event)

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


def test_dim_lowers_the_brightness_dial_without_touching_warmth():
    lamp = _FakeLamp(mood=(0.6, 0.8))
    chat._apply_light_command(lamp, "dim", show_gui=False, seconds=0.0)
    warmth, brightness = lamp.get_mood()
    assert brightness < 0.8
    assert warmth == 0.6


def test_repeated_dim_never_reaches_zero_brightness():
    # Repeated "dim" should settle near-off, not hit literal 0 brightness --
    # that's what "off" is explicitly for, dim should stay a distinct action.
    lamp = _FakeLamp(mood=(0.6, 0.8))
    for _ in range(20):
        chat._apply_light_command(lamp, "dim", show_gui=False, seconds=0.0)
    _, brightness = lamp.get_mood()
    assert brightness >= chat._BRIGHTNESS_FLOOR
    assert lamp.calls[-1] != (0, 0, 0)


def test_on_sets_brightness_high_while_preserving_warmth():
    # "on" used to scale whatever raw color was showing -- now it's purely
    # a brightness-dial move, so hue (warmth) is untouched by construction
    # instead of needing separate hue-preserving math.
    lamp = _FakeLamp(mood=(0.2, 0.3))
    target = chat._apply_light_command(lamp, "on", show_gui=False, seconds=0.0)
    warmth, brightness = lamp.get_mood()
    assert warmth == 0.2
    assert brightness == chat._ON_BRIGHTNESS
    assert target == chat.color.from_dials(0.2, chat._ON_BRIGHTNESS)


def test_on_from_off_still_produces_real_brightness():
    # Dial-based brightness has no "scaling zero by anything is still
    # zero" failure mode -- "on" after "off" just sets the dial outright.
    lamp = _FakeLamp(mood=(0.5, 0.0))
    target = chat._apply_light_command(lamp, "on", show_gui=False, seconds=0.0)
    assert target != (0, 0, 0)
    _, brightness = lamp.get_mood()
    assert brightness == chat._ON_BRIGHTNESS


class _FakeReminderEngine:
    def __init__(self):
        self.added = []
        self.cancelled = []

    def add(self, kind, message, interval_s=None, duration_s=None, object_class=None, due_in_s=None,
            check_question=None, alert_on_detection=False):
        self.added.append((kind, message, interval_s, duration_s, object_class, due_in_s, check_question,
                            alert_on_detection))
        return type("R", (), {"id": 1, "kind": kind, "message": message, "interval_s": interval_s,
                               "object_class": object_class, "check_question": check_question,
                               "alert_on_detection": alert_on_detection,
                               "due_at": (time.time() + due_in_s) if due_in_s else None,
                               "expires_at": (time.time() + duration_s) if duration_s else None})()

    def cancel_all(self, kind=None):
        self.cancelled.append(kind)
        return 2


def test_apply_reminder_action_create_adds_to_the_engine():
    engine = _FakeReminderEngine()
    chat._apply_reminder_action(
        {"action": "create", "kind": "recurring", "message": "stand up", "interval_s": 1800.0}, engine)
    assert engine.added == [("recurring", "stand up", 1800.0, None, None, None, None, False)]


def test_apply_reminder_action_create_passes_a_duration_through():
    engine = _FakeReminderEngine()
    chat._apply_reminder_action(
        {"action": "create", "kind": "presence", "message": "come back", "interval_s": None,
         "duration_s": 20.0}, engine)
    assert engine.added == [("presence", "come back", None, 20.0, None, None, None, False)]


def test_apply_reminder_action_create_passes_object_check_fields_through():
    engine = _FakeReminderEngine()
    chat._apply_reminder_action(
        {"action": "create", "kind": "object_check", "message": "did you finish your water?",
         "interval_s": None, "object_class": "bottle", "due_in_s": 3600.0,
         "check_question": "is this bottle full or empty"}, engine)
    assert engine.added == [("object_check", "did you finish your water?", None, None, "bottle", 3600.0,
                              "is this bottle full or empty", False)]


def test_apply_reminder_action_create_passes_alert_on_detection_through():
    engine = _FakeReminderEngine()
    chat._apply_reminder_action(
        {"action": "create", "kind": "object_check", "message": "get off your phone",
         "interval_s": None, "object_class": "cell phone", "due_in_s": 300.0,
         "check_question": None, "alert_on_detection": True}, engine)
    assert engine.added == [("object_check", "get off your phone", None, None, "cell phone", 300.0,
                              None, True)]


def test_apply_reminder_action_cancel_targets_the_engine():
    engine = _FakeReminderEngine()
    chat._apply_reminder_action({"action": "cancel", "kind": "presence"}, engine)
    assert engine.cancelled == ["presence"]


def test_apply_reminder_action_with_no_engine_does_not_raise():
    # standalone chat.py has no reminder_engine (see run()'s docstring) --
    # the tool call still succeeded from the LLM's point of view, this
    # just can't act on it. Must not crash the conversation turn.
    chat._apply_reminder_action({"action": "create", "kind": "presence", "message": "x", "interval_s": None}, None)


def test_strip_emoji_removes_emoji_and_collapses_the_gap():
    assert chat._strip_emoji("Sure thing! \U0001F319 all set") == "Sure thing! all set"
    assert chat._strip_emoji("✨ nice ✨") == "nice"


def test_strip_emoji_leaves_plain_text_untouched():
    assert chat._strip_emoji("Still off to the left, about 5 minutes ago.") == \
        "Still off to the left, about 5 minutes ago."


def test_reactive_tones_are_exactly_yelling_tense_and_quiet():
    assert set(chat._REACTIVE_TONES) == {"yelling", "tense", "quiet"}
    assert "calm" not in chat._REACTIVE_TONES
    assert "energetic" not in chat._REACTIVE_TONES


def test_jerk_back_pulls_the_pose_away_from_wherever_it_currently_is():
    # Additive off get_current_pose(), not a fixed absolute target -- the
    # resulting pose should differ from both the starting pose and from
    # NEUTRAL_POSE by exactly the offset.
    lamp = _FakeLamp(current_pose=np.array([10.0, -30.0, 60.0, 0.0, 0.0, 0.0]))
    chat._jerk_back(lamp)
    assert len(lamp.pose_calls) == 1
    np.testing.assert_allclose(lamp.pose_calls[0], np.array([10.0, -30.0, 60.0, 0.0, 0.0, 0.0]) + chat._JERK_BACK_OFFSET)


def test_droop_is_gentler_and_slower_than_jerk_back():
    # Same additive-offset idea, but the droop offset should be smaller in
    # magnitude than the jerk -- a sag reads as sad, not startled.
    assert np.linalg.norm(chat._DROOP_OFFSET) < np.linalg.norm(chat._JERK_BACK_OFFSET)


def test_flash_tone_on_yelling_does_not_play_a_sound_or_gesture():
    # Yelling's sound+gesture fire immediately, live, off raw mic loudness
    # (_on_loud_during_listen / voice.record_until_silence's on_loud) --
    # _flash_tone only supplies the lingering color tint, so it must not
    # also play "startled" or jerk the pose a second time after the fact.
    lamp = _FakeLamp(mood=(0.5, 0.5))
    chat._flash_tone(lamp, "yelling", show_gui=False, seconds=0.0)
    assert lamp.sounds == []
    assert lamp.pose_calls == []


def test_flash_tone_on_quiet_plays_a_whine_and_droops():
    lamp = _FakeLamp(mood=(0.5, 0.5))
    chat._flash_tone(lamp, "quiet", show_gui=False, seconds=0.0)
    assert lamp.sounds == ["whine"]
    assert len(lamp.pose_calls) == 1


def test_flash_tone_on_tense_plays_no_sound_or_gesture():
    # tense only ever got a color nudge, even before yelling/whine existed
    # -- confirms adding sound/gesture support to the other two labels
    # didn't accidentally give tense one too.
    lamp = _FakeLamp(mood=(0.5, 0.5))
    chat._flash_tone(lamp, "tense", show_gui=False, seconds=0.0)
    assert lamp.sounds == []
    assert lamp.pose_calls == []


def test_on_loud_during_listen_jerks_back_and_plays_startled_then_chirps():
    # Two sounds in order: the sharp startle snap first, then the actual
    # "attention_seek" chirp right after -- see _on_loud_during_listen's
    # own comment for why one sound alone read as an ambiguous blip.
    lamp = _FakeLamp()
    chat._on_loud_during_listen(lamp, rms=0.05)
    assert lamp.sounds == ["startled", "attention_seek"]
    assert len(lamp.pose_calls) == 1


class _FakeReminder:
    def __init__(self, tracked_bearing=15.0, check_question="is this bottle full or empty", message="check it"):
        self.kind = "object_check"
        self.tracked_bearing = tracked_bearing
        self.check_question = check_question
        self.message = message
        self.due_for_check = True
        self.active = True


class _FakeReminderEngineWithList:
    def __init__(self, reminders):
        self.reminders = reminders
        self.saved = 0

    def save(self):
        self.saved += 1


class _FakeAgentForJudgment:
    def __init__(self, answer="looks about half full"):
        self.answer = answer
        self.questions_asked = []

    def judge_view(self, question):
        self.questions_asked.append(question)
        return self.answer


def test_resolve_object_check_judgment_points_at_the_bearing_and_speaks(monkeypatch):
    monkeypatch.setattr(chat.time, "sleep", lambda s: None)
    spoken = []
    monkeypatch.setattr(chat.voice, "speak", lambda text: spoken.append(text))

    lamp = _FakeLamp()
    reminder = _FakeReminder()
    engine = _FakeReminderEngineWithList([reminder])
    agent = _FakeAgentForJudgment(answer="looks about half full")

    chat._resolve_object_check_judgment(reminder, engine, lamp, vision_memory=None, agent=agent)

    assert len(lamp.pose_calls) == 1  # pointed toward tracked_bearing
    assert lamp.sounds == ["recall_point"]
    assert agent.questions_asked == ["is this bottle full or empty"]
    assert spoken == ["check it looks about half full"]
    assert not reminder.active  # resolved -- one-shot
    assert engine.saved == 1


def test_resolve_object_check_judgment_falls_back_to_the_plain_message_if_judgment_fails(monkeypatch):
    monkeypatch.setattr(chat.time, "sleep", lambda s: None)
    spoken = []
    monkeypatch.setattr(chat.voice, "speak", lambda text: spoken.append(text))

    lamp = _FakeLamp()
    reminder = _FakeReminder(message="did you finish your water?")
    engine = _FakeReminderEngineWithList([reminder])
    agent = _FakeAgentForJudgment(answer=None)  # no camera / API failure

    chat._resolve_object_check_judgment(reminder, engine, lamp, vision_memory=None, agent=agent)

    assert spoken == ["did you finish your water?"]  # no dangling "None" appended
    assert not reminder.active


def test_resolve_object_check_judgment_claims_the_reminder_before_resolving(monkeypatch):
    # due_for_check must be cleared immediately (before the slow judge_view
    # call), not after -- see the function's own docstring on why a second
    # poll tick mid-resolution must not double-handle it.
    monkeypatch.setattr(chat.time, "sleep", lambda s: None)
    monkeypatch.setattr(chat.voice, "speak", lambda text: None)

    lamp = _FakeLamp()
    reminder = _FakeReminder()
    engine = _FakeReminderEngineWithList([reminder])

    seen_due_for_check_during_judgment = []

    class _WatchingAgent:
        def judge_view(self, question):
            seen_due_for_check_during_judgment.append(reminder.due_for_check)
            return "answer"

    chat._resolve_object_check_judgment(reminder, engine, lamp, vision_memory=None, agent=_WatchingAgent())

    assert seen_due_for_check_during_judgment == [False]


def test_resolve_object_check_judgment_skips_pointing_with_no_tracked_bearing(monkeypatch):
    monkeypatch.setattr(chat.time, "sleep", lambda s: None)
    monkeypatch.setattr(chat.voice, "speak", lambda text: None)

    lamp = _FakeLamp()
    reminder = _FakeReminder(tracked_bearing=None)
    engine = _FakeReminderEngineWithList([reminder])

    chat._resolve_object_check_judgment(reminder, engine, lamp, vision_memory=None,
                                         agent=_FakeAgentForJudgment("answer"))

    assert lamp.pose_calls == []


def test_reminder_judgment_loop_resolves_a_due_reminder_and_then_stops(monkeypatch):
    monkeypatch.setattr(chat.time, "sleep", lambda s: None)
    monkeypatch.setattr(chat.voice, "speak", lambda text: None)

    shutdown_event = threading.Event()
    lamp = _FakeLamp()
    reminder = _FakeReminder()
    engine = _FakeReminderEngineWithList([reminder])
    agent = _FakeAgentForJudgment("answer")

    # The loop's own shutdown_event.wait() is the poll-interval sleep --
    # stopping it after one pass (rather than mocking time) is what keeps
    # this test from actually blocking for _REMINDER_JUDGMENT_POLL_S.
    monkeypatch.setattr(shutdown_event, "wait", lambda timeout: shutdown_event.set())

    chat._reminder_judgment_loop(engine, lamp, vision_memory=None, agent=agent, shutdown_event=shutdown_event)

    assert not reminder.active
    assert agent.questions_asked == ["is this bottle full or empty"]


def test_reminder_judgment_loop_ignores_reminders_not_due_for_check():
    shutdown_event = threading.Event()
    shutdown_event.set()  # stop after the first (only) pass
    lamp = _FakeLamp()
    reminder = _FakeReminder()
    reminder.due_for_check = False
    engine = _FakeReminderEngineWithList([reminder])
    agent = _FakeAgentForJudgment("answer")

    chat._reminder_judgment_loop(engine, lamp, vision_memory=None, agent=agent, shutdown_event=shutdown_event)

    assert reminder.active  # untouched
    assert agent.questions_asked == []


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
        def __init__(self, store, get_frame=None, apply_light=None):
            self.last_light_action = None
            self.last_reminder_action = None
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
                               no_multi_user=True, wake_word="hey lamp")

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
        def __init__(self, store, get_frame=None, apply_light=None):
            self.last_light_action = None
            self.last_reminder_action = None
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
                               no_multi_user=True, wake_word="hey lamp")

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

    def _fake_listen(speaker_detector, max_s=None, no_speech_timeout_s=None, shutdown_event=None, **_kwargs):
        calls.append(max_s)
        return "quit", None, 0, np.zeros(1600, dtype="float32")

    monkeypatch.setattr(chat, "_listen", _fake_listen)

    class _FakeAgent:
        def __init__(self, store, get_frame=None, apply_light=None):
            self.last_light_action = None
            self.last_reminder_action = None
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
                               no_multi_user=True, wake_word="hey lamp")

    chat.run(args, lamp=_FakeLamp(), store=_FakeStore(), fsm=_FakeFSM(State.ENGAGED))

    assert calls == [chat.config.CONVERSATION_FOLLOWUP_TIMEOUT_S]


def test_wait_for_wake_word_is_given_a_working_is_engaged_callback(monkeypatch):
    # Not engaged at the moment wait_for_wake_word is entered (DISENGAGED
    # here), so it must be called -- but it must be handed a real
    # is_engaged callback (not None), so that if the person becomes
    # engaged *while that call is already blocking* it can short-circuit
    # instead of staying stuck demanding the spoken phrase for the rest
    # of that blocking call (see voice.wait_for_wake_word's own docstring
    # on why this matters -- confirmed missing in live testing).
    fsm = _FakeFSM(State.DISENGAGED)
    seen_is_engaged = []

    def _fake_wait_for_wake_word(wake_word, shutdown_event=None, is_engaged=None):
        seen_is_engaged.append(is_engaged)
        # Simulate the real function detecting engagement mid-call, the
        # way it would if the caller actually looked at the lamp while
        # this was already blocking.
        return "engaged"

    monkeypatch.setattr(chat.voice, "preload", lambda: None)
    monkeypatch.setattr(chat.voice, "wait_for_wake_word", _fake_wait_for_wake_word)

    calls = {"n": 0}

    def _fake_listen(speaker_detector, max_s=None, no_speech_timeout_s=None, shutdown_event=None, **_kwargs):
        calls["n"] += 1
        return "quit", None, 0, np.zeros(1600, dtype="float32")

    monkeypatch.setattr(chat, "_listen", _fake_listen)

    class _FakeAgent:
        def __init__(self, store, get_frame=None, apply_light=None):
            self.last_light_action = None
            self.last_reminder_action = None
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
                               no_multi_user=True, wake_word="hey lamp")

    chat.run(args, lamp=_FakeLamp(), store=_FakeStore(), fsm=fsm)

    assert len(seen_is_engaged) == 1
    assert seen_is_engaged[0] is not None and callable(seen_is_engaged[0])
    # "engaged" (not "quit") must fall through to the same wake-sound +
    # listen path "wake" already takes -- it did reach _listen(), not
    # crash or loop forever.
    assert calls["n"] == 1


def test_engaged_invitation_listen_gets_a_keep_waiting_callback(monkeypatch):
    # Real bug, reproduced live: "you're engaged, go ahead and talk"
    # listens with the same generous timeout as an actual follow-up (up to
    # CONVERSATION_FOLLOWUP_TIMEOUT_S) but never re-checked engagement
    # during that wait -- someone engaged when the call started could go
    # fully idle and still be heard with no wake word minutes later.
    # keep_waiting=_check_engaged closes that; this confirms it's actually
    # passed (not None) for this branch specifically.
    monkeypatch.setattr(chat.voice, "preload", lambda: None)

    def _must_not_be_called(*a, **k):
        raise AssertionError("wait_for_wake_word must not be called while already engaged")

    monkeypatch.setattr(chat.voice, "wait_for_wake_word", _must_not_be_called)

    seen_keep_waiting = []

    def _fake_listen(speaker_detector, max_s=None, no_speech_timeout_s=None, shutdown_event=None, **kwargs):
        seen_keep_waiting.append(kwargs.get("keep_waiting"))
        return "quit", None, 0, np.zeros(1600, dtype="float32")

    monkeypatch.setattr(chat, "_listen", _fake_listen)

    class _FakeStore:
        def list_known_classes(self):
            return []

        def close(self):
            pass

    args = argparse.Namespace(no_gui=True, voice=True, mute=True, ask=None,
                               no_multi_user=True, wake_word="hey lamp")

    chat.run(args, lamp=_FakeLamp(), store=_FakeStore(), fsm=_FakeFSM(State.ENGAGED))

    assert len(seen_keep_waiting) == 1
    assert seen_keep_waiting[0] is not None and callable(seen_keep_waiting[0])


def test_a_real_followup_listen_gets_a_keep_waiting_that_ignores_engagement(monkeypatch):
    # The opposite case: once actually in an open conversation
    # (in_conversation True), losing eye contact briefly while thinking of
    # a reply is expected, not a sign of walking away -- that listen must
    # NOT be cut short by _check_engaged the way the engaged-invitation one
    # deliberately is above. It's not plain None either, though -- it still
    # needs to be a real callback so pause ('p' in main.py) can interrupt
    # an open follow-up window, not just the engaged-invitation case (see
    # _not_paused's own comment at the real call site for the live bug
    # this covers: in_conversation staying True meant pause used to do
    # nothing for up to a full follow-up timeout).
    monkeypatch.setattr(chat.voice, "preload", lambda: None)
    monkeypatch.setattr(chat.voice, "wait_for_wake_word", lambda *a, **k: "wake")
    monkeypatch.setattr(chat.voice, "speak", lambda text: None)

    seen_keep_waiting = []
    calls = {"n": 0}

    def _fake_listen(speaker_detector, max_s=None, no_speech_timeout_s=None, shutdown_event=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "what's the time", None, 0, np.zeros(1600, dtype="float32")
        seen_keep_waiting.append(kwargs.get("keep_waiting"))
        return "quit", None, 0, np.zeros(1600, dtype="float32")

    monkeypatch.setattr(chat, "_listen", _fake_listen)

    class _FakeAgent:
        def __init__(self, store, get_frame=None, apply_light=None):
            self.last_light_action = None
            self.last_reminder_action = None
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
                               no_multi_user=True, wake_word="hey lamp")

    chat.run(args, lamp=_FakeLamp(), store=_FakeStore(), fsm=_FakeFSM(State.DISENGAGED))

    assert calls["n"] == 2
    assert len(seen_keep_waiting) == 1
    keep_waiting = seen_keep_waiting[0]
    assert keep_waiting is not None
    assert keep_waiting() is True  # no paused_event given to chat.run here -- never blocks


def test_a_real_followup_listens_keep_waiting_returns_false_once_paused(monkeypatch):
    # The actual live bug: pausing mid-conversation used to do nothing --
    # in_conversation stayed True, and the follow-up listen's keep_waiting
    # was plain None, so nothing ever re-checked pause during the wait.
    # Same setup as the test above, but flips paused_event after capturing
    # the callback and checks it changes the result -- see the comment by
    # that flip below for why it isn't set before chat.run runs.
    monkeypatch.setattr(chat.voice, "preload", lambda: None)
    monkeypatch.setattr(chat.voice, "wait_for_wake_word", lambda *a, **k: "wake")
    monkeypatch.setattr(chat.voice, "speak", lambda text: None)

    seen_keep_waiting = []
    calls = {"n": 0}

    def _fake_listen(speaker_detector, max_s=None, no_speech_timeout_s=None, shutdown_event=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "what's the time", None, 0, np.zeros(1600, dtype="float32")
        seen_keep_waiting.append(kwargs.get("keep_waiting"))
        return "quit", None, 0, np.zeros(1600, dtype="float32")

    monkeypatch.setattr(chat, "_listen", _fake_listen)

    class _FakeAgent:
        def __init__(self, store, get_frame=None, apply_light=None):
            self.last_light_action = None
            self.last_reminder_action = None
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
                               no_multi_user=True, wake_word="hey lamp")
    # NOT set going in -- if it were, chat.run's own top-of-loop pause
    # check would skip straight past the wake-word/first-listen flow
    # before in_conversation ever became True, and the loop would spin on
    # that check forever (no shutdown_event given here to break it). Set
    # afterward instead, on the captured callback itself -- this is also a
    # more faithful test: in the real thing, paused_event can flip *while*
    # record_until_silence is already mid-poll on this exact closure.
    paused_event = threading.Event()

    chat.run(args, lamp=_FakeLamp(), store=_FakeStore(), fsm=_FakeFSM(State.DISENGAGED),
             paused_event=paused_event)

    assert calls["n"] == 2
    keep_waiting = seen_keep_waiting[0]
    assert keep_waiting is not None
    assert keep_waiting() is True  # not paused during the run -- confirms the callback is live, not hardcoded

    paused_event.set()
    assert keep_waiting() is False  # same callback instance, now reflects pause


def test_pause_skips_opening_the_mic_for_a_fresh_wake_word_listen(monkeypatch):
    # Not in a conversation at all, pause already set from the start --
    # the loop must spin on the pause check and never call
    # wait_for_wake_word/_listen in the first place.
    monkeypatch.setattr(chat.voice, "preload", lambda: None)
    wake_word_calls = {"n": 0}
    monkeypatch.setattr(chat.voice, "wait_for_wake_word", lambda *a, **k: wake_word_calls.update(n=wake_word_calls["n"] + 1))
    listen_calls = {"n": 0}
    monkeypatch.setattr(chat, "_listen", lambda *a, **k: listen_calls.update(n=listen_calls["n"] + 1))

    class _FakeAgent:
        def __init__(self, store, get_frame=None, apply_light=None):
            self.last_light_action = None
            self.last_reminder_action = None
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
                               no_multi_user=True, wake_word="hey lamp")
    paused_event = threading.Event()
    paused_event.set()
    shutdown_event = threading.Event()

    # Flip shutdown after a few spins so the loop actually exits -- there's
    # no other way out while paused_event stays set the whole time.
    spins = {"n": 0}

    def _count_and_maybe_stop(s):
        spins["n"] += 1
        if spins["n"] >= 3:
            shutdown_event.set()

    monkeypatch.setattr(chat.time, "sleep", _count_and_maybe_stop)

    chat.run(args, lamp=_FakeLamp(), store=_FakeStore(), fsm=_FakeFSM(State.DISENGAGED),
             shutdown_event=shutdown_event, paused_event=paused_event)

    assert wake_word_calls["n"] == 0
    assert listen_calls["n"] == 0
    assert spins["n"] >= 3


def test_mute_skips_opening_the_mic_for_a_fresh_wake_word_listen(monkeypatch):
    # Same as the pause test above, but via voice.set_muted -- mute is
    # main.py's separate 'm' toggle (independent of 'p'/paused_event, see
    # _voice_suspended's own comment), and must gate the exact same
    # top-of-loop skip.
    monkeypatch.setattr(chat.voice, "preload", lambda: None)
    wake_word_calls = {"n": 0}
    monkeypatch.setattr(chat.voice, "wait_for_wake_word", lambda *a, **k: wake_word_calls.update(n=wake_word_calls["n"] + 1))
    listen_calls = {"n": 0}
    monkeypatch.setattr(chat, "_listen", lambda *a, **k: listen_calls.update(n=listen_calls["n"] + 1))

    class _FakeAgent:
        def __init__(self, store, get_frame=None, apply_light=None):
            self.last_light_action = None
            self.last_reminder_action = None
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
                               no_multi_user=True, wake_word="hey lamp")
    chat.voice.set_muted(True)
    shutdown_event = threading.Event()

    spins = {"n": 0}

    def _count_and_maybe_stop(s):
        spins["n"] += 1
        if spins["n"] >= 3:
            shutdown_event.set()

    monkeypatch.setattr(chat.time, "sleep", _count_and_maybe_stop)

    chat.run(args, lamp=_FakeLamp(), store=_FakeStore(), fsm=_FakeFSM(State.DISENGAGED),
             shutdown_event=shutdown_event)

    assert wake_word_calls["n"] == 0
    assert listen_calls["n"] == 0
    assert spins["n"] >= 3


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

    def _fake_listen(speaker_detector, max_s=None, no_speech_timeout_s=None, shutdown_event=None, **_kwargs):
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
        def __init__(self, store, get_frame=None, apply_light=None):
            self.last_light_action = None
            self.last_reminder_action = None
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
                               no_multi_user=True, wake_word="hey lamp")

    chat.run(args, lamp=_FakeLamp(), store=_FakeStore())

    assert flashed == []


def test_recall_points_at_the_object_before_speaking_and_confirms_if_live(monkeypatch):
    # "the lamp should look to the phone FIRST and checked if it was
    # there" -- used to speak the (memory-only) reply first and only then
    # point/check. Order must be point-then-speak, and a live find should
    # get an audible confirmation the memory-only reply text couldn't have
    # included. Points toward the *remembered* bearing (not the live one)
    # -- the whole reason to point before checking is so the pan-crop
    # tracking the lamp's aim has a chance to actually include the object
    # by the time the live check runs (see _point_toward's docstring).
    monkeypatch.setattr(chat.voice, "preload", lambda: None)
    monkeypatch.setattr(chat.voice, "wait_for_wake_word", lambda *a, **k: "wake")

    order = []
    monkeypatch.setattr(chat.voice, "speak", lambda text: order.append(("speak", text)))

    real_point_toward = chat._point_toward

    def _tracked_point_toward(lamp, bearing_deg, fsm=None):
        order.append(("point", bearing_deg))
        return real_point_toward(lamp, bearing_deg, fsm=fsm)

    monkeypatch.setattr(chat, "_point_toward", _tracked_point_toward)
    # 25.0 resolves to "off to the right" (bearing_to_direction), the
    # specific case that exposes the "off to the off to the right"
    # duplication bug -- a bearing that resolves to a phrase not already
    # starting with "off to the" (e.g. "slightly right of center")
    # wouldn't have caught it.
    monkeypatch.setattr(chat, "_find_live_bearing", lambda vm, cls, window_s=1.6: 25.0)

    calls = {"n": 0}

    def _fake_listen(speaker_detector, max_s=None, no_speech_timeout_s=None, shutdown_event=None, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "where is my phone", None, 0, np.zeros(1600, dtype="float32")
        return "quit", None, 0, np.zeros(1600, dtype="float32")

    monkeypatch.setattr(chat, "_listen", _fake_listen)
    monkeypatch.setattr(chat.emotion, "analyze",
                         lambda audio: type("T", (), {"label": "calm", "rms": 0.0,
                                                       "speaking_rate_hz": 0.0,
                                                       "pitch_variability_hz": None})())

    class _FakeObs:
        object_class = "cell phone"
        bearing_deg = 30.0
        timestamp = time.time()

    class _FakeAgent:
        def __init__(self, store, get_frame=None, apply_light=None):
            self.last_light_action = None
            self.last_reminder_action = None
            self.last_recall = type("R", (), {"observation": _FakeObs()})()

        def ask(self, question):
            return "I last saw your phone on the desk."

    monkeypatch.setattr(chat, "MemoryAgent", _FakeAgent)

    class _FakeStore:
        def list_known_classes(self):
            return []

        def close(self):
            pass

    args = argparse.Namespace(no_gui=True, voice=True, mute=True, ask=None,
                               no_multi_user=True, wake_word="hey lamp")

    chat.run(args, lamp=_FakeLamp(), store=_FakeStore())

    # point -> an audible "let me check" filler -> the final confirmed reply,
    # in that order. The filler makes the check itself audible instead of
    # just a gap between question and answer; the final reply is still what
    # actually confirms the find.
    assert [kind for kind, _ in order] == ["point", "speak", "speak"]
    assert order[0][1] == 30.0  # pointed at the remembered bearing -- see docstring above
    # Deterministic status sentence, not the LLM's own reply -- see
    # handle()'s comment on why that reply can't be trusted for tense.
    assert order[2][1].startswith("It's right there,")
    assert "I can see it now" in order[2][1]
    # bearing_to_direction already returns a full phrase ("off to the
    # right") -- wrapping it in another "off to the {direction}" produced
    # a real duplication bug ("off to the off to the right"), caught live.
    assert order[2][1] == "It's right there, off to the right -- I can see it now."


def test_recall_does_not_add_a_live_confirmation_when_not_currently_visible(monkeypatch):
    monkeypatch.setattr(chat.voice, "preload", lambda: None)
    spoken = []
    monkeypatch.setattr(chat.voice, "speak", lambda text: spoken.append(text))
    monkeypatch.setattr(chat.voice, "wait_for_wake_word", lambda *a, **k: "wake")
    monkeypatch.setattr(chat, "_find_live_bearing", lambda vm, cls, window_s=1.6: None)

    calls = {"n": 0}

    def _fake_listen(speaker_detector, max_s=None, no_speech_timeout_s=None, shutdown_event=None, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "where is my phone", None, 0, np.zeros(1600, dtype="float32")
        return "quit", None, 0, np.zeros(1600, dtype="float32")

    monkeypatch.setattr(chat, "_listen", _fake_listen)
    monkeypatch.setattr(chat.emotion, "analyze",
                         lambda audio: type("T", (), {"label": "calm", "rms": 0.0,
                                                       "speaking_rate_hz": 0.0,
                                                       "pitch_variability_hz": None})())

    class _FakeObs:
        object_class = "cell phone"
        bearing_deg = 30.0
        timestamp = time.time()

    class _FakeAgent:
        def __init__(self, store, get_frame=None, apply_light=None):
            self.last_light_action = None
            self.last_reminder_action = None
            self.last_recall = type("R", (), {"observation": _FakeObs()})()

        def ask(self, question):
            return "I last saw your phone on the desk."

    monkeypatch.setattr(chat, "MemoryAgent", _FakeAgent)

    class _FakeStore:
        def list_known_classes(self):
            return []

        def close(self):
            pass

    args = argparse.Namespace(no_gui=True, voice=True, mute=True, ask=None,
                               no_multi_user=True, wake_word="hey lamp")

    chat.run(args, lamp=_FakeLamp(), store=_FakeStore())

    # The "let me check" filler still plays (it's audible regardless of
    # outcome). The reply itself is the deterministic "not seen right now"
    # sentence, not the LLM's own (misleadingly present-tense-sounding)
    # reply text -- see handle()'s comment.
    assert spoken[0] == "Let me check."
    assert spoken[1].startswith("I don't see it right now, but it was last seen")


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
    assert chat._find_live_bearing(_StuckVisionMemory(), "cell phone", window_s=0.05) is None


def test_listen_skips_transcription_on_pure_ambient_noise(monkeypatch):
    # A patient engaged/follow-up listen can run its whole window on
    # nothing but ambient noise -- Whisper doesn't reliably return "" on
    # that (it can hallucinate a short phrase), which live testing showed
    # as "it keeps thinking I'm talking even when I don't say anything."
    # Below the gate, transcribe() must never even get called.
    quiet_audio = np.random.uniform(-0.0005, 0.0005, 1600).astype("float32")
    monkeypatch.setattr(chat.voice, "record_until_silence", lambda **kwargs: quiet_audio)

    def _must_not_be_called(*a, **k):
        raise AssertionError("transcribe() must not be called on audio that never crossed the gate")
    monkeypatch.setattr(chat.voice, "transcribe", _must_not_be_called)

    text, bearing, n_faces, audio = chat._listen(None)
    assert text == ""


def test_listen_still_transcribes_real_speech(monkeypatch):
    loud_audio = np.random.uniform(-0.5, 0.5, 1600).astype("float32")
    monkeypatch.setattr(chat.voice, "record_until_silence", lambda **kwargs: loud_audio)
    monkeypatch.setattr(chat.voice, "transcribe", lambda audio, **k: "hello lamp")

    text, bearing, n_faces, audio = chat._listen(None)
    assert text == "hello lamp"
