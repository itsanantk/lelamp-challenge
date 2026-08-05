"""Unit tests for behavior/reminders.py. Run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

import config
import behavior.reminders as reminders_mod
from behavior.reminders import Reminder, ReminderEngine


class _ImmediateThread:
    """Stand-in for threading.Thread that runs its target synchronously on
    .start() instead of on a real background thread -- _fire() fires off
    voice.speak() this way specifically so it doesn't block the render
    loop (see its own comment), but that makes the spoken text arrive on
    a real timing-dependent thread from a test's point of view. Rebinding
    reminders.threading (not the shared global threading module) to a
    fake module-like object exposing just this class keeps the test
    deterministic without touching real threading.Thread at all, even
    temporarily."""

    def __init__(self, target=None, args=(), daemon=None):
        self._target = target
        self._args = args

    def start(self):
        if self._target is not None:
            self._target(*self._args)


class _FakeThreadingModule:
    Thread = _ImmediateThread


class _FakeLamp:
    def __init__(self):
        self.pose_calls = []
        self.sounds = []
        self._pose = np.zeros(6)

    def set_target_pose(self, angles, *args, **kwargs):
        self.pose_calls.append(np.array(angles))
        self._pose = np.array(angles)

    def get_current_pose(self):
        return self._pose

    def play_sound(self, event):
        self.sounds.append(event)


class _FakeDetection:
    def __init__(self, object_class, bearing_deg):
        self.object_class = object_class
        self.bearing_deg = bearing_deg


class _FakeVisionMemory:
    def __init__(self, detections=None):
        self.last_detections = detections or []
        self.immediate_scans_requested = 0

    def request_immediate_scan(self):
        self.immediate_scans_requested += 1


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    # Same reasoning as test_adaptation.py's _isolate_state_path -- every
    # test in this file that fires a reminder calls save() as a side
    # effect, so redirect to a throwaway path rather than touching the
    # real logs/reminders.json. Also fakes out the threading used to fire
    # voice.speak() (see _ImmediateThread) and voice.speak itself, so
    # nothing here touches real audio hardware or Piper/SAPI5 loading.
    monkeypatch.setattr(reminders_mod, "REMINDERS_PATH", tmp_path / "reminders.json")
    monkeypatch.setattr(reminders_mod, "threading", _FakeThreadingModule())
    spoken = []
    monkeypatch.setattr(reminders_mod.voice, "speak", lambda text: spoken.append(text))
    return spoken


def test_add_creates_an_active_reminder_and_persists_it(_isolate):
    engine = ReminderEngine()
    r = engine.add(kind="recurring", message="stand up", interval_s=1800.0)

    assert r.active
    assert r.kind == "recurring"
    assert engine.active_count() == 1
    assert reminders_mod.REMINDERS_PATH.exists()


def test_recurring_reminder_fires_once_the_interval_elapses():
    engine = ReminderEngine()
    r = engine.add(kind="recurring", message="stand up", interval_s=100.0)
    lamp = _FakeLamp()

    engine.tick(now=r.created_at + 50.0, face_found=True, lamp=lamp)
    assert lamp.sounds == []  # too soon

    engine.tick(now=r.created_at + 100.0, face_found=True, lamp=lamp)
    assert lamp.sounds == ["attention_seek"]
    assert len(lamp.pose_calls) == 1
    assert r.last_fired_at == r.created_at + 100.0


def test_recurring_reminder_fires_again_after_resetting():
    engine = ReminderEngine()
    r = engine.add(kind="recurring", message="stand up", interval_s=100.0)
    lamp = _FakeLamp()

    engine.tick(now=r.created_at + 100.0, face_found=True, lamp=lamp)
    engine.tick(now=r.created_at + 150.0, face_found=True, lamp=lamp)  # too soon since the reset
    assert len(lamp.sounds) == 1

    engine.tick(now=r.created_at + 200.0, face_found=True, lamp=lamp)
    assert len(lamp.sounds) == 2


_DEBOUNCE = reminders_mod.PRESENCE_ABSENCE_DEBOUNCE_S


def test_presence_reminder_does_not_fire_on_a_brief_flicker():
    # face_found is a raw per-frame signal -- a frame or two of no
    # detection (e.g. head motion briefly confusing the face landmarker)
    # must not count as "left," only a sustained absence should.
    engine = ReminderEngine()
    engine.add(kind="presence", message="come back")
    lamp = _FakeLamp()

    engine.tick(now=0.0, face_found=True, lamp=lamp)
    engine.tick(now=0.1, face_found=False, lamp=lamp)   # one flickered frame...
    engine.tick(now=0.2, face_found=True, lamp=lamp)    # ...and back -- never a real departure

    assert lamp.sounds == []


def test_presence_reminder_fires_once_after_a_sustained_absence():
    engine = ReminderEngine()
    engine.add(kind="presence", message="come back")
    lamp = _FakeLamp()

    engine.tick(now=0.0, face_found=True, lamp=lamp)
    engine.tick(now=1.0, face_found=False, lamp=lamp)                    # just left -- too soon
    assert lamp.sounds == []
    engine.tick(now=1.0 + _DEBOUNCE + 0.1, face_found=False, lamp=lamp)  # been gone long enough now
    assert lamp.sounds == ["attention_seek"]


def test_presence_reminder_does_not_refire_every_tick_while_still_away():
    engine = ReminderEngine()
    engine.add(kind="presence", message="come back")
    lamp = _FakeLamp()

    engine.tick(now=0.0, face_found=True, lamp=lamp)
    engine.tick(now=1.0, face_found=False, lamp=lamp)
    engine.tick(now=1.0 + _DEBOUNCE + 0.1, face_found=False, lamp=lamp)
    engine.tick(now=1.0 + _DEBOUNCE + 5.0, face_found=False, lamp=lamp)

    assert len(lamp.sounds) == 1  # not one per tick spent away


def test_presence_reminder_rearms_after_the_user_returns():
    engine = ReminderEngine()
    engine.add(kind="presence", message="come back")
    lamp = _FakeLamp()

    engine.tick(now=0.0, face_found=True, lamp=lamp)
    engine.tick(now=1.0, face_found=False, lamp=lamp)
    engine.tick(now=1.0 + _DEBOUNCE + 0.1, face_found=False, lamp=lamp)  # leaves -- fires
    engine.tick(now=10.0, face_found=True, lamp=lamp)                    # returns
    engine.tick(now=11.0, face_found=False, lamp=lamp)
    engine.tick(now=11.0 + _DEBOUNCE + 0.1, face_found=False, lamp=lamp)  # leaves again -- fires again

    assert len(lamp.sounds) == 2


def test_presence_reminder_does_not_refire_within_the_cooldown_even_after_returning_and_leaving_again():
    # The actual point of PRESENCE_FIRE_COOLDOWN_S: a much shorter
    # PRESENCE_ABSENCE_DEBOUNCE_S now confirms a departure fast, which
    # means a choppy return-then-drop-out-again mid-motion (the same real
    # flicker the old, longer debounce alone used to have to fully absorb)
    # could otherwise trigger a second "come back" nudge a moment after
    # the first one. The cooldown blocks that even though _was_present
    # correctly re-armed on the return.
    engine = ReminderEngine()
    engine.add(kind="presence", message="come back")
    lamp = _FakeLamp()

    engine.tick(now=0.0, face_found=True, lamp=lamp)
    engine.tick(now=1.0, face_found=False, lamp=lamp)
    engine.tick(now=1.0 + _DEBOUNCE + 0.1, face_found=False, lamp=lamp)  # leaves -- fires
    assert len(lamp.sounds) == 1

    engine.tick(now=2.0, face_found=True, lamp=lamp)                     # a flickered return...
    engine.tick(now=2.1, face_found=False, lamp=lamp)
    engine.tick(now=2.1 + _DEBOUNCE + 0.1, face_found=False, lamp=lamp)  # ...and gone again, still within 5s

    assert len(lamp.sounds) == 1  # cooldown suppressed the second nudge


def test_presence_reminder_fires_again_once_the_cooldown_has_actually_elapsed():
    engine = ReminderEngine()
    engine.add(kind="presence", message="come back")
    lamp = _FakeLamp()

    engine.tick(now=0.0, face_found=True, lamp=lamp)
    engine.tick(now=1.0, face_found=False, lamp=lamp)
    engine.tick(now=1.0 + _DEBOUNCE + 0.1, face_found=False, lamp=lamp)  # leaves -- fires
    fire_t = 1.0 + _DEBOUNCE + 0.1

    # Returns and leaves again well *past* PRESENCE_FIRE_COOLDOWN_S after
    # the first fire -- a genuinely new departure, not flicker, so it must
    # fire again rather than being permanently silenced.
    return_t = fire_t + reminders_mod.PRESENCE_FIRE_COOLDOWN_S + 1.0
    engine.tick(now=return_t, face_found=True, lamp=lamp)
    engine.tick(now=return_t + 1.0, face_found=False, lamp=lamp)
    engine.tick(now=return_t + 1.0 + _DEBOUNCE + 0.1, face_found=False, lamp=lamp)

    assert len(lamp.sounds) == 2


def test_firing_speaks_the_reminder_message(_isolate):
    spoken = _isolate
    engine = ReminderEngine()
    engine.add(kind="presence", message="hey, come back and sit down")
    lamp = _FakeLamp()

    engine.tick(now=0.0, face_found=True, lamp=lamp)
    engine.tick(now=1.0, face_found=False, lamp=lamp)
    engine.tick(now=1.0 + _DEBOUNCE + 0.1, face_found=False, lamp=lamp)

    assert spoken == ["hey, come back and sit down"]


def test_reminder_with_a_duration_auto_deactivates_after_it_elapses():
    # "only check for the next 20 seconds" -- without an expiration this
    # ran forever, which is the actual bug this covers.
    engine = ReminderEngine()
    r = engine.add(kind="presence", message="come back", duration_s=20.0)
    lamp = _FakeLamp()

    assert engine.active_count() == 1
    engine.tick(now=r.created_at + 21.0, face_found=True, lamp=lamp)

    assert engine.active_count() == 0
    assert not r.active


def test_reminder_with_a_duration_still_fires_normally_before_it_expires():
    engine = ReminderEngine()
    r = engine.add(kind="presence", message="come back", duration_s=100.0)
    lamp = _FakeLamp()

    engine.tick(now=r.created_at, face_found=True, lamp=lamp)
    engine.tick(now=r.created_at + 5.0, face_found=False, lamp=lamp)
    engine.tick(now=r.created_at + 5.0 + reminders_mod.PRESENCE_ABSENCE_DEBOUNCE_S + 0.1,
                face_found=False, lamp=lamp)

    assert lamp.sounds == ["attention_seek"]


def test_reminder_with_no_duration_never_expires_on_its_own():
    engine = ReminderEngine()
    r = engine.add(kind="recurring", message="stand up", interval_s=1.0)

    engine.tick(now=r.created_at + 1_000_000.0, face_found=True, lamp=_FakeLamp())

    assert r.active


def test_expired_reminder_gets_saved_even_if_nothing_fired():
    engine = ReminderEngine()
    r = engine.add(kind="presence", message="come back", duration_s=10.0)
    engine.save = lambda: saved.append(True)
    saved = []

    engine.tick(now=r.created_at + 11.0, face_found=True, lamp=_FakeLamp())

    assert saved == [True]


def test_inactive_reminders_never_fire():
    engine = ReminderEngine()
    r = engine.add(kind="recurring", message="stand up", interval_s=1.0)
    r.active = False
    lamp = _FakeLamp()

    engine.tick(now=r.created_at + 100.0, face_found=True, lamp=lamp)

    assert lamp.sounds == []


def test_cancel_all_deactivates_every_active_reminder_and_returns_the_count():
    engine = ReminderEngine()
    engine.add(kind="recurring", message="a", interval_s=1.0)
    engine.add(kind="presence", message="b")

    cancelled = engine.cancel_all()

    assert cancelled == 2
    assert engine.active_count() == 0


def test_cancel_all_can_target_just_one_kind():
    engine = ReminderEngine()
    engine.add(kind="recurring", message="a", interval_s=1.0)
    engine.add(kind="presence", message="b")

    cancelled = engine.cancel_all(kind="recurring")

    assert cancelled == 1
    assert engine.active_count() == 1
    assert engine.reminders[1].kind == "presence"
    assert engine.reminders[1].active


def test_cancelled_reminder_never_fires_again():
    engine = ReminderEngine()
    engine.add(kind="recurring", message="stand up", interval_s=1.0)
    engine.cancel_all()
    lamp = _FakeLamp()

    engine.tick(now=1000.0, face_found=True, lamp=lamp)

    assert lamp.sounds == []


def test_load_with_no_file_returns_an_empty_engine():
    engine = ReminderEngine.load()
    assert engine.reminders == []
    assert engine.active_count() == 0


def test_save_load_roundtrip_preserves_reminder_state():
    engine = ReminderEngine()
    engine.add(kind="recurring", message="stand up", interval_s=1800.0)
    engine.add(kind="presence", message="come back")
    engine.save()

    loaded = ReminderEngine.load()

    assert len(loaded.reminders) == 2
    assert {r.kind for r in loaded.reminders} == {"recurring", "presence"}
    assert loaded._next_id == engine._next_id


def test_new_ids_continue_from_the_loaded_counter_not_restart_at_one():
    engine = ReminderEngine()
    engine.add(kind="presence", message="a")
    engine.add(kind="presence", message="b")
    engine.save()

    loaded = ReminderEngine.load()
    r3 = loaded.add(kind="presence", message="c")

    assert r3.id == 3


def test_active_summaries_only_lists_active_reminders():
    engine = ReminderEngine()
    engine.add(kind="recurring", message="stand up", interval_s=1800.0)
    r2 = engine.add(kind="presence", message="come back")
    r2.active = False

    summaries = engine.active_summaries()

    assert len(summaries) == 1
    assert "stand up" in summaries[0]


def test_object_check_does_not_confirm_placement_before_the_object_is_seen():
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check your water", object_class="bottle", due_in_s=3600.0)
    lamp = _FakeLamp()
    vm = _FakeVisionMemory(detections=[])

    engine.tick(now=r.created_at + 5.0, face_found=True, lamp=lamp, vision_memory=vm)

    assert not r.placement_confirmed
    assert r.tracked_bearing is None


def test_object_check_starts_the_settle_clock_on_first_sighting():
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check your water", object_class="bottle", due_in_s=3600.0)
    lamp = _FakeLamp()
    vm = _FakeVisionMemory(detections=[_FakeDetection("bottle", 15.0)])

    engine.tick(now=r.created_at, face_found=True, lamp=lamp, vision_memory=vm)

    assert r.tracked_bearing == 15.0
    assert not r.placement_confirmed  # seen, but not yet held still long enough


def test_object_check_confirms_placement_once_it_holds_still_long_enough():
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check your water", object_class="bottle", due_in_s=3600.0)
    lamp = _FakeLamp()
    vm = _FakeVisionMemory(detections=[_FakeDetection("bottle", 15.0)])

    engine.tick(now=r.created_at, face_found=True, lamp=lamp, vision_memory=vm)
    engine.tick(now=r.created_at + reminders_mod.config.REMINDER_PLACEMENT_SETTLE_S + 0.1,
                face_found=True, lamp=lamp, vision_memory=vm)

    assert r.placement_confirmed
    assert r.tracked_bearing == 15.0
    assert lamp.sounds == []  # confirming placement isn't firing -- just bookkeeping


def test_object_check_does_not_confirm_placement_just_under_the_settle_threshold():
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check your water", object_class="bottle", due_in_s=3600.0)
    lamp = _FakeLamp()
    vm = _FakeVisionMemory(detections=[_FakeDetection("bottle", 15.0)])

    engine.tick(now=r.created_at, face_found=True, lamp=lamp, vision_memory=vm)
    engine.tick(now=r.created_at + reminders_mod.config.REMINDER_PLACEMENT_SETTLE_S - 0.1,
                face_found=True, lamp=lamp, vision_memory=vm)

    assert not r.placement_confirmed


def test_object_check_a_real_move_resets_the_settle_clock():
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check your water", object_class="bottle", due_in_s=3600.0)
    lamp = _FakeLamp()

    engine.tick(now=r.created_at, face_found=True, lamp=lamp,
                vision_memory=_FakeVisionMemory([_FakeDetection("bottle", 15.0)]))
    # Nearly settled, then it moves to a clearly different spot -- should
    # NOT confirm at the old timing, the clock restarts from the move.
    almost_settled = r.created_at + reminders_mod.config.REMINDER_PLACEMENT_SETTLE_S - 0.1
    engine.tick(now=almost_settled, face_found=True, lamp=lamp,
                vision_memory=_FakeVisionMemory([_FakeDetection("bottle", 60.0)]))
    engine.tick(now=almost_settled + 0.2, face_found=True, lamp=lamp,
                vision_memory=_FakeVisionMemory([_FakeDetection("bottle", 60.0)]))

    assert not r.placement_confirmed
    assert r.tracked_bearing == 60.0


def test_object_check_with_a_check_question_confirms_placement_on_first_sighting():
    # The actual reported bug: an actively-handled object (a phone) never
    # holds still long enough to pass REMINDER_PLACEMENT_SETTLE_S, so
    # disturbance-watching never started. A check_question reminder must
    # confirm on the very first sighting -- no settle wait at all.
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="don't touch your phone", object_class="cell phone",
                    due_in_s=300.0, check_question="is the phone still untouched, sitting where it was left")
    lamp = _FakeLamp()
    vm = _FakeVisionMemory(detections=[_FakeDetection("cell phone", 15.0)])

    engine.tick(now=r.created_at, face_found=True, lamp=lamp, vision_memory=vm)

    assert r.placement_confirmed
    assert r.tracked_bearing == 15.0


def test_object_check_with_a_check_question_confirms_despite_bearing_jitter_between_sightings():
    # Simulates the real failure mode: the object is seen, then seen again
    # at a meaningfully different bearing (natural jitter from being held,
    # not necessarily a real move) before it's ever detected twice in a
    # row at the same spot. The old settle-based logic would keep
    # resetting its clock forever in this pattern and never confirm.
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="don't touch your phone", object_class="cell phone",
                    due_in_s=300.0, check_question="is the phone still untouched, sitting where it was left")
    lamp = _FakeLamp()

    engine.tick(now=r.created_at, face_found=True, lamp=lamp,
                vision_memory=_FakeVisionMemory([_FakeDetection("cell phone", 15.0)]))
    assert r.placement_confirmed  # already true after just the first tick

    # A later, differently-angled sighting must not be treated as "not
    # settled yet" -- there's no settle concept left for this case.
    engine.tick(now=r.created_at + 0.1, face_found=True, lamp=lamp,
                vision_memory=_FakeVisionMemory([_FakeDetection("cell phone", 40.0)]))
    assert r.placement_confirmed


def test_object_check_fires_by_pointing_at_the_bearing_once_the_deadline_hits():
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="did you finish your water?",
                    object_class="bottle", due_in_s=3600.0)
    r.tracked_bearing = 22.0
    r.placement_confirmed = True
    lamp = _FakeLamp()
    vm = _FakeVisionMemory()

    engine.tick(now=r.created_at + 3600.0 + 1.0, face_found=True, lamp=lamp, vision_memory=vm)

    assert lamp.sounds == ["attention_seek"]
    assert len(lamp.pose_calls) == 1
    assert vm.immediate_scans_requested == 1  # best-effort rescan nudge


def test_object_check_with_a_check_question_defers_instead_of_firing_directly():
    # reminders.py has no LLM/API-key access (see module docstring) --
    # it can't itself answer "is this bottle full or empty," so it must
    # NOT point/speak/deactivate here, just flip due_for_check and leave
    # the reminder alone for chat.py's own poller to claim.
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="did you finish your water?",
                    object_class="bottle", due_in_s=3600.0, check_question="is this bottle full or empty")
    r.tracked_bearing = 22.0
    r.placement_confirmed = True
    lamp = _FakeLamp()

    engine.tick(now=r.created_at + 3600.0 + 1.0, face_found=True, lamp=lamp, vision_memory=_FakeVisionMemory())

    assert r.due_for_check
    assert r.active  # not deactivated -- still pending until chat.py resolves it
    assert lamp.sounds == []
    assert lamp.pose_calls == []


def test_object_check_with_a_check_question_defers_at_the_deadline_even_unconfirmed():
    # The actual bug report: "make sure I'm still reading my book in 10
    # seconds" -- a book in active use rarely holds still long enough to
    # confirm placement, so placement_confirmed may still be False when
    # due_at arrives. due_for_check must flip anyway (using whatever
    # tracked_bearing is known, possibly still None) rather than waiting
    # on a confirmation that might never come.
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="are you still reading?",
                    object_class="book", due_in_s=10.0, check_question="is the person reading")
    lamp = _FakeLamp()

    engine.tick(now=r.created_at + 10.5, face_found=True, lamp=lamp, vision_memory=_FakeVisionMemory())

    assert r.due_for_check
    assert not r.placement_confirmed
    assert r.active  # still pending until chat.py resolves it


def test_object_check_with_a_check_question_only_flips_due_for_check_once():
    # chat.py's poller clears due_for_check the instant it claims a
    # reminder (see _resolve_object_check_judgment), *before* it finishes
    # resolving it (still active while judge_view()/voice.speak() run).
    # A later tick(), still seeing now >= due_at, must not re-arm
    # due_for_check while that resolution is still in flight.
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check", object_class="bottle", due_in_s=1.0,
                    check_question="is this bottle full or empty")
    r.tracked_bearing = 10.0
    r.placement_confirmed = True
    lamp = _FakeLamp()

    engine.tick(now=r.created_at + 2.0, face_found=True, lamp=lamp, vision_memory=_FakeVisionMemory())
    assert r.due_for_check  # sanity: the first tick did cross the deadline
    r.due_for_check = False  # simulate chat.py's poller having claimed it (still r.active -- in progress)
    engine.tick(now=r.created_at + 3.0, face_found=True, lamp=lamp, vision_memory=_FakeVisionMemory())

    # Must not flip back to True just because now is still >= due_at --
    # only the very first crossing should ever set it.
    assert not r.due_for_check


def test_object_check_with_a_check_question_dispatches_early_on_sustained_move():
    # "make sure I don't go on my phone for five minutes" -- picking it up
    # at second 10 should trigger the check_question then, not stay silent
    # until due_at five minutes later.
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check", object_class="cell phone", due_in_s=300.0,
                    check_question="is the phone still untouched, sitting where it was left")
    r.tracked_bearing = 20.0
    r.placement_confirmed = True
    lamp = _FakeLamp()

    moved = _FakeVisionMemory([_FakeDetection("cell phone", 30.0)])  # 10deg away, past WATCH_REAIM_DEG
    engine.tick(now=r.created_at + 10.0, face_found=True, lamp=lamp, vision_memory=moved)
    assert not r.due_for_check  # single tick -- still inside the debounce

    engine.tick(now=r.created_at + 10.0 + config.WATCH_LOST_GRACE_S, face_found=True, lamp=lamp, vision_memory=moved)
    assert r.due_for_check
    assert r.active  # not deactivated -- chat.py's poller resolves and deactivates it
    assert r.due_at is not None and r.due_at > r.created_at + 10.0  # nowhere near due_at yet


def test_object_check_with_a_check_question_dispatches_early_when_the_object_goes_missing():
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check", object_class="cell phone", due_in_s=300.0,
                    check_question="is the phone still untouched, sitting where it was left")
    r.tracked_bearing = 20.0
    r.placement_confirmed = True
    lamp = _FakeLamp()
    gone = _FakeVisionMemory([])  # picked up and out of frame entirely

    engine.tick(now=r.created_at + 10.0, face_found=True, lamp=lamp, vision_memory=gone)
    assert not r.due_for_check  # single tick -- still inside the debounce
    engine.tick(now=r.created_at + 10.0 + config.WATCH_LOST_GRACE_S, face_found=True, lamp=lamp, vision_memory=gone)
    assert r.due_for_check


def test_object_check_disturbance_debounce_resets_if_the_object_is_seen_back_in_place():
    # A brief wobble/off-angle miss must not fire this -- only a move
    # that's still there config.WATCH_LOST_GRACE_S later counts.
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check", object_class="cell phone", due_in_s=300.0,
                    check_question="is the phone still untouched, sitting where it was left")
    r.tracked_bearing = 20.0
    r.placement_confirmed = True
    lamp = _FakeLamp()

    moved = _FakeVisionMemory([_FakeDetection("cell phone", 30.0)])
    engine.tick(now=r.created_at + 10.0, face_found=True, lamp=lamp, vision_memory=moved)
    back = _FakeVisionMemory([_FakeDetection("cell phone", 20.0)])  # back at the confirmed spot
    engine.tick(now=r.created_at + 11.0, face_found=True, lamp=lamp, vision_memory=back)
    engine.tick(now=r.created_at + 10.0 + config.WATCH_LOST_GRACE_S, face_found=True, lamp=lamp, vision_memory=moved)

    assert not r.due_for_check  # the earlier move doesn't carry over -- the clock restarted


def test_object_check_without_a_check_question_never_dispatches_early_on_disturbance():
    # No check_question means no judgment to make early -- "check on my
    # keys in an hour" doesn't imply "and yell if I touch them earlier."
    # These just wait for due_at like before, even if the object moves.
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check", object_class="keys", due_in_s=300.0)
    r.tracked_bearing = 20.0
    r.placement_confirmed = True
    lamp = _FakeLamp()
    moved = _FakeVisionMemory([_FakeDetection("keys", 30.0)])

    engine.tick(now=r.created_at + 10.0 + config.WATCH_LOST_GRACE_S, face_found=True, lamp=lamp, vision_memory=moved)

    assert not r.due_for_check
    assert r.active
    assert lamp.sounds == []


def test_object_check_deactivates_itself_after_firing_one_shot():
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check", object_class="bottle", due_in_s=10.0)
    r.tracked_bearing = 0.0
    r.placement_confirmed = True
    lamp = _FakeLamp()

    engine.tick(now=r.created_at + 11.0, face_found=True, lamp=lamp, vision_memory=_FakeVisionMemory())

    assert not r.active
    # Ticking again well past the deadline must not fire it a second time.
    engine.tick(now=r.created_at + 100.0, face_found=True, lamp=lamp, vision_memory=_FakeVisionMemory())
    assert lamp.sounds == ["attention_seek"]


def test_object_check_still_fires_at_the_deadline_even_if_placement_never_confirmed():
    # An object that's actively handled (picked up, turned, moved) rather
    # than set down and left alone may never hold still long enough to
    # confirm placement. The deadline must not be held hostage to that --
    # it fires on schedule regardless, falling back to a wiggle instead of
    # pointing at a bearing it never learned (bearing=None).
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check", object_class="bottle", due_in_s=1.0)
    lamp = _FakeLamp()

    engine.tick(now=r.created_at + 100.0, face_found=True, lamp=lamp, vision_memory=_FakeVisionMemory())

    assert lamp.sounds == ["attention_seek"]
    assert not r.placement_confirmed
    assert not r.active  # no check_question -- fires directly and deactivates


def test_object_check_with_alert_on_detection_fires_immediately_on_first_sighting():
    # The actual reported bug: "don't let me touch my phone" needs to fire
    # the instant YOLO catches the phone, no placement/settle/debounce at
    # all -- detection itself is the violation.
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="get off your phone", object_class="cell phone",
                    due_in_s=300.0, alert_on_detection=True)
    lamp = _FakeLamp()
    vm = _FakeVisionMemory([_FakeDetection("cell phone", 12.0)])

    engine.tick(now=r.created_at + 1.0, face_found=True, lamp=lamp, vision_memory=vm)

    assert lamp.sounds == ["attention_seek"]
    assert len(lamp.pose_calls) == 1  # pointed at the sighting, not a generic wiggle
    assert r.tracked_bearing == 12.0
    assert not r.active  # one-shot -- caught once, reported, done


def test_object_check_with_alert_on_detection_does_nothing_while_unseen():
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="get off your phone", object_class="cell phone",
                    due_in_s=300.0, alert_on_detection=True)
    lamp = _FakeLamp()

    engine.tick(now=r.created_at + 10.0, face_found=True, lamp=lamp, vision_memory=_FakeVisionMemory([]))

    assert lamp.sounds == []
    assert r.active


def test_object_check_with_alert_on_detection_deactivates_quietly_if_never_seen():
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="get off your phone", object_class="cell phone",
                    due_in_s=1.0, alert_on_detection=True)
    lamp = _FakeLamp()

    engine.tick(now=r.created_at + 2.0, face_found=True, lamp=lamp, vision_memory=_FakeVisionMemory([]))

    assert lamp.sounds == []  # never caught -- nothing to report
    assert not r.active


def test_object_check_with_alert_on_detection_is_included_in_watched_classes():
    # Needs fast scan cadence even more than a disturbance watch does --
    # there's no debounce to protect, so scan cadence is the entire delay
    # between a pickup and the lamp noticing it (see watched_classes's docstring).
    engine = ReminderEngine()
    engine.add(kind="object_check", message="get off your phone", object_class="cell phone",
               due_in_s=300.0, alert_on_detection=True)

    assert engine.watched_classes() == ["cell phone"]


def test_object_check_without_vision_memory_never_confirms_placement():
    # vision_memory is None in a session with no camera loop -- must
    # degrade to a harmless no-op, not crash.
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check", object_class="bottle", due_in_s=3600.0)
    lamp = _FakeLamp()

    engine.tick(now=r.created_at + 5.0, face_found=True, lamp=lamp, vision_memory=None)

    assert not r.placement_confirmed
    assert lamp.sounds == []


def test_pending_placement_classes_only_includes_unconfirmed_object_checks():
    engine = ReminderEngine()
    r1 = engine.add(kind="object_check", message="a", object_class="bottle", due_in_s=3600.0)
    r2 = engine.add(kind="object_check", message="b", object_class="cell phone", due_in_s=3600.0)
    engine.add(kind="recurring", message="c", interval_s=60.0)  # no object_class -- must not appear

    assert set(engine.pending_placement_classes()) == {"bottle", "cell phone"}

    r2.placement_confirmed = True
    assert engine.pending_placement_classes() == ["bottle"]

    r1.active = False
    assert engine.pending_placement_classes() == []


def test_watched_classes_only_includes_confirmed_object_checks_with_a_check_question():
    engine = ReminderEngine()
    unconfirmed = engine.add(kind="object_check", message="a", object_class="bottle", due_in_s=3600.0,
                              check_question="is this bottle full or empty")
    no_question = engine.add(kind="object_check", message="b", object_class="keys", due_in_s=3600.0)
    watched = engine.add(kind="object_check", message="c", object_class="cell phone", due_in_s=3600.0,
                          check_question="is the phone still untouched, sitting where it was left")
    engine.add(kind="recurring", message="d", interval_s=60.0)  # no object_class -- must not appear

    # Not yet confirmed (still awaiting placement) -- not disturbance-
    # watched yet, so must not force fast scanning either.
    assert engine.watched_classes() == []

    unconfirmed.placement_confirmed = True
    no_question.placement_confirmed = True
    watched.placement_confirmed = True
    # unconfirmed now qualifies too (confirmed + has a check_question);
    # no_question never will, since there's no judgment to watch for.
    assert set(engine.watched_classes()) == {"bottle", "cell phone"}

    watched._check_dispatched = True  # already claimed -- no longer worth forcing fast scans for
    assert engine.watched_classes() == ["bottle"]


def test_active_summaries_describes_object_check_status():
    engine = ReminderEngine()
    engine.add(kind="object_check", message="check your water", object_class="bottle", due_in_s=3600.0)

    summary = engine.active_summaries()[0]

    assert "bottle" in summary
    assert "watching for placement" in summary


def test_active_summaries_shows_a_countdown_once_placement_is_confirmed():
    # The debugging aid live testing asked for -- "is this actually near
    # its deadline, or just slow" needs to be visible without reading
    # timestamps out of the console.
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check your water", object_class="bottle", due_in_s=120.0)
    r.placement_confirmed = True

    summary = engine.active_summaries(now=r.created_at + 60.0)[0]

    assert "due in 60s" in summary


def test_active_summaries_shows_overdue_once_the_deadline_has_passed():
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check your water", object_class="bottle", due_in_s=10.0)
    r.placement_confirmed = True

    summary = engine.active_summaries(now=r.created_at + 15.0)[0]

    assert "overdue by 5s" in summary


def test_fmt_duration_switches_to_minutes_past_two_minutes():
    assert reminders_mod._fmt_duration(45.0) == "45s"
    assert reminders_mod._fmt_duration(125.0) == "2min"


def test_tick_prints_a_timestamped_deadline_reached_line_for_object_check(capsys):
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check", object_class="bottle", due_in_s=1.0,
                    check_question="is this bottle full or empty")
    r.tracked_bearing = 0.0
    r.placement_confirmed = True

    engine.tick(now=r.created_at + 5.0, face_found=True, lamp=_FakeLamp(), vision_memory=_FakeVisionMemory())

    out = capsys.readouterr().out
    assert "deadline reached" in out
    assert "4.0s after due_at" in out


if __name__ == "__main__":
    print("Run with pytest -- this file relies on monkeypatch fixtures.")
