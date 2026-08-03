"""Unit tests for behavior/reminders.py. Run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

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
    engine.tick(now=r.created_at + reminders_mod.config.WATCH_STATIONARY_TIMEOUT_S + 0.1,
                face_found=True, lamp=lamp, vision_memory=vm)

    assert r.placement_confirmed
    assert r.tracked_bearing == 15.0
    assert lamp.sounds == []  # confirming placement isn't firing -- just bookkeeping


def test_object_check_a_real_move_resets_the_settle_clock():
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check your water", object_class="bottle", due_in_s=3600.0)
    lamp = _FakeLamp()

    engine.tick(now=r.created_at, face_found=True, lamp=lamp,
                vision_memory=_FakeVisionMemory([_FakeDetection("bottle", 15.0)]))
    # Nearly settled, then it moves to a clearly different spot -- should
    # NOT confirm at the old timing, the clock restarts from the move.
    almost_settled = r.created_at + reminders_mod.config.WATCH_STATIONARY_TIMEOUT_S - 0.1
    engine.tick(now=almost_settled, face_found=True, lamp=lamp,
                vision_memory=_FakeVisionMemory([_FakeDetection("bottle", 60.0)]))
    engine.tick(now=almost_settled + 0.2, face_found=True, lamp=lamp,
                vision_memory=_FakeVisionMemory([_FakeDetection("bottle", 60.0)]))

    assert not r.placement_confirmed
    assert r.tracked_bearing == 60.0


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


def test_object_check_does_not_fire_before_placement_is_confirmed_even_past_the_deadline():
    # An object never seen/settled has nothing to check on -- must not
    # fire (and definitely must not point at bearing=None).
    engine = ReminderEngine()
    r = engine.add(kind="object_check", message="check", object_class="bottle", due_in_s=1.0)
    lamp = _FakeLamp()

    engine.tick(now=r.created_at + 100.0, face_found=True, lamp=lamp, vision_memory=_FakeVisionMemory())

    assert lamp.sounds == []
    assert r.active  # still waiting -- not expired (no duration set) and never got to fire


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


def test_active_summaries_describes_object_check_status():
    engine = ReminderEngine()
    engine.add(kind="object_check", message="check your water", object_class="bottle", due_in_s=3600.0)

    summary = engine.active_summaries()[0]

    assert "bottle" in summary
    assert "watching for placement" in summary


if __name__ == "__main__":
    print("Run with pytest -- this file relies on monkeypatch fixtures.")
