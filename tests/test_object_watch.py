"""Unit tests for behavior/object_watch.py. Run with: python -m pytest tests/ -v"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from behavior.object_watch import WATCH_NUDGE, ObjectWatcher
from behavior.state_machine import DEFAULT_MOOD, ENGAGED_NUDGE, State
from lamp import color
from perception.vision_memory import Detection

WATCH_COLOR = color.from_dials(*color.nudge(*DEFAULT_MOOD, *WATCH_NUDGE))
IDLE_COLOR = color.from_dials(*DEFAULT_MOOD)
ENGAGED_COLOR = color.from_dials(*color.nudge(*DEFAULT_MOOD, *ENGAGED_NUDGE))


class _FakeLamp:
    def __init__(self):
        self.calls = []
        self.breathing = True
        self.mood = DEFAULT_MOOD

    def set_target_pose(self, angles, duration=0.6, anticipation=True, overshoot=True):
        self.calls.append(("pose", tuple(angles)))

    def set_light(self, rgb, transition_s=0.4):
        self.calls.append(("light", rgb))

    def play_sound(self, event):
        self.calls.append(("sound", event))

    def set_breathing(self, enabled):
        self.breathing = enabled

    def set_mood(self, warmth, brightness):
        self.mood = (warmth, brightness)

    def get_mood(self):
        return self.mood

    def update(self, dt):
        pass


def _phone(bearing_deg):
    return [Detection(object_class="cell phone", confidence=0.9, bearing_deg=bearing_deg, bbox_xyxy=(0, 0, 1, 1))]


def test_starts_watching_on_first_sighting():
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(20.0), State.IDLE, dt=0.05)
    assert watcher.active
    assert any(c[0] == "light" and c[1] == WATCH_COLOR for c in lamp.calls)
    assert any(c[0] == "pose" for c in lamp.calls)


def test_ignores_tracked_object_during_attention_seeking():
    """ATTENTION_SEEKING is a short, precisely-timed animation
    (BehaviorFSM._tick_attention_seek) -- the one state where letting the
    watcher grab the pose would visibly break a deliberate gesture."""
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(20.0), State.ATTENTION_SEEKING, dt=0.05)
    assert not watcher.active
    assert lamp.calls == []


def test_tracks_the_object_while_engaged():
    """ENGAGED does NOT block tracking -- unlike ATTENTION_SEEKING,
    BehaviorFSM._enter_engaged only sets the pose once, on the
    transition, and never re-asserts it. There's nothing to fight over,
    so the watcher stays free to track a phone while someone is engaged
    with the lamp -- that's the whole point of "still follow my phone if
    I'm looking at it"."""
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(20.0), State.ENGAGED, dt=0.05)
    assert watcher.active
    assert any(c[0] == "light" and c[1] == WATCH_COLOR for c in lamp.calls)


def test_continues_tracking_across_a_brief_run_of_missed_detections():
    """A single bad-angle miss (or a few) used to instantly flip the
    gesture off and restart it once detection resumed, which is exactly
    the flicker the phone-tracking complaint was about. As long as the
    gap stays under WATCH_LOST_GRACE_S, tracking should just continue
    without ever settling back or replaying the acquire gesture."""
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(20.0), State.IDLE, dt=0.05)
    assert watcher.active
    lamp.calls.clear()

    # a run of missed scans shorter than the grace period
    t = 0.0
    while t < config.WATCH_LOST_GRACE_S - 0.3:
        watcher.update([], State.IDLE, dt=0.05)
        t += 0.05
    assert watcher.active, "should still be tracking through a brief detection gap"

    # detected again -- should re-aim, not replay the acquire sound/flourish
    watcher.update(_phone(20.0), State.IDLE, dt=0.05)
    assert watcher.active
    assert not any(c[0] == "sound" for c in lamp.calls), "brief gap shouldn't replay the acquire cue"


def test_settles_back_after_being_lost_for_longer_than_the_grace_period():
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(20.0), State.IDLE, dt=0.05)
    lamp.calls.clear()

    t = 0.0
    while t < config.WATCH_LOST_GRACE_S + 0.5 and watcher.active:
        watcher.update([], State.IDLE, dt=0.05)
        t += 0.05

    assert not watcher.active
    assert any(c[0] == "light" and c[1] == IDLE_COLOR for c in lamp.calls)


def test_reaims_continuously_while_the_object_moves():
    """This is the "still follow my phone if I'm looking at it" case --
    holding/moving the phone around should keep updating where the lamp
    points, not just glance once and stop."""
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(0.0), State.IDLE, dt=0.05)
    lamp.calls.clear()

    bearing = 0.0
    reaim_count = 0
    for _ in range(10):
        bearing += config.WATCH_REAIM_DEG + 1.0  # comfortably over the re-aim threshold each step
        before = len(lamp.calls)
        watcher.update(_phone(bearing), State.IDLE, dt=0.05)
        if len(lamp.calls) > before:
            reaim_count += 1
    assert reaim_count >= 8, f"expected continuous re-aiming as the object moves, got {reaim_count}/10"


def test_does_not_reaim_on_sub_threshold_jitter():
    """Frame-to-frame bbox noise shouldn't make the arm twitch while the
    object is basically just sitting there."""
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(20.0), State.IDLE, dt=0.05)
    lamp.calls.clear()

    jitter = config.WATCH_REAIM_DEG - 1.0
    watcher.update(_phone(20.0 + jitter), State.IDLE, dt=0.05)
    assert lamp.calls == []


def test_sub_threshold_steps_still_accumulate_into_a_reaim_once_they_cross_the_threshold():
    """_current_bearing only updates on an actual re-aim, so small
    steps compare against a fixed reference and accumulate -- a slowly
    drifting object (not just single-frame noise) must still eventually
    trigger a re-aim once the cumulative drift crosses WATCH_REAIM_DEG,
    rather than never re-aiming because each individual step looked like
    jitter."""
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(20.0), State.IDLE, dt=0.05)
    lamp.calls.clear()

    reaim_count = 0
    for i in range(1, 11):
        watcher.update(_phone(20.0 + i * 1.0), State.IDLE, dt=0.05)  # 1deg/step, threshold is 4deg
        if any(c[0] == "pose" for c in lamp.calls):
            reaim_count += 1
        lamp.calls.clear()
    # crosses the 4deg threshold at step 5 (5deg cumulative from the
    # original 20.0 reference) and again at step 10 (5deg past the new
    # 25.0 reference set by that first re-aim) -- exactly 2 re-aims,
    # not 0 and not 10
    assert reaim_count == 2, f"expected drift to cross the threshold twice, got {reaim_count}/10"


def test_fast_scan_drops_once_a_visible_object_stops_moving():
    """A stationary-but-still-visible object shouldn't pin the loop at
    the fast scan interval forever -- fast-scan is for catching motion in
    progress, not for staring at something that isn't moving."""
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(20.0), State.IDLE, dt=0.05)
    assert watcher.should_scan_fast()

    t = 0.0
    while t < config.WATCH_FAST_SCAN_SETTLE_S + 0.5:
        watcher.update(_phone(20.0), State.IDLE, dt=0.05)  # same bearing every tick -- not moving
        t += 0.05

    assert watcher.active, "should still be tracking -- it's still visible"
    assert not watcher.should_scan_fast(), "should have dropped out of fast-scan once it stopped moving"


def test_does_not_stomp_the_fsm_during_attention_seeking():
    """A phone gets placed (tracking starts), and *before* the lost-grace
    window runs out, the FSM enters ATTENTION_SEEKING. The watcher must
    not touch the lamp at all while that animation owns it, and must not
    stomp its pose/light once the grace window elapses either."""
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)

    watcher.update(_phone(20.0), State.IDLE, dt=0.05)
    assert watcher.active
    lamp.calls.clear()

    # ATTENTION_SEEKING starts -- detections keep coming (phone's still on
    # the desk) but must be ignored while that animation owns the lamp.
    watcher.update(_phone(20.0), State.ATTENTION_SEEKING, dt=0.05)

    t = 0.05
    while t < config.WATCH_LOST_GRACE_S + 0.5:
        watcher.update(_phone(20.0), State.ATTENTION_SEEKING, dt=0.05)
        t += 0.05

    assert not watcher.active
    assert lamp.calls == []


def test_resumes_tracking_once_attention_seeking_ends():
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(20.0), State.IDLE, dt=0.05)
    assert watcher.active

    watcher.update(_phone(20.0), State.ATTENTION_SEEKING, dt=0.05)  # interrupted
    watcher.update(_phone(20.0), State.IDLE, dt=0.05)                # ...and released again, still detected

    assert watcher.active


def test_returns_to_engaged_look_when_lost_while_still_engaged():
    """Losing the phone while the person is still ENGAGED must not
    settle to NEUTRAL_POSE/IDLE_COLOR -- the FSM never re-applies its own
    "looking at you" pose (see BehaviorFSM._enter_engaged, only called on
    the transition), so doing that would leave the lamp visibly looking
    away from someone who's still right there, with nothing to correct
    it. It should return to the engaged look-at-user pose/color instead."""
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(20.0), State.ENGAGED, dt=0.05, user_bearing_deg=-5.0)
    assert watcher.active
    lamp.calls.clear()

    t = 0.0
    while t < config.WATCH_LOST_GRACE_S + 0.5 and watcher.active:
        watcher.update([], State.ENGAGED, dt=0.05, user_bearing_deg=-5.0)
        t += 0.05

    assert not watcher.active
    assert any(c[0] == "light" and c[1] == ENGAGED_COLOR for c in lamp.calls)
    assert not any(c[0] == "light" and c[1] == IDLE_COLOR for c in lamp.calls), \
        "must not fall back to the idle color while the person is still engaged"
    assert any(c[0] == "pose" for c in lamp.calls)


def test_breathing_suppressed_while_tracking_then_restored_on_settle():
    """The idle breathing wobble stacking on top of frequent short
    re-aims is what read as jitter tracking a moving phone -- breathing
    should be off for as long as tracking is active, and back on once it
    stops (however it stops)."""
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(20.0), State.IDLE, dt=0.05)
    assert lamp.breathing is False

    t = 0.0
    while t < config.WATCH_LOST_GRACE_S + 0.5 and watcher.active:
        watcher.update([], State.IDLE, dt=0.05)
        t += 0.05
    assert not watcher.active
    assert lamp.breathing is True


def test_gives_up_watching_once_a_visible_object_stops_moving():
    """Still visible, but hasn't moved in a while -- continuing to fixate
    on it reads as staring, not attentiveness. Should settle back to a
    normal resting look on its own, without ever losing detection."""
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(20.0), State.IDLE, dt=0.05)
    assert watcher.active
    lamp.calls.clear()

    t = 0.0
    while t < config.WATCH_STATIONARY_TIMEOUT_S + 0.5 and watcher.active:
        watcher.update(_phone(20.0), State.IDLE, dt=0.05)  # same bearing every tick -- never lost, never moves
        t += 0.05

    assert not watcher.active, "should give up watching a stationary-but-still-visible object"
    assert any(c[0] == "light" and c[1] == IDLE_COLOR for c in lamp.calls)


def test_small_bump_after_stationary_give_up_does_not_restart_the_acquire_flourish():
    """A tiny nudge on an otherwise-still phone shouldn't replay the full
    acquire sound/flourish every few seconds -- that reads as twitchy,
    not attentive. Only a real, deliberate move should."""
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(20.0), State.IDLE, dt=0.05)

    t = 0.0
    while t < config.WATCH_STATIONARY_TIMEOUT_S + 0.5 and watcher.active:
        watcher.update(_phone(20.0), State.IDLE, dt=0.05)
        t += 0.05
    assert not watcher.active
    lamp.calls.clear()

    # a small bump, comfortably under the reacquire margin
    small_bump = config.WATCH_REAIM_DEG * config.WATCH_REACQUIRE_MARGIN - 1.0
    watcher.update(_phone(20.0 + small_bump), State.IDLE, dt=0.05)
    assert not watcher.active, "a small bump shouldn't restart tracking"
    assert lamp.calls == []

    # a real, deliberate move past the margin should still re-acquire
    real_move = config.WATCH_REAIM_DEG * config.WATCH_REACQUIRE_MARGIN + 5.0
    watcher.update(_phone(20.0 + real_move), State.IDLE, dt=0.05)
    assert watcher.active, "a real move past the reacquire margin should restart tracking"


def test_reaim_turns_the_head_proportionally_to_the_move():
    """wrist_roll/head_twist used to stay at 0 while tracking -- only the
    arm (base_yaw/shoulder/elbow) ever moved, which read as flat/static
    even as the arm visibly swung. A re-aim should now tilt/twist the
    head a little too, scaled by (and in the direction of) the move."""
    lamp = _FakeLamp()
    watcher = ObjectWatcher(lamp)
    watcher.update(_phone(0.0), State.IDLE, dt=0.05)
    lamp.calls.clear()

    watcher.update(_phone(20.0), State.IDLE, dt=0.05)  # a real rightward move
    pose = next(c[1] for c in lamp.calls if c[0] == "pose")
    assert pose[4] > 0, f"expected wrist_roll to tilt toward the move direction, got {pose}"
    assert pose[5] > 0, f"expected head_twist to turn toward the move direction, got {pose}"


if __name__ == "__main__":
    test_starts_watching_on_first_sighting()
    test_ignores_tracked_object_during_attention_seeking()
    test_tracks_the_object_while_engaged()
    test_continues_tracking_across_a_brief_run_of_missed_detections()
    test_settles_back_after_being_lost_for_longer_than_the_grace_period()
    test_reaims_continuously_while_the_object_moves()
    test_does_not_reaim_on_sub_threshold_jitter()
    test_sub_threshold_steps_still_accumulate_into_a_reaim_once_they_cross_the_threshold()
    test_fast_scan_drops_once_a_visible_object_stops_moving()
    test_does_not_stomp_the_fsm_during_attention_seeking()
    test_resumes_tracking_once_attention_seeking_ends()
    test_returns_to_engaged_look_when_lost_while_still_engaged()
    print("ALL OBJECT WATCH TESTS PASSED")
