"""Self-initiated timed checks -- "make sure I get up every 30 minutes,"
"make sure I don't leave my desk." Created via natural language through
conversation/agent.py's create_reminder tool (see chat.py's handle(),
which reads agent.last_reminder_action the same way it already does for
last_light_action), ticked once per frame from main.py's own loop --
presence detection specifically needs frame-rate timing, not chat.py's
much slower listen/reply cadence -- and persisted to logs/reminders.json
so a restart doesn't silently drop one that's still pending.

Three kinds:
  - "recurring": fires every interval_s, then resets and repeats
    ("every 30 minutes").
  - "presence": fires once each time the user leaves (a face stops being
    detected) while active, then re-arms once they're back -- edge-
    triggered on the "just left" transition, not level-triggered on
    "currently away" (which would fire every single tick you're gone).
  - "object_check": watches for object_class to settle in place (a
    placement, e.g. setting down a water bottle), remembers where, then
    points back at it and announces the deadline once due_at hits. One-
    shot -- deactivates itself once fired, unlike the other two. Placement
    detection here is deliberately independent of behavior/object_watch.py's
    own tracking (which can only actively watch one object at a time and
    might be busy with something else) -- this keeps its own minimal
    bookkeeping off the same detections main.py already has, reusing
    ObjectWatcher's stillness thresholds for consistency, not its state.
    Can't yet judge the object's *contents* (e.g. whether a bottle is full
    or empty) -- that needs a vision-LLM look, a deliberately separate
    later addition.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

import config
from conversation import voice
from lamp import kinematics

REMINDERS_PATH = config.LOGS_DIR / "reminders.json"

KINDS = ("recurring", "presence", "object_check")

# face_found is a raw per-frame signal, not the hysteresis-smoothed
# engaged/disengaged state (see perception/engagement.py) -- standing up
# moves your head through angles that can drop out of detection for a
# frame or two before settling on "actually gone," which fired presence
# reminders 2-3 times in the same second on live testing. Requiring
# face_found to have been continuously false for this long, not just
# once, is the same dwell-time debounce idea the engagement pipeline
# already uses for its own ENGAGE_EXIT_DWELL_FRAMES -- longer here since
# "got up and left" is a bigger, rarer, more deliberate event than "looked
# away," so there's no cost to waiting a bit longer to be sure.
PRESENCE_ABSENCE_DEBOUNCE_S = 1.5

# Additive off get_current_pose(), same "nudge, don't replace" convention
# as every other reactive gesture in this codebase (chat.py's
# _jerk_back/_droop, object_watch.py's play_wave_back) -- a single
# overshoot-eased target reads as a lively wiggle on its own, no
# multi-keyframe sequencer needed.
_WIGGLE_OFFSET = np.array([18.0, 0.0, 0.0, 0.0, 12.0, -18.0])


@dataclass
class Reminder:
    id: int
    kind: str            # "recurring" | "presence" | "object_check"
    message: str         # spoken/printed when it fires
    created_at: float    # time.time() -- wall clock, so it's meaningful across a restart
    interval_s: float | None = None     # recurring only
    last_fired_at: float | None = None  # recurring only; None means "never fired yet"
    expires_at: float | None = None     # wall-clock deadline after which this auto-deactivates,
                                          # e.g. "only check for the next 20 seconds" -- None runs
                                          # indefinitely until explicitly cancelled
    object_class: str | None = None      # object_check only -- normalized COCO class, e.g. "bottle"
    due_at: float | None = None          # object_check only -- when to check on it
    tracked_bearing: float | None = None  # object_check only -- where it settled, once known
    placement_confirmed: bool = False     # object_check only -- has it been seen to settle yet
    active: bool = True
    _was_present: bool = True  # presence only -- edge-detection state, see module docstring
    _absent_since: float | None = None  # presence only -- debounce state, see PRESENCE_ABSENCE_DEBOUNCE_S
    _settled_since: float | None = None  # object_check only -- when tracked_bearing last changed


def _wiggle(lamp) -> None:
    """The physical half of a reminder firing -- a quick, playful
    side-to-side wobble."""
    current = lamp.get_current_pose()
    lamp.set_target_pose(current + _WIGGLE_OFFSET, duration=0.35, anticipation=False, overshoot=True)


def _point_at(lamp, bearing_deg: float) -> None:
    """The physical half of an object_check firing -- points toward the
    remembered placement instead of the generic wiggle, the same
    "point first, so the pan-crop tracking the lamp's aim has a chance to
    include the object" idea as chat.py's _point_toward for recall."""
    pose = kinematics.pose_for_look_at(bearing_deg, -10.0, alertness=0.85)
    lamp.set_target_pose(pose, duration=0.5, anticipation=True, overshoot=True)


def _fire(reminder: Reminder, lamp) -> None:
    print(f"[reminder] {reminder.message}")
    if reminder.kind == "object_check" and reminder.tracked_bearing is not None:
        _point_at(lamp, reminder.tracked_bearing)
    else:
        _wiggle(lamp)
    lamp.play_sound("attention_seek")  # already the "did you notice me?" cue -- no new sound asset needed
    # Backgrounded, not a direct call -- this runs from main.py's render-
    # loop thread, and voice.speak() blocks for the length of the
    # utterance (a real TTS clip, not instant). Blocking the render loop
    # for a couple of seconds every time a reminder fires would show up as
    # a visible stutter; voice.speak()'s own lock (see conversation/
    # voice.py) still serializes this against chat.py's own speak calls,
    # so this is fire-and-forget-safe, not a race.
    threading.Thread(target=voice.speak, args=(reminder.message,), daemon=True).start()


@dataclass
class ReminderEngine:
    reminders: list = field(default_factory=list)
    _next_id: int = 1

    @classmethod
    def load(cls) -> "ReminderEngine":
        if REMINDERS_PATH.exists():
            try:
                data = json.loads(REMINDERS_PATH.read_text())
                reminders = [Reminder(**r) for r in data.get("reminders", [])]
                return cls(reminders=reminders, _next_id=data.get("next_id", 1))
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        return cls()

    def save(self) -> None:
        REMINDERS_PATH.write_text(json.dumps({
            "reminders": [asdict(r) for r in self.reminders],
            "next_id": self._next_id,
        }))

    def add(self, kind: str, message: str, interval_s: float | None = None,
            duration_s: float | None = None, object_class: str | None = None,
            due_in_s: float | None = None) -> Reminder:
        created_at = time.time()
        expires_at = created_at + duration_s if duration_s is not None else None
        due_at = created_at + due_in_s if due_in_s is not None else None
        r = Reminder(id=self._next_id, kind=kind, message=message, created_at=created_at,
                      interval_s=interval_s, expires_at=expires_at, object_class=object_class, due_at=due_at)
        self._next_id += 1
        self.reminders.append(r)
        self.save()
        return r

    def cancel_all(self, kind: str | None = None) -> int:
        """Deactivates every active reminder, or only those of `kind` if
        given. Returns how many were cancelled, so a caller (the
        create_reminder tool) can tell the user something concrete
        ("cancelled 2 reminders") instead of a blind "ok, done." """
        cancelled = 0
        for r in self.reminders:
            if r.active and (kind is None or r.kind == kind):
                r.active = False
                cancelled += 1
        if cancelled:
            self.save()
        return cancelled

    def active_count(self) -> int:
        return sum(1 for r in self.reminders if r.active)

    def pending_placement_classes(self) -> list[str]:
        """Object classes at least one active object_check reminder is
        still waiting to see settle -- main.py folds these into
        ObjectWatcher.tracked_classes so the arm visibly reacts to/follows
        the object while this is happening. Purely a cosmetic "the lamp
        noticed too" bonus layered on top of this module's own
        independent placement bookkeeping (see _update_placement) -- not
        the source of truth for whether placement is confirmed."""
        return list({r.object_class for r in self.reminders
                     if r.active and r.kind == "object_check" and not r.placement_confirmed
                     and r.object_class is not None})

    def active_summaries(self, now: float | None = None) -> list[str]:
        """One short line per active reminder, for main.py's HUD panel --
        not the full message (that can be a whole sentence), just enough
        to recognize which one it is at a glance. now is optional -- only
        needed to show remaining time on a reminder that has an
        expiration; omit it (e.g. right after add(), with nothing to tick
        against yet) and that part is just left off."""
        lines = []
        for r in self.reminders:
            if not r.active:
                continue
            label = r.message if len(r.message) <= 40 else r.message[:37] + "..."
            if r.kind == "recurring":
                line = f"#{r.id} every {r.interval_s / 60:.0f}min: {label}"
            elif r.kind == "object_check":
                status = "watching for placement" if not r.placement_confirmed else "waiting for deadline"
                line = f"#{r.id} {r.object_class} ({status}): {label}"
            else:
                line = f"#{r.id} on leaving desk: {label}"
            if r.expires_at is not None and now is not None:
                remaining = max(0.0, r.expires_at - now)
                line += f" (expires in {remaining:.0f}s)" if remaining < 120 else f" (expires in {remaining / 60:.0f}min)"
            lines.append(line)
        return lines

    def _update_placement(self, r: Reminder, now: float, vision_memory) -> None:
        """object_check only -- has r.object_class's detected position
        stayed put long enough to count as "set down." Deliberately reuses
        ObjectWatcher's own stillness thresholds (config.WATCH_REAIM_DEG,
        config.WATCH_STATIONARY_TIMEOUT_S) for a consistent feel, but not
        its state -- see the module docstring on why. vision_memory is
        None in a session with no camera loop (or --no-memory), in which
        case placement can never be confirmed; this just no-ops rather
        than crashing."""
        if vision_memory is None:
            return
        detections = vision_memory.last_detections or []
        match = next((d for d in detections if d.object_class == r.object_class), None)
        if match is None:
            return  # not currently visible this tick -- nothing new to learn, keep waiting
        if r.tracked_bearing is None or abs(match.bearing_deg - r.tracked_bearing) > config.WATCH_REAIM_DEG:
            r.tracked_bearing = match.bearing_deg  # first sighting, or a real move -- reset the settle clock
            r._settled_since = now
        elif r._settled_since is not None and now - r._settled_since >= config.WATCH_STATIONARY_TIMEOUT_S:
            r.placement_confirmed = True

    def tick(self, now: float, face_found: bool, lamp, vision_memory=None) -> None:
        """now is wall-clock time.time(), not time.perf_counter() -- both
        interval_s comparisons and on-disk persistence need to make sense
        across a process restart, which perf_counter's arbitrary
        per-process origin can't do. vision_memory is only needed for
        object_check reminders (placement tracking + a best-effort rescan
        nudge at the deadline) -- every other kind ignores it."""
        fired_any = False
        expired_any = False
        for r in self.reminders:
            if not r.active:
                continue
            if r.expires_at is not None and now >= r.expires_at:
                r.active = False
                expired_any = True
                continue
            if r.kind == "recurring":
                last = r.last_fired_at if r.last_fired_at is not None else r.created_at
                if r.interval_s is not None and now - last >= r.interval_s:
                    _fire(r, lamp)
                    r.last_fired_at = now
                    fired_any = True
            elif r.kind == "presence":
                if face_found:
                    r._absent_since = None
                    r._was_present = True
                else:
                    if r._absent_since is None:
                        r._absent_since = now
                    elif r._was_present and now - r._absent_since >= PRESENCE_ABSENCE_DEBOUNCE_S:
                        _fire(r, lamp)
                        fired_any = True
                        r._was_present = False
            elif r.kind == "object_check":
                if not r.placement_confirmed:
                    self._update_placement(r, now, vision_memory)
                    if r.placement_confirmed:
                        fired_any = True  # not a fire, but placement_confirmed/tracked_bearing changed -- persist it
                elif r.due_at is not None and now >= r.due_at:
                    if vision_memory is not None:
                        vision_memory.request_immediate_scan()  # best-effort -- doesn't block this fire
                    _fire(r, lamp)
                    r.active = False  # one-shot: deadline reached and reported, nothing left to watch for
                    fired_any = True
        if fired_any or expired_any:
            self.save()
