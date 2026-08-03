"""Self-initiated timed checks -- "make sure I get up every 30 minutes,"
"make sure I don't leave my desk." Created via natural language through
conversation/agent.py's create_reminder tool (see chat.py's handle(),
which reads agent.last_reminder_action the same way it already does for
last_light_action), ticked once per frame from main.py's own loop --
presence detection specifically needs frame-rate timing, not chat.py's
much slower listen/reply cadence -- and persisted to logs/reminders.json
so a restart doesn't silently drop one that's still pending.

Two kinds so far:
  - "recurring": fires every interval_s, then resets and repeats
    ("every 30 minutes").
  - "presence": fires once each time the user leaves (a face stops being
    detected) while active, then re-arms once they're back -- edge-
    triggered on the "just left" transition, not level-triggered on
    "currently away" (which would fire every single tick you're gone).

A third kind -- track where an object gets set down, then check on it
(and its contents, via a vision-LLM look) at a deadline -- is a
deliberately separate later addition; it needs its own placement-tracking
and vision-judgment machinery that doesn't exist yet.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

import config
from conversation import voice

REMINDERS_PATH = config.LOGS_DIR / "reminders.json"

KINDS = ("recurring", "presence")

# Additive off get_current_pose(), same "nudge, don't replace" convention
# as every other reactive gesture in this codebase (chat.py's
# _jerk_back/_droop, object_watch.py's play_wave_back) -- a single
# overshoot-eased target reads as a lively wiggle on its own, no
# multi-keyframe sequencer needed.
_WIGGLE_OFFSET = np.array([18.0, 0.0, 0.0, 0.0, 12.0, -18.0])


@dataclass
class Reminder:
    id: int
    kind: str            # "recurring" | "presence"
    message: str         # spoken/printed when it fires
    created_at: float    # time.time() -- wall clock, so it's meaningful across a restart
    interval_s: float | None = None     # recurring only
    last_fired_at: float | None = None  # recurring only; None means "never fired yet"
    active: bool = True
    _was_present: bool = True  # presence only -- edge-detection state, see module docstring


def _wiggle(lamp) -> None:
    """The physical half of a reminder firing -- a quick, playful
    side-to-side wobble."""
    current = lamp.get_current_pose()
    lamp.set_target_pose(current + _WIGGLE_OFFSET, duration=0.35, anticipation=False, overshoot=True)


def _fire(reminder: Reminder, lamp) -> None:
    print(f"[reminder] {reminder.message}")
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

    def add(self, kind: str, message: str, interval_s: float | None = None) -> Reminder:
        r = Reminder(id=self._next_id, kind=kind, message=message, created_at=time.time(),
                      interval_s=interval_s)
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

    def active_summaries(self) -> list[str]:
        """One short line per active reminder, for main.py's HUD panel --
        not the full message (that can be a whole sentence), just enough
        to recognize which one it is at a glance."""
        lines = []
        for r in self.reminders:
            if not r.active:
                continue
            label = r.message if len(r.message) <= 40 else r.message[:37] + "..."
            if r.kind == "recurring":
                lines.append(f"#{r.id} every {r.interval_s / 60:.0f}min: {label}")
            else:
                lines.append(f"#{r.id} on leaving desk: {label}")
        return lines

    def tick(self, now: float, face_found: bool, lamp) -> None:
        """now is wall-clock time.time(), not time.perf_counter() -- both
        interval_s comparisons and on-disk persistence need to make sense
        across a process restart, which perf_counter's arbitrary
        per-process origin can't do."""
        fired_any = False
        for r in self.reminders:
            if not r.active:
                continue
            if r.kind == "recurring":
                last = r.last_fired_at if r.last_fired_at is not None else r.created_at
                if r.interval_s is not None and now - last >= r.interval_s:
                    _fire(r, lamp)
                    r.last_fired_at = now
                    fired_any = True
            elif r.kind == "presence":
                if r._was_present and not face_found:
                    _fire(r, lamp)
                    fired_any = True
                r._was_present = face_found
        if fired_any:
            self.save()
