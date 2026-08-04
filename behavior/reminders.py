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
    checks in once due_at hits. One-shot -- deactivates itself once fired,
    unlike the other two. Placement detection here is deliberately
    independent of behavior/object_watch.py's own tracking (which can
    only actively watch one object at a time and might be busy with
    something else) -- this keeps its own minimal bookkeeping off the
    same detections main.py already has, reusing ObjectWatcher's
    stillness thresholds for consistency, not its state.

    Firing an object_check with no check_question is simple enough that
    this module handles it directly (see _fire): point at the remembered
    bearing, speak the reminder's own message. One *with* a
    check_question ("is this bottle full or empty") needs an actual
    vision-LLM call, which this module has no LLM/API-key access to make
    -- this module has no dependency on conversation/agent.py at all, on
    purpose, the same separation-of-concerns split as everywhere else in
    this codebase (perception/ produces readings, doesn't know an LLM
    exists). tick() only flips due_for_check True once the deadline hits
    for that case, then leaves the reminder alone; chat.py's own
    background poller thread (it has agent/lamp/vision_memory) claims it,
    resolves the judgment, and deactivates it. See chat.py's
    _resolve_object_check_judgment.

    A check_question also means something got disturbed is itself worth
    checking on, not just the deadline -- "make sure I don't go on my
    phone for five minutes" should catch you picking it up at second 10,
    not stay silent until the full window elapses. Once placement is
    confirmed, tick() keeps watching the object via the same detections
    (see _check_for_disturbance); a sustained move away from the
    confirmed bearing, or a sustained disappearance, dispatches the
    check_question early through the exact same due_for_check path a
    real deadline would -- "is this still where it was left" already
    reads correctly whether it's asked because time ran out or because
    something just moved. No check_question means there's no judgment
    to make early ("check on my keys in an hour" doesn't imply "and
    yell if I touch them earlier") -- those just wait for due_at like
    before.
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
# already uses for its own ENGAGE_EXIT_DWELL_FRAMES.
#
# Originally 1.5s (long enough to be genuinely sure before ever firing),
# but that made real departures feel slow to notice. Shrunk to roughly
# "confirmed absent for 2-3 checks a second" instead, with
# PRESENCE_FIRE_COOLDOWN_S below doing the flicker-proofing that the
# longer debounce used to -- a much faster first reaction, with the
# multi-fire risk covered a different way rather than just accepted back.
PRESENCE_ABSENCE_DEBOUNCE_S = 0.4

# Once a presence reminder actually fires, it won't fire again until this
# much time has passed, regardless of how choppy face detection looks in
# between (return-then-immediately-drop-out-again mid-motion, the same
# real flicker PRESENCE_ABSENCE_DEBOUNCE_S alone used to have to fully
# absorb on its own). The two constants split one job into two: the
# debounce above answers "was this a real departure," fast; this answers
# "have I already told them," and doesn't need to be fast at all -- a
# person who's actually gone doesn't need re-nagging every few seconds,
# just the one nudge.
PRESENCE_FIRE_COOLDOWN_S = 5.0


def _fmt_duration(seconds: float) -> str:
    """"3s" / "12min" -- whichever reads more naturally at that
    magnitude, for active_summaries()'s countdown/overdue display."""
    seconds = max(0.0, seconds)
    return f"{seconds:.0f}s" if seconds < 120 else f"{seconds / 60:.0f}min"


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
    last_fired_at: float | None = None  # recurring: last interval fire. presence: last
                                          # nudge, gates PRESENCE_FIRE_COOLDOWN_S. Shared
                                          # field -- a Reminder is only ever one kind, so
                                          # the two uses never overlap. None means "never
                                          # fired yet" either way.
    expires_at: float | None = None     # wall-clock deadline after which this auto-deactivates,
                                          # e.g. "only check for the next 20 seconds" -- None runs
                                          # indefinitely until explicitly cancelled
    object_class: str | None = None      # object_check only -- normalized COCO class, e.g. "bottle"
    due_at: float | None = None          # object_check only -- when to check on it
    tracked_bearing: float | None = None  # object_check only -- where it settled, once known
    placement_confirmed: bool = False     # object_check only -- has it been seen to settle yet
    check_question: str | None = None    # object_check only -- e.g. "is this bottle full or
                                           # empty," answered via a vision-LLM look at the
                                           # deadline. None means "just confirm it's there,"
                                           # handled directly (see _fire) without needing chat.py's
                                           # LLM access at all.
    due_for_check: bool = False          # object_check only -- deadline hit AND check_question is
                                           # set, so firing needs a vision-LLM call this module
                                           # can't make itself (no LLM/API-key access here -- see
                                           # module docstring). tick() sets this once, then leaves
                                           # the reminder alone; chat.py's own background poller
                                           # (its thread has agent/lamp/vision_memory access) claims
                                           # it (clearing this flag immediately, well before it's
                                           # actually finished resolving -- see _check_dispatched
                                           # below for why that alone isn't enough to prevent a
                                           # second dispatch), resolves the judgment, and
                                           # deactivates it.
    _check_dispatched: bool = False      # object_check only -- separate from due_for_check on
                                           # purpose: chat.py's poller clears due_for_check the
                                           # instant it claims a reminder, well before it actually
                                           # finishes (judge_view()/voice.speak() can take several
                                           # seconds) -- a later tick() that only checked
                                           # due_for_check would see it False again, still >=
                                           # due_at, and re-arm it mid-resolution. This flag is set
                                           # alongside due_for_check but never cleared by the
                                           # poller, so a dispatch can only ever happen once per
                                           # reminder.
    active: bool = True
    _was_present: bool = True  # presence only -- edge-detection state, see module docstring
    _absent_since: float | None = None  # presence only -- debounce state, see PRESENCE_ABSENCE_DEBOUNCE_S
    _settled_since: float | None = None  # object_check only -- when tracked_bearing last changed
    _disturbed_since: float | None = None  # object_check only, post-confirmation -- see
                                             # _check_for_disturbance


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
            due_in_s: float | None = None, check_question: str | None = None) -> Reminder:
        created_at = time.time()
        expires_at = created_at + duration_s if duration_s is not None else None
        due_at = created_at + due_in_s if due_in_s is not None else None
        r = Reminder(id=self._next_id, kind=kind, message=message, created_at=created_at,
                      interval_s=interval_s, expires_at=expires_at, object_class=object_class,
                      due_at=due_at, check_question=check_question)
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

    def watched_classes(self) -> list[str]:
        """Object classes at least one active object_check reminder is
        currently disturbance-watching (placement confirmed, a
        check_question set, deadline not yet dispatched -- see
        _check_for_disturbance). main.py folds these into
        vision_memory.fast_mode alongside ObjectWatcher.should_scan_fast()
        so the scan cadence stays fast for the whole watch window, not
        just while the object happens to still be actively moving.

        Real bug this fixes, reported live: "make sure I don't go on my
        phone" felt slow to notice a pickup. Root cause was cadence, not
        the disturbance debounce itself -- ObjectWatcher's own fast-scan
        trigger (should_scan_fast) drops back to the slow interval once an
        object's been stationary for WATCH_FAST_SCAN_SETTLE_S (1.5s), and
        placement confirmation itself requires WATCH_STATIONARY_TIMEOUT_S
        (3.0s) of stillness -- so by the time disturbance-watching even
        starts, fast-scan has always already lapsed. The very first catch
        of an actual pickup then had to wait for the next slow-cadence
        scan (up to YOLO_SCAN_INTERVAL_S, worse if scene-change gating
        skipped a few), *before* the disturbance debounce even started
        counting. This keeps the scan itself fast the whole time a
        disturbance-relevant watch is active, so that lag doesn't stack on
        top of the debounce."""
        return list({r.object_class for r in self.reminders
                     if r.active and r.kind == "object_check" and r.placement_confirmed
                     and r.check_question is not None and not r._check_dispatched
                     and r.object_class is not None})

    def active_summaries(self, now: float | None = None) -> list[str]:
        """One short line per active reminder, for main.py's HUD panel --
        not the full message (that can be a whole sentence), just enough
        to recognize which one it is at a glance. now is optional -- only
        needed to show remaining/elapsed time; omit it (e.g. right after
        add(), with nothing to tick against yet) and those parts are just
        left off. The due-at/overdue-by countdown exists specifically as
        the debugging aid live testing asked for -- "did this actually
        fire near its deadline, or is it just slow?" is otherwise
        invisible without watching timestamps in the console."""
        lines = []
        for r in self.reminders:
            if not r.active:
                continue
            label = r.message if len(r.message) <= 40 else r.message[:37] + "..."
            if r.kind == "recurring":
                line = f"#{r.id} every {r.interval_s / 60:.0f}min: {label}"
            elif r.kind == "object_check":
                if r.due_for_check:
                    status = "checking now"
                elif r.due_at is not None and now is not None:
                    remaining = r.due_at - now
                    time_str = f"due in {_fmt_duration(remaining)}" if remaining >= 0 \
                        else f"overdue by {_fmt_duration(-remaining)}"
                    # Placement and the deadline now run independently (tick()
                    # no longer gates the due_at check on confirmation), so
                    # both can be true at once -- say so rather than implying
                    # placement finished when it may not have.
                    status = time_str if r.placement_confirmed else f"watching, {time_str}"
                elif not r.placement_confirmed:
                    status = "watching for placement"
                else:
                    status = "waiting for deadline"
                line = f"#{r.id} {r.object_class} ({status}): {label}"
            else:
                line = f"#{r.id} on leaving desk: {label}"
            if r.expires_at is not None and now is not None:
                line += f" (expires in {_fmt_duration(r.expires_at - now)})"
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

    def _check_for_disturbance(self, r: Reminder, now: float, vision_memory) -> bool:
        """object_check only, post-confirmation -- has the object moved
        away from its confirmed r.tracked_bearing (frozen once
        placement_confirmed is set -- _update_placement never runs again
        after that, so this compares against the original spot, not a
        drifting one), or gone missing entirely. Reuses
        config.WATCH_LOST_GRACE_S as the debounce for both -- long enough
        that a single off-angle miss or a one-frame detection wobble
        doesn't trip it (object detection flickers harder than face
        detection; PRESENCE_ABSENCE_DEBOUNCE_S exists for exactly this
        reason on the face side), short enough that a real pickup is
        caught in seconds, not left to run out the clock. Returns True
        once on the tick that crosses the debounce; r._disturbed_since
        carries the pending state across ticks either way, and resets the
        moment the object is seen back in place."""
        if vision_memory is None:
            return False
        detections = vision_memory.last_detections or []
        match = next((d for d in detections if d.object_class == r.object_class), None)
        disturbed = match is None or abs(match.bearing_deg - r.tracked_bearing) > config.WATCH_REAIM_DEG
        if not disturbed:
            r._disturbed_since = None
            return False
        if r._disturbed_since is None:
            r._disturbed_since = now
            return False
        return now - r._disturbed_since >= config.WATCH_LOST_GRACE_S

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
                        # Confirmed absent -- still one detection per
                        # departure regardless of the cooldown below, so
                        # this doesn't keep re-evaluating every tick for
                        # the same continuous absence once it's already
                        # been noticed once.
                        r._was_present = False
                        on_cooldown = r.last_fired_at is not None and \
                            now - r.last_fired_at < PRESENCE_FIRE_COOLDOWN_S
                        if not on_cooldown:
                            _fire(r, lamp)
                            fired_any = True
                            r.last_fired_at = now
            elif r.kind == "object_check":
                if not r.placement_confirmed:
                    self._update_placement(r, now, vision_memory)
                    if r.placement_confirmed:
                        fired_any = True  # not a fire, but placement_confirmed/tracked_bearing changed -- persist it
                # Deliberately NOT an elif against the placement branch above.
                # Placement confirmation requires the object to sit still for
                # WATCH_STATIONARY_TIMEOUT_S -- fine for a bottle set down and
                # left alone, but an object being actively used (a book being
                # read, page turns and hand adjustments) can keep resetting
                # that settle timer and may never confirm at all. Gating the
                # deadline on confirmation meant due_at could be missed by as
                # long as placement took to (maybe never) settle. The deadline
                # now fires on its own schedule regardless, using whatever
                # tracked_bearing is known (possibly None/unconfirmed) --
                # _fire/_resolve_object_check_judgment both already fall back
                # to a plain glance when there's no bearing to point at.
                # A check_question implies judging the object's state
                # matters, which makes a disturbance itself worth an early
                # check -- "make sure I don't go on my phone" should catch
                # you picking it up at second 10, not report it silently
                # five minutes later. No check_question means there's no
                # judgment to make early, so those skip straight to the
                # due_at branch below and behave exactly as before.
                if r.placement_confirmed and r.check_question is not None and not r._check_dispatched \
                        and self._check_for_disturbance(r, now, vision_memory):
                    r.due_for_check = True
                    r._check_dispatched = True
                    fired_any = True
                    print(f"[reminder] #{r.id} disturbance detected at "
                          f"{time.strftime('%H:%M:%S', time.localtime(now))} -- dispatching to vision judgment")
                elif r.due_at is not None and now >= r.due_at and not r._check_dispatched:
                    if r.check_question is not None:
                        # Can't resolve this here -- needs a vision-LLM
                        # call this module has no access to make (see
                        # module docstring). Flip the flag once and leave
                        # it; chat.py's background poller claims and
                        # resolves it on its own thread. _check_dispatched
                        # (not cleared by the poller, unlike due_for_check
                        # itself) is what actually guarantees this branch
                        # never runs twice for the same reminder -- see
                        # its own field comment.
                        r.due_for_check = True
                        r._check_dispatched = True
                        fired_any = True
                        # Timestamped on purpose -- live testing asked for
                        # exactly this: a way to see whether a late-feeling
                        # reminder was actually late hitting its deadline,
                        # or the deadline landed on time and the *judgment*
                        # (vision-LLM round trip + TTS, in chat.py's poller)
                        # is what took a few extra seconds. Compare this
                        # line's clock time against _resolve_object_check_
                        # judgment's own "[reminder] <resolved text>" print.
                        late_by = now - r.due_at
                        print(f"[reminder] #{r.id} deadline reached at "
                              f"{time.strftime('%H:%M:%S', time.localtime(now))} "
                              f"({late_by:.1f}s after due_at) -- dispatching to vision judgment")
                    else:
                        if vision_memory is not None:
                            vision_memory.request_immediate_scan()  # best-effort -- doesn't block this fire
                        _fire(r, lamp)
                        r.active = False  # one-shot: deadline reached and reported, nothing left to watch for
                        fired_any = True
        if fired_any or expired_any:
            self.save()
