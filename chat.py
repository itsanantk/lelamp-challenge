"""Ask the lamp what it remembers.

Standalone, this owns its own SimulatedLamp and a short popup window --
run it on its own for a text/voice-only chat against whatever main.py
has written to the memory database so far.

Embedded in main.py (--chat), run()'s lamp/store/fsm params are supplied
by main.py instead of constructed here, so this runs as a background
thread against the SAME lamp instance main.py's own loop is already
rendering continuously -- no second window, no second render loop. See
docs/ARCHITECTURE.md for why this used to be two separate processes and
why it isn't anymore.

Usage:
    python chat.py                 interactive chat, lamp points + a short popup window
    python chat.py --voice         talk to it instead of typing -- mic in, TTS out
    python chat.py --voice --wake-word "hey lamp"   change the wake phrase
    python chat.py --voice --multi-user   also figure out *who* is talking when more than one face is in frame
    python chat.py --no-gui        text-only, no popup window (e.g. headless/CI)
    python chat.py --ask "..."     ask one question non-interactively and exit

Voice mode is wake-word gated, not a fixed listen window: it sits quiet
until it hears the wake phrase, chirps once, then records until you stop
talking (trailing silence, not a countdown) -- see conversation/voice.py.
Say "quit" any time (no need to wake it first) to leave voice mode.

Set LELAMP_LLM_PROVIDER=openai to use OpenAI instead of Anthropic if
that's the key with credit on it (see config.py). Voice mode runs
entirely locally (Whisper + SAPI5 TTS) regardless of which LLM you use.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

import cv2
import numpy as np

import config

if sys.stdout.encoding.lower() != "utf-8":
    # Windows consoles often default to a legacy codepage that can't print
    # an em dash, mangling otherwise-correct model output into "?" or "*"
    sys.stdout.reconfigure(encoding="utf-8")
import viz
from behavior.state_machine import ENGAGED_COLOR, State
from lamp import SimulatedLamp, kinematics
from memory.store import MemoryStore
from conversation.agent import MemoryAgent
from conversation import emotion, voice
from perception.multi_face import SpeakerDetector

FOUND_COLOR = (210, 235, 255)  # BGR, bright warm spotlight -- distinct from
                                 # the orange attention-seek pulse and the
                                 # green object-watch tint
IDLE_COLOR = (40, 110, 200)

# Voice-controlled lighting (control_light tool in conversation/agent.py).
# "dim"/"on" act relative to whatever's currently showing; the rest are
# absolute mood presets. "on" has no fixed color of its own -- see
# _brighten_to_max -- since a flat preset there would mean "brighten"
# replaces whatever hue was showing with plain gray instead of actually
# brightening it.
LIGHT_PRESETS = {
    "cozy": (40, 120, 230),    # BGR, warm amber-orange
    "focus": (230, 225, 210),  # BGR, crisp bright white, slightly cool
    "calm": (200, 150, 90),    # BGR, soft cool blue
    "off": (0, 0, 0),
}
_DIM_STEP = 0.55  # each "dim" scales current brightness down by this factor
_DIM_FLOOR = 4    # per-channel floor so repeated "dim" settles near-off without hitting literal (0,0,0) -- that's what "off" is for


def _brighten_to_max(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Scales rgb up so its brightest channel hits 255, preserving hue --
    the inverse of the "dim" step. Falls back to ENGAGED_COLOR when there's
    essentially nothing to brighten (i.e. coming from "off"), since scaling
    up (0, 0, 0) by any factor is still (0, 0, 0)."""
    peak = max(rgb)
    if peak <= _DIM_FLOOR:
        return ENGAGED_COLOR
    scale = 255 / peak
    # round(), not int() -- truncating float imprecision (255/100 landing
    # at 2.549999... instead of exactly 2.55) made the peak channel land
    # on 254 instead of the 255 it's mathematically supposed to hit.
    return tuple(min(round(c * scale), 255) for c in rgb)


def _wait_out_attention_seek(fsm, timeout: float = 1.5) -> None:
    """fsm is only ever non-None when embedded in main.py, sharing its
    BehaviorFSM. ATTENTION_SEEKING is a short, precisely-timed animation
    (state_machine._tick_attention_seek) that owns pose/light for its
    duration and doesn't expect anything else to touch them mid-gesture --
    the same conflict object_watch.py already solved by deferring to it
    rather than fighting over control (see that module's docstring). A
    voice command briefly waits it out here instead of stomping on it or
    being silently overwritten a moment later."""
    if fsm is None:
        return
    t0 = time.perf_counter()
    while fsm.state == State.ATTENTION_SEEKING and time.perf_counter() - t0 < timeout:
        time.sleep(0.05)


def _hold(lamp: SimulatedLamp, seconds: float, show_gui: bool, standalone: bool) -> None:
    """Holds a flourish's lamp state on screen for `seconds`. Standalone
    chat.py has no other loop advancing the lamp's animation or drawing
    it, so this manually ticks update() + optionally shows a popup window,
    same as it always has. Embedded in main.py (standalone=False), the
    shared render loop is already doing both continuously every frame --
    this just waits, and never touches cv2 at all (only main.py's own
    thread does)."""
    if not standalone:
        time.sleep(seconds)
        return
    t0 = time.perf_counter()
    last = t0
    while time.perf_counter() - t0 < seconds:
        now = time.perf_counter()
        lamp.update(now - last)
        last = now
        if show_gui:
            cv2.imshow("LeLamp recalls...", lamp.render())
            if cv2.waitKey(16) & 0xFF == ord("q"):
                break
    if show_gui:
        cv2.destroyWindow("LeLamp recalls...")


def _apply_light_command(lamp: SimulatedLamp, action: str, show_gui: bool, seconds: float = 1.0,
                          standalone: bool = True, fsm=None) -> tuple[int, int, int]:
    """Returns the color it actually applied, so callers can track it as
    the new baseline (see run()'s base_light_color) instead of reading it
    back mid-transition from get_current_light()."""
    _wait_out_attention_seek(fsm)
    # set_light() only sets a transition target -- nothing advances that
    # transition or draws it without a render loop, so without one the
    # light silently changes internally while the window (if any is even
    # open) never shows it and update() never ticks it forward either.
    if action == "dim":
        current = lamp.get_current_light()
        target = tuple(max(int(c * _DIM_STEP), _DIM_FLOOR) for c in current)
    elif action == "on":
        target = _brighten_to_max(lamp.get_current_light())
    else:
        target = LIGHT_PRESETS[action]
    lamp.set_light(target, transition_s=0.4 if action in ("dim", "on") else 0.5)
    _hold(lamp, seconds, show_gui, standalone)
    return target


def _look_attentive(lamp: SimulatedLamp, bearing_deg: float | None, set_light: bool = True, fsm=None,
                     light_color: tuple[int, int, int] = ENGAGED_COLOR) -> None:
    """Held during the post-wake conversation window so the lamp visibly
    looks like it's still listening, not just idling until you say the
    wake word again. set_light=False when an explicit control_light
    command just fired that turn -- otherwise "make it cozy" would flicker
    straight back to light_color within a second. light_color defaults to
    the plain listening color but should be run()'s current
    base_light_color once one exists, so a still-active custom preset
    ("cozy") doesn't get silently wiped by the very next follow-up turn
    that doesn't also change the light."""
    _wait_out_attention_seek(fsm)
    pose = kinematics.pose_for_look_at(bearing_deg if bearing_deg is not None else 0.0, -10.0, alertness=0.75)
    lamp.set_target_pose(pose, duration=0.4, anticipation=False, overshoot=False)
    if set_light:
        lamp.set_light(light_color, transition_s=0.3)


def _relax_to_idle(lamp: SimulatedLamp, fsm=None) -> None:
    _wait_out_attention_seek(fsm)
    if fsm is not None and fsm.state == State.ENGAGED:
        # Still being looked at -- BehaviorFSM's own engaged look (pose +
        # ENGAGED_COLOR, set once on entry) is already exactly right, and
        # it won't re-assert that on its own since it only does so on its
        # own transitions (see object_watch.py's docstring for the same
        # reasoning applied to object tracking). Relaxing to idle here
        # would fight it for no reason.
        return
    lamp.set_light(IDLE_COLOR, transition_s=0.6)
    lamp.set_target_pose(kinematics.NEUTRAL_POSE, duration=0.6, overshoot=False)


def _point_and_render(lamp: SimulatedLamp, bearing_deg: float, show_gui: bool, seconds: float = 2.0,
                       standalone: bool = True, fsm=None) -> None:
    _wait_out_attention_seek(fsm)
    target = kinematics.pose_for_look_at(bearing_deg, -10.0, alertness=0.85)
    lamp.set_target_pose(target, duration=0.5, anticipation=True, overshoot=True)
    lamp.set_light(FOUND_COLOR, transition_s=0.3)
    lamp.play_sound("recall_point")
    _hold(lamp, seconds, show_gui, standalone)
    lamp.set_light(IDLE_COLOR, transition_s=0.6)
    lamp.set_target_pose(kinematics.NEUTRAL_POSE, duration=0.6, overshoot=False)


# Only these read as something worth visibly reacting to -- "calm" and
# "energetic" are just normal speech and shouldn't tint the light at all,
# every voice turn used to flash *some* color regardless of tone, which
# meant plain calm conversation kept turning the lamp green/blue for no
# reason. quiet is the closest existing bucket to "sad" (see
# conversation/emotion.py's own module docstring on the 4-label system's
# limits -- there's no real "sad" label, this is the nearest proxy).
_REACTIVE_TONES = ("tense", "quiet")


def _flash_tone(lamp: SimulatedLamp, label: str, show_gui: bool, seconds: float = 0.6,
                 standalone: bool = True, fsm=None) -> None:
    _wait_out_attention_seek(fsm)
    tone_color = emotion.BGR_BY_LABEL.get(label, (180, 220, 180))
    lamp.set_light(tone_color, transition_s=0.2)
    _hold(lamp, seconds, show_gui, standalone)
    # Settle into a dimmed version of the same tone color instead of a
    # fixed neutral amber -- this is what makes the warmth actually track
    # how you sounded rather than just flashing and reverting.
    settled = tuple(max(int(c * 0.5), 20) for c in tone_color)
    lamp.set_light(settled, transition_s=0.6)


def _find_live_bearing(vision_memory, object_class: str, timeout_s: float = 0.5) -> float | None:
    """Forces a fresh scan and checks whether object_class is actually
    visible right now, rather than just trusting whatever the last
    opportunistic scan happened to catch -- at normal cadence, a static
    scene can go a couple seconds between real scans (see
    VisionMemory.maybe_scan's scene-change skip logic), and "is my phone
    still there" is exactly the question a stale read answers wrong.
    vision_memory is None in standalone chat.py (no live camera loop to
    ask), where this always falls through to the remembered bearing --
    same as it always has."""
    if vision_memory is None:
        return None
    before = vision_memory.scan_count
    vision_memory.request_immediate_scan()
    t0 = time.perf_counter()
    while vision_memory.scan_count == before and time.perf_counter() - t0 < timeout_s:
        time.sleep(0.02)
    for d in vision_memory.last_detections:
        if d.object_class == object_class:
            return d.bearing_deg
    return None


def _listen(speaker_detector: SpeakerDetector | None, max_s: float | None = None,
            no_speech_timeout_s: float | None = None,
            shutdown_event: threading.Event | None = None) -> tuple[str, float | None, int, np.ndarray]:
    """Records the mic until the question trails off into silence, and --
    if a speaker_detector is available -- samples video for who's-talking
    over the *same* window concurrently, since "who was talking" only
    means something measured at the same moment as the actual question.
    A shared stop_event lets the (fixed-length-guessing) video side end
    exactly when the (variable-length) audio side actually does, instead
    of running to its own separate timeout. max_s/no_speech_timeout_s let
    a follow-up turn in an already-open conversation wait longer than the
    post-wake default (see CONVERSATION_FOLLOWUP_TIMEOUT_S) -- both need
    to move together, since record_until_silence's max_s is an absolute
    ceiling on the whole call, no-speech-wait included, and the speaker
    detector's own timeout has to cover at least as long or it'll return a
    stale bearing before the mic side is done waiting. shutdown_event, if
    given, aborts recording early (process is shutting down). Returns (transcript,
    speaker bearing or None, faces seen, raw audio -- the last one for
    emotion.analyze)."""
    kwargs = {}
    if max_s is not None:
        kwargs["max_s"] = max_s
    if no_speech_timeout_s is not None:
        kwargs["no_speech_timeout_s"] = no_speech_timeout_s
    if shutdown_event is not None:
        kwargs["shutdown_event"] = shutdown_event
    detect_cap = max_s if max_s is not None else config.VOICE_MAX_RECORD_S

    if speaker_detector is not None and speaker_detector.available:
        result: dict = {}
        stop_event = threading.Event()

        def _record() -> None:
            result["audio"] = voice.record_until_silence(stop_event=stop_event, **kwargs)

        t = threading.Thread(target=_record)
        t.start()
        bearing, n_faces = speaker_detector.detect_active_speaker(stop_event, max_duration_s=detect_cap)
        t.join()
        audio = result["audio"]
    else:
        audio = voice.record_until_silence(**kwargs)
        bearing, n_faces = None, 0

    text = voice.transcribe(audio).strip()
    return text, bearing, n_faces, audio


def run(args: argparse.Namespace, lamp: SimulatedLamp | None = None, store: MemoryStore | None = None,
        fsm=None, vision_memory=None, shutdown_event: threading.Event | None = None) -> None:
    """lamp/store/fsm/vision_memory are supplied by main.py when this runs
    embedded on a background thread (--chat) instead of as its own
    process -- see the module docstring. Left as None, this owns and
    constructs lamp/store itself exactly as before, unaffected by any of
    this; vision_memory has no standalone equivalent (there's no live
    camera loop to ask), so recall always falls back to the remembered
    bearing in that mode, same as it always has.

    store gets its own sqlite3 connection either way, never one handed in
    from another thread -- sqlite3 connections are confined to the thread
    that created them, and main.py's own VisionMemory already has its own
    connection to the same underlying db file, so this just mirrors that
    rather than needing check_same_thread=False."""
    standalone = lamp is None
    if standalone and not args.no_gui:
        viz.enable_dpi_awareness()  # must happen before any window gets created -- see viz.py
    owns_store = store is None
    if store is None:
        store = MemoryStore()
    agent = MemoryAgent(store)
    owns_lamp = lamp is None
    if lamp is None:
        lamp = SimulatedLamp(mute=args.mute)

    if args.voice:
        voice.preload()

    speaker_detector = None
    if args.multi_user:
        if not args.voice:
            print("[chat] --multi-user only does anything with --voice, ignoring it")
        else:
            speaker_detector = SpeakerDetector()
            if not speaker_detector.available:
                print("[chat] --multi-user: couldn't open the camera (main.py might already have it), "
                      "continuing without speaker ID")

    known = store.list_known_classes()
    print(f"[chat] LeLamp remembers {len(known)} object type(s): {', '.join(known) or '(nothing yet -- run main.py first)'}")
    if args.voice:
        print(f'[chat] Voice mode: say "{args.wake_word}" to wake it up and ask a question, '
              f'or "quit" any time to exit.\n')
    else:
        print("[chat] Ask about an object's location, or 'quit' to exit.\n")

    # Tracks whatever color should be considered "normal" right now --
    # ENGAGED_COLOR until the user explicitly asks for a preset, then that
    # preset, until the next explicit change. _look_attentive uses this
    # instead of a hardcoded ENGAGED_COLOR so a still-active "cozy" isn't
    # silently wiped by the very next follow-up turn that doesn't also
    # touch the light.
    base_light_color = ENGAGED_COLOR

    def handle(question: str, speaker_bearing: float | None = None, tone_label: str | None = None) -> None:
        nonlocal base_light_color
        if speaker_bearing is not None:
            # Quick glance toward whoever was actually talking, before
            # answering -- separate gesture from the "found it" point.
            _wait_out_attention_seek(fsm)
            glance = kinematics.pose_for_look_at(speaker_bearing, -10.0, alertness=0.6)
            lamp.set_target_pose(glance, duration=0.3, anticipation=True, overshoot=True)

        if tone_label in _REACTIVE_TONES:
            _flash_tone(lamp, tone_label, show_gui=not args.no_gui, standalone=standalone, fsm=fsm)

        try:
            reply = agent.ask(question)
        except Exception as e:
            # A failed API call (bad/unfunded key, network blip, rate
            # limit) used to take the whole process down here, which from
            # the outside just looks like "I said something and it never
            # answered." Surface it instead of dying mid-conversation.
            print(f"[chat] couldn't reach the LLM: {e}\n")
            if args.voice:
                voice.speak("Sorry, I couldn't reach the model just now.")
            return

        print(f"LAMP: {reply}\n")
        if args.voice:
            voice.speak(reply)
        if agent.last_light_action is not None:
            base_light_color = _apply_light_command(lamp, agent.last_light_action, show_gui=not args.no_gui,
                                                      standalone=standalone, fsm=fsm)
        obs = agent.last_recall.observation
        if obs is not None:
            # Try to actually find it before just pointing at memory --
            # a phone sitting still is exactly the case a stale scan can
            # miss (see _find_live_bearing). The reply text above is
            # already generated from the remembered sighting either way
            # (re-prompting the LLM on a live/not-live split isn't worth
            # the extra round trip) -- only the gesture's target bearing
            # changes, live if found, remembered if not.
            live_bearing = _find_live_bearing(vision_memory, obs.object_class)
            bearing = live_bearing if live_bearing is not None else obs.bearing_deg
            _point_and_render(lamp, bearing, show_gui=not args.no_gui, standalone=standalone, fsm=fsm)

    if args.ask:
        handle(args.ask)
        if owns_lamp:
            lamp.close()
        if owns_store:
            store.close()
        if speaker_detector is not None:
            speaker_detector.close()
        return

    try:
        in_conversation = False  # True while listening for a follow-up without needing the wake word again
        while True:
            if shutdown_event is not None and shutdown_event.is_set():
                break
            speaker_bearing = None
            tone_label = None
            if args.voice:
                # Only meaningful when embedded in main.py (fsm is None
                # standalone). If the person's already looking at it,
                # that *is* the invitation to talk -- requiring "hey lamp"
                # on top of already being looked at is exactly the
                # redundant gate a person listening to you wouldn't need.
                # Treated the same as an already-open conversation: same
                # patient follow-up timeout (a quick glance shouldn't kick
                # off a question 3s later), same reasoning for not
                # touching pose/light going in -- BehaviorFSM._enter_engaged
                # already set an accurate look-at-you pose + ENGAGED_COLOR
                # on the transition in, so there's nothing to add before
                # anything's actually been said.
                engaged_now = fsm is not None and fsm.state == State.ENGAGED
                if not in_conversation and not engaged_now:
                    woken = voice.wait_for_wake_word(args.wake_word, shutdown_event=shutdown_event)
                    if woken == "quit":
                        break
                    lamp.play_sound("wake")
                    time.sleep(0.2)  # let the chirp clear the speaker before the mic opens -- otherwise its tail can leak into the recording and falsely trip the silence detector
                    _look_attentive(lamp, None, fsm=fsm, light_color=base_light_color)
                    print("[voice] listening...")
                    question, speaker_bearing, n_faces, audio = _listen(speaker_detector, shutdown_event=shutdown_event)
                else:
                    print("[voice] still listening for a follow-up..." if in_conversation
                          else "[voice] engaged -- listening without a wake word...")
                    question, speaker_bearing, n_faces, audio = _listen(
                        speaker_detector,
                        max_s=config.CONVERSATION_FOLLOWUP_TIMEOUT_S,
                        no_speech_timeout_s=config.CONVERSATION_FOLLOWUP_TIMEOUT_S,
                        shutdown_event=shutdown_event)

                if n_faces > 1:
                    print(f"[multi-user] {n_faces} faces in frame, "
                          f"responding to whoever's mouth was moving (bearing {speaker_bearing:.0f}°)")
                if not question:
                    if in_conversation:
                        print("[voice] no follow-up heard, going back to sleep\n")
                        in_conversation = False
                        _relax_to_idle(lamp, fsm=fsm)
                    elif engaged_now:
                        print("[voice] nothing heard yet, still listening while engaged\n")
                    else:
                        print("[voice] didn't catch that, try again\n")
                    continue
                tone = emotion.analyze(audio)
                tone_label = tone.label
                pitch_var_str = f"{tone.pitch_variability_hz:.1f}" if tone.pitch_variability_hz is not None else "n/a"
                print(f"[voice] tone: {tone.label} (rms={tone.rms:.4f} rate={tone.speaking_rate_hz:.2f}Hz "
                      f"pitch_var={pitch_var_str})")
                print(f"YOU (voice): {question}")
            else:
                try:
                    question = input("YOU: ").strip()
                except EOFError:
                    break
                if not question:
                    continue

            if question.lower().strip(".!? ") in ("quit", "exit", "q"):
                break
            handle(question, speaker_bearing, tone_label)
            if args.voice:
                in_conversation = True
                _look_attentive(lamp, speaker_bearing, set_light=agent.last_light_action is None, fsm=fsm,
                                 light_color=base_light_color)
    finally:
        if owns_lamp:
            lamp.close()
        if owns_store:
            store.close()
        if speaker_detector is not None:
            speaker_detector.close()
        if standalone and not args.no_gui:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ask the lamp what it remembers")
    p.add_argument("--no-gui", action="store_true", help="disable the lamp-pointing popup window")
    p.add_argument("--voice", action="store_true", help="talk instead of typing -- mic input + TTS output")
    p.add_argument("--mute", action="store_true", help="disable sound cues")
    p.add_argument("--ask", type=str, default=None, help="ask one question non-interactively and exit")
    p.add_argument("--multi-user", action="store_true",
                    help="with --voice: identify which face was talking when more than one is in frame")
    p.add_argument("--wake-word", type=str, default=config.WAKE_WORD,
                    help=f'with --voice: phrase that wakes it up to listen (default: "{config.WAKE_WORD}")')
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
