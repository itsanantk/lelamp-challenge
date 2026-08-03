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
entirely locally (Whisper + Piper TTS, falling back to SAPI5 if Piper's
model isn't downloaded) regardless of which LLM you use.
"""
from __future__ import annotations

import argparse
import re
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
from behavior.state_machine import ENGAGED_NUDGE, IDLE_COLOR, State
from lamp import SimulatedLamp, color, kinematics
from memory.store import MemoryStore, bearing_to_direction
from conversation.agent import MemoryAgent
from conversation import emotion, voice
from perception.multi_face import SpeakerDetector

# "Found it" reads as a bright, slightly cooler spotlight -- a nudge off
# whatever mood is currently showing, not an unrelated fixed hue (see
# lamp/color.py's module docstring for why nudging instead of replacing
# is what actually keeps the palette feeling coherent).
FOUND_NUDGE = (-0.2, 0.35)
# tense reads as a little warmer/brighter (agitated); quiet as cooler/
# dimmer (subdued) -- "calm"/"energetic" never reach here, see
# _REACTIVE_TONES below.
_TONE_NUDGES = {"tense": (0.15, 0.25), "quiet": (-0.15, -0.25)}

# Voice-controlled lighting (control_light tool in conversation/agent.py).
# All of these set the persistent (warmth, brightness) *mood* dial (see
# lamp/color.py, SimulatedLamp.get_mood/set_mood) -- "dim"/"on" adjust the
# brightness dial relative to whatever's current; the named presets set
# both dials outright. LIGHT_PRESETS below (BGR, for callers/tests that
# want the actual color a preset produces) is derived from the same
# warmth/brightness values _apply_light_command uses, not a separate list.
LIGHT_PRESETS = {name: color.from_dials(*wb) for name, wb in color.MOOD_PRESETS.items()}
LIGHT_PRESETS["off"] = (0, 0, 0)
_DIM_STEP = 0.15       # each "dim" lowers the brightness dial by this much
_BRIGHTNESS_FLOOR = 0.03  # floor so repeated "dim" settles near-off without hitting
                            # literal 0 brightness -- that's what "off" is explicitly for
_ON_BRIGHTNESS = 0.95   # "on" sets brightness dial to (near-)full, warmth untouched


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
    """Sets the lamp's persistent mood (warmth, brightness) dial and
    returns the color it actually applied. "dim"/"on" only ever touch the
    brightness dial, relative to whatever's current; the named presets set
    both dials outright. Every other transient state (tracking, tone,
    recall) reads the mood back via lamp.get_mood() and nudges a delta off
    it, so it never needs tracking here beyond calling lamp.set_mood()."""
    _wait_out_attention_seek(fsm)
    warmth, brightness = lamp.get_mood()
    if action == "dim":
        brightness = max(brightness - _DIM_STEP, _BRIGHTNESS_FLOOR)
    elif action == "on":
        brightness = _ON_BRIGHTNESS
    elif action == "off":
        brightness = 0.0
    else:
        warmth, brightness = color.MOOD_PRESETS[action]
    lamp.set_mood(warmth, brightness)
    target = color.from_dials(warmth, brightness)
    # set_light() only sets a transition target -- nothing advances that
    # transition or draws it without a render loop, so without one the
    # light silently changes internally while the window (if any is even
    # open) never shows it and update() never ticks it forward either.
    lamp.set_light(target, transition_s=0.4 if action in ("dim", "on") else 0.5)
    _hold(lamp, seconds, show_gui, standalone)
    return target


def _look_attentive(lamp: SimulatedLamp, bearing_deg: float | None, set_light: bool = True, fsm=None,
                     light_color: tuple[int, int, int] | None = None) -> None:
    """Held during the post-wake conversation window so the lamp visibly
    looks like it's still listening, not just idling until you say the
    wake word again. set_light=False when an explicit control_light
    command just fired that turn -- otherwise "make it cozy" would flicker
    straight back within a second. light_color defaults to a nudge off
    the lamp's current mood (read fresh via get_mood() every call, not a
    value tracked separately), so a still-active custom preset ("cozy")
    isn't silently wiped by the very next follow-up turn."""
    _wait_out_attention_seek(fsm)
    pose = kinematics.pose_for_look_at(bearing_deg if bearing_deg is not None else 0.0, -10.0, alertness=0.75)
    lamp.set_target_pose(pose, duration=0.4, anticipation=False, overshoot=False)
    if set_light:
        if light_color is None:
            light_color = color.from_dials(*color.nudge(*lamp.get_mood(), *ENGAGED_NUDGE))
        lamp.set_light(light_color, transition_s=0.3)


def _relax_to_idle(lamp: SimulatedLamp, fsm=None) -> None:
    _wait_out_attention_seek(fsm)
    if fsm is not None and fsm.state == State.ENGAGED:
        # Still being looked at -- BehaviorFSM's own engaged look (pose +
        # color, set once on entry) is already exactly right, and it won't
        # re-assert that on its own since it only does so on its own
        # transitions (see object_watch.py's docstring for the same
        # reasoning applied to object tracking). Relaxing to idle here
        # would fight it for no reason.
        return
    lamp.set_light(color.from_dials(*lamp.get_mood()), transition_s=0.6)
    lamp.set_target_pose(kinematics.NEUTRAL_POSE, duration=0.6, overshoot=False)


def _point_toward(lamp: SimulatedLamp, bearing_deg: float, fsm=None) -> None:
    """Moves toward bearing_deg without yet declaring anything found --
    called before the live check (see handle()), not after, so the check
    actually happens once the lamp is pointed the right way. This matters
    more now than it used to: object detection is restricted to a crop
    that follows wherever the lamp is currently aimed (see
    vision_memory.py's pan_zoom), so checking before moving there was
    checking blind -- the object could be completely outside the frame
    YOLO was even given, no matter how long the check waited."""
    _wait_out_attention_seek(fsm)
    target = kinematics.pose_for_look_at(bearing_deg, -10.0, alertness=0.75)
    lamp.set_target_pose(target, duration=0.5, anticipation=True, overshoot=True)


def _settle_from_point(lamp: SimulatedLamp, show_gui: bool, seconds: float, standalone: bool,
                        confirmed: bool) -> None:
    """confirmed=True plays the "found it" flourish (a brighter light
    nudge + chime) before holding; confirmed=False just holds at the
    remembered-bearing pose quietly. Pointing at a memory and celebrating
    exactly like a live find used to look identical either way -- the
    gesture read as "I see it right now" even on turns where the live
    check came back empty and the reply text said "last seen"."""
    mood = lamp.get_mood()
    if confirmed:
        lamp.set_light(color.from_dials(*color.nudge(*mood, *FOUND_NUDGE)), transition_s=0.3)
        lamp.play_sound("recall_point")
    _hold(lamp, seconds, show_gui, standalone)
    lamp.set_light(color.from_dials(*mood), transition_s=0.6)
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
    mood = lamp.get_mood()
    dw, db = _TONE_NUDGES.get(label, (0.0, 0.0))
    lamp.set_light(color.from_dials(*color.nudge(*mood, dw, db)), transition_s=0.2)
    _hold(lamp, seconds, show_gui, standalone)
    # Settle into a smaller nudge off the same mood instead of a fixed
    # neutral amber -- this is what makes the reaction actually track how
    # you sounded rather than just flashing and reverting to nothing.
    lamp.set_light(color.from_dials(*color.nudge(*mood, dw * 0.4, db * 0.4)), transition_s=0.6)


def _find_live_bearing(vision_memory, object_class: str, window_s: float = 1.6,
                        poll_interval_s: float = 0.15) -> float | None:
    """Repeatedly forces fresh scans for up to window_s, checking whether
    object_class is actually visible right now -- not a single snapshot.
    Returns as soon as it's seen (no need to burn the whole window once
    it's found); returns None only once the window fully elapses without
    ever seeing it. This is what makes the "look, hold, decide" sequence
    in handle() a genuinely stable read instead of a single roll of the
    dice -- one forced scan can land before the lamp/pan-crop has finished
    settling on the new bearing and come back empty even when the object
    is right there; repeating it a handful of times over ~1-2 seconds
    (matching how long a person would actually glance and check) gives
    it real room to catch up.

    vision_memory is None in standalone chat.py (no live camera loop to
    ask), where this always falls through to the remembered bearing --
    same as it always has."""
    if vision_memory is None:
        return None
    deadline = time.perf_counter() + window_s
    while time.perf_counter() < deadline:
        before = vision_memory.scan_count
        vision_memory.request_immediate_scan()
        scan_deadline = min(deadline, time.perf_counter() + poll_interval_s * 2)
        while vision_memory.scan_count == before and time.perf_counter() < scan_deadline:
            time.sleep(0.02)
        for d in vision_memory.last_detections:
            if d.object_class == object_class:
                return d.bearing_deg
        time.sleep(poll_interval_s)
    return None


# Belt-and-suspenders: SYSTEM_PROMPT already tells the model never to use
# emojis (they read as garbled noise or get silently dropped by TTS), but
# instructions aren't a hard guarantee -- this strips any that slip
# through before a reply is ever printed or spoken.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # pictographs, emoticons, transport, supplemental symbols
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicator symbols (flag letters)
    "\U00002B00-\U00002BFF"  # misc symbols and arrows
    "\U0000FE0F"              # variation selector-16 (forces emoji presentation)
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji(text: str) -> str:
    return re.sub(r" {2,}", " ", _EMOJI_PATTERN.sub("", text)).strip()


def _seconds_ago_phrase(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        n = round(seconds)
        return f"{n} second{'s' if n != 1 else ''}"
    minutes = seconds / 60.0
    if minutes < 90:
        n = round(minutes)
        return f"{n} minute{'s' if n != 1 else ''}"
    n = round(minutes / 60.0)
    return f"{n} hour{'s' if n != 1 else ''}"


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

    # A patient follow-up/engaged listen can run its full window (up to
    # CONVERSATION_FOLLOWUP_TIMEOUT_S = 60s) on pure ambient noise if
    # nothing was ever actually said -- record_until_silence's own
    # no-speech bail-out only fires that fast when it's genuinely short
    # (the original post-wake case); a longer window means a lot more
    # ambient audio can accumulate before it gives up. Whisper does not
    # reliably return an empty string on that; it can hallucinate a short
    # phrase from near-silent/noise-only audio, which is what live testing
    # actually showed as "it keeps thinking I'm talking even when I don't
    # say anything." Skip transcription entirely -- same peak-RMS gate the
    # recording itself uses to decide "speech started" -- rather than
    # trusting the model to notice silence on its own.
    if audio.size and voice._peak_rms(audio, config.VOICE_SAMPLE_RATE) >= config.VOICE_GATE_RMS_THRESHOLD:
        text = voice.transcribe(audio).strip()
    else:
        text = ""
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
    agent = MemoryAgent(store, get_frame=(lambda: vision_memory.last_frame) if vision_memory is not None else None)
    owns_lamp = lamp is None
    if lamp is None:
        lamp = SimulatedLamp(mute=args.mute)

    if args.voice:
        voice.preload()

    speaker_detector = None
    if args.voice and not args.no_multi_user:
        # Embedded (--chat) passes vision_memory, so this shares main.py's
        # own camera frame instead of opening a second capture handle on
        # the same device -- see multi_face.py's module docstring for why
        # that used to be the likely cause of this silently not working
        # when run the normal way (main.py --chat --voice).
        get_frame = (lambda: vision_memory.last_frame) if vision_memory is not None else None
        speaker_detector = SpeakerDetector(get_frame=get_frame)
        if not speaker_detector.available:
            print("[chat] couldn't set up multi-user speaker detection (camera/model unavailable), "
                  "continuing without it")

    known = store.list_known_classes()
    print(f"[chat] LeLamp remembers {len(known)} object type(s): {', '.join(known) or '(nothing yet -- run main.py first)'}")
    if args.voice:
        print(f'[chat] Voice mode: say "{args.wake_word}" to wake it up and ask a question, '
              f'or "quit" any time to exit.\n')
    else:
        print("[chat] Ask about an object's location, or 'quit' to exit.\n")

    def handle(question: str, speaker_bearing: float | None = None, tone_label: str | None = None) -> None:
        if speaker_bearing is not None:
            # Quick glance toward whoever was actually talking, before
            # answering -- separate gesture from the "found it" point.
            _wait_out_attention_seek(fsm)
            glance = kinematics.pose_for_look_at(speaker_bearing, -10.0, alertness=0.6)
            lamp.set_target_pose(glance, duration=0.3, anticipation=True, overshoot=True)

        if tone_label in _REACTIVE_TONES:
            _flash_tone(lamp, tone_label, show_gui=not args.no_gui, standalone=standalone, fsm=fsm)

        try:
            reply = _strip_emoji(agent.ask(question))
        except Exception as e:
            # A failed API call (bad/unfunded key, network blip, rate
            # limit) used to take the whole process down here, which from
            # the outside just looks like "I said something and it never
            # answered." Surface it instead of dying mid-conversation.
            print(f"[chat] couldn't reach the LLM: {e}\n")
            if args.voice:
                voice.speak("Sorry, I couldn't reach the model just now.")
            return

        # Look FIRST, then tell you -- not the other way around. This used
        # to speak the reply (generated purely from the remembered
        # sighting) before ever checking or pointing at all, which read as
        # "it says it found it, THEN the arm moves" -- backwards from how
        # a person would actually answer ("let me check... yep, right
        # there"). The physical point also happens *before* the live
        # check now, not after: object detection is restricted to a crop
        # that follows wherever the lamp is aimed (vision_memory.py's
        # pan_zoom), so checking before moving there was checking blind,
        # no matter how long it waited.
        obs = agent.last_recall.observation
        if obs is not None:
            print(f"[chat] looking toward the last known spot for {obs.object_class}...")
            _point_toward(lamp, obs.bearing_deg, fsm=fsm)
            if args.voice:
                # Deliberately audible, not just a silent pause -- makes
                # the "let me actually go check" step something you can
                # hear happening instead of a gap that just reads as lag
                # between asking and getting an answer.
                voice.speak("Let me check.")
            # Repeatedly rescans for up to ~1.6s (see _find_live_bearing) --
            # a real "hold still and look for a couple seconds" window, not
            # a single roll of the dice that can land before the lamp/crop
            # has even finished settling on the new bearing.
            live_bearing = _find_live_bearing(vision_memory, obs.object_class)
            print(f"[chat] {'found it live' if live_bearing is not None else 'not currently visible'} "
                  f"-- {'confirming' if live_bearing is not None else 'going with the last known spot'}")
            _settle_from_point(lamp, show_gui=not args.no_gui, seconds=1.2, standalone=standalone,
                                confirmed=live_bearing is not None)
            # The status sentence is constructed here, deterministically,
            # instead of using the LLM's own reply -- that reply is always
            # composed from the remembered sighting alone (it has no way to
            # know the live-check result, which only just happened), so it
            # always reads in a "spotted X ago" / "still over there" past-
            # tense frame. Prefixing "it's right there" onto that used to
            # produce "it's right there -- spotted 40 seconds ago," which
            # contradicts itself; and the unprefixed miss case ("still off
            # to the left") read as claiming current presence when there
            # wasn't any. live_bearing is the one fact that actually
            # differs turn to turn, so it -- not the LLM -- decides the
            # tense.
            if live_bearing is not None:
                direction = bearing_to_direction(live_bearing)
                reply = f"It's right there, off to the {direction} -- I can see it now."
            else:
                direction = bearing_to_direction(obs.bearing_deg)
                ago = _seconds_ago_phrase(time.time() - obs.timestamp)
                reply = f"I don't see it right now, but it was last seen {direction}, about {ago} ago."

        print(f"LAMP: {reply}\n")
        if args.voice:
            voice.speak(reply)
        if agent.last_light_action is not None:
            _apply_light_command(lamp, agent.last_light_action, show_gui=not args.no_gui,
                                  standalone=standalone, fsm=fsm)

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
                    _look_attentive(lamp, None, fsm=fsm)
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
                _look_attentive(lamp, speaker_bearing, set_light=agent.last_light_action is None, fsm=fsm)
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
    p.add_argument("--no-multi-user", action="store_true",
                    help="with --voice: disable pointing toward whoever's actually talking "
                         "(via mouth movement) when more than one face is in frame")
    p.add_argument("--wake-word", type=str, default=config.WAKE_WORD,
                    help=f'with --voice: phrase that wakes it up to listen (default: "{config.WAKE_WORD}")')
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
