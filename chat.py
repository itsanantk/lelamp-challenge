"""Ask the lamp what it remembers.

Separate process from main.py (see conversation/agent.py), reading the
same memory database main.py wrote to. When a location is recalled, the
simulated lamp turns to physically point at the remembered bearing and
lights up there instead of just answering in text.

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
from behavior.state_machine import ENGAGED_COLOR
from lamp import SimulatedLamp, kinematics
from memory.store import MemoryStore
from conversation.agent import MemoryAgent
from conversation import emotion, voice
from perception.multi_face import SpeakerDetector

FOUND_COLOR = (210, 235, 255)  # BGR, bright warm spotlight -- distinct from
                                 # the orange attention-seek pulse and the
                                 # green object-watch tint
IDLE_COLOR = (40, 110, 200)


def _look_attentive(lamp: SimulatedLamp, bearing_deg: float | None) -> None:
    """Held during the post-wake conversation window so the lamp visibly
    looks like it's still listening, not just idling until you say the
    wake word again."""
    pose = kinematics.pose_for_look_at(bearing_deg if bearing_deg is not None else 0.0, -10.0, alertness=0.75)
    lamp.set_target_pose(pose, duration=0.4, anticipation=False, overshoot=False)
    lamp.set_light(ENGAGED_COLOR, transition_s=0.3)


def _relax_to_idle(lamp: SimulatedLamp) -> None:
    lamp.set_light(IDLE_COLOR, transition_s=0.6)
    lamp.set_target_pose(kinematics.NEUTRAL_POSE, duration=0.6, overshoot=False)


def _point_and_render(lamp: SimulatedLamp, bearing_deg: float, show_gui: bool, seconds: float = 2.0) -> None:
    target = kinematics.pose_for_look_at(bearing_deg, -10.0, alertness=0.85)
    lamp.set_target_pose(target, duration=0.5, anticipation=True, overshoot=True)
    lamp.set_light(FOUND_COLOR, transition_s=0.3)
    lamp.play_sound("recall_point")

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

    lamp.set_light(IDLE_COLOR, transition_s=0.6)
    lamp.set_target_pose(kinematics.NEUTRAL_POSE, duration=0.6, overshoot=False)


def _flash_tone(lamp: SimulatedLamp, label: str, show_gui: bool, seconds: float = 0.6) -> None:
    lamp.set_light(emotion.BGR_BY_LABEL.get(label, (180, 220, 180)), transition_s=0.2)
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
    lamp.set_light((40, 110, 200), transition_s=0.4)


def _listen(speaker_detector: SpeakerDetector | None, max_s: float | None = None,
            no_speech_timeout_s: float | None = None) -> tuple[str, float | None, int, np.ndarray]:
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
    stale bearing before the mic side is done waiting. Returns (transcript,
    speaker bearing or None, faces seen, raw audio -- the last one for
    emotion.analyze)."""
    kwargs = {}
    if max_s is not None:
        kwargs["max_s"] = max_s
    if no_speech_timeout_s is not None:
        kwargs["no_speech_timeout_s"] = no_speech_timeout_s
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


def run(args: argparse.Namespace) -> None:
    if not args.no_gui:
        viz.enable_dpi_awareness()  # must happen before any window gets created -- see viz.py
    store = MemoryStore()
    agent = MemoryAgent(store)
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

    def handle(question: str, speaker_bearing: float | None = None, tone_label: str | None = None) -> None:
        if speaker_bearing is not None:
            # Quick glance toward whoever was actually talking, before
            # answering -- separate gesture from the "found it" point.
            glance = kinematics.pose_for_look_at(speaker_bearing, -10.0, alertness=0.6)
            lamp.set_target_pose(glance, duration=0.3, anticipation=True, overshoot=True)

        if tone_label is not None:
            _flash_tone(lamp, tone_label, show_gui=not args.no_gui)

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
        obs = agent.last_recall.observation
        if obs is not None:
            _point_and_render(lamp, obs.bearing_deg, show_gui=not args.no_gui)

    if args.ask:
        handle(args.ask)
        lamp.close()
        store.close()
        if speaker_detector is not None:
            speaker_detector.close()
        return

    try:
        in_conversation = False  # True while listening for a follow-up without needing the wake word again
        while True:
            speaker_bearing = None
            tone_label = None
            if args.voice:
                if not in_conversation:
                    woken = voice.wait_for_wake_word(args.wake_word)
                    if woken == "quit":
                        break
                    lamp.play_sound("wake")
                    time.sleep(0.2)  # let the chirp clear the speaker before the mic opens -- otherwise its tail can leak into the recording and falsely trip the silence detector
                    _look_attentive(lamp, None)
                    print("[voice] listening...")
                    question, speaker_bearing, n_faces, audio = _listen(speaker_detector)
                else:
                    print("[voice] still listening for a follow-up...")
                    question, speaker_bearing, n_faces, audio = _listen(
                        speaker_detector,
                        max_s=config.CONVERSATION_FOLLOWUP_TIMEOUT_S,
                        no_speech_timeout_s=config.CONVERSATION_FOLLOWUP_TIMEOUT_S)

                if n_faces > 1:
                    print(f"[multi-user] {n_faces} faces in frame, "
                          f"responding to whoever's mouth was moving (bearing {speaker_bearing:.0f}°)")
                if not question:
                    if in_conversation:
                        print("[voice] no follow-up heard, going back to sleep\n")
                        in_conversation = False
                        _relax_to_idle(lamp)
                    else:
                        print("[voice] didn't catch that, try again\n")
                    continue
                tone = emotion.analyze(audio)
                tone_label = tone.label
                print(f"[voice] tone: {tone.label}")
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
                _look_attentive(lamp, speaker_bearing)
    finally:
        lamp.close()
        store.close()
        if speaker_detector is not None:
            speaker_detector.close()
        if not args.no_gui:
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
