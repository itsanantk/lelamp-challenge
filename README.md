# LeLamp Challenge — Anant Khanna

A 6-DOF lamp that tracks whether someone's looking at it, reacts with
motion/light/sound, keeps a memory of objects it's seen, and can answer
questions about that memory through an LLM. Full writeup with diagrams is
in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

I don't have a physical LeLamp, so the arm/light/speaker are simulated
behind a small hardware abstraction layer (`lamp/hal.py`). A real driver
would implement the same interface — see `lamp/real_backend.py` for what
that'd actually take (servo bus, LED bridge, speaker).

## Setup

```
pip install -r requirements.txt
```

You'll need two model files in `models/` (already there locally, but
gitignored since they're big binaries — redownload on a fresh clone):

```
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -L -o models/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt
```

Object detection is YOLO11s (small), not the nano tier — noticeably
better hit rate for a small latency cost, see the tradeoffs section in
the architecture doc for the actual numbers and why the scan cadence
isn't higher than it is.

For `chat.py` (step 4) you need either `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` set, with actual credit on the account. Set
`LELAMP_LLM_PROVIDER=openai` to use OpenAI instead of Anthropic if that's
the one that's funded.

## Running it

```
python main.py                        # steps 1-3, live window
python main.py --record demo.mp4      # also saves the composite feed to recordings/
python main.py --label                # SPACE toggles ground truth, for eval later
```

Then, once there's something in memory, in a second terminal:

```
python chat.py                        # step 4 — ask what the lamp remembers
python chat.py --voice                # talk to it instead — mic in, spoken reply out
python chat.py --voice --multi-user   # + figure out who's talking if more than one face is in frame
```

The window is two rows: webcam feed + simulated arm side by side on top
(engagement state + YOLO boxes drawn right on the feed), and a full-width
FSM/memory/debug panel below — captured and rendered at 1920x1080 (this
webcam's max), not upscaled from something smaller, so it stays sharp at
the larger window size. The webcam panel is mirrored for display (like
any normal selfie camera) so it doesn't read as backwards — all the
actual bearing/pointing math still runs on the original, un-mirrored
frame underneath, so where the lamp turns to point is unaffected.
`chat.py` pops a short second window where the lamp turns to point at
whatever it recalled, with a bright spotlight-white light on it.

Sound cues (the lamp's own chirps, `lamp/sim_backend.py`) are confirmed
audible on this machine's built-in speakers. **They were not reliably
audible over Bluetooth headphones/earbuds in testing** — short sounds
with any internal silence went completely silent on that output, even
after several fixes; only one continuous gapless sweep came through, and
faintly. Use `--mute` if you're on Bluetooth output and don't want to
rely on them, or plug in / switch to wired output.

Voice mode runs speech-to-text locally (Whisper) so it works no matter
which LLM key you've got funded — first run downloads the model
(~460MB, small.en -- bumped up from base.en for accuracy; ~45s to
download once, ~5s to load on later runs).
It's wake-word gated, not a fixed listen window: it sits quiet until you
say **"hey lamp"**, chirps once, then records until you stop talking
(trailing silence, not a countdown) — say "quit" any time to leave a
voice session (you don't have to wake it up first just to quit).
Matching is deliberately forgiving: only the distinctive word ("lamp")
actually has to land, and it's checked against the last two things heard
concatenated, so the phrase getting split across a chunk boundary still
counts. Change the phrase with `--wake-word "..."` if you want something
else. The volume gate uses peak RMS over short subframes rather than one
average over the whole chunk, so a brief word in an otherwise-quiet
window doesn't get diluted below the threshold. After it answers, it
keeps listening for about a minute (`CONVERSATION_FOLLOWUP_TIMEOUT_S`) so
you can ask a follow-up without saying "hey lamp" again — the lamp holds
an alert pose/color the whole time so it visibly looks like it's still
paying attention, and drops back to idle if nothing's said.

**If nothing seems to happen when you talk to it**, the terminal is the
first place to look — voice mode prints, per chunk: `[voice] rms=0.0031`
(too quiet to bother transcribing) or `[voice] rms=0.041 heard: "..."`
(what Whisper actually transcribed). It also prints which microphone it's
recording from once at startup. Reading those tells you which of three
things is actually wrong instead of guessing:
- **RMS numbers never get anywhere near the gate** — it's very likely
  recording from the wrong microphone (check the printed device name; a
  paired-but-not-selected Bluetooth headset is a common cause) or the
  gate is still miscalibrated for your setup. Lower
  `VOICE_GATE_RMS_THRESHOLD` in `config.py` (separate from
  `AUDIO_GATE_RMS_THRESHOLD`, which is for the interruption-awareness
  feature, not voice commands) if the device is right but the numbers
  are still low.
- **RMS crosses the gate but the transcript is never close to "lamp"** —
  that's Whisper mishearing you, not the volume gate. Try `--wake-word`
  with a different, more distinctive word.
- **The transcript is right but it still doesn't wake up** — that would
  be an actual bug; worth reporting with the exact printed line.

### Watching your phone

If a phone shows up in frame, `main.py` bumps the object-detection rate
up and has the lamp track it continuously while it's visible — not just
a one-time glance. Move it around and the lamp keeps following; put it
down and it keeps watching; pick it back up and look at it and it keeps
tracking, **including while you're engaged with the lamp** (looking at
it) — a person doesn't fully block phone-tracking, only the short
attention-seeking animation does, since that's the one moment the lamp
is mid-gesture and would visibly glitch if interrupted. It only settles
back once the phone hasn't been seen for a couple seconds
(`WATCH_LOST_GRACE_S`), which is also what rides out a bad-angle miss or
two without the gesture flickering off and restarting every time a
single scan comes back empty — and if that happens while you're still
engaged, it returns to looking at you rather than dropping to a neutral
idle pose, since the lamp shouldn't look like it stopped paying attention
to you just because your phone left frame. The phone also gets its own,
more lenient detection confidence bar (`TRACKED_CLASS_CONF_THRESHOLD`)
than the rest of scene memory, since an off-angle phone is exactly the
kind of detection that scores just under the general threshold. Nothing
extra needed for the memory part — every sighting still gets logged the
normal way, so the last place it saw the phone is just whatever
`chat.py` reports back later, spotlight and all.

**If detection still feels slow or laggy in general**, check whether
you're on battery power — this machine's CPU gets meaningfully
throttled on battery under Windows' Balanced power plan (measured
~2x+ slower YOLO scans than on AC in testing), which shows up as
sluggishness across the whole app, not just object detection. Plug in
before judging responsiveness. Separately, at 1920x1080 the per-frame HUD
compositing (mirror + overlays + lamp render + debug panel) got real
enough to matter — the debug text panel now redraws every 4th frame
instead of every frame, and a couple of redundant full-frame copies were
cut, dropping p95 loop latency roughly 30%.

### Bonus behavior

All four bonus items from the challenge doc are built and on by default;
each degrades gracefully (mic/camera unavailable, whatever) instead of
crashing the rest of the demo. See `docs/ARCHITECTURE.md` for the honest
scope of each — what they actually do versus a "real" version.

- **Interruption awareness** — a background mic-RMS gate
  (`perception/audio_monitor.py`) holds off attention-seeking while the
  room's making noise (you talking, TV on). `python main.py --no-audio-gate`
  turns it off.
- **Multi-user speaker ID** — `chat.py --voice --multi-user` tracks up to
  4 faces and picks whoever's mouth was actually moving during the
  question, so it glances at and reasons about the right person when more
  than one face is in frame (`perception/multi_face.py`).
- **Emotion from voice tone** — every voice turn gets a coarse read
  (energetic/tense/quiet/calm) from loudness, pitch variability, and
  speaking rate (`conversation/emotion.py`), and the lamp flashes a
  matching color before it answers. Printed in the terminal too
  (`[voice] tone: ...`).
- **Self-learning attention-seeking** — `behavior/adaptation.py` tracks
  whether attention-seeking attempts actually get a response and nudges
  the delay/cooldown/max-tries within fixed bounds accordingly, persisted
  to `logs/adaptation_state.json` across runs. HUD shows the currently
  learned values. `python main.py --no-adapt` disables it,
  `--fresh-adaptation` wipes the learned state and starts over.

### Recording the actual video

`main.py --record` writes an annotated composite video automatically as a
backup, but it has no audio. For the real submission I used Win+G to
screen-record instead, since that picks up the sound cues — ran `main.py`
for steps 1-3, then `chat.py` for step 4 in the same take (there's a cut
between the two windows, see the "why recall is a separate session" bit
in the architecture doc).

## Evaluation

```
python main.py --label     # run for a minute or two, toggling SPACE to match reality
python -m eval.engagement_eval     # precision/recall/F1/flicker-rate from that session
python -m eval.latency_eval        # p50/p95/p99 latency per stage, same session
```

Both scripts grab the most recent `logs/session_*.csv` by default, or
take a path as an argument.

## Tests

```
python -m pytest tests/ -v             # everything, no camera/mic needed
```

`tests/test_state_machine.py` — FSM transition logic. `tests/test_adaptation.py`
— self-learning engine, isolated and end-to-end against the real FSM.
`tests/test_object_watch.py` — phone-watch gesture, including that it
correctly backs off and doesn't stomp the FSM's own pose/light when a
person shows up mid-watch. `tests/test_viz.py` — display-only mirroring.
`tests/test_voice.py` — wake-word matching and RMS gating logic (the
mic-hardware pieces are live-verified instead, same as
`perception/audio_monitor.py` — see the architecture doc).
`tests/test_emotion.py` — tone classification thresholds and the
peak-RMS measurement fix.

## Layout

```
config.py                 tuning constants + paths
main.py                   camera -> engagement -> FSM -> lamp -> recording
chat.py                   conversational recall, separate session
viz.py                    HUD overlay + display-only mirroring

perception/engagement.py       MediaPipe head pose -> hysteresis
perception/vision_memory.py    interval-triggered YOLO scanning
perception/audio_monitor.py    mic-RMS gate for interruption awareness (bonus)
perception/multi_face.py       tracks faces, picks the active speaker (bonus)

lamp/hal.py                abstract actuator interface
lamp/kinematics.py         6-DOF forward kinematics
lamp/motion.py             anticipation + overshoot easing
lamp/sim_backend.py        render + synthesized-chirp sound implementation
lamp/real_backend.py       stub for real servo/LED/speaker hardware

behavior/state_machine.py  IDLE / ENGAGED / DISENGAGED / ATTENTION_SEEKING
behavior/object_watch.py   continuously tracks a tracked object (phone) while visible
behavior/adaptation.py     bounded self-learning of attention-seek timing (bonus)

memory/store.py            SQLite scene memory
conversation/agent.py      Claude/OpenAI tool-use agent over the memory store
conversation/voice.py      wake-word gated mic input (Whisper) + TTS output
conversation/emotion.py    heuristic voice-tone read from raw audio (bonus)

eval/engagement_eval.py    precision/recall/F1/flicker-rate
eval/latency_eval.py       per-stage latency percentiles

tests/test_state_machine.py
tests/test_adaptation.py
tests/test_object_watch.py
tests/test_viz.py
tests/test_voice.py
tests/test_emotion.py
docs/ARCHITECTURE.md       full writeup
```
