# LeLamp Challenge — Anant Khanna

A 6-DOF lamp that tracks whether someone's looking at it, reacts with
motion/light/sound, keeps a memory of objects it's seen, and can answer
questions about that memory through an LLM. Full writeup with diagrams is
in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); a criteria-by-criteria
mapping to the challenge brief is in
[docs/requirements_traceability.md](docs/requirements_traceability.md);
quick-reference evaluation numbers (no prose to dig through) are in
[docs/EVALUATION.md](docs/EVALUATION.md).

I don't have a physical LeLamp, so the arm/light/speaker are simulated
behind a small hardware abstraction layer (`lamp/hal.py`). A real driver
would implement the same interface — see `lamp/real_backend.py` for what
that'd actually take (servo bus, LED bridge, speaker).

## Setup

```
pip install -r requirements.txt
```

You'll need model files in `models/` (already there locally, but
gitignored since they're big binaries — redownload on a fresh clone):

```
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -L -o models/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
curl -L -o models/en_US-amy-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
curl -L -o models/en_US-amy-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

The hand-landmarker model is only needed for hand-wave detection
(perception/hand_wave.py) — if it's missing, main.py prints a note and
carries on with that feature disabled (`--no-hand-wave` to skip it
outright). The Piper voice model is spoken-output only (conversation/voice.py)
— if it's missing, TTS falls back to the built-in, noticeably more robotic
SAPI5 voice instead of failing outright.

Object detection is YOLO11s (small), not the nano tier — noticeably
better hit rate for a small latency cost, see the tradeoffs section in
the architecture doc for the actual numbers and why the scan cadence
isn't higher than it is.

For step 4 (recall) you need either `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` set, with actual credit on the account. Set
`LELAMP_LLM_PROVIDER=openai` to use OpenAI instead of Anthropic if that's
the one that's funded.

## Running it

```
python main.py                        # steps 1-3, live window
python main.py --chat --voice         # + step 4, same window -- talk to it, mic in/spoken reply out
python main.py --chat                 # + step 4 as typed chat instead of voice
python main.py --chat --voice --no-multi-user   # skip pointing at whoever's actually talking
                                                  # (via mouth movement) when more than one face is in frame -- on by default
python main.py --record demo.mp4      # also saves the composite feed to recordings/
python main.py --label                # SPACE toggles ground truth, for eval later
```

**Keys** (interactive window, live):

| Key | Does |
|---|---|
| `q` | quit |
| `c` | clear stored memory, reset learned attention-seek timing, cancel all reminders |
| `p` | pause/resume — freezes the reactive/learning loop *and* wake-word/voice listening, and excludes frames from `--record`'s output, so you can get a camera set up without the lamp reacting to the fumbling. The lamp's own idle motion/breathing keeps running (it stays visually alive, just stops noticing anything new), and an already-open conversation is left to finish rather than cut off mid-reply |
| `m` | mute/unmute — silences wake-word listening and every spoken reply (conversation *and* reminder announcements), independent of `p`. Tracking/reminders/everything else keeps running; use this instead of `p` if the lamp should keep noticing you on camera but just not talk |
| `h` | show/hide the debug HUD panel — cuts the composite window's pixel count by about a third and skips its (relatively costly) redraw; useful while screen-recording, since an external recorder's own cost is driven by what's actually on screen, not anything this process can throttle in itself |
| `SPACE` | (with `--label`) toggle "I am currently looking at the lamp" ground truth |

`--chat` runs the conversational agent (`chat.py`) on a background thread
against the same lamp and window the live loop is already rendering —
one process, one lamp, everything (engagement, object tracking, voice
commands) actuating the same instance. `chat.py` still runs standalone
(`python chat.py`, `python chat.py --voice`) if you just want to query
whatever's already in memory without the camera loop running.

The window is two rows: webcam feed + simulated arm side by side on top
(engagement state + YOLO boxes drawn right on the feed), and a full-width
FSM/memory/debug panel below — captured and rendered at 1920x1080 (this
webcam's max), not upscaled from something smaller, so it stays sharp at
the larger window size. The webcam panel is mirrored for display (like
any normal selfie camera) so it doesn't read as backwards — all the
actual bearing/pointing math still runs on the original, un-mirrored
frame underneath, so where the lamp turns to point is unaffected. Asking
about a recalled object turns the same on-screen lamp to point at it,
with a bright spotlight-white light — no second window.

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
cut, dropping p95 loop latency roughly 30%. With `--chat --voice`,
expect a stutter in the first ~15s specifically — that's Whisper loading
on the conversation thread, competing for CPU with MediaPipe/YOLO on the
main one; it clears once the "[voice] ready" line prints.

**If it specifically gets slower the moment you start screen-recording**,
that's a separate issue from the above — OpenCV was defaulting to using
every logical core for its own per-frame work (resize, color conversion,
compositing), which left no real headroom for whatever's capturing the
screen. `main.py` now caps this to 4 threads at startup
(`cv2.setNumThreads(4)`), same reasoning as `vision_memory.py`'s existing
`torch.set_num_threads(4)`. If it's still slow with that in place, check
whether your recorder is using software (CPU) encoding rather than a
hardware encoder — that's the one remaining variable outside this app's
control. The `h` key (hide the debug HUD panel) also directly cuts what
the recorder has to capture and encode, independent of anything CPU-side.

### Bonus behavior

All four bonus items from the challenge doc are built and on by default;
each degrades gracefully (mic/camera unavailable, whatever) instead of
crashing the rest of the demo. See `docs/ARCHITECTURE.md` for the honest
scope of each — what they actually do versus a "real" version.

- **Interruption awareness** — a background mic-RMS gate
  (`perception/audio_monitor.py`) holds off attention-seeking while the
  room's making noise (you talking, TV on). "Busy" stays true for
  `AUDIO_GATE_SUSTAIN_S` (3.0s) after the last loud block, not just the
  instant, so a normal mid-sentence pause doesn't read as the room going
  quiet and re-arm attention-seeking a beat too early. `python main.py
  --no-audio-gate` turns it off.
- **Multi-user speaker detection** — on by default with `--chat --voice`,
  tracks up to 4 faces and picks whoever's mouth was actually moving
  during the question, so it glances at and points toward the right
  person when more than one face is in frame (`perception/multi_face.py`).
  Shares main.py's own camera frame rather than opening a second capture
  handle. `--no-multi-user` to disable.
- **Emotion from voice tone** — every voice turn gets a coarse read
  (yelling/energetic/tense/quiet/calm) from loudness, pitch variability,
  and speaking rate (`conversation/emotion.py`), and the lamp flashes a
  matching color before it answers. Loud enough (`config.
  VOICE_FLINCH_RMS_THRESHOLD`) gets a physical reaction too — a startled
  flinch (jerk-back + a sharp "startled" sound) followed by the same
  curious chirp used elsewhere for noticing something, fired live off raw
  mic loudness the instant it crosses the threshold, not after the
  sentence finishes recording and gets classified — and above that same
  loudness floor, the text classification itself skips straight to
  "tense" rather than still weighing speaking rate, so a loud, slow, or
  drawn-out raised voice doesn't get diluted back down to "calm" by an
  averaged score. An actual yell (a separate, much higher threshold,
  `config.VOICE_YELL_RMS_THRESHOLD`) gets its own "yelling" label. A
  quiet/subdued tone gets a soft whine + droop. Printed in the terminal
  too (`[voice] tone: ...`).
- **Self-learning attention-seeking** — `behavior/adaptation.py` tracks
  whether attention-seeking attempts actually get a response and nudges
  the delay/cooldown/max-tries within fixed bounds accordingly, persisted
  to `logs/adaptation_state.json` across runs. HUD shows the currently
  learned values. `python main.py --no-adapt` disables it,
  `--fresh-adaptation` wipes the learned state and starts over at
  startup; the `c` key does the same live, mid-session, alongside
  clearing scene memory.

### Self-initiated reminders

Not one of the four challenge bonuses, but built the same way -- ask it to
watch something and check back on its own, no further prompting needed
(`behavior/reminders.py`, requires `--chat --voice` or `--chat` since it's
created via conversation). Three kinds:

- **Recurring** — "make sure I get up every 30 minutes." Fires on a fixed
  interval, resets, repeats.
- **Presence** — "make sure I don't get up from my desk" / "tell me if I
  leave." Fires once presence has been lost for a sustained ~0.4s (not a
  single flickered frame -- raw per-frame face detection briefly drops out
  during the head motion of actually standing up), then re-arms once
  you're back. That debounce alone used to be 1.5s, long enough on its own
  to fully absorb the flicker risk, but that made a real departure feel
  slow to notice. Shortened it and split the job in two instead: a fast
  ~0.4s debounce confirms a departure quickly, and a separate 5s cooldown
  after any fire blocks a second one from a choppy
  return-then-drop-out-again mid-motion, without slowing down the first
  reaction.
- **Object check** — "make sure I drink my water by 6" / "check on my
  water bottle in an hour." Watches wherever the object (resolved through
  the same class aliasing `recall_object_location` uses, so "water
  bottle" -> the tracked `bottle` class) settles -- reusing
  `behavior/object_watch.py`'s own stillness thresholds for a consistent
  feel, but its own independent bookkeeping, since `ObjectWatcher` can
  only actively watch one object at a time and might be busy with
  something else. Once it's confirmed as set down, the lamp stops
  actively tracking it (it briefly joins `ObjectWatcher.tracked_classes`
  while awaiting placement, purely for the visible "the lamp noticed too"
  reaction) and just waits. At the deadline it points back at the
  remembered spot and checks in -- one-shot, not repeating.

  If the request implies judging the object's *state*, not just
  confirming it's there ("make sure I drink all my water" implies "is it
  empty yet"), the model attaches a `check_question` at creation time
  ("is this water bottle full or empty"). `behavior/reminders.py` has no
  LLM/API-key access on purpose (same perception-doesn't-know-an-LLM-
  exists separation as the rest of this codebase) -- it can't answer that
  itself, so at the deadline it just flags the reminder as due and leaves
  it alone. A background thread in `chat.py` (independent of the voice
  loop, which can be blocked inside a single mic listen for up to a
  minute) polls for exactly that, claims it, points the lamp, takes a
  fresh look, and asks the model a single one-shot vision question --
  deliberately not routed through the actual conversation history, since
  a background check the user didn't just ask about shouldn't show up as
  something the model "remembers" saying later. An object check with no
  `check_question` (just "check on my keys") skips all of that and fires
  directly, no LLM round trip needed.

  A `check_question` also means a disturbance is itself worth checking on,
  not just the deadline -- "make sure I drink all my water in the next
  hour" catches you finishing it early instead of staying silent for the
  full window. Once placement is confirmed, the same detections keep
  being watched for a sustained move away from the confirmed spot, or a
  sustained disappearance (debounced against a single bad detection the
  same way presence debounces a flickered face); either one dispatches
  the existing `check_question` early through the exact same path a real
  deadline would -- "is this still where it was left" reads correctly
  whether it's asked because time ran out or because something just
  moved. No `check_question` means there's nothing to judge early, so
  those still just wait for the deadline.

- **Object check, prohibition mode** ("make sure I don't go on my phone
  for five minutes," "don't let me touch my phone while I'm working") --
  a genuinely different mechanism from the disturbance-watch above, not
  a variant of it, set via `alert_on_detection` instead of
  `check_question`. A disturbance watch has a known baseline (the bottle
  sitting there) and reports a *change* from it; a prohibition has no
  baseline at all -- the object simply being detected is itself the
  violation, so there's no placement to confirm, no settle wait, no
  disturbance debounce, and no vision-LLM judgment call to make (YOLO's
  own detection already answers the only question there is). Fires
  directly the instant the object shows up in a scan while the reminder
  is active, one-shot, using `_fire()`'s own point-and-speak the same
  way a plain deadline-only object check does. This exists because the
  disturbance-watch model genuinely doesn't work for this case: something
  you're actively handling jitters in bearing between sightings just from
  being picked up, so the settle-wait's stability requirement could go
  unmet indefinitely and the watch would silently never activate --
  reported live as "the reminder just doesn't do anything."

Any kind can also be scoped to a limited window -- "only check for the
next 20 seconds," "just for the next hour" -- after which it deactivates
on its own instead of running until explicitly cancelled.

The HUD's reminders panel shows a live countdown ("due in 8s") or
"overdue by Xs" once an object check's placement is confirmed, and the
console prints two timestamped lines for a judged check -- "deadline
reached at HH:MM:SS" (from `behavior/reminders.py`'s tick) and "resolved
at HH:MM:SS" (from `chat.py`'s judgment poller) -- specifically so a
reminder that feels late is easy to attribute: was the deadline itself
late, or did the vision-LLM round trip + TTS after it take a few seconds
(the more common case)?

Firing plays a physical reaction (a wiggle, or a point at the remembered
spot for an object check), a sound cue, and speaks the message aloud on a
background thread (so firing one doesn't stutter the render loop for the
length of the TTS clip). "Stop reminding me"/"cancel that" cancels active
reminders through the same tool. Persisted to `logs/reminders.json` across
runs, same pattern as adaptation state; `--no-reminders` disables it,
`--fresh-reminders` wipes it at startup, and the `c` key cancels
everything active, live.

The current time is included in the LLM's context specifically so it can
resolve a deadline like "by 6pm" into minutes-from-now itself -- nothing
else in its context otherwise tells it what time it currently is.

### Recording the actual video

`main.py --record` writes an annotated composite video automatically as a
backup, but it has no audio. For the real submission I used Win+G to
screen-record instead, since that picks up the sound cues — `main.py
--chat --voice` runs all four steps in one continuous take now, no cut
between windows. Use `p` to pause while getting the camera/tripod
positioned (idle motion keeps running so the lamp doesn't look frozen on
camera, but it stops reacting to the setup itself), `h` to hide the debug
HUD panel for a cleaner frame and a lighter capture load, and `m` if you
want it to keep tracking/reacting visually without talking during a
specific segment.

## Evaluation

Real numbers, from a real labeled session (`session_20260804_134637.csv`,
1224 frames over 70.1s, a deliberate mixed pattern of quick glances and
longer stretches). Full methodology, discussion, and the separate voice/
recall-pipeline latency breakdown are in `docs/ARCHITECTURE.md` §4 — this
is the short version.

**Engagement detection reliability**

| | precision | recall | F1 | accuracy |
|---|---|---|---|---|
| raw (n=1224) | 0.851 | 0.856 | 0.853 | 0.880 |
| windowed, excludes ±500ms around a label change (n=1002) | 0.933 | 0.972 | **0.952** | 0.962 |

Flicker: 11.1 predicted-state changes/min. The raw/windowed gap is mostly
the hysteresis working as designed (see §4) — not noise being excluded to
flatter the number.

**Latency (p50 / p95 / p99 / max, ms)**

| Stage | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| Engagement (frame -> yaw/pitch) | 1224 | 13.8 | 17.1 | 19.2 | 23.9 |
| YOLO memory scan (when it runs) | 63 | 44.3 | 55.9 | 61.4 | 64.2 |
| Full per-frame loop | 1224 | 44.4 | 84.4 | 132.1 | 153.6 |

**End-to-end voice/recall latency** (measured directly against the real
local models + LLM API on this machine, not from the per-frame CSV log —
see §4 for why those need separate methodology):

| Stage | Latency |
|---|---|
| Wake-word chunk transcription (tiny.en) | ~155ms |
| Full-question transcription (small.en) | ~1.0s |
| LLM reply, no tool call | ~1.6s |
| LLM reply, tool-triggering turn | ~3.4s |
| Vision judgment (judge_view, downscaled frame) | ~2.5s |
| TTS synthesis (compute only) | 105-167ms |

Rough feel for a full voice round trip: **4-6 seconds**, wake-word to
reply-starts-playing, for a typical question.

To reproduce or re-run against a fresh session:

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
peak-RMS measurement fix. `tests/test_reminders.py` — recurring/presence/
object-check firing and edge-detection logic (including `alert_on_detection`),
cancellation, and the save/load round-trip.

That's a handful of highlights, not the full list -- there are 263 tests
across 18 files, one per module (`tests/test_agent.py`,
`tests/test_hand_wave.py`, `tests/test_ambient_light.py`,
`tests/test_scene_change.py`, `tests/test_chat_light.py`, and so on --
see Layout below for the complete module list each one covers).

## Layout

```
config.py                 tuning constants + paths
main.py                   camera -> engagement -> FSM -> lamp -> recording
chat.py                   conversational recall -- standalone, or embedded via main.py --chat
viz.py                    HUD overlay + display-only mirroring
audio_output.py           shared lock around sd.play() so chirps and TTS never talk over each other
conftest.py               puts the project root on sys.path for pytest

perception/engagement.py       MediaPipe head pose -> hysteresis
perception/vision_memory.py    interval-triggered YOLO scanning
perception/scene_change.py     frame-diff gate -- skips a YOLO scan when the scene hasn't visibly changed
perception/audio_monitor.py    mic-RMS gate for interruption awareness (bonus)
perception/multi_face.py       tracks faces, picks the active speaker (bonus)
perception/hand_wave.py        MediaPipe hand landmarker -- wave detection while ENGAGED
perception/ambient_light.py    samples frame luminance, nudges the lamp's own brightness

lamp/hal.py                abstract actuator interface
lamp/kinematics.py         6-DOF forward kinematics
lamp/motion.py             anticipation + overshoot easing
lamp/color.py              two-dial (warmth, brightness) color model
lamp/sim_backend.py        render + synthesized-chirp sound implementation
lamp/real_backend.py       stub for real servo/LED/speaker hardware

behavior/state_machine.py  IDLE / ENGAGED / DISENGAGED / ATTENTION_SEEKING
behavior/object_watch.py   continuously tracks a tracked object (phone) while visible
behavior/idle_scan.py      sweeps gaze across waypoints when nothing else has the lamp's attention
behavior/adaptation.py     bounded self-learning of attention-seek timing (bonus)
behavior/reminders.py      self-initiated timed/recurring/presence/object-check checks, created via conversation

memory/store.py            SQLite scene memory
conversation/agent.py      Claude/OpenAI tool-use agent over the memory store
conversation/voice.py      wake-word gated mic input (Whisper) + TTS output (Piper, SAPI5 fallback)
conversation/emotion.py    heuristic voice-tone read from raw audio (bonus)

eval/engagement_eval.py    precision/recall/F1/flicker-rate
eval/latency_eval.py       per-stage latency percentiles

tests/                     263 tests total, no camera/mic required -- one file per module above,
                            e.g. tests/test_reminders.py, tests/test_agent.py, tests/test_hand_wave.py
docs/ARCHITECTURE.md       full writeup
docs/requirements_traceability.md   challenge brief -> where it's addressed
docs/EVALUATION.md         quick-reference eval numbers, no prose
```
