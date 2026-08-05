# Requirements traceability

Maps the challenge brief directly to where each part is addressed, so it's
checkable in one pass instead of hunting through the full writeup.

## Deliverables

| Deliverable | Where |
|---|---|
| System architecture diagram | `docs/ARCHITECTURE.md` §1 (mermaid) |
| Data flow (high/low-level) | `docs/ARCHITECTURE.md` §2 (per-frame loop + recall sequence diagrams) |
| Design decisions and tradeoffs | `docs/ARCHITECTURE.md` §3 |
| Demo video | see README "Recording the actual video" |
| Engagement detection reliability metrics | `eval/engagement_eval.py` — precision/recall/F1/flicker-rate |
| End-to-end latency measurements | `eval/latency_eval.py` — p50/p95/p99/max per stage |

## What we're looking for

| Criteria | Where |
|---|---|
| Architecture — well-structured, cleanly separated modules | `perception/` only produces readings, `behavior/` only talks to the `LampActuator` interface, `memory/store.py` is the single read/write point, `conversation/` only reasons over tool results. See `docs/ARCHITECTURE.md` §1 for the boundaries and why. |
| Perception pipeline — reliable real-time engagement detection | `perception/engagement.py`: MediaPipe head pose, two-threshold hysteresis (deadband + dwell frames) against single-cutoff flicker. Measured in `eval/engagement_eval.py`. |
| Behavior design — expressive and natural reactions | `lamp/motion.py`: anticipation + overshoot easing, randomized per-move so nothing repeats identically. `behavior/state_machine.py`: attention-seeking escalates in intensity across attempts instead of repeating the same gesture. `lamp/sim_backend.py`: synthesized sound cues, randomized per play. |
| Memory system — accurate store/retrieve | `memory/store.py` (SQLite) + `conversation/agent.py` tool-use: the LLM can only answer from `recall_object_location`/`list_seen_objects` results, so it can't fabricate a location not actually in the store. |
| Tradeoff reasoning | `docs/ARCHITECTURE.md` §3 — each design decision states the alternative considered and why it lost (e.g. head pose vs. iris tracking, discrete-burst vs. continuous object tracking). |
| Code quality | `tests/` (263 tests, no camera/mic required), type-hinted function signatures throughout, `lamp/hal.py` as the actuator boundary so `lamp/real_backend.py` is a drop-in swap. |

## Bonus challenges

| Bonus | Where |
|---|---|
| Multi-user interaction (who's speaking) | `perception/multi_face.py` — mouth-openness variance across tracked faces, on by default with `chat.py --voice` (`--no-multi-user` to skip it) |
| Emotion detection from voice tone | `conversation/emotion.py` — loudness/pitch-variability/speaking-rate heuristic |
| Self-learning behavior | `behavior/adaptation.py` — bounded adjustment of attention-seek delay/cooldown/max-attempts based on response rate |
| Interruption awareness | `perception/audio_monitor.py` — energy-based mic gate holds off attention-seeking while the room's active |

## Extra features (beyond the brief)

Not asked for by the challenge doc, built the same way as everything above
— same perception/behavior/conversation boundaries, same test discipline.
See README "Self-initiated reminders" for the full behavior writeup.

| Feature | Where |
|---|---|
| Recurring reminders ("every 30 minutes") | `behavior/reminders.py` — fixed-interval, ticked from `main.py`'s per-frame loop, persisted to `logs/reminders.json` |
| Presence reminders ("tell me if I leave") | `behavior/reminders.py` — debounced absence detection (`PRESENCE_ABSENCE_DEBOUNCE_S`) against raw per-frame `face_found` so a standup doesn't multi-fire, re-arms on return |
| Object-check reminders ("check my water bottle in an hour") | `behavior/reminders.py` watches the object settle (reuses `behavior/object_watch.py`'s stillness thresholds) and points back at it at the deadline; if the request implies judging the object's *state* rather than just its presence, `conversation/agent.py`'s `judge_view()` takes a one-shot vision-LLM look, kept out of the real conversation history on purpose |
| Prohibition reminders ("don't let me touch my phone") | `behavior/reminders.py`'s `alert_on_detection` mode — a genuinely different mechanism from the disturbance-watch above, not a variant of it: no baseline placement, no settle wait, no vision-LLM judgment, fires directly the instant the object is seen at all while the reminder is active |
| Time-scoped reminders ("just for the next 20 seconds") | `Reminder.expires_at` — any kind can auto-deactivate after a stated window instead of running until cancelled |
| Creation/cancellation via conversation | `conversation/agent.py`'s `create_reminder` tool → `chat.py`'s `_apply_reminder_action`, mirroring the existing light-command tool/apply split |
| Recording-friendly controls | `main.py`'s live `p` (pause reactive loop + voice listening), `m` (mute voice output only), and `h` (hide the debug HUD panel) keys — none asked for by the brief, added specifically to make clean demo recording possible |
