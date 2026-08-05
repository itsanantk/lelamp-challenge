# Evaluation results

Quick-reference numbers only. Full methodology, discussion of what the
raw/windowed split means, and the reasoning behind each measurement are in
[ARCHITECTURE.md](ARCHITECTURE.md) §4 — this file exists so the actual
results aren't buried in prose.

All numbers below are from one real labeled session
(`logs/session_20260804_134637.csv`, 1224 frames over 70.1s, a deliberate
mixed pattern of quick glances and longer stretches, 41% of frames labeled
"looking"). Reproduce with:

```
python main.py --label     # run for a minute or two, toggling SPACE to match reality
python -m eval.engagement_eval     # precision/recall/F1/flicker-rate from that session
python -m eval.latency_eval        # p50/p95/p99 latency per stage, same session
```

## Engagement detection reliability

| | precision | recall | F1 | accuracy |
|---|---|---|---|---|
| raw (n=1224) | 0.851 | 0.856 | 0.853 | 0.880 |
| windowed, excludes ±500ms around a label change (n=1002) | 0.933 | 0.972 | **0.952** | 0.962 |

Flicker: 11.1 predicted-state changes/min.

The raw/windowed gap (0.853 -> 0.952 F1) is mostly the hysteresis working
as designed, not noise: excluding the 500ms around each real transition
removes exactly the frames where the dwell-frame requirement is *supposed*
to lag the label by a beat. See ARCHITECTURE.md §4 for the full argument.

## Per-frame pipeline latency (ms)

| Stage | n | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| Engagement (frame -> yaw/pitch) | 1224 | 13.8 | 17.1 | 19.2 | 23.9 |
| YOLO memory scan (when it runs) | 63 | 44.3 | 55.9 | 61.4 | 64.2 |
| Full per-frame loop | 1224 | 44.4 | 84.4 | 132.1 | 153.6 |

## End-to-end voice/recall latency

Measured directly against the real local models + LLM API on this
machine (not from the per-frame CSV log — `chat.py`/`conversation/voice.py`
run on their own thread, on their own cadence, so they're not in that log
at all; see ARCHITECTURE.md §4 for why these needed separate methodology).

| Stage | Latency |
|---|---|
| Wake-word chunk transcription (tiny.en, 1.8s chunk) | ~155ms (149-165ms) |
| Full-question transcription (small.en, ~3s of speech) | ~1030ms (1012-1048ms) |
| LLM reply, no tool call | ~1.6s |
| LLM reply, tool-triggering turn | ~3.4s (2.99-3.83s) |
| Vision judgment (judge_view, 1024px downscaled frame) | ~2.5s (2.08-2.99s) |
| TTS synthesis (compute only, not playback) | 105-167ms |
| Resulting audio playback duration | 2.0-4.4s (scales with reply length) |

Rough feel for a full voice round trip: **4-6 seconds**, wake-word to
reply-starts-playing, for a typical question; longer for anything needing
a vision judgment or a longer answer.

## Test suite

263 tests, no camera/mic required — `python -m pytest tests/ -v`.
