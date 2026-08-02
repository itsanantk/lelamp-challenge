# Demo recording script

Walks through all four steps from the challenge brief in one continuous
take — `main.py --chat --voice` now runs the live loop and the
conversational agent against the same lamp/window (see "Recall runs on a
background thread, sharing one lamp" in `docs/ARCHITECTURE.md`).

## Setup

- Plug in (battery throttling measurably slows detection, see README)
- Close anything else using the webcam/mic
- `python main.py --chat --voice --record demo.mp4` (writes video
  automatically, but has no audio — for the real take, screen-record
  instead so sound cues and narration come through, per README)

## One take, steps 1-4

1. **Engagement detection** — look at the lamp, then look away. Watch the
   HUD's `STATE:` line flip ENGAGED → DISENGAGED. Do this once cleanly on
   camera before anything else.
2. **Attention-seeking** — stay disengaged. After `ATTENTION_SEEK_DELAY_S`
   the lamp perks up and pulses (first attempt is subtle; if it takes a
   couple of tries before re-engaging, later attempts are visibly more
   pronounced — worth letting one run to a second attempt on camera).
   Re-engage to cancel it.
3. **Memory formation** — hold a couple of recognizable objects (phone,
   bottle, cup) in view for a few seconds each so a scan actually catches
   them. Check the HUD's "seen recently" list updates.
4. Optional: move the phone around while engaged — the lamp should track
   it continuously without breaking eye contact with you.
5. **Recall** — say **"hey lamp"**, wait for the chirp, then ask about
   one of the objects shown earlier ("where's my phone?"). Confirm the
   answer matches what was actually shown, and that the same lamp on
   screen turns to point at it — no window switch, no cut.
6. Ask a follow-up without saying the wake word again, to show the
   conversation staying open.
7. Say **"quit"** to leave voice mode, then `q` to end the recording.

## Notes

- If a take has a miss (phone not detected, wake word missed), it's
  cheaper to redo the whole take now that it's one continuous recording
  rather than trying to splice — the eval scripts don't care which take
  produced the demo video.
- `chat.py` still runs standalone (`python chat.py --voice`) if you want
  to test recall in isolation without the live camera loop running.
