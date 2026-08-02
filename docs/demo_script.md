# Demo recording script

Walks through the four steps from the challenge brief in order. Two
processes, one cut between them (see "why recall is a separate process"
in `docs/ARCHITECTURE.md`).

## Setup

- Plug in (battery throttling measurably slows detection, see README)
- Close anything else using the webcam/mic
- `python main.py --record demo.mp4` (writes video automatically, but has
  no audio — for the real take, screen-record instead so sound cues and
  narration come through, per README)

## Part 1: `main.py` (steps 1-3)

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

Stop recording, close `main.py` cleanly (`q`).

## Part 2: `chat.py --voice` (step 4)

1. Start it, wait for "ready".
2. Say **"hey lamp"**, wait for the chirp, then ask about one of the
   objects shown in part 1 ("where's my phone?"). Confirm the answer
   matches what was actually shown.
3. Ask a follow-up without saying the wake word again, to show the
   conversation window staying open.
4. Say **"quit"** to end cleanly.

## Notes for the cut

- Keep the two clips separate rather than trying to fake continuity —
  the brief's step 4 says "user *later* asks," so a cut is honest to the
  scenario, not a shortcut.
- If a take has a miss (phone not detected, wake word missed), it's
  cheaper to redo the specific part than the whole recording — the
  eval scripts don't care which take produced the demo video.
