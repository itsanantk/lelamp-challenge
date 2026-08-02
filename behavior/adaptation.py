"""Bonus: a bounded, honest form of self-learning.

Not a black-box policy -- a small persisted history of "did an
attention-seek attempt get a response," which nudges three named
BehaviorFSM parameters (attention_seek_delay_s, attention_seek_max_attempts,
attention_seek_cooldown_s) within fixed bounds. Every adjustment traces
back to a countable statistic (response rate over the last 10 attempts),
not a learned weight nobody could explain. Persisted to
logs/adaptation_state.json across runs, which is what makes this actually
"improve over time" rather than resetting every session -- behavior on
session 10 reflects what happened in sessions 1-9.

What this deliberately doesn't do: no reward shaping, no exploration
strategy, no generalization across users or contexts. A real learned
policy needs a defined reward signal and enough state to generalize;
this optimizes exactly two things (how long to wait, how hard to try)
against exactly one signal (did trying work), which is proportionate to
what a desk lamp's attention-seeking behavior actually needs tuned.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import config

STATE_PATH = config.LOGS_DIR / "adaptation_state.json"

# Bounds: learning nudges within these, never past them.
MIN_DELAY_S, MAX_DELAY_S = 2.0, 8.0
MIN_ATTEMPTS, MAX_ATTEMPTS = 1, 5
MIN_COOLDOWN_S, MAX_COOLDOWN_S = 3.0, 10.0

RESPONSE_WINDOW_S = 20.0  # re-engaging within this long after an attempt counts as "it worked"
ROLLING_WINDOW = 10        # how many recent attempts the response rate is computed over
HISTORY_LIMIT = 50         # how much history gets persisted (rolling window only needs ROLLING_WINDOW,
                            # but keeping more around makes the eval/debug story richer)
MIN_SAMPLES_TO_ADAPT = 3   # don't start nudging until there's at least this many data points


@dataclass
class AdaptationEngine:
    delay_s: float = config.ATTENTION_SEEK_DELAY_S
    max_attempts: int = config.ATTENTION_SEEK_MAX_ATTEMPTS
    cooldown_s: float = config.ATTENTION_SEEK_COOLDOWN_S
    history: list = field(default_factory=list)  # [{"t": float, "responded": bool}, ...]
    _pending_attempt_t: float | None = None

    @classmethod
    def load(cls) -> "AdaptationEngine":
        if STATE_PATH.exists():
            try:
                data = json.loads(STATE_PATH.read_text())
                return cls(
                    delay_s=data.get("delay_s", config.ATTENTION_SEEK_DELAY_S),
                    max_attempts=data.get("max_attempts", config.ATTENTION_SEEK_MAX_ATTEMPTS),
                    cooldown_s=data.get("cooldown_s", config.ATTENTION_SEEK_COOLDOWN_S),
                    history=data.get("history", []),
                )
            except (json.JSONDecodeError, OSError):
                pass
        return cls()

    def save(self) -> None:
        STATE_PATH.write_text(json.dumps({
            "delay_s": self.delay_s,
            "max_attempts": self.max_attempts,
            "cooldown_s": self.cooldown_s,
            "history": self.history[-HISTORY_LIMIT:],
        }))

    # -- events, called from main.py as it observes FSM state transitions --

    def on_attention_seek_started(self, now: float | None = None) -> None:
        self._pending_attempt_t = now if now is not None else time.monotonic()

    def on_engaged(self, now: float | None = None) -> None:
        """Call whenever the FSM transitions into ENGAGED. If there's a
        pending attempt, this is either the response to it (re-engaged in
        time) or, if on_check_timeout already resolved it as a miss,
        a no-op."""
        now = now if now is not None else time.monotonic()
        if self._pending_attempt_t is not None:
            responded = (now - self._pending_attempt_t) <= RESPONSE_WINDOW_S
            self._record(now, responded)

    def on_check_timeout(self, now: float | None = None) -> None:
        """Call every tick while disengaged. If a pending attempt's
        response window has elapsed with no re-engagement, that's a miss."""
        now = now if now is not None else time.monotonic()
        if self._pending_attempt_t is not None and (now - self._pending_attempt_t) > RESPONSE_WINDOW_S:
            self._record(now, False)

    def _record(self, now: float, responded: bool) -> None:
        self.history.append({"t": now, "responded": responded})
        self.history = self.history[-HISTORY_LIMIT:]
        self._pending_attempt_t = None
        self._adapt()
        self.save()

    def _adapt(self) -> None:
        recent = self.history[-ROLLING_WINDOW:]
        if len(recent) < MIN_SAMPLES_TO_ADAPT:
            return
        rate = sum(1 for h in recent if h["responded"]) / len(recent)

        if rate >= 0.7:
            # Responds well to prompts -- try sooner, a little more often.
            self.delay_s = round(max(MIN_DELAY_S, self.delay_s - 0.3), 2)
            self.cooldown_s = round(max(MIN_COOLDOWN_S, self.cooldown_s - 0.2), 2)
            if len(recent) >= ROLLING_WINDOW:
                self.max_attempts = min(MAX_ATTEMPTS, self.max_attempts + 1)
        elif rate <= 0.2:
            # Tends to ignore it -- back off, both in timing and persistence.
            self.delay_s = round(min(MAX_DELAY_S, self.delay_s + 0.3), 2)
            self.cooldown_s = round(min(MAX_COOLDOWN_S, self.cooldown_s + 0.2), 2)
            if len(recent) >= ROLLING_WINDOW:
                self.max_attempts = max(MIN_ATTEMPTS, self.max_attempts - 1)

    def apply_to(self, fsm) -> None:
        fsm.attention_seek_delay_s = self.delay_s
        fsm.attention_seek_max_attempts = self.max_attempts
        fsm.attention_seek_cooldown_s = self.cooldown_s

    def summary(self) -> dict:
        recent = self.history[-ROLLING_WINDOW:]
        rate = round(sum(1 for h in recent if h["responded"]) / len(recent), 2) if recent else None
        return {
            "delay_s": self.delay_s, "max_attempts": self.max_attempts,
            "cooldown_s": self.cooldown_s, "recent_response_rate": rate,
            "samples": len(self.history),
        }
