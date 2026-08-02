"""Two-dial color model: every lamp color is `from_dials(warmth,
brightness)`, not an arbitrary named BGR constant.

warmth: 0.0 = cool white/blue, 1.0 = warm yellow/orange.
brightness: 0.0 = off, 1.0 = full.

These are the only two lighting parameters exposed to the user (voice
presets, "dim"/"on"). Everything else -- engaged look, object tracking,
attention-seeking, a tense/quiet tone reaction -- is expressed as a small
`nudge()` off whatever the current mood dial values are (see
SimulatedLamp.get_mood/set_mood), not a jump to an unrelated hue. That's
what actually fixes "warm light + tracks my phone -> aqua -> back to
white": a transient state used to pick its own independent color and
forget what was showing before it; nudging keeps every state visually
related to the one the user actually chose.
"""
from __future__ import annotations

import numpy as np

# BGR endpoints of the warmth spectrum.
_COOL_BGR = (255, 230, 200)  # bright white with a cool blue cast
_WARM_BGR = (25, 130, 255)   # deep warm orange


def from_dials(warmth: float, brightness: float) -> tuple[int, int, int]:
    warmth = float(np.clip(warmth, 0.0, 1.0))
    brightness = float(np.clip(brightness, 0.0, 1.0))
    base = tuple(c * (1 - warmth) + w * warmth for c, w in zip(_COOL_BGR, _WARM_BGR))
    return tuple(max(0, min(255, int(round(c * brightness)))) for c in base)


def nudge(warmth: float, brightness: float, dw: float = 0.0, db: float = 0.0) -> tuple[float, float]:
    """Offsets (warmth, brightness) by (dw, db) and clamps to [0, 1] --
    the shape every transient state (tracking, recall, a tone reaction)
    uses instead of replacing the mood with an unrelated absolute color."""
    return (float(np.clip(warmth + dw, 0.0, 1.0)), float(np.clip(brightness + db, 0.0, 1.0)))


# Voice-commanded mood presets (control_light tool). "dim"/"on" instead
# adjust the brightness dial relative to whatever's current -- see
# chat.py's _apply_light_command.
MOOD_PRESETS = {
    "cozy": (0.85, 0.55),   # warm amber-orange
    "focus": (0.05, 0.95),  # crisp bright white, slightly cool
    "calm": (0.55, 0.35),   # soft warm, dim
}
