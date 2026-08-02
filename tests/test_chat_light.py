"""Unit tests for chat.py's voice-controlled lighting. Run with:
python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chat


class _FakeLamp:
    def __init__(self, current_light=(100, 100, 100)):
        self.calls = []
        self._light = current_light

    def set_light(self, rgb, transition_s=0.4):
        self.calls.append(rgb)
        self._light = rgb

    def get_current_light(self):
        return self._light

    def update(self, dt):
        pass

    def render(self):
        return None


def test_preset_actions_use_their_exact_defined_color():
    lamp = _FakeLamp()
    chat._apply_light_command(lamp, "cozy", show_gui=False, seconds=0.0)
    assert lamp.calls[-1] == chat.LIGHT_PRESETS["cozy"]


def test_off_is_fully_dark():
    lamp = _FakeLamp()
    chat._apply_light_command(lamp, "off", show_gui=False, seconds=0.0)
    assert lamp.calls[-1] == (0, 0, 0)


def test_dim_scales_down_from_whatever_is_currently_showing():
    lamp = _FakeLamp(current_light=(100, 100, 100))
    chat._apply_light_command(lamp, "dim", show_gui=False, seconds=0.0)
    dimmed = lamp.calls[-1]
    assert all(c < 100 for c in dimmed)
    assert all(c > 0 for c in dimmed)


def test_repeated_dim_never_reaches_pure_black():
    # Repeated "dim" should settle near-off, not hit literal (0,0,0) --
    # that's what "off" is explicitly for, dim should stay a distinct action.
    lamp = _FakeLamp(current_light=(100, 100, 100))
    for _ in range(20):
        chat._apply_light_command(lamp, "dim", show_gui=False, seconds=0.0)
    assert all(c >= chat._DIM_FLOOR for c in lamp.calls[-1])
    assert lamp.calls[-1] != (0, 0, 0)


def test_render_loop_runs_even_without_a_gui_window():
    # show_gui=False should still tick the lamp's own animation forward
    # (update()), just skip the cv2 display calls -- confirms the render
    # loop itself doesn't silently no-op when there's no window.
    lamp = _FakeLamp()
    ticked = []
    lamp.update = lambda dt: ticked.append(dt)
    chat._apply_light_command(lamp, "cozy", show_gui=False, seconds=0.02)
    assert len(ticked) > 0
