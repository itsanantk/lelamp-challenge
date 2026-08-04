"""Unit tests for conversation/agent.py's tool-execution logic. Doesn't
construct a real MemoryAgent (its __init__ builds a live LLM client) --
uses __new__ to bypass that and set only what _execute_tool touches.
Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from conversation.agent import MemoryAgent, RecallResult, LIGHT_ACTIONS


def _agent():
    agent = MemoryAgent.__new__(MemoryAgent)
    agent.store = None
    agent.get_frame = None
    agent.apply_light = None
    agent.last_recall = RecallResult()
    agent.last_light_action = None
    agent.last_reminder_action = None
    return agent


def test_control_light_accepts_a_known_action():
    agent = _agent()
    result, image = agent._execute_tool("control_light", {"action": "cozy"})
    assert result == {"applied": True, "action": "cozy"}
    assert image is None
    assert agent.last_light_action == "cozy"


def test_control_light_rejects_an_unknown_action():
    agent = _agent()
    result, image = agent._execute_tool("control_light", {"action": "strobe"})
    assert result["applied"] is False
    assert agent.last_light_action is None


def test_control_light_is_case_insensitive():
    agent = _agent()
    agent._execute_tool("control_light", {"action": "COZY"})
    assert agent.last_light_action == "cozy"


def test_control_light_invokes_apply_light_immediately_when_set():
    calls = []
    agent = _agent()
    agent.apply_light = lambda action: calls.append(action)
    agent._execute_tool("control_light", {"action": "dim"})
    assert calls == ["dim"]


def test_control_light_does_not_apply_an_invalid_action():
    calls = []
    agent = _agent()
    agent.apply_light = lambda action: calls.append(action)
    agent._execute_tool("control_light", {"action": "strobe"})
    assert calls == []


def test_every_light_action_is_a_valid_string():
    assert all(isinstance(a, str) and a for a in LIGHT_ACTIONS)


def test_create_reminder_recurring_records_the_intended_action():
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {
        "action": "create", "kind": "recurring", "interval_minutes": 30, "message": "stand up",
    })
    assert result == {"created": True, "kind": "recurring"}
    assert image is None
    assert agent.last_reminder_action == {
        "action": "create", "kind": "recurring", "message": "stand up", "interval_s": 1800.0,
        "duration_s": None, "object_class": None, "due_in_s": None, "check_question": None,
    }


def test_create_reminder_with_a_duration_records_it_in_seconds():
    # "only check for the next 20 seconds" -- duration_minutes is optional
    # and independent of interval_minutes (which only recurring uses).
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {
        "action": "create", "kind": "presence", "message": "come back", "duration_minutes": 20 / 60,
    })
    assert result == {"created": True, "kind": "presence"}
    assert agent.last_reminder_action["duration_s"] == 20.0


def test_create_reminder_rejects_a_non_positive_duration():
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {
        "action": "create", "kind": "presence", "message": "come back", "duration_minutes": 0,
    })
    assert result["created"] is True  # duration_minutes=0 is falsy -- treated as "not given," not rejected
    assert agent.last_reminder_action["duration_s"] is None

    result, image = agent._execute_tool("create_reminder", {
        "action": "create", "kind": "presence", "message": "come back", "duration_minutes": -5,
    })
    assert result["created"] is False


def test_create_reminder_object_check_normalizes_the_object_name():
    # "water bottle" is a known alias for the actual tracked COCO class
    # "bottle" (see memory/store.py's CLASS_ALIASES) -- the tool should
    # resolve it server-side, the same normalize_class() recall_object_location
    # already relies on, not expect the LLM to know the exact vocabulary.
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {
        "action": "create", "kind": "object_check", "object_name": "water bottle",
        "deadline_minutes": 60, "message": "did you finish your water?",
    })
    assert result == {"created": True, "kind": "object_check"}
    assert agent.last_reminder_action["object_class"] == "bottle"
    assert agent.last_reminder_action["due_in_s"] == 3600.0
    assert agent.last_reminder_action["check_question"] is None  # not given -- omitted, not an error


def test_create_reminder_object_check_records_a_check_question_when_given():
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {
        "action": "create", "kind": "object_check", "object_name": "water bottle",
        "deadline_minutes": 60, "message": "did you finish your water?",
        "check_question": "is this water bottle full or empty",
    })
    assert result == {"created": True, "kind": "object_check"}
    assert agent.last_reminder_action["check_question"] == "is this water bottle full or empty"


def test_create_reminder_object_check_requires_an_object_name():
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {
        "action": "create", "kind": "object_check", "deadline_minutes": 60, "message": "x",
    })
    assert result["created"] is False
    assert agent.last_reminder_action is None


def test_create_reminder_object_check_requires_a_deadline():
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {
        "action": "create", "kind": "object_check", "object_name": "bottle", "message": "x",
    })
    assert result["created"] is False
    assert agent.last_reminder_action is None


def test_create_reminder_presence_does_not_require_an_interval():
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {
        "action": "create", "kind": "presence", "message": "come back and sit down",
    })
    assert result == {"created": True, "kind": "presence"}
    assert agent.last_reminder_action["interval_s"] is None


def test_create_reminder_recurring_without_an_interval_is_rejected():
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {
        "action": "create", "kind": "recurring", "message": "stand up",
    })
    assert result["created"] is False
    assert agent.last_reminder_action is None  # rejected -- nothing recorded for chat.py to apply


def test_create_reminder_rejects_an_unknown_kind():
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {"action": "create", "kind": "weather", "message": "x"})
    assert result["created"] is False
    assert agent.last_reminder_action is None


def test_create_reminder_requires_a_message():
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {"action": "create", "kind": "presence"})
    assert result["created"] is False
    assert agent.last_reminder_action is None


def test_create_reminder_cancel_defaults_to_every_kind():
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {"action": "cancel"})
    assert result == {"cancelled": True}
    assert agent.last_reminder_action == {"action": "cancel", "kind": None}


def test_create_reminder_cancel_can_target_one_kind():
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {"action": "cancel", "kind": "recurring"})
    assert agent.last_reminder_action == {"action": "cancel", "kind": "recurring"}


def test_create_reminder_rejects_an_unknown_action():
    agent = _agent()
    result, image = agent._execute_tool("create_reminder", {"action": "snooze"})
    assert "error" in result
    assert agent.last_reminder_action is None


class _FakeBlock:
    def __init__(self, type_, text=""):
        self.type = type_
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.stop_reason = "end_turn"
        self.content = [_FakeBlock("text", text)]


class _FakeMessages:
    def create(self, **kwargs):
        return _FakeResponse("just a plain reply, no tools needed")


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def test_ask_resets_last_light_action_and_last_recall_at_the_start_of_each_turn():
    # A turn that doesn't call control_light or recall_object_location
    # (small talk, or a turn that only calls the other tool) must not
    # leave stale state from a previous turn -- otherwise chat.py would
    # re-apply an old light command or re-point at an old object that had
    # nothing to do with the current question. Exercises the real ask()
    # path (via a fake LLM client that never calls a tool), not a
    # reimplementation of the reset logic.
    agent = _agent()
    agent.provider = "anthropic"
    agent.messages = []
    agent.client = _FakeClient()
    agent.last_light_action = "cozy"
    agent.last_recall.observation = object()
    agent.last_reminder_action = {"action": "create", "kind": "recurring", "message": "x", "interval_s": 60.0}

    agent.ask("just chatting, nothing about lights or objects")

    assert agent.last_light_action is None
    assert agent.last_recall.observation is None
    assert agent.last_reminder_action is None


def test_describe_current_view_unavailable_with_no_frame_provider():
    agent = _agent()  # get_frame=None -- standalone chat.py, no live camera loop
    result, image = agent._execute_tool("describe_current_view", {})
    assert result == {"available": False, "reason": "no live camera in this session"}
    assert image is None


def test_describe_current_view_unavailable_when_no_frame_captured_yet():
    agent = _agent()
    agent.get_frame = lambda: None
    result, image = agent._execute_tool("describe_current_view", {})
    assert result["available"] is False
    assert image is None


def test_describe_current_view_returns_jpeg_bytes_for_a_real_frame():
    agent = _agent()
    agent.get_frame = lambda: np.zeros((48, 64, 3), dtype=np.uint8)
    result, image = agent._execute_tool("describe_current_view", {})
    assert result == {"available": True}
    assert isinstance(image, bytes) and len(image) > 0
    assert image[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_judge_view_returns_none_with_no_frame_provider():
    agent = _agent()  # get_frame=None -- standalone chat.py, no live camera loop
    assert agent.judge_view("is this bottle full or empty") is None


def test_judge_view_returns_none_when_no_frame_captured_yet():
    agent = _agent()
    agent.get_frame = lambda: None
    assert agent.judge_view("is this bottle full or empty") is None


def test_judge_view_returns_the_models_answer():
    agent = _agent()
    agent.get_frame = lambda: np.zeros((48, 64, 3), dtype=np.uint8)
    agent.provider = "anthropic"
    agent.client = _FakeClient()  # _FakeMessages.create always answers "just a plain reply, no tools needed"

    answer = agent.judge_view("is this bottle full or empty")

    assert answer == "just a plain reply, no tools needed"


def test_judge_view_does_not_touch_the_conversation_history():
    # Deliberately not routed through ask()/self.messages -- a background
    # check the user didn't just ask about shouldn't show up as something
    # the model "remembers" saying in a later, unrelated question.
    agent = _agent()
    agent.get_frame = lambda: np.zeros((48, 64, 3), dtype=np.uint8)
    agent.provider = "anthropic"
    agent.client = _FakeClient()
    agent.messages = []

    agent.judge_view("is this bottle full or empty")

    assert agent.messages == []


def test_judge_view_returns_none_and_does_not_raise_if_the_api_call_fails():
    class _BoomMessages:
        def create(self, **kwargs):
            raise RuntimeError("network blip")

    agent = _agent()
    agent.get_frame = lambda: np.zeros((48, 64, 3), dtype=np.uint8)
    agent.provider = "anthropic"
    agent.client = type("C", (), {"messages": _BoomMessages()})()

    assert agent.judge_view("is this bottle full or empty") is None


class _FakeStore:
    def get_latest_by_class(self, object_class):
        assert isinstance(object_class, str), "must never reach the store as None"
        return None


def test_recall_with_an_explicit_null_object_name_does_not_crash():
    # .get("object_name", "") only falls back to "" when the key is
    # *missing* -- a tool call with an explicit {"object_name": null}
    # still passes None through, which used to crash inside
    # normalize_class (None.strip()).
    agent = _agent()
    agent.store = _FakeStore()
    result, image = agent._execute_tool("recall_object_location", {"object_name": None})
    assert result == {"found": False}
    assert image is None


class _FakeToolUseBlock:
    def __init__(self, name, tool_input, id_="toolu_test123"):
        self.type = "tool_use"
        self.name = name
        self.input = tool_input
        self.id = id_


class _SequencedMessages:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        return self._responses.pop(0)


class _SequencedClient:
    def __init__(self, responses):
        self.messages = _SequencedMessages(responses)


def test_a_tool_that_raises_still_gets_a_paired_tool_result():
    # Real bug, reproduced live: if _execute_tool raised, the code used to
    # skip straight past appending the matching tool_result, leaving that
    # turn's tool_use permanently unpaired in self.messages (an instance
    # attribute kept across turns). Every later call in the session then
    # resent that same broken history and failed with Anthropic's "tool_use
    # ids were found without tool_result blocks" 400 -- not just that one
    # turn, every turn after it too. An error result must keep the pairing
    # intact instead.
    agent = _agent()
    agent.provider = "anthropic"
    agent.messages = []
    tool_use = _FakeToolUseBlock("recall_object_location", {"object_name": "phone"})
    first_response = type("R", (), {"stop_reason": "tool_use", "content": [tool_use]})()
    second_response = _FakeResponse("here's what I found")
    agent.client = _SequencedClient([first_response, second_response])

    def _boom(name, tool_input):
        raise RuntimeError("simulated tool failure")

    agent._execute_tool = _boom

    reply = agent.ask("where's my phone?")

    assert reply == "here's what I found"
    tool_result_msg = agent.messages[2]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["tool_use_id"] == "toolu_test123"
    assert "error" in tool_result_msg["content"][0]["content"]


def test_a_tool_that_returns_an_unserializable_result_still_gets_a_paired_tool_result():
    # Same bug class, different trigger: _execute_tool succeeding but
    # returning something json.dumps can't handle used to raise *after*
    # the try/except (it only wrapped _execute_tool itself), skipping
    # results.append for that block and leaving its tool_use unpaired --
    # exactly what surfaced live as a 400 on a *later*, unrelated turn.
    agent = _agent()
    agent.provider = "anthropic"
    agent.messages = []
    tool_use = _FakeToolUseBlock("create_reminder", {"action": "create"})
    first_response = type("R", (), {"stop_reason": "tool_use", "content": [tool_use]})()
    second_response = _FakeResponse("done")
    agent.client = _SequencedClient([first_response, second_response])

    agent._execute_tool = lambda name, tool_input: ({"created": True, "bad": object()}, None)

    reply = agent.ask("remind me about something")

    assert reply == "done"
    tool_result_msg = agent.messages[2]
    assert tool_result_msg["content"][0]["tool_use_id"] == "toolu_test123"
    assert "error" in tool_result_msg["content"][0]["content"]


def test_control_light_applies_before_a_later_describe_current_view_in_the_same_turn():
    # Real bug, reported live: "turn on your light and check what it says"
    # -- describe_current_view used to always capture the pre-change frame,
    # because control_light only ever recorded last_light_action for
    # chat.py to apply *after* ask() returned, well after any
    # describe_current_view in the same turn had already run. apply_light
    # (see MemoryAgent.__init__) fixes this by applying immediately, inside
    # the tool loop itself -- this exercises the real two-tool-call loop,
    # not just each tool in isolation, to prove the ordering.
    agent = _agent()
    agent.provider = "anthropic"
    agent.messages = []
    calls = []
    agent.apply_light = lambda action: calls.append(("apply_light", action))
    agent.get_frame = lambda: calls.append(("get_frame",)) or np.zeros((4, 4, 3), dtype=np.uint8)
    light_block = _FakeToolUseBlock("control_light", {"action": "on"}, id_="toolu_light")
    describe_block = _FakeToolUseBlock("describe_current_view", {}, id_="toolu_describe")
    first_response = type("R", (), {"stop_reason": "tool_use", "content": [light_block, describe_block]})()
    second_response = _FakeResponse("light's on, here's what I see")
    agent.client = _SequencedClient([first_response, second_response])

    agent.ask("turn on your light and check what it says")

    assert [c[0] for c in calls] == ["apply_light", "get_frame"]
