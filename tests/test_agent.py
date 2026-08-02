"""Unit tests for conversation/agent.py's tool-execution logic. Doesn't
construct a real MemoryAgent (its __init__ builds a live LLM client) --
uses __new__ to bypass that and set only what _execute_tool touches.
Run with: python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conversation.agent import MemoryAgent, RecallResult, LIGHT_ACTIONS


def _agent():
    agent = MemoryAgent.__new__(MemoryAgent)
    agent.store = None
    agent.last_recall = RecallResult()
    agent.last_light_action = None
    return agent


def test_control_light_accepts_a_known_action():
    agent = _agent()
    result = agent._execute_tool("control_light", {"action": "cozy"})
    assert result == {"applied": True, "action": "cozy"}
    assert agent.last_light_action == "cozy"


def test_control_light_rejects_an_unknown_action():
    agent = _agent()
    result = agent._execute_tool("control_light", {"action": "strobe"})
    assert result["applied"] is False
    assert agent.last_light_action is None


def test_control_light_is_case_insensitive():
    agent = _agent()
    agent._execute_tool("control_light", {"action": "COZY"})
    assert agent.last_light_action == "cozy"


def test_every_light_action_is_a_valid_string():
    assert all(isinstance(a, str) and a for a in LIGHT_ACTIONS)


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

    agent.ask("just chatting, nothing about lights or objects")

    assert agent.last_light_action is None
    assert agent.last_recall.observation is None
