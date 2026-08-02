"""Conversational memory recall. An LLM answers questions about what the
lamp has seen by calling two tools that hit the SQLite memory store
directly -- it can't invent a location, only report what's actually in
memory/store.py.

Runs as its own process, separate from main.py's live camera loop. They
only share the SQLite file, not memory or threads. Partly because that's
literally how the demo is framed ("user *later* asks..."), and partly
because I didn't want a slow network round trip to an LLM anywhere near
the loop that engagement detection depends on for feeling responsive.

Supports either Anthropic or OpenAI as the backend (config.LLM_PROVIDER,
or set LELAMP_LLM_PROVIDER=openai) since API credit availability was the
actual bottleneck while building this, not architecture.
"""
from __future__ import annotations

import json
import time

from memory.store import MemoryStore, Observation, bearing_to_direction
import config

SYSTEM_PROMPT = """You are LeLamp, a small desk lamp robot with a camera and a memory \
of what it has recently seen on and around the desk. You're warm, brief, a little curious \
-- talk like a helpful companion, not a search engine.

Your only source of truth is the recall_object_location and list_seen_objects tools, which \
query your actual observation log. You have no other way to know where anything is.

If recall_object_location returns found=false, that object was NOT seen. Say that plainly \
and directly -- "I haven't spotted your X" -- as its own sentence, before anything else. \
Never phrase a miss as if it were a sighting ("I saw it near..."), and never use a *different* \
object's sighting (e.g. a person) as evidence for where the thing you were actually asked \
about is -- that's a guess wearing a sighting's clothes. Only after the plain "haven't seen \
it" may you separately mention, clearly marked as a different object, what you have seen via \
list_seen_objects, in case it's useful.

When you do report a real sighting (found=true), use the plain-language direction you're \
given (e.g. "off to the left") and roughly how long ago you saw it, not raw degrees or \
timestamps. Keep answers to 1-3 sentences.

seen_alongside lists other objects from that same camera scan, each with its own direction. \
Only call one of them "near" the main object if its direction actually matches (or is one \
step away, e.g. both "slightly left" and "center"); if a direction differs, describe it \
separately with its own direction instead of lumping it in as nearby -- being in the same \
wide shot is not the same as being close together."""

TOOL_SPECS = [
    {
        "name": "recall_object_location",
        "description": "Look up the most recent sighting of a named object in the lamp's "
                        "observation memory. Returns whether it was found, how long ago, "
                        "its rough direction, and what else was in the same camera scan -- "
                        "each of those with its own separate direction, since the camera's "
                        "field of view is wide enough that two objects in the same scan can "
                        "still be on opposite sides of it.",
        "params": {
            "object_name": {"type": "string", "description": "e.g. 'phone', 'cup', 'keys'"},
        },
        "required": ["object_name"],
    },
    {
        "name": "list_seen_objects",
        "description": "List every distinct object class the lamp has ever observed. Use "
                        "this for broad questions ('what have you seen?') or when a specific "
                        "lookup comes back empty and you want to suggest alternatives.",
        "params": {},
        "required": [],
    },
]


def _anthropic_tools() -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": {"type": "object", "properties": t["params"], "required": t["required"]},
        }
        for t in TOOL_SPECS
    ]


def _openai_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {"type": "object", "properties": t["params"], "required": t["required"]},
            },
        }
        for t in TOOL_SPECS
    ]


class RecallResult:
    """Set by the last successful recall_object_location call, so chat.py
    can point the lamp at it."""

    def __init__(self):
        self.observation: Observation | None = None
        self.direction_phrase: str | None = None


class MemoryAgent:
    def __init__(self, store: MemoryStore, provider: str | None = None):
        self.store = store
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.messages: list[dict] = []
        self.last_recall = RecallResult()

        if self.provider == "openai":
            import openai
            self.client = openai.OpenAI()
        else:
            import anthropic
            self.client = anthropic.Anthropic()

    def _execute_tool(self, name: str, tool_input: dict) -> dict:
        if name == "recall_object_location":
            obs = self.store.get_latest_by_class(tool_input.get("object_name", ""))
            if obs is None:
                self.last_recall = RecallResult()
                return {"found": False}
            direction = bearing_to_direction(obs.bearing_deg)
            seen_ago_s = max(0.0, time.time() - obs.timestamp)
            cooccurring = self.store.get_cooccurring(obs.frame_group_id, exclude_class=obs.object_class)
            self.last_recall = RecallResult()
            self.last_recall.observation = obs
            self.last_recall.direction_phrase = direction
            return {
                "found": True,
                "object_class": obs.object_class,
                "seen_ago_seconds": round(seen_ago_s, 1),
                "direction": direction,
                "confidence": round(obs.confidence, 2),
                "seen_alongside": [
                    {"object_class": cls, "direction": bearing_to_direction(bearing)}
                    for cls, bearing in cooccurring
                ],
            }
        if name == "list_seen_objects":
            return {"objects": self.store.list_known_classes()}
        return {"error": f"unknown tool {name}"}

    def ask(self, user_text: str) -> str:
        if self.provider == "openai":
            return self._ask_openai(user_text)
        return self._ask_anthropic(user_text)

    # -- Anthropic -----------------------------------------------------

    def _ask_anthropic(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        tools = _anthropic_tools()

        response = self.client.messages.create(
            model=config.ANTHROPIC_MODEL, max_tokens=512, system=SYSTEM_PROMPT,
            tools=tools, messages=self.messages,
        )
        while response.stop_reason == "tool_use":
            self.messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self._execute_tool(block.name, block.input)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
            self.messages.append({"role": "user", "content": results})
            response = self.client.messages.create(
                model=config.ANTHROPIC_MODEL, max_tokens=512, system=SYSTEM_PROMPT,
                tools=tools, messages=self.messages,
            )

        self.messages.append({"role": "assistant", "content": response.content})
        return "".join(b.text for b in response.content if b.type == "text")

    # -- OpenAI ----------------------------------------------------------

    def _ask_openai(self, user_text: str) -> str:
        if not self.messages:
            self.messages.append({"role": "system", "content": SYSTEM_PROMPT})
        self.messages.append({"role": "user", "content": user_text})
        tools = _openai_tools()

        response = self.client.chat.completions.create(
            model=config.OPENAI_MODEL, messages=self.messages, tools=tools, tool_choice="auto",
        )
        message = response.choices[0].message

        while message.tool_calls:
            self.messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = self._execute_tool(call.function.name, args)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL, messages=self.messages, tools=tools, tool_choice="auto",
            )
            message = response.choices[0].message

        self.messages.append({"role": "assistant", "content": message.content})
        return message.content or ""
