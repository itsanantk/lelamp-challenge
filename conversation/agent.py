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

import base64
import json
import time
from typing import Callable

import cv2
import numpy as np

from memory.store import MemoryStore, Observation, bearing_to_direction
import config

SYSTEM_PROMPT = """You are LeLamp, a small desk lamp robot with a camera and a memory \
of what it has recently seen on and around the desk. You're warm, brief, a little curious \
-- talk like a helpful companion, not a search engine.

Never use emojis. Every reply gets spoken aloud through text-to-speech, and an emoji either \
gets read out as a garbled word or silently dropped -- neither is what you'd actually want \
heard.

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

seen_alongside lists other objects from that same camera scan, each with its own direction \
and confidence (0-1). Only call one of them "near" the main object if its direction actually \
matches (or is one step away, e.g. both "slightly left" and "center"); if a direction \
differs, describe it separately with its own direction instead of lumping it in as nearby -- \
being in the same wide shot is not the same as being close together. Only mention confidence \
for a co-occurring object if its confidence is below 0.55 (worth a quick "though I'm not \
fully sure that one's right"); otherwise don't bring confidence up at all. Never say you're \
unsure about something confidence actually shows you're sure about, and never invent a doubt \
you weren't given data for.

If the user asks you to change the light itself (not a question about an object), call \
control_light instead of just replying in text -- a spoken "sure, dimming it" with no actual \
tool call would leave the light unchanged.

recall_object_location only knows a fixed list of common object types (the classes your \
object detector recognizes) -- it has no idea what a "black water bottle" or "the blue \
notebook" is beyond the generic class. If recall_object_location comes back found=false, or \
the user describes something more specific than a bare class name, call \
describe_current_view to actually look at the live camera feed before giving up -- the thing \
might be sitting in plain sight even though it isn't one of your tracked types. Describe only \
what you can actually see in that image; if it's not visible there either, say so plainly, \
the same as you would for a recall_object_location miss. Don't call describe_current_view \
more than once per question -- one live look is enough."""

# "dim" and "on" are relative to whatever's currently showing (dim steps
# down, on restores to a plain default); the rest are absolute presets.
LIGHT_ACTIONS = ("dim", "off", "on", "cozy", "focus", "calm")

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
    {
        "name": "control_light",
        "description": "Change the lamp's own light. Use for direct requests like 'dim the "
                        "light', 'turn it off', 'turn it back on', or a mood ('make it cozy', "
                        "'I need focus lighting', 'something calm'). Not for questions about "
                        "objects -- only for controlling the lamp's light itself.",
        "params": {
            "action": {"type": "string", "enum": list(LIGHT_ACTIONS),
                       "description": "one of: " + ", ".join(LIGHT_ACTIONS)},
        },
        "required": ["action"],
    },
    {
        "name": "describe_current_view",
        "description": "Look at what the camera currently sees, live -- for objects outside "
                        "your fixed detection classes, a more specific description than a bare "
                        "class name (color, brand, which one of several), or open-ended "
                        "questions about what's around right now. Returns the live image "
                        "itself for you to describe, not a text answer.",
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
    def __init__(self, store: MemoryStore, provider: str | None = None,
                 get_frame: Callable[[], np.ndarray | None] | None = None):
        """get_frame: optional zero-arg callable returning the current raw
        camera frame (or None), used only by describe_current_view. None
        in standalone chat.py (no live camera loop to ask) and in any
        session without --chat's vision_memory wired through -- the tool
        just reports itself unavailable rather than the agent needing a
        real camera dependency to construct at all."""
        self.store = store
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.get_frame = get_frame
        self.messages: list[dict] = []
        self.last_recall = RecallResult()
        self.last_light_action: str | None = None  # set by the last control_light call, so chat.py can apply it

        if self.provider == "openai":
            import openai
            self.client = openai.OpenAI()
        else:
            import anthropic
            self.client = anthropic.Anthropic()

    def _execute_tool(self, name: str, tool_input: dict) -> tuple[dict, bytes | None]:
        """Returns (json_result, jpeg_bytes_or_None) -- only
        describe_current_view ever returns real image bytes; every other
        tool's second element is always None. Kept as one return shape
        (not a special path just for the vision tool) so both provider
        loops below have exactly one place that decides whether to attach
        an image, not two slightly different tool-dispatch code paths."""
        if name == "recall_object_location":
            # `.get(key, "")` only falls back to "" when the key is
            # *missing* -- a tool call with an explicit {"object_name":
            # null} still gets None through, which normalize_class then
            # crashes on (.strip() on None). `or ""` catches None and
            # empty string the same way.
            obs = self.store.get_latest_by_class(str(tool_input.get("object_name") or ""))
            if obs is None:
                self.last_recall = RecallResult()
                return {"found": False}, None
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
                    {"object_class": cls, "direction": bearing_to_direction(bearing), "confidence": round(conf, 2)}
                    for cls, bearing, conf in cooccurring
                ],
            }, None
        if name == "list_seen_objects":
            return {"objects": self.store.list_known_classes()}, None
        if name == "control_light":
            action = str(tool_input.get("action", "")).strip().lower()
            if action not in LIGHT_ACTIONS:
                return {"applied": False, "error": f"'{action}' isn't a valid action"}, None
            self.last_light_action = action
            return {"applied": True, "action": action}, None
        if name == "describe_current_view":
            if self.get_frame is None:
                return {"available": False, "reason": "no live camera in this session"}, None
            frame = self.get_frame()
            if frame is None:
                return {"available": False, "reason": "no frame captured yet"}, None
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                return {"available": False, "reason": "couldn't encode the current frame"}, None
            return {"available": True}, jpeg.tobytes()
        return {"error": f"unknown tool {name}"}, None

    def ask(self, user_text: str) -> str:
        # Reset per-turn side state up front, not just inside the tool
        # branches that set it -- otherwise a turn that doesn't call
        # recall_object_location (a light command, small talk) would leave
        # last_recall holding an old turn's observation, and chat.py would
        # re-point at a stale object that had nothing to do with this turn.
        self.last_recall = RecallResult()
        self.last_light_action = None
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
            # Anthropic requires every tool_use in that message to have a
            # matching tool_result in the next one -- if _execute_tool
            # raised here and skipped straight to the except, the tool_use
            # above would be left permanently unpaired in self.messages
            # (an instance attribute kept across turns), poisoning every
            # future call in the session with the same 400, not just this
            # one. An error result keeps the pairing intact either way and
            # lets the model react to the failure instead of the whole
            # exchange dying.
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result, image_bytes = self._execute_tool(block.name, block.input)
                    except Exception as e:
                        result, image_bytes = {"error": f"tool failed: {e}"}, None
                    if image_bytes is not None:
                        # Anthropic tool_result content can be a list of
                        # blocks, not just a text string -- an image block
                        # here is what actually lets the model *see* the
                        # frame describe_current_view captured, not just
                        # read a caption about it.
                        content = [
                            {"type": "text", "text": json.dumps(result)},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                          "data": base64.b64encode(image_bytes).decode("ascii")}},
                        ]
                    else:
                        content = json.dumps(result)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
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
            # Same reasoning as the Anthropic loop below -- every tool_call
            # here needs a matching "tool" role reply in self.messages, or
            # the gap persists across turns for the rest of the session.
            for call in message.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                    result, image_bytes = self._execute_tool(call.function.name, args)
                except Exception as e:
                    result, image_bytes = {"error": f"tool failed: {e}"}, None
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
                if image_bytes is not None:
                    # OpenAI's tool-role messages are text-only -- an
                    # image has to ride along on a normal user message
                    # instead, which is why this follows the tool result
                    # rather than being part of it. The model still sees
                    # it before composing its next reply either way.
                    b64 = base64.b64encode(image_bytes).decode("ascii")
                    self.messages.append({"role": "user", "content": [
                        {"type": "text", "text": "(current camera view, for describe_current_view)"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]})
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL, messages=self.messages, tools=tools, tool_choice="auto",
            )
            message = response.choices[0].message

        self.messages.append({"role": "assistant", "content": message.content})
        return message.content or ""
