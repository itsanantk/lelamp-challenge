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
import datetime
import json
import time
from typing import Callable

import cv2
import numpy as np

from behavior import reminders
from memory.store import MemoryStore, Observation, bearing_to_direction, normalize_class
import config

SYSTEM_PROMPT = """You are LeLamp, a small desk lamp robot with a camera and a memory \
of what it has recently seen on and around the desk. You're warm, brief, a little curious \
-- talk like a helpful companion, not a search engine.

{time_context}

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

If the user asks you to set up a recurring check-in or an ongoing "make sure I..." habit \
(e.g. "make sure I get up every 30 minutes", "make sure I don't leave my desk", "stop \
reminding me to stand up"), call create_reminder instead of just replying in text -- same \
reasoning as control_light, a spoken acknowledgment with no tool call wouldn't actually set \
anything up. If they only want it active for a limited time ("only check for the next 20 \
seconds", "just for the next hour"), pass that as duration_minutes -- without it, a reminder \
stays active until explicitly cancelled. This is for ongoing self-initiated checks the lamp runs on its own, not for a \
one-time question you can just answer directly.

If the user is holding or references a specific object and wants you to check on it later \
(e.g. "make sure I drink all my water by 6", "check on my water bottle in an hour"), that's \
create_reminder with kind="object_check" -- pass the object as object_name (a plain noun like \
"water bottle" or "cup", the same way you'd pass object_name to recall_object_location) and \
the deadline as deadline_minutes, computed from the current time given above (e.g. "by 6pm" \
when it's currently 3:45pm is 135 minutes). You'll be watching where they set the object down \
and pointing back at it when the deadline hits -- you can't yet judge the object's contents \
(whether a bottle is full or empty), so don't promise that in your reply, just confirm you'll \
check on it.

recall_object_location only knows a fixed list of common object types (the classes your \
object detector recognizes) -- it has no idea what a "black water bottle" or "the blue \
notebook" is beyond the generic class. If recall_object_location comes back found=false, or \
the user describes something more specific than a bare class name, call \
describe_current_view to actually look at the live camera feed before giving up -- the thing \
might be sitting in plain sight even though it isn't one of your tracked types. Describe only \
what you can actually see in that image; if it's not visible there either, say so plainly, \
the same as you would for a recall_object_location miss. Don't call describe_current_view \
more than once per question -- one live look is enough."""


def _system_prompt() -> str:
    """SYSTEM_PROMPT with the current time filled in -- computed fresh per
    call (not baked in at import time) since "now" obviously changes
    across a long-running conversation. Needed for create_reminder's
    object_check deadlines: "by 6pm" only means something if the model
    knows what time it currently is, which nothing else in its context
    tells it."""
    now = datetime.datetime.now().strftime("%A, %I:%M %p").replace(" 0", " ")
    return SYSTEM_PROMPT.format(time_context=f"The current time is {now}.")


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
        "name": "create_reminder",
        "description": "Create or cancel a self-initiated timed check the lamp runs on its "
                        "own, without being asked again -- 'make sure I get up every 30 "
                        "minutes' (recurring), 'make sure I don't get up from my desk' / "
                        "'tell me if I leave' (presence), or 'make sure I drink my water by "
                        "6' / 'check on my bottle in an hour' (object_check). Also handles "
                        "cancelling ('stop reminding me', 'cancel that', 'never mind about "
                        "the check-ins').",
        "params": {
            "action": {"type": "string", "enum": ["create", "cancel"],
                       "description": "'create' a new reminder, or 'cancel' existing one(s)"},
            "kind": {"type": "string", "enum": list(reminders.KINDS),
                     "description": "'recurring': fires repeatedly on a fixed interval (e.g. "
                                     "every 30 minutes). 'presence': fires once each time the "
                                     "user leaves the desk (a face stops being detected), then "
                                     "re-arms once they're back. 'object_check': watches where a "
                                     "named object gets set down, then points back at it and "
                                     "checks in at a deadline -- for 'make sure I do something "
                                     "with X by/in a certain time.' Required for a 'create' "
                                     "action; for 'cancel', omit to cancel every active reminder "
                                     "instead of just one kind."},
            "interval_minutes": {"type": "number",
                                  "description": "required when creating a 'recurring' reminder "
                                                  "-- how often it fires, in minutes"},
            "object_name": {"type": "string",
                             "description": "required when creating an 'object_check' reminder "
                                             "-- the object to watch, as a plain noun (e.g. "
                                             "'water bottle', 'cup'), the same way you'd pass "
                                             "object_name to recall_object_location"},
            "deadline_minutes": {"type": "number",
                                  "description": "required when creating an 'object_check' "
                                                  "reminder -- minutes from now until the check, "
                                                  "computed from the current time given above "
                                                  "(e.g. 'by 6pm' when it's 3:45pm now is 135)"},
            "check_question": {"type": "string",
                                "description": "optional, 'object_check' only -- if the request "
                                                "implies judging something about the object's "
                                                "state, not just confirming it's still there, "
                                                "phrase that as a short question you'd want "
                                                "answered from a live look at it (e.g. 'make sure "
                                                "I drink all my water' -> 'is this water bottle "
                                                "full or empty'). Omit when there's nothing to "
                                                "visually judge (e.g. 'check on my keys' just "
                                                "needs to confirm where they are)."},
            "duration_minutes": {"type": "number",
                                  "description": "optional, for 'recurring'/'presence' only -- if "
                                                  "the user only wants this active for a limited "
                                                  "time ('only check for the next 20 seconds', "
                                                  "'just for the next hour'), how long from now "
                                                  "until it should stop checking on its own, in "
                                                  "minutes. Omit for a reminder that should stay "
                                                  "active until explicitly cancelled. Not used for "
                                                  "'object_check' -- deadline_minutes already gives "
                                                  "it a natural one-shot end point."},
            "message": {"type": "string",
                        "description": "required when creating a reminder -- what to say out "
                                        "loud when it fires, phrased the way you'd actually say "
                                        "it (e.g. 'time to stand up and stretch for a bit' or "
                                        "'hey, come back and sit down')"},
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
        self.last_reminder_action: dict | None = None  # set by the last create_reminder call, so chat.py can apply it

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
        if name == "create_reminder":
            action = str(tool_input.get("action", "")).strip().lower()
            if action == "create":
                kind = str(tool_input.get("kind", "")).strip().lower()
                if kind not in reminders.KINDS:
                    return {"created": False, "error": f"kind must be one of {list(reminders.KINDS)}"}, None
                message = str(tool_input.get("message") or "").strip()
                if not message:
                    return {"created": False, "error": "message is required"}, None
                interval_s = None
                if kind == "recurring":
                    minutes = tool_input.get("interval_minutes")
                    if not minutes or float(minutes) <= 0:
                        return {"created": False,
                                "error": "interval_minutes is required (and must be positive) "
                                         "for a recurring reminder"}, None
                    interval_s = float(minutes) * 60.0
                object_class = None
                due_in_s = None
                check_question = None
                if kind == "object_check":
                    object_name = str(tool_input.get("object_name") or "").strip()
                    if not object_name:
                        return {"created": False, "error": "object_name is required for an object_check reminder"}, None
                    deadline_minutes = tool_input.get("deadline_minutes")
                    if not deadline_minutes or float(deadline_minutes) <= 0:
                        return {"created": False,
                                "error": "deadline_minutes is required (and must be positive) "
                                         "for an object_check reminder"}, None
                    object_class = normalize_class(object_name)
                    due_in_s = float(deadline_minutes) * 60.0
                    check_question = str(tool_input.get("check_question") or "").strip() or None
                duration_s = None
                duration_minutes = tool_input.get("duration_minutes")
                if duration_minutes:
                    if float(duration_minutes) <= 0:
                        return {"created": False, "error": "duration_minutes must be positive if given"}, None
                    duration_s = float(duration_minutes) * 60.0
                self.last_reminder_action = {"action": "create", "kind": kind, "message": message,
                                              "interval_s": interval_s, "duration_s": duration_s,
                                              "object_class": object_class, "due_in_s": due_in_s,
                                              "check_question": check_question}
                return {"created": True, "kind": kind}, None
            if action == "cancel":
                kind = tool_input.get("kind")
                kind = str(kind).strip().lower() if kind else None
                if kind is not None and kind not in reminders.KINDS:
                    return {"cancelled": False, "error": f"kind must be one of {list(reminders.KINDS)}"}, None
                self.last_reminder_action = {"action": "cancel", "kind": kind}
                return {"cancelled": True}, None
            return {"error": f"unknown action '{action}' -- must be 'create' or 'cancel'"}, None
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
        self.last_reminder_action = None
        if self.provider == "openai":
            return self._ask_openai(user_text)
        return self._ask_anthropic(user_text)

    # -- Anthropic -----------------------------------------------------

    def _ask_anthropic(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        tools = _anthropic_tools()
        # One snapshot for the whole turn (including any tool-loop
        # follow-ups below) -- a turn resolves in a few seconds at most,
        # no need to re-read the clock mid-turn.
        system_prompt = _system_prompt()

        response = self.client.messages.create(
            model=config.ANTHROPIC_MODEL, max_tokens=512, system=system_prompt,
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
                model=config.ANTHROPIC_MODEL, max_tokens=512, system=system_prompt,
                tools=tools, messages=self.messages,
            )

        self.messages.append({"role": "assistant", "content": response.content})
        return "".join(b.text for b in response.content if b.type == "text")

    # -- OpenAI ----------------------------------------------------------

    def _ask_openai(self, user_text: str) -> str:
        # Refreshed every turn, not just set once -- unlike Anthropic's
        # system param (passed fresh on every messages.create() call
        # below), OpenAI's system message lives at messages[0] and would
        # otherwise keep whatever time was current on the *first* turn for
        # the rest of the conversation.
        if not self.messages:
            self.messages.append({"role": "system", "content": _system_prompt()})
        else:
            self.messages[0]["content"] = _system_prompt()
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

    # -- One-shot vision judgment (stage 3 of behavior/reminders.py's
    # object_check) ------------------------------------------------------

    def judge_view(self, question: str) -> str | None:
        """A single vision-LLM call answering `question` against the
        current camera frame -- e.g. "is this water bottle full or
        empty," fired when an object_check reminder's deadline hits (see
        chat.py's _resolve_object_check_judgment). Deliberately NOT routed
        through ask()/self.messages: this runs on reminders.py's own
        background poller thread, not in response to anything the user
        just said, and folding a background check into the actual
        conversation history would have the model referencing "the
        bottle I checked a minute ago" in an unrelated later question.
        One-shot, no tools, no memory of its own past answers -- exactly
        the same image-capture path as describe_current_view, just
        returning a direct answer instead of tool-result image bytes for
        the main conversation loop to react to.

        Returns None if there's no live camera in this session, no frame
        captured yet, or the API call itself fails -- callers already
        have a fallback message (the reminder's own) for when this comes
        back empty, same as any other imperfect vision read in this
        project."""
        if self.get_frame is None:
            return None
        frame = self.get_frame()
        if frame is None:
            return None
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None
        b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
        prompt = (f"Looking only at what's visible in this image, answer in one short, "
                  f"spoken-out-loud-friendly sentence: {question}")
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]}],
                )
                return (response.choices[0].message.content or "").strip() or None
            response = self.client.messages.create(
                model=config.ANTHROPIC_MODEL, max_tokens=200,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                ]}],
            )
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            return text or None
        except Exception as e:
            print(f"[agent] judge_view failed: {e}")
            return None
