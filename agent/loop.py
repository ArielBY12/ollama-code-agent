"""UI-agnostic agent loop.

Yields typed events so any frontend (TUI, web, tests) can consume them.
Never imports Textual. Never calls print().
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Generator, Union

import ollama  # type: ignore[import-untyped]

from agent.prompts import SYSTEM_PROMPT
from agent.tools.definitions import TOOL_SCHEMAS
from agent.tools.executor import ToolExecutor
from config import AgentConfig


# ── event types ─────────────────────────────────────────────────────

@dataclass
class ToolCallEvent:
    """The LLM decided to invoke a tool."""

    name: str
    args: dict


@dataclass
class ToolResultEvent:
    """A tool finished executing."""

    name: str
    result: str


@dataclass
class AssistantMessageEvent:
    """Final (non-tool-call) LLM response text."""

    content: str


@dataclass
class ErrorEvent:
    """Something went wrong during the turn."""

    message: str


AgentEvent = Union[
    ToolCallEvent, ToolResultEvent, AssistantMessageEvent, ErrorEvent
]


# ── text-mode tool-call extraction ──────────────────────────────────
# Some Ollama models (notably qwen2.5-coder:7b) emit tool calls as JSON
# inside ```json fences in the message body instead of populating the
# structured `message.tool_calls` field. We parse those out so the loop
# can still drive multi-step tool use.

_FENCE_RE = re.compile(r"```(?:json|tool_call)?\s*\n?(.*?)```", re.DOTALL)


def _coerce_call(item: object) -> dict | None:
    """Shape one parsed object into an SDK-style tool-call dict, or None."""
    if not isinstance(item, dict):
        return None
    name = item.get("name")
    args = item.get("arguments", item.get("parameters", {}))
    if not (isinstance(name, str) and name and isinstance(args, (dict, str))):
        return None
    return {"function": {"name": name, "arguments": args}}


def _scan_bare_json(content: str) -> list[dict]:
    """Walk *content* looking for top-level JSON objects that look like
    tool calls (have a "name" key). Handles the case where the model emits
    the call with no surrounding fence or tag.
    """
    decoder = json.JSONDecoder()
    results: list[dict] = []
    i = 0
    length = len(content)
    while i < length:
        j = content.find("{", i)
        if j == -1:
            break
        try:
            obj, end = decoder.raw_decode(content, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        items = obj if isinstance(obj, list) else [obj]
        for item in items:
            call = _coerce_call(item)
            if call is not None:
                results.append(call)
        i = end
    return results


def _extract_text_tool_calls(content: str) -> list[dict]:
    """Pull tool calls out of fenced JSON blocks in *content*.

    Falls back to scanning bare JSON objects if no fenced block yields a
    valid call. Returns a list shaped like the SDK's `message.tool_calls`
    field. Accepts arguments under either "arguments" or "parameters".
    """
    calls: list[dict] = []
    for match in _FENCE_RE.finditer(content):
        block = match.group(1).strip()
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            call = _coerce_call(item)
            if call is not None:
                calls.append(call)
    if calls:
        return calls
    return _scan_bare_json(content)


# ── agent loop ──────────────────────────────────────────────────────

# Hard ceiling on tool-call iterations per user turn. Guards against a
# runaway model that keeps requesting tools forever.
_MAX_TURN_ITERATIONS = 25


class AgentLoop:
    """Manages conversation history and drives multi-step tool use."""

    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._client = ollama.Client(host=config.host)
        self._tools = ToolExecutor(config)
        self._tool_names = frozenset(
            s["function"]["name"] for s in TOOL_SCHEMAS
        )
        self._messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    def clear(self) -> None:
        """Reset conversation history, keeping only the system prompt."""
        self._messages = [self._messages[0]]

    def run_turn(
        self, user_message: str
    ) -> Generator[AgentEvent, None, None]:
        """Send *user_message* and yield events until the LLM stops.

        Handles multi-step tool-call chains automatically.
        """
        self._messages.append(
            {"role": "user", "content": user_message}
        )

        for _ in range(_MAX_TURN_ITERATIONS):
            try:
                response = self._client.chat(
                    model=self._config.model,
                    messages=self._messages,
                    tools=TOOL_SCHEMAS,
                    options={"num_ctx": self._config.ctx},
                )
            except Exception as exc:
                yield ErrorEvent(message=str(exc))
                return

            msg = response.get("message", {})
            self._messages.append(msg)

            tool_calls = msg.get("tool_calls")
            content = msg.get("content", "") or ""

            # Fallback: some models emit tool calls as fenced JSON in
            # `content` instead of using the structured field. Parser
            # returns shape-valid calls; we filter to registered tools
            # so spurious JSON doesn't round-trip as "unknown tool".
            if not tool_calls and content:
                extracted = [
                    c
                    for c in _extract_text_tool_calls(content)
                    if c["function"]["name"] in self._tool_names
                ]
                if extracted:
                    tool_calls = extracted
                    msg["tool_calls"] = extracted

            if not tool_calls:
                if content:
                    yield AssistantMessageEvent(content=content)
                return

            # Process each tool call, then loop for the next LLM turn.
            yield from self._execute_tool_calls(tool_calls)
        else:
            yield ErrorEvent(
                message=(
                    f"Stopped: exceeded {_MAX_TURN_ITERATIONS} "
                    "tool-call iterations in one turn."
                )
            )

    def _execute_tool_calls(
        self, tool_calls: list[dict]
    ) -> Generator[AgentEvent, None, None]:
        """Run requested tools and append results to history."""
        for tc in tool_calls:
            func_info = tc.get("function", {})
            name = func_info.get("name", "")
            raw_args = func_info.get("arguments", {})
            call_id = tc.get("id")

            # Normalise args: the SDK may hand us a JSON string.
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}

            yield ToolCallEvent(name=name, args=raw_args)

            result = self._tools.dispatch(name, raw_args)

            yield ToolResultEvent(name=name, result=result)

            tool_msg: dict = {
                "role": "tool",
                "name": name,
                "content": result,
            }
            if call_id:
                tool_msg["tool_call_id"] = call_id
            self._messages.append(tool_msg)
