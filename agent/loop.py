"""UI-agnostic agent loop.

Yields typed events so any frontend (TUI, web, tests) can consume them.
Never imports Textual. Never calls print().
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Generator, Iterable, Union

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
class AssistantChunkEvent:
    """A streamed delta of assistant text."""

    content: str


@dataclass
class AssistantMessageEvent:
    """Final (non-tool-call) LLM response text.

    Emitted once per streamed response with the fully assembled content,
    so consumers that don't care about incremental rendering can ignore
    AssistantChunkEvent and still get the full message.
    """

    content: str


@dataclass
class ConfirmRequestEvent:
    """Ask the consumer to approve or deny a sensitive tool call.

    The loop blocks on ``reply`` after yielding this event. The consumer
    MUST append a single bool to ``approved`` and then set ``reply`` —
    True to run the tool, False to synthesize a denial result without
    executing.
    """

    name: str
    args: dict
    reply: threading.Event = field(default_factory=threading.Event)
    approved: list[bool] = field(default_factory=list)


@dataclass
class ErrorEvent:
    """Something went wrong during the turn."""

    message: str


AgentEvent = Union[
    ToolCallEvent,
    ToolResultEvent,
    AssistantChunkEvent,
    AssistantMessageEvent,
    ConfirmRequestEvent,
    ErrorEvent,
]


# Tools whose execution requires explicit user approval each call.
_SENSITIVE_TOOLS: frozenset[str] = frozenset(
    {"run_bash", "write_file", "patch_file"}
)

# Default reply timeout for confirmation — generous, but bounded so a
# detached frontend can't wedge the worker forever.
_CONFIRM_TIMEOUT_SECONDS = 300.0


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
_MAX_TURN_ITERATIONS = 100


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

        Handles multi-step tool-call chains and streaming automatically.
        """
        self._messages.append(
            {"role": "user", "content": user_message}
        )

        for _ in range(_MAX_TURN_ITERATIONS):
            try:
                stream = self._client.chat(
                    model=self._config.model,
                    messages=self._messages,
                    tools=TOOL_SCHEMAS,
                    stream=True,
                    keep_alive="30m",
                    options={"num_ctx": self._config.ctx},
                )
            except Exception as exc:
                yield ErrorEvent(message=str(exc))
                return

            try:
                content, tool_calls, final_msg = yield from self._consume_stream(
                    stream
                )
            except Exception as exc:
                yield ErrorEvent(message=str(exc))
                return

            # Store the assembled assistant turn on the conversation
            # history. Some SDKs return a full message per chunk; keep the
            # last one but overwrite its content/tool_calls with our own
            # assembly so history is consistent regardless of SDK shape.
            msg = dict(final_msg) if isinstance(final_msg, dict) else {}
            msg.setdefault("role", "assistant")
            msg["content"] = content
            if tool_calls is not None:
                msg["tool_calls"] = tool_calls
            self._messages.append(msg)

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

    def _consume_stream(
        self, stream: Iterable[dict]
    ) -> Generator[AgentEvent, None, tuple[str, list[dict] | None, dict]]:
        """Drain the streamed chat iterator, yielding chunks as they arrive.

        Returns the assembled (content, tool_calls, final_raw_message).
        """
        content_parts: list[str] = []
        tool_calls: list[dict] | None = None
        final_msg: dict = {}

        for chunk in stream:
            if not isinstance(chunk, dict):
                # Some SDKs yield objects with a .model_dump() method; best
                # effort — if it's not dict-shaped, skip the delta but keep
                # looping so we drain the stream.
                continue
            msg = chunk.get("message") or {}
            final_msg = msg if msg else final_msg

            delta = msg.get("content") or ""
            if delta:
                content_parts.append(delta)
                yield AssistantChunkEvent(content=delta)

            calls = msg.get("tool_calls")
            if calls:
                # Structured tool calls arrive whole, not delta-by-delta,
                # so the last non-empty one wins.
                tool_calls = list(calls)

        return ("".join(content_parts), tool_calls, final_msg)

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

            if name in _SENSITIVE_TOOLS:
                approved = yield from self._await_confirmation(name, raw_args)
                if not approved:
                    result = f"ERROR: user denied execution of {name}"
                    yield ToolResultEvent(name=name, result=result)
                    self._append_tool_message(name, result, call_id)
                    continue

            result = self._tools.dispatch(name, raw_args)

            yield ToolResultEvent(name=name, result=result)
            self._append_tool_message(name, result, call_id)

    def _await_confirmation(
        self, name: str, args: dict
    ) -> Generator[AgentEvent, None, bool]:
        """Yield a ConfirmRequestEvent and block on the consumer's reply."""
        event = ConfirmRequestEvent(name=name, args=args)
        yield event
        event.reply.wait(timeout=_CONFIRM_TIMEOUT_SECONDS)
        return bool(event.approved and event.approved[0])

    def _append_tool_message(
        self, name: str, result: str, call_id: str | None
    ) -> None:
        tool_msg: dict = {
            "role": "tool",
            "name": name,
            "content": result,
        }
        if call_id:
            tool_msg["tool_call_id"] = call_id
        self._messages.append(tool_msg)
