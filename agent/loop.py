"""UI-agnostic agent loop.

Yields typed events so any frontend (TUI, web, tests) can consume them.
Never imports Textual. Never calls print().
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Generator, Iterable, Union

import ollama  # type: ignore[import-untyped]

from agent.prompts import SYSTEM_PROMPT
from agent.tools.definitions import SENSITIVE_TOOL_NAMES, TOOL_SCHEMAS
from agent.tools.executor import ToolExecutor
from config import AgentConfig


# ── event types ─────────────────────────────────────────────────────

@dataclass
class ToolCallEvent:
    name: str
    args: dict


@dataclass
class ToolResultEvent:
    name: str
    result: str


@dataclass
class AssistantChunkEvent:
    content: str


@dataclass
class AssistantMessageEvent:
    """Final assistant text; emitted only when a turn ends without a tool call."""

    content: str


class ConfirmRequestEvent:
    """Ask the consumer to approve or deny a sensitive tool call.

    The loop yields this event and then calls ``wait()``; the consumer
    calls ``approve()`` or ``deny()`` to unblock. Missing reply within
    the timeout is treated as a denial so a detached frontend can't
    wedge the worker forever.
    """

    def __init__(self, name: str, args: dict) -> None:
        self.name = name
        self.args = args
        self._reply = threading.Event()
        self._approved = False

    def approve(self) -> None:
        self._approved = True
        self._reply.set()

    def deny(self) -> None:
        self._approved = False
        self._reply.set()

    def wait(self, timeout: float) -> bool:
        return self._reply.wait(timeout=timeout) and self._approved


@dataclass
class ErrorEvent:
    message: str


AgentEvent = Union[
    ToolCallEvent,
    ToolResultEvent,
    AssistantChunkEvent,
    AssistantMessageEvent,
    ConfirmRequestEvent,
    ErrorEvent,
]


_CONFIRM_TIMEOUT_SECONDS = 300.0


# ── text-mode tool-call extraction ──────────────────────────────────
# Some Ollama models (notably qwen2.5-coder:7b) emit tool calls as JSON
# inside ```json fences in the message body instead of populating the
# structured `message.tool_calls` field. We parse those out so the loop
# can still drive multi-step tool use.

_FENCE_RE = re.compile(r"```(?:json|tool_call)?\s*\n?(.*?)```", re.DOTALL)

# `<tool_call>...</tool_call>` envelope (Qwen/Hermes). Body may be JSON or
# further tag-based markup — we hand the inner text back to the main parser.
_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)

# `<function=NAME> ... </function>` block (Llama 3.1 built-in). Body is
# either JSON arguments or a sequence of `<parameter=KEY>VALUE</parameter>`
# children.
_FUNCTION_TAG_RE = re.compile(
    r"<function\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*>(.*?)</function>",
    re.DOTALL,
)

_PARAMETER_TAG_RE = re.compile(
    r"<parameter\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*>(.*?)</parameter>",
    re.DOTALL,
)


def _coerce_param_value(raw: str) -> object:
    """Best-effort decode of a tag-wrapped parameter value.

    Parameter bodies arrive as raw text. Try JSON first so numbers, bools,
    and nested objects round-trip; fall back to the stripped string so path
    arguments like `.` stay usable.
    """
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_function_tag_body(name: str, body: str) -> dict | None:
    params = _PARAMETER_TAG_RE.findall(body)
    if params:
        args: dict = {k: _coerce_param_value(v) for k, v in params}
        return {"function": {"name": name, "arguments": args}}
    # No <parameter> children — try JSON arguments instead.
    stripped = body.strip()
    if not stripped:
        return {"function": {"name": name, "arguments": {}}}
    try:
        args_obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(args_obj, dict):
        return None
    return {"function": {"name": name, "arguments": args_obj}}


def _extract_tag_tool_calls(content: str) -> list[dict]:
    """Parse XML-ish tool-call tags some models emit instead of JSON fences.

    Handles two shapes that show up in Ollama model output:
      * ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``
      * ``<function=NAME>`` + ``<parameter=KEY>VALUE</parameter>`` children
    A ``<tool_call>`` envelope may wrap either — we recurse into its body.
    """
    calls: list[dict] = []

    for match in _TOOL_CALL_TAG_RE.finditer(content):
        inner = match.group(1)
        fn_matches = list(_FUNCTION_TAG_RE.finditer(inner))
        if fn_matches:
            for fm in fn_matches:
                call = _parse_function_tag_body(fm.group(1), fm.group(2))
                if call is not None:
                    calls.append(call)
            continue
        try:
            data = json.loads(inner.strip())
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            call = _coerce_call(item)
            if call is not None:
                calls.append(call)

    # Also catch `<function=...>` blocks that appear outside a `<tool_call>`
    # envelope (Llama 3.1 chat template emits them bare).
    consumed_spans = [m.span() for m in _TOOL_CALL_TAG_RE.finditer(content)]
    for fm in _FUNCTION_TAG_RE.finditer(content):
        start, end = fm.span()
        if any(s <= start and end <= e for s, e in consumed_spans):
            continue
        call = _parse_function_tag_body(fm.group(1), fm.group(2))
        if call is not None:
            calls.append(call)

    return calls


def _coerce_call(item: object) -> dict | None:
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
    tag_calls = _extract_tag_tool_calls(content)
    if tag_calls:
        return tag_calls
    return _scan_bare_json(content)


# ── agent loop ──────────────────────────────────────────────────────

# Hard ceiling on tool-call iterations per user turn. Guards against a
# runaway model that keeps requesting tools forever.
_MAX_TURN_ITERATIONS = 100


class AgentLoop:
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

            # Some SDKs return a full message per chunk; keep the last
            # one but overwrite content/tool_calls with our own assembly.
            msg = dict(final_msg) if isinstance(final_msg, dict) else {}
            msg.setdefault("role", "assistant")
            msg["content"] = content
            if tool_calls is not None:
                msg["tool_calls"] = tool_calls
            self._messages.append(msg)

            # Fallback for models that emit tool calls as fenced JSON in
            # `content` instead of the structured field. Filter to known
            # tools so spurious JSON doesn't round-trip as "unknown tool".
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
                # ollama>=0.4 yields pydantic ChatResponse objects. Flatten
                # to a plain dict so downstream code (which expects dict
                # tool_calls with a "function" key) keeps working.
                if hasattr(chunk, "model_dump"):
                    chunk = chunk.model_dump()
                else:
                    continue
            msg = chunk.get("message") or {}
            final_msg = msg if msg else final_msg

            delta = msg.get("content") or ""
            if delta:
                content_parts.append(delta)
                yield AssistantChunkEvent(content=delta)

            calls = msg.get("tool_calls")
            if calls:
                # Structured tool calls arrive whole, not delta-by-delta.
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

            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}

            yield ToolCallEvent(name=name, args=raw_args)

            if name in SENSITIVE_TOOL_NAMES:
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
        event = ConfirmRequestEvent(name=name, args=args)
        yield event
        return event.wait(timeout=_CONFIRM_TIMEOUT_SECONDS)

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
