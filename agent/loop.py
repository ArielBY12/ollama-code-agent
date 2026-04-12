"""UI-agnostic agent loop.

Yields typed events so any frontend (TUI, web, tests) can consume them.
Never imports Textual. Never calls print().
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Generator, Union

import ollama  # type: ignore[import-untyped]

from agent.prompts import SYSTEM_PROMPT
from agent.tools.definitions import TOOL_SCHEMAS
from agent.tools.executor import TOOL_EXECUTORS
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


# ── agent loop ──────────────────────────────────────────────────────

class AgentLoop:
    """Manages conversation history and drives multi-step tool use."""

    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._client = ollama.Client(host=config.host)
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

        while True:
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
            if not tool_calls:
                # No tool calls — treat as final assistant response.
                content = msg.get("content", "")
                if content:
                    yield AssistantMessageEvent(content=content)
                return

            # Process each tool call, then loop for the next LLM turn.
            yield from self._execute_tool_calls(tool_calls)

    def _execute_tool_calls(
        self, tool_calls: list[dict]
    ) -> Generator[AgentEvent, None, None]:
        """Run requested tools and append results to history."""
        for tc in tool_calls:
            func_info = tc.get("function", {})
            name = func_info.get("name", "")
            raw_args = func_info.get("arguments", {})

            # Normalise args: the SDK may hand us a JSON string.
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}

            yield ToolCallEvent(name=name, args=raw_args)

            executor = TOOL_EXECUTORS.get(name)
            if executor is None:
                result = f"ERROR: unknown tool '{name}'"
            else:
                try:
                    result = executor(**raw_args)
                except TypeError as exc:
                    result = f"ERROR: bad arguments — {exc}"

            yield ToolResultEvent(name=name, result=result)

            self._messages.append(
                {"role": "tool", "content": result}
            )
