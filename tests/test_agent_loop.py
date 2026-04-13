"""Integration tests for agent.loop.AgentLoop.run_turn using a fake client."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from agent.loop import (
    AgentLoop,
    AssistantMessageEvent,
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
    _MAX_TURN_ITERATIONS,
)
from config import AgentConfig


class FakeClient:
    """Stand-in for ollama.Client that replays scripted chat responses."""

    def __init__(self, host: str = "", responses: list[Any] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[dict] = []

    def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeClient exhausted — test scripted too few responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _msg(content: str = "", tool_calls: list[dict] | None = None) -> dict:
    return {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}


def _make_loop(responses: list[Any], workdir: Path | None = None) -> tuple[AgentLoop, FakeClient]:
    cfg = AgentConfig(model="m", host="h", workdir=workdir or Path("."), ctx=32768)
    fake = FakeClient(responses=responses)
    with patch("agent.loop.ollama.Client", return_value=fake):
        loop = AgentLoop(cfg)
    return loop, fake


class RunTurnTests(unittest.TestCase):
    def test_direct_answer_without_tools(self) -> None:
        loop, _ = _make_loop([_msg(content="hello there")])
        events = list(loop.run_turn("hi"))
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], AssistantMessageEvent)
        self.assertEqual(events[0].content, "hello there")

    def test_single_structured_tool_call_then_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "f.txt"
            target.write_text("payload", encoding="utf-8")
            tc = [{
                "id": "call_1",
                "function": {"name": "read_file", "arguments": {"path": str(target)}},
            }]
            loop, fake = _make_loop([
                _msg(tool_calls=tc),
                _msg(content="file said payload"),
            ])
            events = list(loop.run_turn("read it"))
            kinds = [type(e).__name__ for e in events]
            self.assertEqual(
                kinds,
                ["ToolCallEvent", "ToolResultEvent", "AssistantMessageEvent"],
            )
            result_event = events[1]
            assert isinstance(result_event, ToolResultEvent)
            self.assertEqual(result_event.result, "payload")
            # tool_call_id was threaded back into history
            tool_msg = [m for m in loop._messages if m.get("role") == "tool"][0]
            self.assertEqual(tool_msg["tool_call_id"], "call_1")
            self.assertEqual(tool_msg["name"], "read_file")
            self.assertEqual(len(fake.calls), 2)

    def test_text_mode_fallback_fires_for_known_tool(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "f.txt"
            target.write_text("hi", encoding="utf-8")
            inline = (
                'Sure. {"name": "read_file", "arguments": '
                f'{{"path": "{target.as_posix()}"}}}}'
            )
            loop, _ = _make_loop([
                _msg(content=inline),
                _msg(content="done"),
            ])
            events = list(loop.run_turn("read"))
            self.assertTrue(any(isinstance(e, ToolCallEvent) for e in events))
            self.assertTrue(any(isinstance(e, ToolResultEvent) for e in events))
            self.assertTrue(any(isinstance(e, AssistantMessageEvent) for e in events))

    def test_text_mode_unknown_tool_name_is_rejected(self) -> None:
        inline = '{"name": "greet", "arguments": {"who": "world"}}'
        loop, _ = _make_loop([_msg(content=inline)])
        events = list(loop.run_turn("say hi"))
        # Unknown-name JSON falls through to the AssistantMessageEvent path,
        # NOT dispatched as a bogus tool call.
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], AssistantMessageEvent)
        # No tool events emitted.
        self.assertFalse(any(isinstance(e, ToolCallEvent) for e in events))

    def test_client_exception_becomes_error_event(self) -> None:
        loop, _ = _make_loop([RuntimeError("boom")])
        events = list(loop.run_turn("hi"))
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], ErrorEvent)
        self.assertIn("boom", events[0].message)

    def test_iteration_cap_yields_error(self) -> None:
        # Model always asks for another tool call — we should stop at the cap.
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "f.txt"
            target.write_text("x", encoding="utf-8")
            forever_call = _msg(tool_calls=[{
                "function": {"name": "read_file", "arguments": {"path": str(target)}},
            }])
            loop, fake = _make_loop([forever_call] * (_MAX_TURN_ITERATIONS + 5))
            events = list(loop.run_turn("loop"))
            self.assertIsInstance(events[-1], ErrorEvent)
            self.assertIn("exceeded", events[-1].message)
            self.assertEqual(len(fake.calls), _MAX_TURN_ITERATIONS)

    def test_clear_keeps_system_prompt_only(self) -> None:
        loop, _ = _make_loop([_msg(content="ok")])
        list(loop.run_turn("hi"))
        self.assertGreater(len(loop._messages), 1)
        loop.clear()
        self.assertEqual(len(loop._messages), 1)
        self.assertEqual(loop._messages[0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
