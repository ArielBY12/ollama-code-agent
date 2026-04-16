"""Integration tests for agent.loop.AgentLoop.run_turn using a fake client."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from agent.loop import (
    AgentLoop,
    AssistantChunkEvent,
    AssistantMessageEvent,
    ConfirmRequestEvent,
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
    _MAX_TURN_ITERATIONS,
)
from config import AgentConfig


class FakeClient:
    """Stand-in for ollama.Client that replays scripted streaming responses.

    Each scripted response is either an Exception (raised on .chat()) or a
    list of chunk dicts — each chunk shaped like Ollama's stream output
    (``{"message": {"role": "assistant", "content": str, "tool_calls": ...}}``).
    Callers can pass a plain ``{"message": {...}}`` dict as shorthand and
    it will be lifted to a one-chunk stream automatically.
    """

    def __init__(self, host: str = "", responses: list[Any] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[dict] = []

    def chat(self, **kwargs: Any) -> Iterator[dict]:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeClient exhausted — test scripted too few responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, dict):
            chunks = [item]
        else:
            chunks = list(item)
        return iter(chunks)


def _msg(content: str = "", tool_calls: list[dict] | None = None) -> dict:
    return {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}


def _stream_chunks(content: str, tool_calls: list[dict] | None = None) -> list[dict]:
    """Break *content* into per-character chunks; append tool_calls at end."""
    chunks: list[dict] = [
        {"message": {"role": "assistant", "content": ch, "tool_calls": None}}
        for ch in content
    ]
    if tool_calls is not None:
        chunks.append(
            {"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}}
        )
    return chunks


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
        self.assertIsInstance(events[-1], AssistantMessageEvent)
        self.assertEqual(events[-1].content, "hello there")

    def test_streaming_emits_chunks_and_final(self) -> None:
        loop, _ = _make_loop([_stream_chunks("hi!")])
        events = list(loop.run_turn("ping"))
        chunk_deltas = [e.content for e in events if isinstance(e, AssistantChunkEvent)]
        self.assertEqual("".join(chunk_deltas), "hi!")
        self.assertIsInstance(events[-1], AssistantMessageEvent)
        self.assertEqual(events[-1].content, "hi!")

    def test_pydantic_chunks_are_accepted(self) -> None:
        # ollama>=0.4 yields ChatResponse pydantic objects, not dicts. The
        # loop must flatten them or the turn ends silently with no output.
        from ollama import ChatResponse, Message

        chunks = [
            ChatResponse(
                model="m",
                created_at="2025-01-01T00:00:00Z",
                message=Message(role="assistant", content=ch),
                done=False,
            )
            for ch in "yo"
        ]
        chunks.append(
            ChatResponse(
                model="m",
                created_at="2025-01-01T00:00:00Z",
                message=Message(role="assistant", content=""),
                done=True,
            )
        )
        loop, _ = _make_loop([chunks])
        events = list(loop.run_turn("hi"))
        chunk_deltas = [e.content for e in events if isinstance(e, AssistantChunkEvent)]
        self.assertEqual("".join(chunk_deltas), "yo")
        self.assertIsInstance(events[-1], AssistantMessageEvent)
        self.assertEqual(events[-1].content, "yo")

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
            self.assertIn("ToolCallEvent", kinds)
            self.assertIn("ToolResultEvent", kinds)
            self.assertIsInstance(events[-1], AssistantMessageEvent)
            result_events = [e for e in events if isinstance(e, ToolResultEvent)]
            self.assertEqual(result_events[0].result, "payload")
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
        self.assertIsInstance(events[-1], AssistantMessageEvent)
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


class ConfirmationTests(unittest.TestCase):
    """Sensitive tools must emit ConfirmRequestEvent and honour approve/deny."""

    def _script_run_bash_then_reply(self) -> list[dict]:
        tc = [{
            "function": {
                "name": "run_bash",
                "arguments": {"command": "echo hi"},
            },
        }]
        return [_msg(tool_calls=tc), _msg(content="done")]

    def _approve_and_collect(
        self, loop: AgentLoop, user_text: str, decision: bool
    ) -> list[object]:
        """Drive run_turn, responding to any ConfirmRequestEvent with *decision*."""
        events: list[object] = []
        gen = loop.run_turn(user_text)
        for ev in gen:
            events.append(ev)
            if isinstance(ev, ConfirmRequestEvent):
                if decision:
                    ev.approve()
                else:
                    ev.deny()
        return events

    def test_sensitive_tool_emits_confirm_request(self) -> None:
        loop, _ = _make_loop(self._script_run_bash_then_reply())
        events = self._approve_and_collect(loop, "run it", True)
        self.assertTrue(any(isinstance(e, ConfirmRequestEvent) for e in events))

    def test_approved_tool_runs(self) -> None:
        loop, _ = _make_loop(self._script_run_bash_then_reply())
        events = self._approve_and_collect(loop, "run it", True)
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        # echo hi produces "hi\n" and an exit-code footer.
        self.assertIn("hi", result.result)
        self.assertIn("exit code: 0", result.result)

    def test_denied_tool_yields_error_result(self) -> None:
        loop, _ = _make_loop(self._script_run_bash_then_reply())
        events = self._approve_and_collect(loop, "run it", False)
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        self.assertTrue(result.result.startswith("ERROR: user denied"))
        # Denial still appends to history so the model sees the outcome.
        tool_msgs = [m for m in loop._messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("user denied", tool_msgs[0]["content"])

    def test_non_sensitive_tool_is_not_gated(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "f.txt"
            target.write_text("x", encoding="utf-8")
            tc = [{
                "function": {
                    "name": "read_file",
                    "arguments": {"path": str(target)},
                },
            }]
            loop, _ = _make_loop([_msg(tool_calls=tc), _msg(content="ok")])
            events = list(loop.run_turn("read"))
            self.assertFalse(
                any(isinstance(e, ConfirmRequestEvent) for e in events)
            )


if __name__ == "__main__":
    unittest.main()
