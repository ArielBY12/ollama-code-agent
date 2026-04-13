"""Tests for agent.loop._extract_text_tool_calls fallback parser."""

from __future__ import annotations

import unittest

from agent.loop import _extract_text_tool_calls


class ExtractTextToolCallsTests(unittest.TestCase):
    def test_no_fence_returns_empty(self) -> None:
        self.assertEqual(_extract_text_tool_calls("just prose"), [])

    def test_single_object(self) -> None:
        content = '```json\n{"name": "read_file", "arguments": {"path": "x"}}\n```'
        self.assertEqual(
            _extract_text_tool_calls(content),
            [{"function": {"name": "read_file", "arguments": {"path": "x"}}}],
        )

    def test_list_of_objects(self) -> None:
        content = (
            '```json\n'
            '[{"name": "a", "arguments": {}},'
            ' {"name": "b", "arguments": {"k": 1}}]\n'
            '```'
        )
        out = _extract_text_tool_calls(content)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["function"]["name"], "a")
        self.assertEqual(out[1]["function"]["arguments"], {"k": 1})

    def test_parameters_alias_accepted(self) -> None:
        content = '```json\n{"name": "x", "parameters": {"p": 2}}\n```'
        out = _extract_text_tool_calls(content)
        self.assertEqual(out[0]["function"]["arguments"], {"p": 2})

    def test_tool_call_fence_label_accepted(self) -> None:
        content = '```tool_call\n{"name": "x", "arguments": {}}\n```'
        out = _extract_text_tool_calls(content)
        self.assertEqual(len(out), 1)

    def test_malformed_json_skipped(self) -> None:
        content = '```json\n{not valid\n```'
        self.assertEqual(_extract_text_tool_calls(content), [])

    def test_non_dict_items_skipped(self) -> None:
        content = '```json\n[1, "str", {"name": "ok", "arguments": {}}]\n```'
        out = _extract_text_tool_calls(content)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["name"], "ok")

    def test_missing_name_skipped(self) -> None:
        content = '```json\n{"arguments": {"k": 1}}\n```'
        self.assertEqual(_extract_text_tool_calls(content), [])

    def test_string_arguments_preserved(self) -> None:
        # Some models emit arguments as a JSON-encoded string.
        content = '```json\n{"name": "x", "arguments": "{\\"k\\":1}"}\n```'
        out = _extract_text_tool_calls(content)
        self.assertEqual(out[0]["function"]["arguments"], '{"k":1}')

    def test_bare_json_object(self) -> None:
        # No fence, just an inline JSON object.
        content = '{"name": "run_bash", "arguments": {"command": "ls"}}'
        out = _extract_text_tool_calls(content)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["name"], "run_bash")
        self.assertEqual(out[0]["function"]["arguments"], {"command": "ls"})

    def test_bare_json_with_prose_around(self) -> None:
        content = (
            'Sure, let me search. '
            '{"name": "search_code", "arguments": {"pattern": "foo"}}'
            ' Done.'
        )
        out = _extract_text_tool_calls(content)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["arguments"], {"pattern": "foo"})

    def test_bare_json_ignores_non_tool_objects(self) -> None:
        # An unrelated JSON object that doesn't have a "name" must be skipped.
        content = '{"hello": "world"}'
        self.assertEqual(_extract_text_tool_calls(content), [])

    def test_bare_json_prefers_fenced_when_both_present(self) -> None:
        content = (
            'Prose {"name": "bare", "arguments": {}}\n'
            '```json\n{"name": "fenced", "arguments": {}}\n```'
        )
        out = _extract_text_tool_calls(content)
        # Fenced wins — bare scan is only a fallback.
        self.assertEqual([c["function"]["name"] for c in out], ["fenced"])


if __name__ == "__main__":
    unittest.main()
