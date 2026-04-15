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


    def test_tool_call_tag_with_json_body(self) -> None:
        content = (
            '<tool_call>\n'
            '{"name": "list_directory", "arguments": {"path": "."}}\n'
            '</tool_call>'
        )
        out = _extract_text_tool_calls(content)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["name"], "list_directory")
        self.assertEqual(out[0]["function"]["arguments"], {"path": "."})

    def test_function_tag_with_parameter_children(self) -> None:
        content = (
            '<tool_call>\n'
            '<function=list_directory>\n'
            '<parameter=path>.</parameter>\n'
            '</function>\n'
            '</tool_call>'
        )
        out = _extract_text_tool_calls(content)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["name"], "list_directory")
        self.assertEqual(out[0]["function"]["arguments"], {"path": "."})

    def test_bare_function_tag_without_tool_call_envelope(self) -> None:
        content = (
            '<function=run_bash>\n'
            '<parameter=command>ls -la</parameter>\n'
            '</function>'
        )
        out = _extract_text_tool_calls(content)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["name"], "run_bash")
        self.assertEqual(out[0]["function"]["arguments"], {"command": "ls -la"})

    def test_function_tag_with_json_body(self) -> None:
        content = '<function=read_file>{"path": "main.py"}</function>'
        out = _extract_text_tool_calls(content)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["function"]["arguments"], {"path": "main.py"})

    def test_parameter_value_numeric_coerced(self) -> None:
        content = (
            '<function=search_code>'
            '<parameter=pattern>foo</parameter>'
            '<parameter=max_results>5</parameter>'
            '</function>'
        )
        out = _extract_text_tool_calls(content)
        self.assertEqual(
            out[0]["function"]["arguments"],
            {"pattern": "foo", "max_results": 5},
        )

    def test_function_tag_not_double_counted_inside_tool_call(self) -> None:
        content = (
            '<tool_call>'
            '<function=read_file>'
            '<parameter=path>x.py</parameter>'
            '</function>'
            '</tool_call>'
        )
        out = _extract_text_tool_calls(content)
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
