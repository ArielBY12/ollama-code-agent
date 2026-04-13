"""Tests for agent.tools.executor.ToolExecutor and helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent.tools.definitions import TOOL_SCHEMAS
from agent.tools.executor import ToolExecutor, _cap
from config import AgentConfig


def _cfg(ctx: int = 32768, workdir: Path | None = None) -> AgentConfig:
    return AgentConfig(
        model="m", host="h", workdir=workdir or Path("."), ctx=ctx
    )


class CapHelperTests(unittest.TestCase):
    def test_under_limit_untouched(self) -> None:
        self.assertEqual(_cap("abc", 10), "abc")

    def test_at_limit_untouched(self) -> None:
        self.assertEqual(_cap("abcde", 5), "abcde")

    def test_over_limit_truncated_with_notice(self) -> None:
        out = _cap("a" * 20, 5)
        self.assertTrue(out.startswith("aaaaa"))
        self.assertIn("[output truncated]", out)


class CapDerivationTests(unittest.TestCase):
    def test_default_ctx_reproduces_legacy_caps(self) -> None:
        t = ToolExecutor(_cfg(32768))
        self.assertEqual(t._file_cap, 43690)
        self.assertEqual(t._search_cap, 6553)
        self.assertEqual(t._dir_line_cap, 409)

    def test_small_ctx_shrinks_all_caps(self) -> None:
        t = ToolExecutor(_cfg(8192))
        self.assertEqual(t._file_cap, 10922)
        self.assertEqual(t._search_cap, 1638)
        self.assertEqual(t._dir_line_cap, 102)

    def test_large_ctx_grows_all_caps(self) -> None:
        t = ToolExecutor(_cfg(131072))
        self.assertEqual(t._file_cap, 174762)
        self.assertEqual(t._search_cap, 26214)
        self.assertEqual(t._dir_line_cap, 1638)

    def test_dir_line_cap_has_floor(self) -> None:
        t = ToolExecutor(_cfg(256))
        self.assertEqual(t._dir_line_cap, 50)

    def test_executor_registry_matches_schemas(self) -> None:
        t = ToolExecutor(_cfg())
        schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        self.assertEqual(set(t._executors), schema_names)


class DispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t = ToolExecutor(_cfg())

    def test_unknown_tool(self) -> None:
        self.assertEqual(
            self.t.dispatch("nope", {}),
            "ERROR: unknown tool 'nope'",
        )

    def test_bad_arguments(self) -> None:
        out = self.t.dispatch("read_file", {"wrong": 1})
        self.assertTrue(out.startswith("ERROR: bad arguments"))

    def test_arbitrary_exception_flattened(self) -> None:
        class Boom(RuntimeError):
            pass

        def boomer(**_: object) -> str:
            raise Boom("kaboom")

        self.t._executors["boom"] = boomer
        out = self.t.dispatch("boom", {})
        self.assertTrue(out.startswith("ERROR: Boom: kaboom"))

    def test_happy_path_returns_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "hi.txt"
            p.write_text("hello", encoding="utf-8")
            self.assertEqual(self.t.dispatch("read_file", {"path": str(p)}), "hello")


class ReadFileTests(unittest.TestCase):
    def test_missing_file(self) -> None:
        t = ToolExecutor(_cfg())
        self.assertTrue(t.read_file("no_such_file.xyz").startswith("ERROR: file not found"))

    def test_reads_content(self) -> None:
        t = ToolExecutor(_cfg())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.txt"
            p.write_text("payload", encoding="utf-8")
            self.assertEqual(t.read_file(str(p)), "payload")

    def test_truncates_at_cap(self) -> None:
        t = ToolExecutor(_cfg(ctx=256))  # file_cap = 341
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "big.txt"
            p.write_text("x" * 1000, encoding="utf-8")
            out = t.read_file(str(p))
            self.assertIn("[output truncated]", out)
            self.assertLess(len(out), 1000)


class WriteFileTests(unittest.TestCase):
    def test_creates_parent_dirs_and_writes(self) -> None:
        t = ToolExecutor(_cfg())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "nested" / "deep" / "out.txt"
            out = t.write_file(str(p), "hello")
            self.assertIn("Wrote 5 chars", out)
            self.assertEqual(p.read_text(encoding="utf-8"), "hello")


class PatchFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t = ToolExecutor(_cfg())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "f.txt"

    def test_missing_file(self) -> None:
        self.assertTrue(
            self.t.patch_file("no.txt", "a", "b").startswith("ERROR: file not found")
        )

    def test_unique_match_replaced(self) -> None:
        self.path.write_text("hello world", encoding="utf-8")
        self.assertEqual(
            self.t.patch_file(str(self.path), "world", "there"),
            "Patch applied successfully.",
        )
        self.assertEqual(self.path.read_text(encoding="utf-8"), "hello there")

    def test_zero_matches_rejected(self) -> None:
        self.path.write_text("abc", encoding="utf-8")
        out = self.t.patch_file(str(self.path), "zzz", "y")
        self.assertEqual(out, "ERROR: old_string not found in file")

    def test_non_unique_rejected(self) -> None:
        self.path.write_text("aa aa aa", encoding="utf-8")
        out = self.t.patch_file(str(self.path), "aa", "bb")
        self.assertTrue(out.startswith("ERROR: old_string appears 3 times"))
        # File left untouched.
        self.assertEqual(self.path.read_text(encoding="utf-8"), "aa aa aa")


class ListDirectoryTests(unittest.TestCase):
    def test_not_a_directory(self) -> None:
        t = ToolExecutor(_cfg())
        self.assertTrue(t.list_directory("no_such_dir").startswith("ERROR: not a directory"))

    def test_empty_dir(self) -> None:
        t = ToolExecutor(_cfg())
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(t.list_directory(d), "(empty directory)")

    def test_lists_entries(self) -> None:
        t = ToolExecutor(_cfg())
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.txt").touch()
            (Path(d) / "b.txt").touch()
            out = t.list_directory(d)
            self.assertIn("a.txt", out)
            self.assertIn("b.txt", out)

    def test_truncates_at_line_cap(self) -> None:
        t = ToolExecutor(_cfg(ctx=256))  # dir_line_cap = 50
        with tempfile.TemporaryDirectory() as d:
            for i in range(200):
                (Path(d) / f"f{i}.txt").touch()
            out = t.list_directory(d)
            self.assertIn("[listing truncated]", out)


class SearchCodeTests(unittest.TestCase):
    def test_not_a_directory(self) -> None:
        t = ToolExecutor(_cfg())
        self.assertTrue(t.search_code("x", "no_such_dir").startswith("ERROR: not a directory"))

    def test_invalid_regex(self) -> None:
        t = ToolExecutor(_cfg())
        with tempfile.TemporaryDirectory() as d:
            out = t.search_code("(", d)
            self.assertTrue(out.startswith("ERROR: invalid regex"))

    def test_no_matches(self) -> None:
        t = ToolExecutor(_cfg())
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "f.txt").write_text("abc\n", encoding="utf-8")
            self.assertEqual(t.search_code("zzz", d), "(no matches)")

    def test_finds_matches_with_filename_line_and_text(self) -> None:
        t = ToolExecutor(_cfg())
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "a.py"
            target.write_text("first\nsecond needle\nthird\n", encoding="utf-8")
            out = t.search_code("needle", d)
            self.assertIn("a.py:2:second needle", out)

    def test_file_pattern_filter(self) -> None:
        t = ToolExecutor(_cfg())
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.py").write_text("hit\n", encoding="utf-8")
            (Path(d) / "a.txt").write_text("hit\n", encoding="utf-8")
            out = t.search_code("hit", d, file_pattern="*.py")
            self.assertIn("a.py", out)
            self.assertNotIn("a.txt", out)


if __name__ == "__main__":
    unittest.main()
