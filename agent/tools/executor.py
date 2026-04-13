"""Tool implementations — each method returns a plain string.

Error messages always start with 'ERROR: '.
Output is capped proportional to AgentConfig.ctx to protect the window.
"""

from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

from config import AgentConfig


def _cap(text: str, limit: int) -> str:
    """Truncate *text* to *limit* chars, appending a notice if trimmed."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [output truncated]"


class ToolExecutor:
    """Owns tool implementations with caps derived from the context window."""

    def __init__(self, config: AgentConfig) -> None:
        ctx = config.ctx
        # Rough heuristic: ~4 chars/token. Fractions chosen so that the
        # default ctx=32768 reproduces the previous hardcoded caps.
        self._file_cap = ctx * 4 // 3
        self._search_cap = ctx * 4 // 20
        self._dir_line_cap = max(50, ctx // 80)

        self._executors: dict[str, Callable[..., str]] = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "patch_file": self.patch_file,
            "run_bash": self.run_bash,
            "list_directory": self.list_directory,
            "search_code": self.search_code,
        }

    def dispatch(self, name: str, args: dict) -> str:
        """Run tool *name* with *args*, returning its string result.

        Any exception — including tool bugs — is flattened to an error
        string so a failing tool can't take the worker thread down.
        """
        executor = self._executors.get(name)
        if executor is None:
            return f"ERROR: unknown tool '{name}'"
        try:
            return executor(**args)
        except TypeError as exc:
            return f"ERROR: bad arguments — {exc}"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    # ── tool implementations ────────────────────────────────────────

    def read_file(self, path: str) -> str:
        target = Path(path)
        if not target.is_file():
            return f"ERROR: file not found — {path}"
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"ERROR: {exc}"
        return _cap(content, self._file_cap)

    def write_file(self, path: str, content: str) -> str:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: {exc}"
        return f"Wrote {len(content)} chars to {path}"

    def patch_file(self, path: str, old_string: str, new_string: str) -> str:
        target = Path(path)
        if not target.is_file():
            return f"ERROR: file not found — {path}"
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            return f"ERROR: {exc}"

        count = text.count(old_string)
        if count == 0:
            return "ERROR: old_string not found in file"
        if count > 1:
            return (
                f"ERROR: old_string appears {count} times — "
                "it must be unique"
            )

        new_text = text.replace(old_string, new_string, 1)
        try:
            target.write_text(new_text, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: {exc}"
        return "Patch applied successfully."

    def run_bash(self, command: str) -> str:
        """Run a shell command. PowerShell on Windows, default shell on Unix."""
        if platform.system() == "Windows":
            cmd_args = ["powershell", "-NoProfile", "-Command", command]
            use_shell = False
        else:
            cmd_args = command
            use_shell = True

        try:
            result = subprocess.run(
                cmd_args,
                shell=use_shell,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out after 30 seconds"
        except OSError as exc:
            return f"ERROR: {exc}"

        parts: list[str] = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(result.stderr)
        parts.append(f"[exit code: {result.returncode}]")
        return _cap("\n".join(parts), self._file_cap)

    def list_directory(self, path: str = ".", recursive: bool = False) -> str:
        target = Path(path)
        if not target.is_dir():
            return f"ERROR: not a directory — {path}"

        lines: list[str] = []
        try:
            entries = target.rglob("*") if recursive else target.iterdir()
            for entry in entries:
                lines.append(entry.as_posix())
                if len(lines) >= self._dir_line_cap:
                    lines.append("... [listing truncated]")
                    break
        except OSError as exc:
            return f"ERROR: {exc}"
        return "\n".join(lines) if lines else "(empty directory)"

    def search_code(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: Optional[str] = None,
    ) -> str:
        """Regex search for *pattern* under *path*, optionally filtered by glob."""
        target = Path(path)
        if not target.is_dir():
            return f"ERROR: not a directory — {path}"

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"ERROR: invalid regex — {exc}"

        glob_pat = file_pattern or "*"
        matches: list[str] = []
        char_count = 0

        try:
            for filepath in target.rglob(glob_pat):
                if not filepath.is_file():
                    continue
                try:
                    text = filepath.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    continue
                for lineno, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        entry = f"{filepath.as_posix()}:{lineno}:{line}"
                        matches.append(entry)
                        char_count += len(entry)
                        if char_count >= self._search_cap:
                            return _cap("\n".join(matches), self._search_cap)
        except OSError as exc:
            return f"ERROR: {exc}"

        return "\n".join(matches) if matches else "(no matches)"
