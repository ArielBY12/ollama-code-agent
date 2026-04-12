"""Tool implementations — each function returns a plain string.

Error messages always start with 'ERROR: '.
Output is capped to prevent blowing up the context window.
"""

from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path
from typing import Optional

# ── size caps ────────────────────────────────────────────────────────
_FILE_CAP = 50_000       # max chars returned from file reads
_SEARCH_CAP = 6_000      # max chars from search results
_DIR_LINE_CAP = 400       # max lines in directory listings


def _cap(text: str, limit: int) -> str:
    """Truncate *text* to *limit* chars, appending a notice if trimmed."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [output truncated]"


# ── tool implementations ────────────────────────────────────────────

def read_file(path: str) -> str:
    """Return file contents, capped at _FILE_CAP characters."""
    target = Path(path)
    if not target.is_file():
        return f"ERROR: file not found — {path}"
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"ERROR: {exc}"
    return _cap(content, _FILE_CAP)


def write_file(path: str, content: str) -> str:
    """Write *content* to *path*, creating parent dirs if needed."""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"ERROR: {exc}"
    return f"Wrote {len(content)} chars to {path}"


def patch_file(path: str, old_string: str, new_string: str) -> str:
    """Replace a unique occurrence of *old_string* in the file."""
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


def run_bash(command: str) -> str:
    """Run a shell command and return combined output + exit code.

    Uses PowerShell on Windows, the default shell on Unix.
    """
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
    return _cap("\n".join(parts), _FILE_CAP)


def list_directory(
    path: str = ".",
    recursive: bool = False,
) -> str:
    """List directory contents, capped at _DIR_LINE_CAP lines."""
    target = Path(path)
    if not target.is_dir():
        return f"ERROR: not a directory — {path}"

    lines: list[str] = []
    try:
        entries = target.rglob("*") if recursive else target.iterdir()
        for entry in entries:
            lines.append(entry.as_posix())
            if len(lines) >= _DIR_LINE_CAP:
                lines.append("... [listing truncated]")
                break
    except OSError as exc:
        return f"ERROR: {exc}"
    return "\n".join(lines) if lines else "(empty directory)"


def search_code(
    pattern: str,
    path: str = ".",
    file_pattern: Optional[str] = None,
) -> str:
    """Regex search for *pattern* under *path*, optionally filtered by glob.

    Pure-Python implementation — no external tools needed.
    """
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
                    if char_count >= _SEARCH_CAP:
                        return _cap("\n".join(matches), _SEARCH_CAP)
    except OSError as exc:
        return f"ERROR: {exc}"

    return "\n".join(matches) if matches else "(no matches)"


# ── dispatch table ──────────────────────────────────────────────────

TOOL_EXECUTORS: dict[str, callable] = {
    "read_file": read_file,
    "write_file": write_file,
    "patch_file": patch_file,
    "run_bash": run_bash,
    "list_directory": list_directory,
    "search_code": search_code,
}
