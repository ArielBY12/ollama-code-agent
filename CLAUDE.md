# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Claude Code-style agentic coding TUI that runs fully offline against a local Ollama server. Default model `qwen3-coder:30b`. The agent drives a multi-step tool-use loop (file + shell tools) rendered in a Textual terminal UI. Designed for workstations with ~64 GB RAM.

## Common commands

```bash
pip install -r requirements.txt          # install deps (ollama, textual, rich)
ollama serve                             # ensure Ollama is up (Windows installer runs it as a service)
ollama pull qwen3-coder:30b              # pull a model if not present
python main.py --model qwen3-coder:30b   # run the TUI
python -m unittest discover tests -v     # run all tests (stdlib unittest, no extra deps)
python -m unittest tests.test_executor -v                                           # single module
python -m unittest tests.test_executor.TestToolExecutor.test_read_file_happy_path   # single test
```

CLI flags: `--model`, `--host` (default `http://localhost:11434`), `--workdir` (agent `chdir`s into this), `--ctx` (token window, default 131072).

No linter or build step is configured. Tests cover pure logic only — the live Ollama client and the Textual UI are intentionally not unit-tested.

> **Windows caveat:** launch the TUI from Windows Terminal, PowerShell, or `cmd.exe`. Git Bash / MSYS shells don't provide a TTY Textual can drive — `python main.py` will print the header and exit silently there.

## Architecture

Three layers, strictly separated — keep them that way when editing:

1. **`config.py`** — `AgentConfig` frozen dataclass, the single source of truth for runtime settings. Built once in `main.py` from CLI args.
2. **`agent/`** — UI-agnostic. `AgentLoop.run_turn()` (`agent/loop.py`) is a generator that yields typed events (`ToolCallEvent`, `ToolResultEvent`, `AssistantChunkEvent`, `AssistantMessageEvent`, `ConfirmRequestEvent`, `ErrorEvent`). Owns the `messages` history and loops calling `ollama.Client.chat(..., stream=True, keep_alive="30m")` until the model stops requesting tools. Hard-capped at `_MAX_TURN_ITERATIONS = 100` per user turn to guard against a runaway model. **Never import Textual here. Never call `print()`.**
3. **`ui/`** — Textual frontend. `AgentApp` (`ui/app.py`) runs `AgentLoop` on a `@work(thread=True)` background worker and marshals events back via `call_from_thread()`. Input is disabled for the duration of a turn.

### Streaming contract

- `client.chat(..., stream=True)` returns an iterator; `_consume_stream` drains it, emitting `AssistantChunkEvent(delta)` per non-empty token and a single `AssistantMessageEvent(full)` at turn end — but **only when no tool call follows**. Tool-calling responses never emit `AssistantMessageEvent`.
- During the stream, `ChatDisplay.add_assistant_chunk` appends deltas into a persistent `Static` rendering **plain text** (no Markdown). Markdown parsing runs once in `finalize_assistant_message`. This avoids O(N²) re-parse per token.
- Structured `message.tool_calls` may arrive in any chunk; last non-empty wins. The fenced-JSON / `<tool_call>` / `<function=...>` fallback parsers (`_extract_text_tool_calls`, `_extract_tag_tool_calls`) run against the assembled content after the stream drains, and results are filtered against registered tool names.

### Sensitive-tool confirmation

- `SENSITIVE_TOOL_NAMES` lives in `agent/tools/definitions.py` alongside `TOOL_SCHEMAS` (single source of truth). Currently: `run_bash`, `write_file`, `patch_file`.
- Before dispatch the loop yields a `ConfirmRequestEvent`. The consumer calls `event.approve()` or `event.deny()`; the loop waits with `event.wait(timeout=_CONFIRM_TIMEOUT_SECONDS)` (300 s). Timeout or no reply is treated as denial. Denial synthesises `"ERROR: user denied execution of <tool>"` so the model can recover.
- Read-only tools (`read_file`, `list_directory`, `search_code`) are never gated.

### Tool layer (`agent/tools/`)

- `definitions.py` — pure data. `TOOL_SCHEMAS` is the JSON schema sent to the LLM; `SENSITIVE_TOOL_NAMES` is the gating list. No imports.
- `executor.py` — Python implementations. Every tool returns a **plain string**; errors start with `"ERROR: "`. `ToolExecutor.dispatch()` flattens every exception (including `TypeError` for bad arg shapes) into an `"ERROR: …"` string so a buggy tool can't kill the worker thread.
- Dispatch is via the `_executors` dict built in `ToolExecutor.__init__`. **Adding a new tool means appending to three places**: `TOOL_SCHEMAS` in `definitions.py`, `_executors` in `executor.py`, and — if it mutates disk or runs code — `SENSITIVE_TOOL_NAMES` in `definitions.py`.
- **Output caps scale with `--ctx`** (see `ToolExecutor.__init__`): `_file_cap = ctx*4//3`, `_search_cap = ctx*4//20`, `_dir_line_cap = max(50, ctx//80)`. Shrinking `--ctx` shrinks every tool's max output — intentional, so a single read can't blow the window.

### UI conventions

- Tool calls and results render as compact single-line rows with CSS classes `tool-call` / `tool-result` and an opt-in `hidden` class. Both start hidden.
- `ChatDisplay` owns the per-kind hidden state (`self._hidden: dict[ToolRowKind, bool]`) and exposes `add_tool_row(kind, name, body)` + `toggle_rows_hidden(kind)`. `AgentApp`'s single `action_toggle_rows(kind)` is bound as `Ctrl+T` (`"call"`) and `Ctrl+R` (`"result"`).
- `ErrorEvent` fires three things: an in-scroll red panel, a Textual `notify(severity="error", timeout=8)` toast, and the terminal bell.
- `ConfirmModal` (`ui/widgets.ConfirmModal`) is a `ModalScreen[bool]` pushed via `call_from_thread`; its dismiss callback calls `event.approve()` / `event.deny()`.

### Shell execution

`run_bash` shells through `powershell -NoProfile -Command` on Windows and the default shell elsewhere, with a hardcoded 30-second timeout.

### System prompt

Lives in `agent/prompts.py` as a single `SYSTEM_PROMPT` constant. Contains strict clarification rules (ask before guessing) and confirmation awareness (sensitive tools are user-gated; denial is a signal, not a retry trigger). Edit there to change agent behavior without touching the loop.

### Entry point

`main.py` does pre-flight only: parses args, sets `OLLAMA_HOST` env var, validates workdir, calls `agent.check_ollama()` for a soft warning if the server/model is missing, then `os.chdir(workdir)` so all tool-relative paths resolve correctly before the TUI launches.
