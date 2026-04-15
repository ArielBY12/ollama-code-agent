# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Claude Code-style agentic coding TUI that runs fully offline against a local Ollama server. The user picks a model (default `qwen3-coder:30b`), and the agent drives a multi-step tool-use loop over file/shell tools, rendered in a Textual terminal UI. Designed for workstations with ~64 GB RAM.

## Common commands

```bash
pip install -r requirements.txt          # install deps (ollama, textual, rich)
ollama serve                             # ensure Ollama is up (Windows installer runs it as a service)
ollama pull qwen3-coder:30b              # pull a model if not present
python main.py --model qwen3-coder:30b   # run the TUI
python -m unittest discover tests -v     # run all tests (stdlib unittest, no extra deps)
python -m unittest tests.test_executor -v                                           # single test module
python -m unittest tests.test_executor.TestToolExecutor.test_read_file_happy_path   # single test
```

CLI flags: `--model`, `--host` (default `http://localhost:11434`), `--workdir` (agent `chdir`s into this), `--ctx` (token window, default 131072).

No linter or build step is configured. Tests cover pure logic only — the live Ollama client and the Textual UI are intentionally not unit-tested.

> **Windows caveat:** launch the TUI from Windows Terminal, PowerShell, or `cmd.exe`. Git Bash / MSYS shells don't provide a TTY Textual can drive — `python main.py` will print the header and exit silently there.

## Architecture

Three layers, strictly separated — keep them that way when editing:

1. **`config.py`** — `AgentConfig` frozen dataclass, the single source of truth for runtime settings. Built once in `main.py` from CLI args.
2. **`agent/`** — UI-agnostic. `AgentLoop.run_turn()` (`agent/loop.py`) is a generator that yields typed events (`ToolCallEvent`, `ToolResultEvent`, `AssistantChunkEvent`, `AssistantMessageEvent`, `ConfirmRequestEvent`, `ErrorEvent`). It owns the `messages` history and loops calling `ollama.Client.chat(..., stream=True)` until the model stops requesting tools. Hard-capped at `_MAX_TURN_ITERATIONS = 100` tool calls per user turn to guard against a runaway model. **It must never import Textual or call `print()`.**
3. **`ui/`** — Textual frontend. `AgentApp` (`ui/app.py`) runs `AgentLoop` on a `@work(thread=True)` background worker and marshals events back via `call_from_thread()`. Input is disabled for the duration of a turn.

### Streaming + confirmation contract

- The loop calls `client.chat(..., stream=True)` and drains the iterator, emitting `AssistantChunkEvent(content=delta)` per non-empty token delta and a single `AssistantMessageEvent(content=full)` at stream end (only when no tool call follows). The UI appends chunks into a persistent `Static` bubble via `ChatDisplay.add_assistant_chunk` and commits the final text with `finalize_assistant_message`.
- Structured `message.tool_calls` may arrive in any chunk; the loop keeps the last non-empty one. The fenced-JSON fallback parser runs against the fully assembled content after the stream drains.
- **Sensitive tools** (`_SENSITIVE_TOOLS = {"run_bash", "write_file", "patch_file"}`) are gated. Before dispatch the loop yields a `ConfirmRequestEvent(name, args, reply: threading.Event, approved: list[bool])` and blocks on `reply.wait(timeout=300s)`. The consumer MUST append one bool to `approved` and set `reply`. A denial synthesizes `"ERROR: user denied execution of <tool>"` so the model gets feedback. Read-only tools (`read_file`, `list_directory`, `search_code`) are never gated.

### UI conventions

- Tool calls and results render as compact one-line rows with CSS classes `tool-call` / `tool-result` and an opt-in `hidden` class. `AgentApp` toggles visibility via `Ctrl+T` / `Ctrl+R`; both start hidden.
- `ErrorEvent` fires three things: an in-scroll red panel, a Textual `notify(severity="error")` toast, and a terminal bell.
- `ConfirmModal` (`ui/widgets.ConfirmModal`) is a `ModalScreen[bool]` pushed via `call_from_thread`; its dismiss callback populates the event and unblocks the worker.

### Tool layer (`agent/tools/`)

- `definitions.py` — pure data, the JSON schemas sent to the LLM. No imports.
- `executor.py` — Python implementations. Every tool returns a **plain string**; errors start with `"ERROR: "`. `ToolExecutor.dispatch()` flattens every exception (including `TypeError` for bad arg shapes) into an `ERROR:` string so a buggy tool can't kill the worker thread.
- Dispatch is via the `_executors` dict built in `ToolExecutor.__init__` — adding a new tool means appending to both `definitions.py` and that dict.
- **Output caps scale with `--ctx`** (see `ToolExecutor.__init__`): `_file_cap = ctx*4//3`, `_search_cap = ctx*4//20`, `_dir_line_cap = max(50, ctx//80)`. Shrinking `--ctx` shrinks every tool's max output — intentional, so a single read can't blow the window.

### Tool-call parsing fallback

Some Ollama models (notably `qwen2.5-coder:7b`) emit tool calls as fenced JSON in `message.content` instead of populating `message.tool_calls`. `agent/loop.py:_extract_text_tool_calls` parses fenced `json` / `tool_call` blocks, and falls back to scanning bare JSON objects with a `"name"` key. Extracted calls are filtered against registered tool names before execution. Pinned by `tests/test_loop_helpers.py` — don't regress this when touching the loop.

### Shell execution

`run_bash` shells through `powershell -NoProfile -Command` on Windows and the default shell elsewhere, with a hardcoded 30-second timeout.

### System prompt

Lives in `agent/prompts.py` as a single `SYSTEM_PROMPT` constant — edit there to change agent behavior without touching the loop.

### Entry point

`main.py` does pre-flight only: parses args, sets `OLLAMA_HOST` env var, validates workdir, calls `agent.check_ollama()` for a soft warning if the server/model is missing, then `os.chdir(workdir)` so all tool-relative paths resolve correctly before the TUI launches.
