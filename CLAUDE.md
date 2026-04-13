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
```

CLI flags: `--model`, `--host` (default `http://localhost:11434`), `--workdir` (agent `chdir`s into this), `--ctx` (token window, default 131072).

There is no test suite, linter, or build step configured.

## Architecture

Three layers, strictly separated — keep them that way when editing:

1. **`config.py`** — `AgentConfig` frozen dataclass, the single source of truth for runtime settings. Built once in `main.py` from CLI args.
2. **`agent/`** — UI-agnostic. `AgentLoop.run_turn()` (`agent/loop.py`) is a generator that yields typed events (`ToolCallEvent`, `ToolResultEvent`, `AssistantMessageEvent`, `ErrorEvent`). It owns the `messages` history and loops calling `ollama.Client.chat(...)` until the model stops requesting tools. **It must never import Textual or call `print()`.**
3. **`ui/`** — Textual frontend. `AgentApp` (`ui/app.py`) runs `AgentLoop` on a `@work(thread=True)` background worker and marshals events back via `call_from_thread()`. Input is disabled for the duration of a turn.

Tool layer (`agent/tools/`) is split deliberately:
- `definitions.py` — pure data, the JSON schemas sent to the LLM. No imports.
- `executor.py` — Python implementations. Every tool returns a **plain string**; errors start with `"ERROR: "`; outputs are capped (`_FILE_CAP=50_000`, `_SEARCH_CAP=6_000`, `_DIR_LINE_CAP=400`) to protect the context window. Dispatch is via the `TOOL_EXECUTORS` dict — adding a new tool means appending to both files and updating that dict.

`run_bash` shells through `powershell -NoProfile -Command` on Windows and the default shell elsewhere, with a hardcoded 30-second timeout.

System prompt lives in `agent/prompts.py` as a single `SYSTEM_PROMPT` constant — edit there to change agent behavior without touching the loop.

`main.py` does pre-flight only: parses args, sets `OLLAMA_HOST` env var, validates workdir, calls `agent.check_ollama()` for a soft warning if the server/model is missing, then `os.chdir(workdir)` so all tool-relative paths resolve correctly before the TUI launches.
