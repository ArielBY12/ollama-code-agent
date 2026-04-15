# Ollama Code Agent

A Claude Code-like agentic coding assistant that runs fully offline using
[Ollama](https://ollama.com) as the local LLM backend and
[Textual](https://textual.textualize.io) as the terminal UI.

Works on **macOS**, **Linux**, and **Windows**.

## Setup (macOS / Linux)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running
ollama serve

# 3. Pull a model (if you haven't already)
ollama pull qwen3-coder:30b

# 4. Run the agent
python main.py --model qwen3-coder:30b
```

## Setup (Windows)

```powershell
# 1. Install Python 3.10+ from https://python.org
# 2. Install Ollama from https://ollama.com/download/windows
#    (runs as a background service automatically after install)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull a recommended model
ollama pull qwen3-coder:30b

# 5. Run the agent — from Windows Terminal, PowerShell, or cmd.exe
python main.py --model qwen3-coder:30b
```

> **Note:** Launch the TUI from a real Windows console (Windows Terminal,
> PowerShell, or `cmd.exe`). Git Bash / MSYS shells do not provide a TTY
> that Textual can drive — `python main.py` will print the header and
> exit silently there.

## Recommended Models

This project targets workstations with ~64 GB RAM, so the defaults lean
toward larger models.

| Model              | Size    | RAM needed | Best for                       |
|--------------------|---------|------------|--------------------------------|
| `qwen3-coder:30b`  | ~19 GB  | 32 GB+     | **Default** — strongest agent  |
| `qwen2.5-coder:32b`| ~19 GB  | 32 GB+     | Alternative dense 32B          |
| `qwen2.5-coder:14b`| ~9 GB   | 16 GB+     | Lighter machines               |
| `qwen2.5-coder:7b` | ~4.5 GB | 8 GB+      | Lowest-resource fallback       |

## Usage

```bash
python main.py [options]
```

### Options

| Flag        | Default                    | Description              |
|-------------|----------------------------|--------------------------|
| `--model`   | `qwen3-coder:30b`         | Ollama model to use      |
| `--host`    | `http://localhost:11434`   | Ollama server URL        |
| `--workdir` | `.`                        | Working directory        |
| `--ctx`     | `131072`                   | Context window (tokens)  |

### Keybindings

| Key     | Action                         |
|---------|--------------------------------|
| Enter   | Send message                   |
| Ctrl+T  | Toggle tool-call rows          |
| Ctrl+R  | Toggle tool-result rows        |
| Ctrl+L  | Clear chat                     |
| Ctrl+N  | New session                    |
| Ctrl+C  | Quit                           |

Tool calls and their results are **hidden by default** — the chat shows
only your messages and the assistant's streamed replies. Press `Ctrl+T`
to reveal the compact tool-call rows (`⚙ read_file(...)`) and `Ctrl+R`
to reveal the one-line tool-result rows. Each toggle is independent.

### Sensitive-tool confirmation

`run_bash`, `write_file`, and `patch_file` are gated by a confirmation
modal before execution — review the command / path / diff preview, then
approve (`y`) or deny (`n` / `Esc`). A denial is reported back to the
model as `ERROR: user denied execution of <tool>` so it can recover.

## Available Tools

The agent has access to the following tools:

- **read_file** — Read file contents
- **write_file** — Create or overwrite a file
- **patch_file** — Replace a unique string in a file
- **run_bash** — Execute a shell command (PowerShell on Windows)
- **list_directory** — List directory contents
- **search_code** — Regex code search across files

## Offline Usage

Once you've installed the dependencies and pulled a model, everything runs
locally. No internet connection is needed. The agent talks to the Ollama
server on your machine — your code never leaves your computer.

## Tests

Unit tests live in `tests/` and use the standard-library `unittest` runner
— no extra dependencies needed.

```bash
python -m unittest discover tests -v
```

Coverage focuses on pure logic: `AgentConfig` wiring, `ToolExecutor` cap
derivation and every tool's happy + error paths, and the fenced-JSON
tool-call fallback parser. The live Ollama client and the Textual UI are
intentionally not unit-tested — those are integration surfaces.

## Architecture

Three layers, strictly separated:

1. **`config.py`** — `AgentConfig` frozen dataclass, the single source of
   truth for runtime settings.
2. **`agent/`** — UI-agnostic. `AgentLoop.run_turn()` yields typed events
   (`ToolCallEvent`, `ToolResultEvent`, `AssistantMessageEvent`,
   `ErrorEvent`) and caps turns at 25 tool-call iterations to guard
   against runaway models.
3. **`ui/`** — Textual frontend. Runs `AgentLoop` on a background worker
   and marshals events back via `call_from_thread()`.

Tool output is capped proportional to `--ctx` so smaller context windows
get proportionally smaller reads/searches — preventing a single tool call
from blowing the model's window.
