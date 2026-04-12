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
ollama pull qwen2.5-coder:7b

# 4. Run the agent
python main.py --model qwen2.5-coder:7b
```

## Setup (Windows)

```powershell
# 1. Install Python 3.10+ from https://python.org
# 2. Install Ollama from https://ollama.com/download/windows
#    (Ollama runs as a background service automatically after install)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull a recommended model
ollama pull qwen2.5-coder:7b

# 5. Run the agent
python main.py --model qwen2.5-coder:7b
```

## Recommended Models

| Model              | Size   | RAM needed | Best for              |
|--------------------|--------|------------|-----------------------|
| `qwen2.5-coder:7b` | ~4.5 GB | 8 GB+     | Most machines         |
| `qwen2.5-coder:14b`| ~9 GB  | 16 GB+     | Better quality        |
| `qwen2.5-coder:32b`| ~19 GB | 32 GB+     | Best quality          |

## Usage

```bash
python main.py [options]
```

### Options

| Flag        | Default                    | Description              |
|-------------|----------------------------|--------------------------|
| `--model`   | `qwen2.5-coder:32b`       | Ollama model to use      |
| `--host`    | `http://localhost:11434`   | Ollama server URL        |
| `--workdir` | `.`                        | Working directory        |
| `--ctx`     | `32768`                    | Context window (tokens)  |

### Keybindings

| Key     | Action                |
|---------|-----------------------|
| Enter   | Send message          |
| Ctrl+L  | Clear chat            |
| Ctrl+N  | New session           |
| Ctrl+C  | Quit                  |

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
