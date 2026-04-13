"""Entry point for the Ollama Code Agent.

Parses CLI arguments, runs pre-flight checks, and launches the TUI.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ollama Code Agent — local agentic coding assistant",
    )
    parser.add_argument(
        "--model",
        default="qwen3-coder:30b",
        help="Ollama model to use (default: qwen3-coder:30b)",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--workdir",
        default=".",
        help="Working directory for the agent (default: .)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=131072,
        help="Context window in tokens (default: 131072)",
    )
    return parser.parse_args()


def check_ollama(host: str, model: str) -> None:
    """Soft pre-flight: warn if Ollama is unreachable or model missing."""
    from agent import check_ollama as _check

    warning = _check(host, model)
    if warning:
        print(warning)


def main() -> None:
    """Run pre-flight checks and launch the TUI."""
    args = parse_args()

    # Set OLLAMA_HOST so the SDK picks it up everywhere.
    os.environ["OLLAMA_HOST"] = args.host

    # Validate workdir.
    workdir = Path(args.workdir).resolve()
    if not workdir.is_dir():
        print(f"Error: workdir does not exist — {workdir}")
        sys.exit(1)

    print(f"Model:   {args.model}")
    print(f"Host:    {args.host}")
    print(f"Workdir: {workdir}")
    print(f"Context: {args.ctx}")
    print()

    check_ollama(args.host, args.model)

    # Change to workdir so relative paths resolve correctly.
    os.chdir(workdir)

    # Build config and launch.
    from config import AgentConfig
    from ui.app import AgentApp

    config = AgentConfig.from_args(args)
    app = AgentApp(config)
    try:
        app.run()
    except BaseException:
        # Textual restores the main screen on exit, so a traceback printed
        # mid-run would otherwise be wiped. Capture, then re-raise.
        import traceback

        tb = traceback.format_exc()
        print("\n--- TUI crashed ---")
        print(tb)
        raise


if __name__ == "__main__":
    main()
