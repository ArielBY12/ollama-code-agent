"""Agent package — LLM interaction loop and tool layer."""

from __future__ import annotations

from typing import List

import ollama  # type: ignore[import-untyped]


def check_ollama(host: str, model: str) -> str | None:
    """Soft pre-flight: return a warning string, or None if all OK.

    Lives here so that main.py never imports ollama directly.
    """
    try:
        client = ollama.Client(host=host)
        response = client.list()
        names: List[str] = [m.model for m in response.models]
        if not any(model in n for n in names):
            return (
                f"Warning: model '{model}' not found on server. "
                "Available: " + ", ".join(names or ["(none)"])
            )
    except Exception as exc:
        return f"Warning: could not reach Ollama at {host} — {exc}"
    return None
