"""Textual application — layout, keybindings, and event wiring.

Threading is owned here: the agent loop runs in a background worker
and posts UI updates via call_from_thread().
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual import work

from agent.loop import (
    AgentLoop,
    AssistantMessageEvent,
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from config import AgentConfig
from ui.widgets import ChatDisplay, PromptInput


class AgentApp(App):
    """Main TUI application for the Ollama coding agent."""

    TITLE = "Ollama Code Agent"

    BINDINGS = [
        Binding("ctrl+l", "clear_chat", "Clear chat"),
        Binding("ctrl+n", "new_session", "New session"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }
    """

    def __init__(self, config: AgentConfig) -> None:
        super().__init__()
        self._config = config
        self._agent = AgentLoop(config)

    def compose(self) -> ComposeResult:
        """Build the widget tree."""
        yield ChatDisplay()
        yield PromptInput()

    def on_mount(self) -> None:
        """Focus the input on startup."""
        self.query_one(PromptInput).focus()

    # ── user input handling ─────────────────────────────────────────

    def on_prompt_input_user_submitted(
        self, event: PromptInput.UserSubmitted
    ) -> None:
        """Handle a user message: render it and kick off the agent."""
        chat = self.query_one(ChatDisplay)
        chat.add_user_message(event.text)

        prompt = self.query_one(PromptInput)
        prompt.disabled = True

        self._run_agent(event.text)

    # ── background agent worker ─────────────────────────────────────

    @work(thread=True)
    def _run_agent(self, user_text: str) -> None:
        """Drive the agent loop on a background thread."""
        chat = self.query_one(ChatDisplay)

        for event in self._agent.run_turn(user_text):
            if isinstance(event, ToolCallEvent):
                self.call_from_thread(
                    chat.add_tool_call, event.name, event.args
                )
            elif isinstance(event, ToolResultEvent):
                self.call_from_thread(
                    chat.add_tool_result, event.name, event.result
                )
            elif isinstance(event, AssistantMessageEvent):
                self.call_from_thread(
                    chat.add_assistant_message, event.content
                )
            elif isinstance(event, ErrorEvent):
                self.call_from_thread(
                    chat.add_error, event.message
                )

        # Re-enable input after the turn finishes.
        prompt = self.query_one(PromptInput)
        self.call_from_thread(self._enable_prompt, prompt)

    @staticmethod
    def _enable_prompt(prompt: PromptInput) -> None:
        """Re-enable and focus the prompt input."""
        prompt.disabled = False
        prompt.focus()

    # ── keybinding actions ──────────────────────────────────────────

    def action_clear_chat(self) -> None:
        """Clear chat display and conversation history (Ctrl+L)."""
        self._agent.clear()
        chat = self.query_one(ChatDisplay)
        chat.remove_children()

    def action_new_session(self) -> None:
        """Start a fresh session (Ctrl+N)."""
        self.action_clear_chat()
