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
    AssistantChunkEvent,
    AssistantMessageEvent,
    ConfirmRequestEvent,
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from config import AgentConfig
from ui.widgets import ChatDisplay, ConfirmModal, PromptInput


class AgentApp(App):
    """Main TUI application for the Ollama coding agent."""

    TITLE = "Ollama Code Agent"

    BINDINGS = [
        Binding("ctrl+l", "clear_chat", "Clear chat"),
        Binding("ctrl+n", "new_session", "New session"),
        Binding("ctrl+t", "toggle_tool_calls", "Toggle tool calls"),
        Binding("ctrl+r", "toggle_tool_results", "Toggle tool results"),
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
        # Tool calls and results are hidden by default; users reveal them
        # explicitly with Ctrl+T / Ctrl+R.
        self._show_tool_calls = False
        self._show_tool_results = False

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
                    chat.add_tool_call,
                    event.name,
                    event.args,
                    not self._show_tool_calls,
                )
            elif isinstance(event, ToolResultEvent):
                self.call_from_thread(
                    chat.add_tool_result,
                    event.name,
                    event.result,
                    not self._show_tool_results,
                )
            elif isinstance(event, AssistantChunkEvent):
                self.call_from_thread(
                    chat.add_assistant_chunk, event.content
                )
            elif isinstance(event, AssistantMessageEvent):
                self.call_from_thread(
                    chat.finalize_assistant_message, event.content
                )
            elif isinstance(event, ConfirmRequestEvent):
                self._handle_confirm(event)
            elif isinstance(event, ErrorEvent):
                self.call_from_thread(
                    chat.add_error, event.message
                )
                # Surface errors outside the scrollback too.
                self.call_from_thread(
                    self.notify,
                    event.message,
                    severity="error",
                    timeout=8,
                )
                self.call_from_thread(self.bell)

        # Re-enable input after the turn finishes.
        prompt = self.query_one(PromptInput)
        self.call_from_thread(self._enable_prompt, prompt)

    def _handle_confirm(self, event: ConfirmRequestEvent) -> None:
        """Show a confirm modal and block the worker on the user's decision."""
        def _show_and_wire() -> None:
            def _on_result(approved: bool | None) -> None:
                event.approved.append(bool(approved))
                event.reply.set()

            self.push_screen(
                ConfirmModal(event.name, event.args), _on_result
            )

        self.call_from_thread(_show_and_wire)
        # Loop's _await_confirmation blocks on event.reply with a timeout —
        # safe to return immediately here.

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

    def action_toggle_tool_calls(self) -> None:
        """Show/hide tool-call rows in the chat (Ctrl+T)."""
        self._show_tool_calls = not self._show_tool_calls
        self.query_one(ChatDisplay).set_tool_calls_hidden(
            not self._show_tool_calls
        )
        state = "shown" if self._show_tool_calls else "hidden"
        self.notify(f"Tool calls {state}", timeout=2)

    def action_toggle_tool_results(self) -> None:
        """Show/hide tool-result rows in the chat (Ctrl+R)."""
        self._show_tool_results = not self._show_tool_results
        self.query_one(ChatDisplay).set_tool_results_hidden(
            not self._show_tool_results
        )
        state = "shown" if self._show_tool_results else "hidden"
        self.notify(f"Tool results {state}", timeout=2)
