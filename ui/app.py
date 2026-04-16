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
from ui.widgets import ChatDisplay, ConfirmModal, PromptInput, ToolRowKind


class AgentApp(App):
    TITLE = "Ollama Code Agent"

    BINDINGS = [
        Binding("ctrl+l", "clear_chat", "Clear chat"),
        Binding("ctrl+n", "new_session", "New session"),
        Binding("ctrl+t", "toggle_rows('call')", "Toggle tool calls"),
        Binding("ctrl+r", "toggle_rows('result')", "Toggle tool results"),
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
        yield ChatDisplay()
        yield PromptInput()

    def on_mount(self) -> None:
        self.query_one(PromptInput).focus()

    def on_prompt_input_user_submitted(
        self, event: PromptInput.UserSubmitted
    ) -> None:
        chat = self.query_one(ChatDisplay)
        chat.add_user_message(event.text)

        prompt = self.query_one(PromptInput)
        prompt.disabled = True

        self._run_agent(event.text)

    @work(thread=True)
    def _run_agent(self, user_text: str) -> None:
        chat = self.query_one(ChatDisplay)

        try:
            for event in self._agent.run_turn(user_text):
                if isinstance(event, ToolCallEvent):
                    self.call_from_thread(
                        chat.add_tool_row, "call", event.name, str(event.args)
                    )
                elif isinstance(event, ToolResultEvent):
                    self.call_from_thread(
                        chat.add_tool_row, "result", event.name, event.result
                    )
                elif isinstance(event, AssistantChunkEvent):
                    self.call_from_thread(chat.add_assistant_chunk, event.content)
                elif isinstance(event, AssistantMessageEvent):
                    self.call_from_thread(chat.finalize_assistant_message, event.content)
                elif isinstance(event, ConfirmRequestEvent):
                    self._handle_confirm(event)
                elif isinstance(event, ErrorEvent):
                    self.call_from_thread(chat.add_error, event.message)
                    self.call_from_thread(
                        self.notify, event.message, severity="error", timeout=8
                    )
                    self.call_from_thread(self.bell)
        except Exception as exc:
            # Worker must never leave the prompt wedged — surface the crash
            # and fall through to re-enable input.
            message = f"Agent worker crashed: {type(exc).__name__}: {exc}"
            self.call_from_thread(chat.add_error, message)
            self.call_from_thread(
                self.notify, message, severity="error", timeout=8
            )
        finally:
            prompt = self.query_one(PromptInput)
            self.call_from_thread(self._enable_prompt, prompt)

    def _handle_confirm(self, event: ConfirmRequestEvent) -> None:
        def _show() -> None:
            def _on_result(approved: bool | None) -> None:
                if approved:
                    event.approve()
                else:
                    event.deny()

            self.push_screen(ConfirmModal(event.name, event.args), _on_result)

        self.call_from_thread(_show)

    @staticmethod
    def _enable_prompt(prompt: PromptInput) -> None:
        prompt.disabled = False
        prompt.focus()

    def action_clear_chat(self) -> None:
        self._agent.clear()
        self.query_one(ChatDisplay).remove_children()

    def action_new_session(self) -> None:
        self.action_clear_chat()

    def action_toggle_rows(self, kind: ToolRowKind) -> None:
        hidden = self.query_one(ChatDisplay).toggle_rows_hidden(kind)
        label = "calls" if kind == "call" else "results"
        state = "hidden" if hidden else "shown"
        self.notify(f"Tool {label} {state}", timeout=2)
