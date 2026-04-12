"""Custom Textual widgets for the coding-agent TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Input, Static
from rich.markdown import Markdown
from rich.panel import Panel


class ChatDisplay(VerticalScroll):
    """Scrollable area that renders conversation bubbles."""

    DEFAULT_CSS = """
    ChatDisplay {
        height: 1fr;
        padding: 1 2;
    }
    """

    def add_user_message(self, text: str) -> None:
        """Append a user bubble to the chat."""
        panel = Panel(
            text,
            title="You",
            border_style="cyan",
            expand=True,
        )
        self.mount(Static(panel))
        self.scroll_end(animate=False)

    def add_assistant_message(self, text: str) -> None:
        """Append an assistant bubble with Markdown rendering."""
        panel = Panel(
            Markdown(text),
            title="Assistant",
            border_style="green",
            expand=True,
        )
        self.mount(Static(panel))
        self.scroll_end(animate=False)

    def add_tool_call(self, name: str, args: dict) -> None:
        """Show which tool was called and with what arguments."""
        body = f"[bold]{name}[/bold]({args})"
        panel = Panel(
            body,
            title="Tool Call",
            border_style="yellow",
            expand=True,
        )
        self.mount(Static(panel))
        self.scroll_end(animate=False)

    def add_tool_result(self, name: str, result: str) -> None:
        """Show a tool's return value."""
        # Cap display length so the UI stays responsive.
        display = result[:2000] + "..." if len(result) > 2000 else result
        panel = Panel(
            display,
            title=f"Result: {name}",
            border_style="dim",
            expand=True,
        )
        self.mount(Static(panel))
        self.scroll_end(animate=False)

    def add_error(self, text: str) -> None:
        """Show an error message."""
        panel = Panel(
            text,
            title="Error",
            border_style="red",
            expand=True,
        )
        self.mount(Static(panel))
        self.scroll_end(animate=False)


class PromptInput(Input):
    """Single-line input with a submit message."""

    DEFAULT_CSS = """
    PromptInput {
        dock: bottom;
        margin: 0 2;
    }
    """

    class UserSubmitted(Message):
        """Fired when the user presses Enter with non-empty text."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(self) -> None:
        super().__init__(
            placeholder="Type a message…",
            id="prompt-input",
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Relay non-empty submissions as a PromptInput.Submitted."""
        text = event.value.strip()
        if text:
            self.clear()
            self.post_message(self.UserSubmitted(text))
