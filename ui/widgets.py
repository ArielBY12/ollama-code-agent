"""Custom Textual widgets for the coding-agent TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static
from rich.markdown import Markdown
from rich.panel import Panel


# Summarising a tool result down to one line keeps the compact row
# readable while the full payload remains in model history.
_TOOL_RESULT_SUMMARY_CHARS = 120


def _one_line(text: str, limit: int = _TOOL_RESULT_SUMMARY_CHARS) -> str:
    """Collapse *text* to a single line, elided to *limit* chars."""
    first = text.splitlines()[0] if text else ""
    if len(first) > limit:
        first = first[: limit - 1] + "…"
    return first


class ChatDisplay(VerticalScroll):
    """Scrollable area that renders conversation bubbles."""

    DEFAULT_CSS = """
    ChatDisplay {
        height: 1fr;
        padding: 1 2;
    }

    ChatDisplay .tool-row {
        color: $text-muted;
        padding: 0 1;
    }

    ChatDisplay .tool-call.hidden,
    ChatDisplay .tool-result.hidden {
        display: none;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        # Tracks the last streaming-assistant widget so chunk deltas
        # append in place instead of creating a new bubble per token.
        self._active_assistant: Static | None = None
        self._active_text: str = ""

    def add_user_message(self, text: str) -> None:
        """Append a user bubble to the chat."""
        self._active_assistant = None
        self._active_text = ""
        panel = Panel(
            text,
            title="You",
            border_style="cyan",
            expand=True,
        )
        self.mount(Static(panel))
        self.scroll_end(animate=False)

    def _render_assistant_panel(self, text: str) -> Panel:
        return Panel(
            Markdown(text) if text else "",
            title="Assistant",
            border_style="green",
            expand=True,
        )

    def add_assistant_chunk(self, delta: str) -> None:
        """Append streamed text to the in-progress assistant bubble."""
        if not delta:
            return
        self._active_text += delta
        if self._active_assistant is None:
            self._active_assistant = Static(
                self._render_assistant_panel(self._active_text)
            )
            self.mount(self._active_assistant)
        else:
            self._active_assistant.update(
                self._render_assistant_panel(self._active_text)
            )
        self.scroll_end(animate=False)

    def finalize_assistant_message(self, text: str) -> None:
        """Commit the assistant bubble after the stream completes.

        If streaming produced a bubble, update it with the authoritative
        full text (guards against delta desync). Otherwise mount a fresh
        bubble — covers non-streaming code paths.
        """
        if self._active_assistant is not None:
            self._active_text = text
            self._active_assistant.update(
                self._render_assistant_panel(text)
            )
        else:
            self.mount(Static(self._render_assistant_panel(text)))
        self._active_assistant = None
        self._active_text = ""
        self.scroll_end(animate=False)

    def add_tool_call(self, name: str, args: dict, hidden: bool) -> None:
        """Show a compact single-line tool-call row."""
        summary = _one_line(f"{name}({args})")
        widget = Static(f"[dim]⚙ {summary}[/dim]", classes="tool-row tool-call")
        if hidden:
            widget.add_class("hidden")
        self.mount(widget)
        self.scroll_end(animate=False)

    def add_tool_result(self, name: str, result: str, hidden: bool) -> None:
        """Show a compact single-line tool-result row."""
        summary = _one_line(result)
        style = "red" if result.startswith("ERROR:") else "dim"
        widget = Static(
            f"[{style}]↳ {name}: {summary}[/{style}]",
            classes="tool-row tool-result",
        )
        if hidden:
            widget.add_class("hidden")
        self.mount(widget)
        self.scroll_end(animate=False)

    def add_error(self, text: str) -> None:
        """Show an error message."""
        self._active_assistant = None
        self._active_text = ""
        panel = Panel(
            text,
            title="Error",
            border_style="red",
            expand=True,
        )
        self.mount(Static(panel))
        self.scroll_end(animate=False)

    def set_tool_calls_hidden(self, hidden: bool) -> None:
        for w in self.query(".tool-call"):
            w.set_class(hidden, "hidden")

    def set_tool_results_hidden(self, hidden: bool) -> None:
        for w in self.query(".tool-result"):
            w.set_class(hidden, "hidden")


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


class ConfirmModal(ModalScreen[bool]):
    """Blocks the UI until the user approves or denies a tool call."""

    BINDINGS = [
        Binding("y", "decide(True)", "Approve"),
        Binding("n", "decide(False)", "Deny"),
        Binding("escape", "decide(False)", "Deny"),
    ]

    DEFAULT_CSS = """
    ConfirmModal {
        align: center middle;
    }
    ConfirmModal #dialog {
        width: 70%;
        max-width: 100;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }
    ConfirmModal #preview {
        margin-top: 1;
        padding: 1;
        border: solid $panel;
        max-height: 15;
        overflow: auto;
    }
    ConfirmModal Horizontal {
        margin-top: 1;
        align-horizontal: right;
        height: auto;
    }
    ConfirmModal Button {
        margin-left: 2;
    }
    """

    def __init__(self, tool_name: str, args: dict) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._args = args

    def compose(self) -> ComposeResult:
        preview = _build_preview(self._tool_name, self._args)
        with Vertical(id="dialog"):
            yield Static(
                f"[b]Confirm tool:[/b] [yellow]{self._tool_name}[/yellow]"
            )
            yield Static(preview, id="preview")
            yield Static("[dim]y = approve · n/Esc = deny[/dim]")
            with Horizontal():
                yield Button("Deny", id="deny", variant="error")
                yield Button("Approve", id="approve", variant="success")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve")

    def action_decide(self, approve: bool) -> None:
        self.dismiss(approve)


def _build_preview(name: str, args: dict) -> str:
    """Human-readable, truncated preview of a sensitive tool invocation."""
    if name == "run_bash":
        cmd = str(args.get("command", "")).strip()
        return f"[b]command:[/b]\n{cmd}"
    if name == "write_file":
        path = args.get("path", "?")
        content = str(args.get("content", ""))
        byte_count = len(content.encode("utf-8"))
        first = _one_line(content, 200)
        return (
            f"[b]path:[/b] {path}\n"
            f"[b]size:[/b] {byte_count} bytes\n"
            f"[b]preview:[/b] {first}"
        )
    if name == "patch_file":
        path = args.get("path", "?")
        old = _one_line(str(args.get("old_string", "")), 200)
        new = _one_line(str(args.get("new_string", "")), 200)
        return (
            f"[b]path:[/b] {path}\n"
            f"[red]- {old}[/red]\n"
            f"[green]+ {new}[/green]"
        )
    # Fallback for any future sensitive tool.
    return str(args)
