"""Make Hebrew readable in terminals that cannot do right-to-left text.

VS Code's integrated terminal (xterm.js) draws characters left to right, so Hebrew shows up
backwards. `fix_rtl_output` wraps stdout so that every line containing Hebrew is reordered to
visual order (the bidi algorithm) before it is written. This is for display only: it applies
to an interactive terminal, never to pipes, files or `--json` output, and the data on disk
keeps the logical order.
"""
from __future__ import annotations

import re
import shutil
import sys
import textwrap
from typing import Any, TextIO

HEBREW_RE = re.compile(r"[֐-׿]")


def visual_order(line: str) -> str:
    """Reorder one line (without its newline) from logical to visual left-to-right order."""
    if not HEBREW_RE.search(line):
        return line
    from bidi.algorithm import get_display

    return get_display(line)


def _reorder_keeping_newline(line: str) -> str:
    """Wrap to the terminal width first, then reorder each wrapped line on its own.

    Reordering a whole long line and letting the terminal wrap it afterwards would put the
    end of the sentence on the first row.
    """
    body = line.rstrip("\r\n")
    if not HEBREW_RE.search(body):
        return line
    indent = body[: len(body) - len(body.lstrip())]
    width = max(shutil.get_terminal_size().columns - 1, 20)
    rows = textwrap.wrap(body, width=width, subsequent_indent=indent, break_long_words=False) or [body]
    return "\n".join(visual_order(row) for row in rows) + line[len(body):]


class RtlFixStream:
    """File-like wrapper that reorders each written line containing Hebrew."""

    def __init__(self, stream: TextIO):
        self._stream = stream

    def write(self, text: str) -> int:
        if HEBREW_RE.search(text):
            text = "".join(_reorder_keeping_newline(line) for line in text.splitlines(keepends=True))
        return self._stream.write(text)

    def __getattr__(self, name: str) -> Any:   # flush, isatty, encoding, ...
        return getattr(self._stream, name)


def fix_rtl_output(enabled: bool = True) -> bool:
    """Wrap sys.stdout if it is an interactive terminal. Returns True if it was wrapped."""
    if not enabled or isinstance(sys.stdout, RtlFixStream) or not sys.stdout.isatty():
        return False
    sys.stdout = RtlFixStream(sys.stdout)
    return True
