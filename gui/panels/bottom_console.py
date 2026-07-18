# -*- coding: utf-8 -*-
"""
Bottom Console Log — displays redirected stdout/stderr from subprocesses.

Color-coded output:
  - White (#c9d1d9):  standard output
  - Yellow (#d29922): standard error
  - Cyan (#58a6ff):   system messages
  - Red (#f85149):    errors / critical
"""

import time
from collections import deque

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QTextCursor, QColor, QTextCharFormat, QFont
from PyQt5.QtWidgets import QPlainTextEdit, QSizePolicy


_MAX_LINES = 5000  # maximum lines to keep in buffer


class BottomConsole(QPlainTextEdit):
    """Read-only console log with color-coded output."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(_MAX_LINES)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(150)

        # Monospace font
        font = QFont("DejaVu Sans Mono", 10)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)

        # Cursor for auto-scroll
        self._cursor = self.textCursor()

        # Pending lines buffer (thread-safe)
        self._pending = deque()
        self._flush_timer = QTimer(self)
        self._flush_timer.timeout.connect(self._flush_pending)
        self._flush_timer.start(100)  # flush every 100ms

    # ── Public slots ──────────────────────────────────────────────

    @pyqtSlot(str)
    def append_line(self, text: str):
        """Append a generic line (used for Qt signal connections)."""
        self.append_stdout(text)

    def append_stdout(self, text: str):
        """Append a stdout line (white)."""
        self._pending.append(("stdout", text))

    def append_stderr(self, text: str):
        """Append a stderr line (yellow)."""
        self._pending.append(("stderr", text))

    def append_system(self, text: str):
        """Append a system message (cyan)."""
        self._pending.append(("system", text))

    def append_error(self, text: str):
        """Append an error message (red)."""
        self._pending.append(("error", text))

    # ── Internal ──────────────────────────────────────────────────

    def _flush_pending(self):
        """Flush the pending line buffer to the text widget.

        This is called by a QTimer to batch updates and avoid
        flooding the GUI thread with individual append operations.
        """
        if not self._pending:
            return

        # Take all pending lines
        lines = list(self._pending)
        self._pending.clear()

        for level, text in lines:
            self._append_colored(level, text)

        # Auto-scroll to bottom
        self._cursor.movePosition(QTextCursor.End)
        self.setTextCursor(self._cursor)
        self.ensureCursorVisible()

    def _append_colored(self, level: str, text: str):
        """Append a single line with color based on level."""
        timestamp = time.strftime("%H:%M:%S")

        fmt = QTextCharFormat()
        color_map = {
            "stdout": QColor("#c9d1d9"),
            "stderr": QColor("#d29922"),
            "system": QColor("#58a6ff"),
            "error": QColor("#f85149"),
        }
        fmt.setForeground(color_map.get(level, QColor("#c9d1d9")))

        # Timestamp prefix in dim gray
        self._cursor.movePosition(QTextCursor.End)
        ts_fmt = QTextCharFormat()
        ts_fmt.setForeground(QColor("#484f58"))
        self._cursor.insertText("[{}] ".format(timestamp), ts_fmt)

        # Content
        self._cursor.insertText(text + "\n", fmt)
