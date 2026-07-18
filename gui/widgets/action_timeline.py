# -*- coding: utf-8 -*-
"""
Action Timeline Widget — horizontal scrolling track display.

Shows 4 colored tracks for action classification:
  - Serve      (blue #58a6ff)
  - Forehand   (orange #f0a050)
  - Backhand   (green #3fb950)
  - Background (gray #484f58)

The timeline scrolls right as the video progresses.
"""

from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSlot
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
)
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy


ACTION_COLORS = {
    "serve": QColor("#58a6ff"),
    "forehand": QColor("#f0a050"),
    "backhand": QColor("#3fb950"),
    "background": QColor("#484f58"),
}

ACTION_LABELS_ZH = {
    "serve": "发球",
    "forehand": "正手",
    "backhand": "反手",
    "background": "背景",
}


class ActionTimelineWidget(QWidget):
    """Scrolling action classification timeline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Data
        self._actions = []          # list of action strings
        self._current_index = 0     # highlight position

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        title = QLabel("动作时序图")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #58a6ff; font-weight: bold; font-size: 12px; border: none;")
        layout.addWidget(title)

        # Refresh
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(200)

        # Scroll offset
        self._scroll_offset = 0

    # ── Public ────────────────────────────────────────────────────

    def load_actions(self, actions: list):
        """Load action labels (e.g. ['serve', 'serve', 'forehand', ...])."""
        self._actions = actions
        self._scroll_offset = 0
        self.update()

    def set_current_frame(self, frame_idx: int):
        """Highlight the current frame's action."""
        self._current_index = frame_idx
        self.update()

    @pyqtSlot(int)
    def on_frame_changed(self, frame_idx: int):
        self._current_index = frame_idx
        # Auto-scroll to keep current position visible
        if frame_idx > self._scroll_offset + 50:
            self._scroll_offset = frame_idx - 50

    # ── Paint ─────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        track_top = 30
        track_h = 14
        track_gap = 4

        # Background
        painter.fillRect(0, 0, w, h, QColor("#0d1117"))

        if not self._actions:
            painter.setPen(QColor("#484f58"))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "等待动作数据...")
            painter.end()
            return

        # How many frames fit horizontally
        px_per_frame = 4
        visible_frames = w // px_per_frame

        start = max(0, self._scroll_offset)
        end = min(len(self._actions), start + visible_frames + 1)

        # Draw tracks for each action type
        track_names = ["serve", "forehand", "backhand", "background"]
        for track_idx, action_name in enumerate(track_names):
            y = track_top + track_idx * (track_h + track_gap)

            # Track label
            label = ACTION_LABELS_ZH.get(action_name, action_name)
            painter.setPen(QColor("#8b949e"))
            font = QFont("DejaVu Sans", 8)
            painter.setFont(font)
            painter.drawText(4, y + track_h - 2, label)

            # Draw action blocks
            label_w = 36
            block_start = None
            current_action = None

            for i in range(start, end):
                action = self._actions[i]
                x = label_w + (i - start) * px_per_frame

                if action != current_action:
                    # Close previous block
                    if block_start is not None and current_action is not None:
                        self._draw_block(painter, block_start, y, x - block_start, track_h, current_action)
                    block_start = x
                    current_action = action

            # Close last block
            if block_start is not None and current_action is not None:
                last_x = label_w + (end - start) * px_per_frame
                self._draw_block(painter, block_start, y, last_x - block_start, track_h, current_action)

        # Current position indicator (red vertical line)
        if start <= self._current_index < end:
            cx = label_w + (self._current_index - start) * px_per_frame
            painter.setPen(QPen(QColor("#f85149"), 2))
            painter.drawLine(cx, track_top, cx, track_top + 4 * (track_h + track_gap))

        painter.end()

    def _draw_block(self, painter, x, y, w, h, action):
        """Draw a colored block for an action segment."""
        color = ACTION_COLORS.get(action, QColor("#484f58"))
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        rect = QRectF(x + 1, y, max(1, w - 2), h)
        painter.drawRoundedRect(rect, 3, 3)
