# -*- coding: utf-8 -*-
"""
2D Mini-Court Widget — bird's-eye view of standard tennis court (1000×2168).

Draws:
  - Court lines (singles / doubles boundaries)
  - Net line
  - Real-time ball position trail (yellow dots)
  - Bounce events: green "X" for IN, red "X" for OUT
"""

from collections import deque

from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, pyqtSlot
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPainterPath,
)
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy


# Standard court (1000 × 2168)
COURT_W = 1000.0
COURT_H = 2168.0

# Singles boundaries (in standard coords)
SINGLES_LEFT = 125.0
SINGLES_RIGHT = 875.0

# Doubles = full width (0..1000)
DOUBLES_LEFT = 0.0
DOUBLES_RIGHT = 1000.0

# Key vertical lines
NET_Y = 1084.0           # net at center
SERVICE_LINE_TOP = 752.0   # service line (top half)
SERVICE_LINE_BOT = 1416.0  # service line (bottom half)
BASELINE_TOP = 0.0
BASELINE_BOT = COURT_H

# Colors
COLOR_COURT_BG = QColor("#1a3a1a")       # dark green
COLOR_COURT_LINE = QColor("#ffffff")      # white lines
COLOR_COURT_FILL = QColor("#1e5631")      # inner court green
COLOR_NET = QColor("#888888")
COLOR_BALL_TRAIL = QColor(255, 220, 0, 120)  # yellow, semi-transparent
COLOR_BALL_CURRENT = QColor("#ffff00")
COLOR_BOUNCE_IN = QColor("#00ff00")
COLOR_BOUNCE_OUT = QColor("#ff4444")
COLOR_TEXT = QColor("#cccccc")


class MiniCourtWidget(QWidget):
    """Bird's-eye tennis court visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 340)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # State
        self._court_mode = "singles"
        self._ball_positions = deque(maxlen=50)  # (court_x, court_y)
        self._bounce_events = []                  # (court_x, court_y, "IN"|"OUT", frame)
        self._current_ball = None                 # (court_x, court_y)

        # Title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        title = QLabel("2D Mini-Court")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #58a6ff; font-weight: bold; font-size: 12px; border: none;")
        layout.addWidget(title)

        # Refresh timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(100)  # 10 fps refresh

    # ── Public slots ──────────────────────────────────────────────

    @pyqtSlot(object)
    def update_trajectory(self, rows: list):
        """Called when tracknet_rknn_output.csv is updated.

        Args:
            rows: List of dicts with 'court_x', 'court_y' keys.
        """
        if not rows:
            return
        # Take latest data points
        for row in rows[-50:]:
            cx = float(row.get("court_x", 0))
            cy = float(row.get("court_y", 0))
            self._ball_positions.append((cx, cy))
        if rows:
            last = rows[-1]
            self._current_ball = (
                float(last.get("court_x", 0)),
                float(last.get("court_y", 0)),
            )

    def add_bounce_event(self, court_x: float, court_y: float, verdict: str, frame: int):
        """Add a bounce event marker."""
        self._bounce_events.append((court_x, court_y, verdict, frame))
        # Keep only last 20
        if len(self._bounce_events) > 20:
            self._bounce_events = self._bounce_events[-20:]

    def set_court_mode(self, mode: str):
        """Set 'singles' or 'doubles' court boundaries."""
        self._court_mode = mode

    # ── Paint ─────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Margins for axis labels
        margin_x = 30
        margin_y = 10
        usable_w = self.width() - margin_x
        usable_h = self.height() - margin_y * 2

        if usable_w < 50 or usable_h < 50:
            return

        # Scale: court → widget (fit width, maintain aspect)
        scale = min(usable_w / COURT_W, usable_h / COURT_H)
        offset_x = margin_x + (usable_w - COURT_W * scale) / 2
        offset_y = margin_y + (usable_h - COURT_H * scale) / 2

        def to_widget(cx, cy):
            return QPointF(offset_x + cx * scale, offset_y + cy * scale)

        # ── Fill court background ──
        painter.fillRect(
            QRectF(offset_x, offset_y, COURT_W * scale, COURT_H * scale),
            COLOR_COURT_BG,
        )

        # ── Court fill (playing area) ──
        left = SINGLES_LEFT if self._court_mode == "singles" else DOUBLES_LEFT
        right = SINGLES_RIGHT if self._court_mode == "singles" else DOUBLES_RIGHT
        court_rect = QRectF(
            offset_x + left * scale,
            offset_y + BASELINE_TOP * scale,
            (right - left) * scale,
            COURT_H * scale,
        )
        painter.fillRect(court_rect, COLOR_COURT_FILL)

        # ── Court lines ──
        line_pen = QPen(COLOR_COURT_LINE, 1.5)
        painter.setPen(line_pen)

        # Outer boundary
        painter.drawRect(QRectF(
            offset_x, offset_y,
            COURT_W * scale, COURT_H * scale,
        ))

        # Singles lines
        p1 = to_widget(SINGLES_LEFT, BASELINE_TOP)
        p2 = to_widget(SINGLES_LEFT, BASELINE_BOT)
        painter.drawLine(p1, p2)
        p1 = to_widget(SINGLES_RIGHT, BASELINE_TOP)
        p2 = to_widget(SINGLES_RIGHT, BASELINE_BOT)
        painter.drawLine(p1, p2)

        # Net
        net_pen = QPen(COLOR_NET, 2)
        painter.setPen(net_pen)
        p1 = to_widget(DOUBLES_LEFT, NET_Y)
        p2 = to_widget(DOUBLES_RIGHT, NET_Y)
        painter.drawLine(p1, p2)

        # Service lines
        painter.setPen(line_pen)
        p1 = to_widget(left, SERVICE_LINE_TOP)
        p2 = to_widget(right, SERVICE_LINE_TOP)
        painter.drawLine(p1, p2)
        p1 = to_widget(left, SERVICE_LINE_BOT)
        p2 = to_widget(right, SERVICE_LINE_BOT)
        painter.drawLine(p1, p2)

        # Center service line
        center_x = (left + right) / 2
        p1 = to_widget(center_x, SERVICE_LINE_TOP)
        p2 = to_widget(center_x, NET_Y)
        painter.drawLine(p1, p2)
        p1 = to_widget(center_x, NET_Y)
        p2 = to_widget(center_x, SERVICE_LINE_BOT)
        painter.drawLine(p1, p2)

        # ── Ball trail ──
        trail_pen = QPen(COLOR_BALL_TRAIL, 2)
        painter.setPen(trail_pen)
        trail_points = [to_widget(x, y) for (x, y) in self._ball_positions]
        for i in range(len(trail_points) - 1):
            painter.drawLine(trail_points[i], trail_points[i + 1])

        # Current ball
        if self._current_ball:
            cx, cy = self._current_ball
            pt = to_widget(cx, cy)
            painter.setBrush(QBrush(COLOR_BALL_CURRENT))
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawEllipse(pt, 5, 5)

        # ── Bounce events ──
        for bx, by, verdict, frame in self._bounce_events:
            pt = to_widget(bx, by)
            color = COLOR_BOUNCE_IN if verdict.upper() == "IN" else COLOR_BOUNCE_OUT
            painter.setPen(QPen(color, 2.5))
            # Draw "X"
            s = 8
            painter.drawLine(QPointF(pt.x() - s, pt.y() - s), QPointF(pt.x() + s, pt.y() + s))
            painter.drawLine(QPointF(pt.x() + s, pt.y() - s), QPointF(pt.x() - s, pt.y() + s))

            # Label
            font = QFont("DejaVu Sans", 8)
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(QPointF(pt.x() + s + 2, pt.y() + 4), verdict)

        # ── Axis labels ──
        font = QFont("DejaVu Sans", 7)
        painter.setFont(font)
        painter.setPen(COLOR_TEXT)
        painter.drawText(QPointF(2, offset_y + COURT_H * scale / 2), "X")
        painter.drawText(QPointF(offset_x + COURT_W * scale / 2, self.height() - 2), "Y")

        painter.end()
