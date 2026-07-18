# -*- coding: utf-8 -*-
"""
Event Table Widget — displays bounce events from bounce_events.csv.

Columns: [帧号] [场地坐标] [置信度] [判定结果]

Clicking a row emits signal_seek_frame(frame_num) so the video widget
can jump to that timestamp.
"""

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QBrush
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QAbstractItemView,
)


class EventTable(QWidget):
    """Bounce events table with click-to-seek."""

    signal_seek_frame = pyqtSignal(int)

    COLUMNS = ["帧号", "场地坐标", "置信度", "判定"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        # Title
        title = QLabel("落地事件表")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "background:rgba(8,27,45,132);color:#7ee7ff;font-weight:bold;font-size:13px;"
            "border:1px solid rgba(126,231,255,115);border-radius:6px;padding:5px;")
        layout.addWidget(title)

        # Table
        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(True)
        self._table.setStyleSheet(
            "QTableWidget{background:rgba(5,17,31,140);"
            "border:1px solid rgba(126,231,255,135);border-radius:8px;}"
            "QTableWidget::item{background:transparent;}")

        # Column sizing
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        # Click handler
        self._table.cellClicked.connect(self._on_cell_clicked)

        layout.addWidget(self._table)

        # Data store
        self._events = []

    # ── Public slot ───────────────────────────────────────────────

    @pyqtSlot(object)
    def update_bounce_events(self, events: list):
        """Replace table contents with new bounce events.

        Args:
            events: list of dicts with keys 'frame', 'court_x', 'court_y',
                    'confidence', 'verdict'.
        """
        self._events = events
        self._table.setRowCount(0)
        self._table.setRowCount(len(events))

        for row_idx, evt in enumerate(events):
            # Frame number
            frame = evt.get("frame", evt.get("frame_num", ""))
            self._set_cell(row_idx, 0, str(frame))

            # Court coordinate (CSV uses x,y, not court_x/court_y)
            cx = evt.get("x", evt.get("court_x", 0))
            cy = evt.get("y", evt.get("court_y", 0))
            coord_text = "({:.0f}, {:.0f})".format(
                float(cx) if cx else 0, float(cy) if cy else 0)
            self._set_cell(row_idx, 1, coord_text)

            # Confidence (CSV uses score)
            conf = evt.get("score", evt.get("confidence", evt.get("conf", 0)))
            try:
                conf_val = float(conf)
                conf_text = "{:.1%}".format(min(conf_val, 1.0))
            except (ValueError, TypeError):
                conf_text = "-"
                conf_val = 0
            item = self._set_cell(row_idx, 2, conf_text)
            try:
                if conf_val >= 0.7:
                    item.setForeground(QBrush(QColor("#3fb950")))
                elif conf_val >= 0.4:
                    item.setForeground(QBrush(QColor("#d29922")))
                else:
                    item.setForeground(QBrush(QColor("#f85149")))
            except (ValueError, TypeError):
                pass

            # Event type as verdict
            evt_type = evt.get("event_type", evt.get("verdict", ""))
            item = self._set_cell(row_idx, 3, str(evt_type))
            v = str(evt_type).upper()
            if "BOUNCE" in v or "IN" in v:
                item.setForeground(QBrush(QColor("#3fb950")))
                item.setFont(QFont("DejaVu Sans", 10, QFont.Bold))
            elif "OUT" in v:
                item.setForeground(QBrush(QColor("#f85149")))
                item.setFont(QFont("DejaVu Sans", 10, QFont.Bold))

    # ── Helpers ───────────────────────────────────────────────────

    def _set_cell(self, row, col, text):
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        self._table.setItem(row, col, item)
        return item

    def _on_cell_clicked(self, row, col):
        """Emit signal to seek video to clicked event's frame."""
        if row < len(self._events):
            evt = self._events[row]
            frame = evt.get("frame", evt.get("frame_num", 0))
            try:
                frame_num = int(frame)
                self.signal_seek_frame.emit(frame_num)
            except (ValueError, TypeError):
                pass
