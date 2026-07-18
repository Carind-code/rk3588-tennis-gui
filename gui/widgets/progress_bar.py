# -*- coding: utf-8 -*-
"""Progress bar — pipeline running + optional cloud upload."""

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtWidgets import QWidget, QSizePolicy


class ProgressBarWidget(QWidget):
    """Shows pipeline progress (indeterminate) + optional cloud bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMinimumHeight(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._running = False
        self._cloud_pct = 0
        self._cloud_visible = False
        self._phase = "就绪"
        self._pulse_pos = 0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)

    def start_running(self, cloud=False):
        self._running = True
        self._cloud_visible = cloud
        self._cloud_pct = 0
        self._phase = "流水线运行中..."
        self._pulse_pos = 0
        self._pulse_timer.start(40)
        self.update()

    def stop_running(self):
        self._running = False
        self._pulse_timer.stop()
        self._phase = "完成"
        self.update()

    def set_cloud_progress(self, pct):
        self._cloud_pct = pct
        self._cloud_visible = True
        self.update()

    def reset(self):
        self._running = False
        self._pulse_timer.stop()
        self._cloud_pct = 0
        self._cloud_visible = False
        self._phase = "就绪"
        self._pulse_pos = 0
        self.update()

    def _pulse(self):
        self._pulse_pos = (self._pulse_pos + 3) % 100
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        painter.fillRect(0, 0, w, h, QColor(7, 20, 34, 78))
        margin = 10
        bar_w = w - margin * 2
        bar_h = 11
        font = QFont("DejaVu Sans", 9)
        painter.setFont(font)

        if self._running:
            # Indeterminate pulsing bar
            pw = int(bar_w * 0.25)
            px = int((bar_w - pw) * self._pulse_pos / 100)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#21262d"))
            if self._cloud_visible:
                label_y, bar_y = 16, 21
            else:
                label_y, bar_y = max(16, h - 31), max(21, h - 26)
            painter.drawRoundedRect(margin, bar_y, bar_w, bar_h, 4, 4)
            painter.setBrush(QColor("#ccff00"))
            painter.drawRoundedRect(margin + px, bar_y, pw, bar_h, 4, 4)
            painter.setPen(QColor("#8b949e"))
            painter.drawText(margin, label_y, "流水线: {}".format(self._phase))
            painter.setPen(QColor("#c9d1d9"))
            painter.drawText(w - 50, label_y, "处理中...")
        else:
            painter.setPen(QColor("#8b949e"))
            painter.drawText(margin, max(16, h - 17), "流水线: {}".format(self._phase))

        if self._cloud_visible and self._cloud_pct > 0:
            cloud_y = h - bar_h - 7
            painter.setPen(QColor("#8b949e"))
            painter.drawText(margin, cloud_y - 2, "云上传:")
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#21262d"))
            painter.drawRoundedRect(margin, cloud_y, bar_w, bar_h, 4, 4)
            painter.setBrush(QColor("#58a6ff"))
            painter.drawRoundedRect(margin, cloud_y, int(bar_w * self._cloud_pct / 100), bar_h, 4, 4)
            painter.setPen(QColor("#c9d1d9"))
            painter.drawText(w - 50, cloud_y - 2, "{}%".format(self._cloud_pct))
        elif self._cloud_visible:
            cloud_y = h - bar_h - 7
            painter.setPen(QColor("#8b949e"))
            painter.drawText(margin, cloud_y - 2, "云上传: 准备中...")

        painter.end()
