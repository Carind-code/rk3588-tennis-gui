# -*- coding: utf-8 -*-
"""
Right Data Panel — compact score, events table, action frequency, cloud.
"""

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QSizePolicy, QStackedWidget,
)

from gui.modes import MODE_A, MODE_B
from gui.widgets.score_box import ScoreBox
from gui.widgets.event_table import EventTable
from gui.widgets.cloud_indicator import CloudIndicator


class RightDataPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(240)

        layout = QVBoxLayout(self)
        # Keep every match/training card inside the right glass edge on the
        # narrow board display.  The asymmetric right gutter protects against
        # the splitter handle and table scrollbar.
        layout.setContentsMargins(8, 2, 12, 2)
        layout.setSpacing(2)

        # Cloud indicator at TOP so bottom console doesn't block it
        self.cloud_indicator = CloudIndicator()
        layout.addWidget(self.cloud_indicator)

        # Header
        self._header = QLabel("比赛判罚看板")
        self._header.setAlignment(Qt.AlignCenter)
        self._header.setMinimumWidth(0)
        self._header.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._header.setStyleSheet(
            "background:rgba(8,27,45,132);color:#d8ff45;font-weight:bold;"
            "font-size:12px;border:1px solid rgba(126,231,255,115);"
            "border-radius:6px;padding:6px;")
        layout.addWidget(self._header)

        # Stacked panels
        self._stack = QStackedWidget()
        self._match_panel = self._build_match_panel()
        self._stack.addWidget(self._match_panel)
        self._training_panel = self._build_training_panel()
        self._stack.addWidget(self._training_panel)
        layout.addWidget(self._stack, 1)

    # ── Match panel ──────────────────────────────────────────────

    def _build_match_panel(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 4, 0)
        l.setSpacing(3)

        self.score_box = ScoreBox()
        l.addWidget(self.score_box)

        self.event_table = EventTable()
        l.addWidget(self.event_table, 1)
        return w

    # ── Training panel ───────────────────────────────────────────

    def _build_training_panel(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(0, 0, 4, 0)
        l.setSpacing(4)

        lbl = QLabel("当前动作")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            "color:#58a6ff;font-weight:bold;font-size:12px;border:none;")
        l.addWidget(lbl)

        self._action_label = QLabel("等待数据...")
        self._action_label.setAlignment(Qt.AlignCenter)
        self._action_label.setMinimumWidth(0)
        self._action_label.setWordWrap(True)
        self._action_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._action_label.setMinimumHeight(32)
        self._action_label.setStyleSheet(
            "background:rgba(9,27,45,150);color:#b7c6d6;font-size:18px;font-weight:bold;"
            "border:1px solid rgba(126,231,255,145);border-radius:8px;padding:8px;")
        l.addWidget(self._action_label)

        lbl2 = QLabel("动作频率统计")
        lbl2.setAlignment(Qt.AlignCenter)
        lbl2.setStyleSheet(
            "color:#58a6ff;font-weight:bold;font-size:11px;border:none;")
        l.addWidget(lbl2)

        # Frequency bars — store references for updates
        self._freq_bars = {}
        for key, label, color in [
            ("forehand", "正手", "#f0a050"),
            ("backhand", "反手", "#3fb950"),
            ("serve", "发球", "#58a6ff"),
            ("background", "背景", "#484f58"),
        ]:
            fw, flabel, fbar = self._make_freq_bar(label, color)
            self._freq_bars[key] = fbar
            l.addWidget(fw)

        l.addStretch()
        return w

    def _make_freq_bar(self, label, color):
        w = QWidget()
        l = QHBoxLayout(w)
        # Reserve a small right gutter so the percentage bar never crosses
        # the glass-panel boundary on the narrow side-view layout.
        l.setContentsMargins(0, 1, 10, 1)
        l.setSpacing(4)
        name = QLabel(label)
        name.setStyleSheet(
            "color:#8b949e;font-size:10px;border:none;min-width:56px;max-width:56px;")
        l.addWidget(name)
        bar = QLabel("0%")
        bar.setAlignment(Qt.AlignCenter)
        bar.setStyleSheet(
            "background:rgba(18,43,64,175);color:{};font-size:9px;font-weight:bold;"
            "border:1px solid {};border-radius:4px;padding:3px 4px;min-width:45px;".format(color, color))
        l.addWidget(bar, 1)
        return w, bar, bar

    def update_frequencies(self, predictions: list):
        if not predictions:
            return
        total = len(predictions)
        counts = {"forehand": 0, "backhand": 0, "serve": 0, "background": 0}
        for p in predictions:
            if p in counts:
                counts[p] += 1
        for key, bar in self._freq_bars.items():
            cnt = counts.get(key, 0)
            pct = cnt / total * 100 if total > 0 else 0
            bar.setText("{:.1f}%".format(pct))
        self._action_label.setText("数据分析成功")
        self._action_label.setStyleSheet(
            "background:rgba(26,127,55,155);color:#efffe9;font-size:16px;font-weight:bold;"
            "border:1px solid #73e49a;border-radius:8px;padding:8px;")
        self._header.setText("个人智练看板 ✓")

    # ── Mode switching ───────────────────────────────────────────

    def set_mode(self, mode_key):
        if mode_key == MODE_A:
            self._stack.setCurrentIndex(0)
            self._header.setText("比赛判罚看板")
        elif mode_key == MODE_B:
            self._stack.setCurrentIndex(1)
            self._header.setText("个人智练看板")

    # ── Public slots ─────────────────────────────────────────────

    @pyqtSlot(object)
    def update_judgement(self, data: dict):
        self.score_box.update_judgement(data)
        self._header.setText("比赛判罚看板 ✓")

    @pyqtSlot(object)
    def update_bounce_events(self, events: list):
        self.event_table.update_bounce_events(events)

    def set_action(self, action: str):
        colors = {"serve": "#58a6ff", "forehand": "#f0a050",
                  "backhand": "#3fb950", "background": "#8b949e"}
        labels = {"serve": "发球 SERVE", "forehand": "正手 FOREHAND",
                  "backhand": "反手 BACKHAND", "background": "背景"}
        color = colors.get(action, "#8b949e")
        label = labels.get(action, action)
        self._action_label.setText(label)
        self._action_label.setStyleSheet(
            "background:rgba(9,27,45,150);color:{};font-size:18px;font-weight:bold;"
            "border:1px solid {};border-radius:8px;padding:8px;".format(color, color))
