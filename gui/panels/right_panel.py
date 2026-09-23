# -*- coding: utf-8 -*-
"""
Right Data Panel — compact score, events table, action frequency, cloud.
"""

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QSizePolicy, QStackedWidget,
)

from gui.modes import MODE_LIVE, MODE_A, MODE_B
from gui.widgets.score_box import ScoreBox
from gui.widgets.event_table import EventTable
from gui.widgets.cloud_indicator import CloudIndicator


class RightDataPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(240)
        self._last_predictions = []
        self._frequency_display = "count"
        self._action_completed = False

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

        self._freq_title = QLabel("动作技术统计")
        self._freq_title.setAlignment(Qt.AlignCenter)
        self._freq_title.setCursor(Qt.ArrowCursor)
        self._freq_title.setStyleSheet(
            "color:#58a6ff;font-weight:bold;font-size:11px;border:none;")
        self._freq_title.mousePressEvent = self._on_frequency_title_clicked
        l.addWidget(self._freq_title)

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
        # Frame count is the default display. A zero must read as "0", not
        # "0%", until a completed pass explicitly enables percentage switching.
        bar = QLabel("0")
        bar.setAlignment(Qt.AlignCenter)
        bar.setStyleSheet(
            "background:rgba(18,43,64,175);color:{};font-size:9px;font-weight:bold;"
            "border:1px solid {};border-radius:4px;padding:3px 4px;min-width:45px;".format(color, color))
        l.addWidget(bar, 1)
        return w, bar, bar

    def update_frequencies(self, predictions: list):
        """Compatibility slot for completed output files."""
        self.update_action_progress(predictions, completed=True)

    def reset_action_progress(self):
        """Clear side-view counters before a new replay or real task starts."""
        self._last_predictions = []
        self._frequency_display = "count"
        self._action_completed = False
        self._freq_title.setText("动作帧数统计")
        self._freq_title.setCursor(Qt.ArrowCursor)
        for bar in self._freq_bars.values():
            bar.setText("0")
        self._action_label.setText("等待数据...")
        self._action_label.setStyleSheet(
            "background:rgba(9,27,45,150);color:#b7c6d6;font-size:18px;font-weight:bold;"
            "border:1px solid rgba(126,231,255,145);border-radius:8px;padding:8px;")

    def update_action_progress(self, predictions: list, completed=False):
        """Show the number of classifier frames assigned to each action."""
        self._last_predictions = list(predictions or [])
        self._action_completed = bool(completed)
        counts = self._count_action_frames(self._last_predictions)
        total_frames = len(self._last_predictions)
        for key, bar in self._freq_bars.items():
            cnt = counts.get(key, 0)
            # Every input item represents one classified video frame. Unknown
            # labels are deliberately excluded, so the four displayed values
            # can never add up to more than total_frames.
            pct = cnt / total_frames * 100 if total_frames > 0 else 0
            bar.setText("{:.1f}%".format(pct) if self._frequency_display == "percent"
                        else str(cnt))
        if self._last_predictions:
            self.set_action(self._last_predictions[-1])
        if self._action_completed:
            mode_text = ("动作占比（点击查看动作帧数）"
                         if self._frequency_display == "percent"
                         else "动作帧数统计（点击查看占比）")
            self._freq_title.setText(mode_text)
            self._freq_title.setCursor(Qt.PointingHandCursor)
            self._header.setText("个人智练看板 ✓")
        else:
            self._freq_title.setText("动作帧数统计")
            self._freq_title.setCursor(Qt.ArrowCursor)

    @staticmethod
    def _count_action_frames(predictions):
        """Count labeled video frames, never more than the input frame count."""
        counts = {"forehand": 0, "backhand": 0, "serve": 0, "background": 0}
        for label in predictions or []:
            label = str(label).strip().lower()
            if label in counts:
                counts[label] += 1
        return counts

    def _on_frequency_title_clicked(self, event):
        """Percentages are available only after the action pass is complete."""
        if self._action_completed:
            self._frequency_display = (
                "percent" if self._frequency_display == "count" else "count")
            self.update_action_progress(self._last_predictions, completed=True)
        event.accept()

    # ── Mode switching ───────────────────────────────────────────

    def set_mode(self, mode_key):
        if mode_key == MODE_LIVE:
            self._stack.setCurrentIndex(0)
            self._header.setText("实时演示监控")
        elif mode_key == MODE_A:
            self._stack.setCurrentIndex(0)
            self._header.setText("比赛判罚看板")
        elif mode_key == MODE_B:
            self._stack.setCurrentIndex(1)
            self._header.setText("个人智练看板")

    def reset_replay(self, mode_key):
        """Clear only the data panel relevant to a new simulated task."""
        if mode_key == MODE_A:
            self.score_box.reset()
            self.event_table.update_bounce_events([])
            self._header.setText("比赛判罚看板")
        elif mode_key == MODE_B:
            self.reset_action_progress()
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
