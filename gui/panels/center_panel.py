# -*- coding: utf-8 -*-
"""
Center Visual Panel — video display + progress bar / action timeline.

Widgets show/hide based on active scene mode.
"""

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QStackedWidget,
)

from gui.modes import MODE_A, MODE_B
from gui.widgets.video_widget import VideoWidget
from gui.widgets.progress_bar import ProgressBarWidget
from gui.widgets.action_timeline import ActionTimelineWidget


class CenterVisualPanel(QWidget):

    def __init__(self, project_root: str, compact: bool = False, parent=None):
        super().__init__(parent)
        self._project_root = project_root
        self._compact = compact
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background:transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Video widget ──
        self.video_widget = VideoWidget()
        self.video_widget.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_widget, 10)

        # ── Bottom bar: stacked between progress & timeline ──
        bottom = QWidget()
        bottom.setObjectName("bottomBar")
        bottom.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # A compact status rail, anchored to the bottom of the preview.  Its
        # outer width deliberately matches the video widget exactly.
        bottom.setFixedHeight(72 if not compact else 78)
        bottom.setStyleSheet(
            "#bottomBar{background:rgba(7,20,34,78);border-top:2px solid #d8ff45;"
            "border-left:1px solid rgba(126,231,255,140);border-right:1px solid rgba(126,231,255,140);}")
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        self._stack = QStackedWidget()
        self._stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Page 0: Progress bar (default / pipeline running)
        self.progress_bar = ProgressBarWidget()
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._stack.addWidget(self.progress_bar)

        # Page 1: Action timeline (side mode)
        self.action_timeline = ActionTimelineWidget()
        self.action_timeline.setMinimumHeight(70)
        self.action_timeline.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._stack.addWidget(self.action_timeline)

        bottom_layout.addWidget(self._stack, 1)
        layout.addWidget(bottom, 1)

        # Internal signals
        self.video_widget.signal_frame_changed.connect(
            self.action_timeline.on_frame_changed)

    # ── Mode switching ───────────────────────────────────────────

    def set_mode(self, mode_key):
        if mode_key == MODE_A:
            self._stack.setCurrentIndex(0)  # progress bar
        elif mode_key == MODE_B:
            self._stack.setCurrentIndex(0)  # progress bar (no timeline)

    def show_progress(self):
        self._stack.setCurrentIndex(0)

    def show_timeline(self):
        self._stack.setCurrentIndex(1)

    def reset_progress(self):
        self.progress_bar.reset()

    def hide_bottom_bar(self):
        """Hide progress/timeline bar for clean camera view."""
        self.findChild(QWidget, "bottomBar").hide() if hasattr(self, 'findChild') else None
        # Find and hide the bottom widget
        for child in self.children():
            if isinstance(child, QWidget) and child.objectName() == "bottomBar":
                child.hide()

    def show_bottom_bar(self):
        """Restore bottom bar."""
        for child in self.children():
            if isinstance(child, QWidget) and child.objectName() == "bottomBar":
                child.show()

    # ── Public slots ─────────────────────────────────────────────

    @pyqtSlot(str, str)
    def on_video_source_changed(self, source_type: str, path: str):
        if source_type == "file" and path:
            self.video_widget.load_video(path)

    def load_action_data(self, actions: list):
        self.action_timeline.load_actions(actions)
