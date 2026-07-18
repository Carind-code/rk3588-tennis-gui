# -*- coding: utf-8 -*-
"""
NPU Monitor Widget — displays RK3588 NPU core utilization and FPS.

Shows three NPU core indicators and current inference frame rate.
Data is updated via the top status bar refresh timer.
"""

import os

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QSizePolicy


class NPUMonitor(QWidget):
    """RK3588 NPU three-core status indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # NPU label
        lbl = QLabel("NPU:")
        lbl.setStyleSheet("color: #8b949e; font-size: 11px; border: none;")
        layout.addWidget(lbl)

        # Core indicators
        self._core_labels = []
        for i in range(3):
            core_lbl = QLabel("C{}".format(i))
            core_lbl.setStyleSheet(
                "background-color: #21262d; color: #484f58; font-size: 10px; "
                "border-radius: 3px; padding: 2px 6px; border: none;"
            )
            layout.addWidget(core_lbl)
            self._core_labels.append(core_lbl)

        # FPS
        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet("color: #c9d1d9; font-size: 11px; font-weight: bold; border: none;")
        layout.addWidget(self._fps_label)

        self._load_path = "/sys/devices/platform/fdab0000.npu/devfreq/fdab0000.npu/load"
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_load)
        self._poll_timer.start(1000)
        self._poll_load()

    # ── Public ────────────────────────────────────────────────────

    def set_core_active(self, core_idx: int, active: bool):
        """Highlight a core as active/inactive."""
        if 0 <= core_idx < 3:
            if active:
                self._core_labels[core_idx].setStyleSheet(
                    "background-color: #1a7f37; color: #ffffff; font-size: 10px; "
                    "border-radius: 3px; padding: 2px 6px; border: none;"
                )
            else:
                self._core_labels[core_idx].setStyleSheet(
                    "background-color: #21262d; color: #484f58; font-size: 10px; "
                    "border-radius: 3px; padding: 2px 6px; border: none;"
                )

    def set_fps(self, fps: float):
        """Update FPS display."""
        self._fps_label.setText("FPS: {:.1f}".format(fps))

    def _poll_load(self):
        """Read RK3588's kernel-exported NPU devfreq load once per second."""
        try:
            with open(self._load_path, "r", encoding="utf-8") as fh:
                raw = fh.read().strip()  # e.g. "63@1000000000Hz"
            load = max(0, min(100, int(raw.split("@", 1)[0])))
        except (OSError, ValueError, IndexError):
            for label in self._core_labels:
                label.setText("--")
            return
        # This RK3588 kernel exports one real hardware load counter for the
        # NPU cluster, not three independent per-core counters. Each chip
        # therefore shows the actual shared-cluster utilisation.
        for index, label in enumerate(self._core_labels):
            label.setText("C{} {}%".format(index, load))
            if load >= 70:
                color = "#dc2626"
            elif load >= 30:
                color = "#d97706"
            else:
                color = "#15803d"
            label.setStyleSheet(
                "background-color:{}; color:#ffffff; font-size:10px; "
                "border-radius:3px; padding:2px 6px; border:none;".format(color))

    def set_all_active(self):
        """Light all three cores as active (default during inference)."""
        for i in range(3):
            self.set_core_active(i, True)

    def set_all_idle(self):
        """Dim all cores (idle state)."""
        for i in range(3):
            self.set_core_active(i, False)
