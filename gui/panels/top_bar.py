# -*- coding: utf-8 -*-
"""
Top Status Bar — system title, NPU monitor, camera/cloud status, batch upload.
"""

import subprocess

from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtWidgets import (
    QToolBar, QLabel, QWidget, QHBoxLayout, QSizePolicy, QPushButton,
)



class TopStatusBar(QToolBar):

    def __init__(self, parent=None):
        super().__init__("QiuWu AI", parent)
        self.setMovable(False)
        self.setFloatable(False)
        self.setStyleSheet(
            "QToolBar{background:#0a1728;border-bottom:1px solid #1d3855;"
            "spacing:10px;padding:5px 22px;}")

        # Title
        title = QLabel("QiuWu AI | 球悟AI")
        title.setStyleSheet(
            "color:#d8ff45;font-size:20px;font-weight:bold;border:none;"
            "padding-right:6px;")
        self.addWidget(title)

        # Mode label
        self._mode_label = QLabel("俯视 · 比赛智判")
        self._mode_label.setStyleSheet(
            "color:#7ee7ff;font-size:11px;border:none;")
        self.addWidget(self._mode_label)

        self.addWidget(self._sep())

        # Camera status
        self._cam_status = QLabel("相机: --")
        self._cam_status.setStyleSheet(
            "color:#8b949e;font-size:11px;border:none;")
        self.addWidget(self._cam_status)

        self.addWidget(self._sep())

        # Cloud status
        self._cloud_status = QLabel("云端: --")
        self._cloud_status.setStyleSheet(
            "color:#8b949e;font-size:11px;border:none;")
        self.addWidget(self._cloud_status)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.addWidget(spacer)

        # Batch upload button
        self._btn_batch = QPushButton("批量上云")
        self._btn_batch.setStyleSheet(
            "QPushButton{background:#235878;color:#fff;border:1px solid #3a7696;"
            "border-radius:4px;padding:4px 12px;font-size:11px;font-weight:bold}"
            "QPushButton:hover{background:#388bfd}")
        self.addWidget(self._btn_batch)

        # FPS tracking
        self._frame_count = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(1000)
        self.set_camera_status(False)
        self._refresh()

    def _sep(self):
        s = QLabel("│")
        s.setStyleSheet("color:#30363d;font-size:14px;border:none;")
        return s

    # ── Public ───────────────────────────────────────────────────

    def set_batch_upload_callback(self, callback):
        self._btn_batch.clicked.connect(callback)

    def set_mode_label(self, text):
        self._mode_label.setText(text)

    def set_camera_status(self, connected, resolution=""):
        if connected:
            t = "相机: 已连接"
            if resolution:
                t += " ({})".format(resolution)
            self._cam_status.setStyleSheet(
                "color:#3fb950;font-size:11px;border:none;")
        else:
            t = "相机: 未连接"
            self._cam_status.setStyleSheet(
                "color:#f85149;font-size:11px;border:none;")
        self._cam_status.setText(t)

    def set_cloud_status(self, status):
        colors = {"connected":"#3fb950","connecting":"#d29922",
                  "disconnected":"#f85149","disabled":"#484f58"}
        labels = {"connected":"云端: 已连接","connecting":"云端: 连接中",
                  "disconnected":"云端: 断开","disabled":"云端: --"}
        self._cloud_status.setText(labels.get(status, status))
        self._cloud_status.setStyleSheet(
            "color:{};font-size:11px;border:none;".format(
                colors.get(status, "#8b949e")))

    @pyqtSlot(object)
    def on_trajectory_data(self, rows):
        pass

    def _refresh(self):
        # The cloud SDK runs in the cloud pipeline process. Presence of its
        # main process is the only truthful connection signal available to
        # the GUI when that process owns the MQTT callback.
        try:
            running = subprocess.run(
                ["pgrep", "-f", "match_judgement/cloud/main.py"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=.2).returncode == 0
        except (OSError, subprocess.SubprocessError):
            running = False
        self.set_cloud_status("connected" if running else "disconnected")
