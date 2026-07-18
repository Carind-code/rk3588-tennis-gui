# -*- coding: utf-8 -*-
"""
Cloud Indicator Widget — MQTT and OSS connection status.

Shows colored indicator dots and status text for:
  - MQTT: Green = connected to Alibaba Cloud IoT, Red = disconnected
  - OSS:  Green = upload complete, Yellow = uploading, Red = failed
"""

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QBrush, QColor, QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy


class CloudIndicator(QWidget):
    """MQTT / OSS connection status display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 2)
        layout.setSpacing(2)

        # MQTT + OSS in one compact row
        row = QHBoxLayout()
        row.setSpacing(6)

        self._mqtt_dot = QLabel("●")
        self._mqtt_dot.setStyleSheet("color:#f85149;font-size:12px;border:none;")
        self._mqtt_dot.setFixedWidth(14)
        row.addWidget(self._mqtt_dot)
        self._mqtt_label = QLabel("MQTT: 未连接")
        self._mqtt_label.setStyleSheet("color:#8b949e;font-size:10px;border:none;")
        row.addWidget(self._mqtt_label)

        self._oss_dot = QLabel("●")
        self._oss_dot.setStyleSheet("color:#484f58;font-size:12px;border:none;")
        self._oss_dot.setFixedWidth(14)
        row.addWidget(self._oss_dot)
        self._oss_label = QLabel("OSS: --")
        self._oss_label.setStyleSheet("color:#8b949e;font-size:10px;border:none;")
        row.addWidget(self._oss_label)
        row.addStretch()
        layout.addLayout(row)

    # ── Public API ────────────────────────────────────────────────

    def set_mqtt_status(self, connected: bool, detail: str = ""):
        if connected:
            self._mqtt_dot.setStyleSheet("color:#3fb950;font-size:12px;border:none;")
            self._mqtt_label.setText("MQTT: 已连接")
            self._mqtt_label.setStyleSheet("color:#c9d1d9;font-size:10px;border:none;")
        else:
            self._mqtt_dot.setStyleSheet("color:#f85149;font-size:12px;border:none;")
            txt = "MQTT: {}".format(detail if detail else "未连接")
            self._mqtt_label.setText(txt)
            self._mqtt_label.setStyleSheet("color:#8b949e;font-size:10px;border:none;")

    def set_oss_status(self, status: str, detail: str = ""):
        colors = {"idle":"#484f58","uploading":"#d29922","done":"#3fb950","failed":"#f85149"}
        labels = {"idle":"OSS: --","uploading":"OSS: 上传中","done":"OSS: 完成","failed":"OSS: 失败"}
        self._oss_dot.setStyleSheet("color:{};font-size:12px;border:none;".format(colors.get(status, "#484f58")))
        self._oss_label.setText(labels.get(status, "OSS: {}".format(status)))
        self._oss_label.setStyleSheet("color:{};font-size:10px;border:none;".format(
            "#c9d1d9" if status != "idle" else "#8b949e"))
