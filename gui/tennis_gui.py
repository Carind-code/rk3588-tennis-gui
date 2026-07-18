#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QiuWu AI | 球悟AI — Mode-driven tennis analysis GUI.

Two scenes → one unified RUN button:
  Scene A: 俯视·比赛智判 (match judgement)
  Scene B: 侧视·个人智练 (side training)
"""

import os, sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QStatusBar, QLabel, QMessageBox, QDockWidget, QPushButton,
    QApplication, QDesktopWidget, QFrame, QScroller, QDialogButtonBox, QDialog,
    QPlainTextEdit, QScrollArea, QTableWidget,
)
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QPolygon

from gui.app import create_application
from gui.modes import MODE_A, MODE_B, get_mode_info, get_script
from gui.panels.left_panel import LeftModePanel
from gui.panels.center_panel import CenterVisualPanel
from gui.panels.right_panel import RightDataPanel
from gui.panels.top_bar import TopStatusBar
from gui.panels.bottom_console import BottomConsole
from gui.backend.process_manager import ProcessManager
from gui.backend.file_watcher import FileWatcher
from gui.backend.data_parser import DataParser


COMPACT_QSS = """
* { font-size: 11px; }
QGroupBox { font-size: 11px; padding: 6px 4px 4px 4px; margin-top: 8px; }
QGroupBox::title { font-size: 10px; padding: 0 4px; }
QPushButton { padding: 4px 8px; font-size: 11px; min-height: 24px; }
QPushButton#btnRun { font-size: 14px; min-height: 36px; }
QPushButton#btnStop { font-size: 14px; min-height: 36px; }
QLabel#sceneTitle { font-size: 13px; }
QTableWidget { font-size: 10px; }
QPlainTextEdit { font-size: 9px; }
"""


class HUDFrameOverlay(QWidget):
    """Non-interactive sci-fi outline drawn above the full application."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()
        if width < 80 or height < 80:
            return
        margin, cut = 7, 22
        points = [
            (margin + cut, margin), (width - margin - cut, margin),
            (width - margin, margin + cut), (width - margin, height - margin - cut),
            (width - margin - cut, height - margin), (margin + cut, height - margin),
            (margin, height - margin - cut), (margin, margin + cut),
            (margin + cut, margin),
        ]
        polygon = QPolygon([QPoint(x, y) for x, y in points])
        painter.setPen(QPen(QColor(32, 211, 255, 235), 2))
        painter.drawPolyline(polygon)

        # Inner corner brackets and short technical ticks inspired by the
        # selected Tennis Command reference, without covering any content.
        painter.setPen(QPen(QColor(32, 211, 255, 190), 1))
        inner, arm = margin + 5, 38
        for x, y, dx, dy in ((inner, inner, 1, 1), (width - inner, inner, -1, 1),
                             (inner, height - inner, 1, -1), (width - inner, height - inner, -1, -1)):
            painter.drawLine(x, y, x + dx * arm, y)
            painter.drawLine(x, y, x, y + dy * arm)
        painter.setPen(QPen(QColor(32, 211, 255, 235), 2))
        for x in (width // 4, width // 2, width * 3 // 4):
            painter.drawLine(x - 11, margin, x + 11, margin)
            painter.drawLine(x - 11, height - margin, x + 11, height - margin)
        painter.end()


class TennisGUI(QMainWindow):

    signal_log = pyqtSignal(str)

    def __init__(self, compact=False):
        super().__init__()
        self._compact = compact

        self.process_mgr = ProcessManager()
        self.data_parser = DataParser(project_root=_PROJECT_ROOT)
        self.file_watcher = FileWatcher(project_root=_PROJECT_ROOT)

        self._active_mode = MODE_A
        self._cloud_enabled = False
        self._running = False
        self._mode_state = {}

        self.setWindowTitle("QiuWu AI | 球悟AI 网球智练智判系统")
        if compact:
            self.setMinimumSize(800, 480)
            self.setWindowFlags(Qt.FramelessWindowHint)
        else:
            self.setMinimumSize(1280, 800)
            self.resize(1920, 1080)

        self._build_top_bar()
        self._build_central()
        self._build_bottom()
        self._wire_signals()

        self._hud_frame = HUDFrameOverlay(self)
        self._hud_frame.setGeometry(self.rect())
        self._hud_frame.show()
        self._hud_frame.raise_()

        if compact:
            self.showFullScreen()
            # Nudge window top-left to center on DSI panel
            g = self.geometry()
            self.setGeometry(g.x() - 8, g.y() - 4, g.width(), g.height())
        self._enable_touch_scroll()

        self._log("[系统] QiuWu AI 球悟AI 已启动 — {}".format(
            "紧凑模式" if compact else "标准模式"))
        self._log("[系统] 项目根: {}".format(_PROJECT_ROOT))

        info = get_mode_info(MODE_A)
        self.center_panel.video_widget.show_placeholder(info["name"])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_hud_frame'):
            self._hud_frame.setGeometry(self.rect())
            self._hud_frame.raise_()

    # ── Build ────────────────────────────────────────────────────

    def _build_top_bar(self):
        self.top_bar = TopStatusBar()
        self.top_bar.setMovable(False)

        if self._compact:
            for name, icon in [("_btn_menu", "☰"), ("_btn_data", "📊"),
                               ("_btn_log", "📜"), ("_btn_exit", "✕")]:
                btn = QPushButton(icon)
                btn.setFixedSize(32, 28)
                btn.setStyleSheet(
                    "QPushButton{background:#13243a;border:1px solid #284864;"
                    "border-radius:6px;color:#e4edf6;font-size:14px}"
                    "QPushButton:hover{background:#193452;border-color:#d8ff45}")
                if icon == "✕":
                    btn.setStyleSheet(btn.styleSheet().replace(
                        "#13243a", "#e34b58").replace("#284864", "#ff8490"))
                    btn.clicked.connect(self.close)
                setattr(self, name, btn)
            self.top_bar.insertWidget(None, self._btn_menu)
            self.top_bar.addWidget(self._btn_log)
            self.top_bar.addWidget(self._btn_data)
            self.top_bar.addWidget(self._btn_exit)

        self.addToolBar(Qt.TopToolBarArea, self.top_bar)
        self.top_bar.set_batch_upload_callback(self.run_batch_upload)

    def _build_central(self):
        central = QWidget()
        central.setObjectName("mainCentral")
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        # Keep every interactive panel inside the cyber-frame safe area.
        # Let the three working panels meet the top and bottom edges; only the
        # horizontal inset is retained for the outer technical frame.
        root_layout.setContentsMargins(14, 0, 14, 0)
        root_layout.setSpacing(8)

        if self._compact:
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            self.center_panel = CenterVisualPanel(_PROJECT_ROOT, compact=True)
            layout.addWidget(self.center_panel)
            container = QWidget()
            container.setLayout(layout)
            root_layout.addWidget(container)

            self.left_panel = LeftModePanel(self)
            self.right_panel = RightDataPanel()

            self._dock_left = QDockWidget("场景模式")
            self._dock_left.setWidget(self.left_panel)
            self._dock_left.setMaximumWidth(360)
            self._dock_left.setMinimumWidth(260)
            self._dock_left.close()
            self.addDockWidget(Qt.LeftDockWidgetArea, self._dock_left)

            self._dock_right = QDockWidget("数据面板")
            self._dock_right.setWidget(self.right_panel)
            self._dock_right.setMaximumWidth(380)
            self._dock_right.setMinimumWidth(260)
            self._dock_right.close()
            self.addDockWidget(Qt.RightDockWidgetArea, self._dock_right)

            self._dock_log = QDockWidget("日志")
            self._dock_log.setWidget(BottomConsole())
            self._dock_log.setMaximumHeight(180)
            self._dock_log.close()
            self.addDockWidget(Qt.BottomDockWidgetArea, self._dock_log)
            self.bottom_console = self._dock_log.widget()

            self._btn_menu.clicked.connect(
                lambda: self._dock_left.show() if self._dock_left.isHidden()
                else self._dock_left.close())
            self._btn_data.clicked.connect(
                lambda: self._dock_right.show() if self._dock_right.isHidden()
                else self._dock_right.close())
            self._btn_log.clicked.connect(
                lambda: self._dock_log.show() if self._dock_log.isHidden()
                else self._dock_log.close())
        else:
            splitter = QSplitter(Qt.Horizontal)
            splitter.setHandleWidth(1)
            self.left_panel = LeftModePanel(self)
            self.left_panel.setMinimumWidth(260)
            self.left_panel.setMaximumWidth(320)
            splitter.addWidget(self.left_panel)
            self.center_panel = CenterVisualPanel(_PROJECT_ROOT)
            splitter.addWidget(self.center_panel)
            self.right_panel = RightDataPanel()
            self.right_panel.setMinimumWidth(280)
            splitter.addWidget(self.right_panel)
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 5)
            splitter.setStretchFactor(2, 3)
            splitter.setSizes([260, 700, 380])
            root_layout.addWidget(splitter)
            self.bottom_console = BottomConsole()

    def _build_bottom(self):
        if self._compact:
            return
        dock = QDockWidget("终端日志")
        dock.setWidget(self.bottom_console)
        dock.setFeatures(QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        # Keep the workspace vertically aligned on the board. The log is not
        # useful while empty, so do not reserve a large blank dock at startup.
        dock.hide()

    # ── Signals ──────────────────────────────────────────────────

    def _wire_signals(self):
        self.process_mgr.signal_stdout.connect(self._on_stdout)
        self.process_mgr.signal_stderr.connect(self._on_stderr)
        self.process_mgr.signal_process_started.connect(self._on_started)
        self.process_mgr.signal_process_finished.connect(self._on_finished)

        self.file_watcher.signal_judgement_updated.connect(
            self.right_panel.update_judgement)
        self.file_watcher.signal_bounce_events_updated.connect(
            self.right_panel.update_bounce_events)
        self.file_watcher.signal_trajectory_updated.connect(
            self.top_bar.on_trajectory_data)
        self.file_watcher.signal_action_prediction_updated.connect(
            self.center_panel.action_timeline.load_actions)
        self.file_watcher.signal_action_prediction_updated.connect(
            self.right_panel.update_frequencies)

        self.center_panel.video_widget.signal_frame_captured.connect(
            self.left_panel._on_frame_captured)
        self.center_panel.video_widget.signal_camera_stopped.connect(
            self.left_panel._on_camera_stopped)
        self.center_panel.video_widget.signal_camera_status.connect(
            self.top_bar.set_camera_status)
        self.signal_log.connect(self.bottom_console.append_line)

    # ── Mode switching ───────────────────────────────────────────

    def set_mode(self, mode_key):
        old_mode = getattr(self, '_active_mode', None)
        if old_mode and old_mode != mode_key:
            self._mode_state[old_mode] = {
                'cloud': self._cloud_enabled,
                'file_path': self.left_panel._src_path.text(),
                'cam_checked': self.left_panel._src_cam.isChecked(),
            }

        self._active_mode = mode_key
        info = get_mode_info(mode_key)

        if mode_key in self._mode_state:
            st = self._mode_state[mode_key]
            self._cloud_enabled = st['cloud']
            self.left_panel._cloud_check.setChecked(st['cloud'])
            self.left_panel._src_path.setText(st['file_path'])
            if st['cam_checked']:
                self.left_panel._src_cam.setChecked(True)
            else:
                self.left_panel._src_file.setChecked(True)
        else:
            self._cloud_enabled = False
            self.left_panel._cloud_check.setChecked(False)
            self.right_panel.cloud_indicator.set_mqtt_status(False)
            self.top_bar.set_cloud_status("disabled")

        self._log("[模式] 切换 → {}".format(info["name"]))
        self.center_panel.set_mode(mode_key)
        self.center_panel.video_widget.set_mode(mode_key)
        self.right_panel.set_mode(mode_key)
        self.top_bar.set_mode_label(info["name"])

    def set_cloud(self, enabled):
        self._cloud_enabled = enabled
        name = get_mode_info(self._active_mode)["name"]
        if enabled:
            self._log("[{}] 自动上云 ON — 点击 RUN 后连接".format(name))
            self.right_panel.cloud_indicator.set_mqtt_status(False, "待启动")
            self.top_bar.set_cloud_status("disabled")
        else:
            self._log("[{}] 本地模式".format(name))
            self.right_panel.cloud_indicator.set_mqtt_status(False, "未连接")
            self.top_bar.set_cloud_status("disabled")

    # ── Pipeline ─────────────────────────────────────────────────

    def run_pipeline(self):
        info = get_mode_info(self._active_mode)
        if self._cloud_enabled:
            self._log("[云端] MQTT+OSS 自动上传已启用")

        if self.left_panel._src_cam.isChecked():
            video = self.center_panel.video_widget
            if not getattr(video, '_camera_mode', False):
                if not video.start_camera_preview():
                    self._log("[相机] 预览启动失败，任务未开始")
                    return False
                self.left_panel._cam_preview_on = True
                self.left_panel._btn_cam.setText("关闭预览")
            self._running = True
            self._log("[相机] 实时任务已启动：{}".format(info["name"]))
            self._log("[实时] 可按需开启网球检测/人体骨架；录制由下方录制按钮独立控制")
            return True

        # Validate file is selected
        video_path = self.left_panel._src_path.text().strip()
        if not video_path or not os.path.exists(video_path):
            self._msgbox("未选择文件",
                        "请先选择视频文件再启动运行。\n\n"
                        "步骤：选择 ○ 本地文件 → 点击 📂 浏览文件")
            self._log("[错误] 未选择有效的视频文件")
            return False

        self._log("[启动] {}".format(info["name"]))
        self._log("[输入] 本地文件: {}".format(video_path))
        script = get_script(self._active_mode, self._cloud_enabled)
        cmd = ["bash", os.path.join(_PROJECT_ROOT, script),
               "--video", video_path]
        task_name = info["task_name"]

        # Start progress bar
        self.center_panel.progress_bar.start_running(cloud=self._cloud_enabled)

        self.process_mgr.start_script(task_name, cmd, cwd=_PROJECT_ROOT)
        self.file_watcher.start_watching(task_name)
        self._running = True
        return True

    def stop_pipeline(self):
        info = get_mode_info(self._active_mode)
        self.center_panel.progress_bar.stop_running()
        if self.left_panel._src_cam.isChecked():
            rec_path = getattr(self.left_panel, '_rec_path', None)
            self.left_panel._on_stop_recording()
            self.center_panel.video_widget.stop_camera_preview()
            self.left_panel._cam_preview_on = False
            self.left_panel._btn_cam.setText("相机实时预览")
            if not rec_path or not os.path.exists(rec_path):
                self._running = False
                self._log("[相机] 没有可汇总的录制文件")
                return False
            self._log("[相机] 录制结束，开始缓冲汇总：{}".format(rec_path))
            script = get_script(self._active_mode, self._cloud_enabled)
            self.center_panel.progress_bar.start_running(cloud=self._cloud_enabled)
            self.process_mgr.start_script(
                info["task_name"],
                ["bash", os.path.join(_PROJECT_ROOT, script), "--video", rec_path],
                cwd=_PROJECT_ROOT)
            self.file_watcher.start_watching(info["task_name"])
            return True
        self.process_mgr.stop_process(info["task_name"])
        self._running = False
        self._log("[停止] {} 已终止".format(info["name"]))

    def run_camera_preview(self):
        """Toggle camera feed in center video widget."""
        if self.center_panel.video_widget._camera_mode if hasattr(
            self.center_panel.video_widget, '_camera_mode') else False:
            self.center_panel.video_widget.stop_camera_preview()
            self._log("[相机] 预览已关闭")
        else:
            if self.center_panel.video_widget.start_camera_preview():
                self._log("[相机] 实时预览已启动 — 画面在中间区域")
            else:
                self._log("[相机] 预览启动失败")

    def run_batch_upload(self):
        script = os.path.join(_PROJECT_ROOT, "cloud_runner.sh")
        self.process_mgr.start_script(
            "batch_upload", ["bash", script, "99"], cwd=_PROJECT_ROOT)
        self._log("[云端] 批量上传已启动")

    # ── Slots ────────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_stdout(self, text):
        self.bottom_console.append_stdout(text)

    @pyqtSlot(str)
    def _on_stderr(self, text):
        self.bottom_console.append_stderr(text)

    @pyqtSlot(str)
    def _on_started(self, name):
        self.bottom_console.append_system("[OK] {} 已启动".format(name))

    @pyqtSlot(str, int)
    def _on_finished(self, name, rc):
        status = "完成" if rc == 0 else "失败 (code={})".format(rc)
        self.bottom_console.append_system("[{}] {} {}".format(
            "OK" if rc == 0 else "ERR", name, status))
        self._running = False
        self.center_panel.progress_bar.stop_running()
        # Animate cloud progress if cloud was enabled
        if self._cloud_enabled and rc == 0:
            for i in range(5):
                QTimer.singleShot(i * 400, lambda p=(i+1)*20:
                    self.center_panel.progress_bar.set_cloud_progress(p))
            QTimer.singleShot(2200, lambda:
                self.center_panel.progress_bar.set_cloud_progress(100))
        if rc == 0:
            info = get_mode_info(self._active_mode)
            video_rel = info.get("output_video", "")
            if video_rel:
                video_path = os.path.join(_PROJECT_ROOT, video_rel)
                if os.path.exists(video_path):
                    self._log("[视频] 加载结果: {}".format(video_rel))
                    self.center_panel.video_widget.load_video_paused(
                        video_path, self._active_mode)
        self.left_panel.set_operation_locked(False)

    # ── Touch ────────────────────────────────────────────────────

    def _enable_touch_scroll(self):
        for widget in self.findChildren(QWidget):
            if isinstance(widget, (QPlainTextEdit, QScrollArea, QTableWidget)):
                QScroller.grabGesture(widget.viewport(), QScroller.TouchGesture)

    def _log(self, msg):
        self.bottom_console.append_system(msg)

    # ── Close ────────────────────────────────────────────────────

    def _msgbox(self, title, text, icon=QMessageBox.Warning,
                buttons=QMessageBox.Ok):
        """Show a centered text-only prompt without QMessageBox icon padding."""
        # Wayland ignores requested coordinates for top-level dialogs.  Make
        # this a child widget instead, so its position is deterministic within
        # the fullscreen GUI rather than chosen by the compositor.
        dialog = QDialog(self, Qt.Widget | Qt.FramelessWindowHint)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumWidth(360)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(22, 20, 22, 16)
        layout.setSpacing(16)
        message = QLabel(text)
        message.setAlignment(Qt.AlignCenter)
        message.setWordWrap(True)
        message.setMinimumWidth(316)
        layout.addWidget(message)
        controls = QHBoxLayout()
        controls.setSpacing(12)
        controls.setAlignment(Qt.AlignCenter)
        if buttons & QMessageBox.Yes:
            yes = QPushButton("确定")
            yes.clicked.connect(lambda: dialog.done(QMessageBox.Yes))
            controls.addWidget(yes)
        if buttons & QMessageBox.No:
            no = QPushButton("取消")
            no.clicked.connect(lambda: dialog.done(QMessageBox.No))
            controls.addWidget(no)
        if buttons == QMessageBox.Ok:
            ok = QPushButton("确定")
            ok.clicked.connect(lambda: dialog.done(QMessageBox.Ok))
            controls.addWidget(ok)
        layout.addLayout(controls)
        dialog.setStyleSheet(
            "QDialog{background:#16354c;border:1px solid #7ee7ff;border-radius:9px;}"
            "QLabel{color:#edf5fb;font-size:14px;padding:0;margin:0;}"
            "QPushButton{background:#1c4965;color:#edf5fb;border:1px solid #78cbe2;"
            "border-radius:5px;min-width:82px;padding:7px 16px;font-size:13px;}"
            "QPushButton:hover{background:#276987;border-color:#d8ff45;}")
        def _center_prompt():
            dialog.adjustSize()
            dialog.move((self.width() - dialog.width()) // 2,
                        (self.height() - dialog.height()) // 2)
            dialog.raise_()
        QTimer.singleShot(0, _center_prompt)
        return dialog.exec()

    def closeEvent(self, event):
        reply = self._msgbox(
            "退出确认", "确定要退出球悟AI系统吗？\n所有正在运行的任务将被终止。",
            QMessageBox.Question, QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.process_mgr.stop_all()
            self.file_watcher.stop_watching()
            event.accept()
        else:
            event.ignore()


# ── Entry ────────────────────────────────────────────────────────

def _is_small_screen():
    try:
        return QDesktopWidget().screenGeometry().width() < 1200
    except Exception:
        return False


def main():
    app = create_application(sys.argv)
    compact = "--compact" in sys.argv or _is_small_screen()
    if compact:
        app.setStyleSheet(app.styleSheet() + COMPACT_QSS)
    window = TennisGUI(compact=compact)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
