# -*- coding: utf-8 -*-
"""
Left Mode Panel — scene selection cards + unified device controls.

Mode-driven: user picks a scene, clicks RUN → system auto-launches
the correct shell script based on mode + cloud toggle.
"""

import json
import os

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QScrollArea, QCheckBox,
    QRadioButton, QButtonGroup, QLineEdit, QFileDialog,
    QFrame, QMessageBox, QDialogButtonBox, QSizePolicy,
)

from gui.modes import MODE_A, MODE_B, get_mode_info


# ── Scene Card widget ──────────────────────────────────────────────

class SceneCard(QFrame):
    """Clickable scene mode card with icon, title, subtitle."""
    clicked = pyqtSignal(str)

    def __init__(self, mode_key, parent=None):
        super().__init__(parent)
        self.mode_key = mode_key
        info = get_mode_info(mode_key)

        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            SceneCard { background: #0d1b2d; border: 2px solid #284864;
                        border-radius: 8px; padding: 8px; }
            SceneCard:hover { border-color: #7ee7ff; background: #10243a; }
            SceneCard[active="true"] { border-color: %s; background: #172b24; }
        """ % info["color"])

        layout = QVBoxLayout(self)
        layout.setSpacing(2)

        # Icon + name
        header = QHBoxLayout()
        icon_lbl = QLabel(info["icon"])
        icon_lbl.setStyleSheet("font-size: 20px; border:none; background:transparent;")
        header.addWidget(icon_lbl)

        name_lbl = QLabel(info["name"])
        name_lbl.setObjectName("sceneTitle")
        name_lbl.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #e0e0e0; border:none; background:transparent;")
        header.addWidget(name_lbl)
        header.addStretch()
        layout.addLayout(header)

        # Subtitle
        sub = QLabel(info["subtitle"])
        sub.setWordWrap(True)
        sub.setStyleSheet(
            "font-size: 10px; color: #8b949e; border:none; background:transparent;")
        layout.addWidget(sub)

        self._active = False

    def set_active(self, active):
        self._active = active
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        self.clicked.emit(self.mode_key)


# ── Left Panel ─────────────────────────────────────────────────────

class LeftModePanel(QScrollArea):
    """Scrollable left panel with scene cards and unified controls."""

    signal_video_source_changed = pyqtSignal(str, str)

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self._mw = main_window  # reference to TennisGUI for mode switching
        self._project_root = main_window._PROJECT_ROOT \
            if hasattr(main_window, '_PROJECT_ROOT') else ""

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(240)

        inner = QWidget()
        self.setWidget(inner)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Scene Cards ───────────────────────────────────────────
        scene_group = QGroupBox("场景模式选择")
        scene_layout = QVBoxLayout(scene_group)

        self._cards = {}
        for mk in [MODE_A, MODE_B]:
            card = SceneCard(mk)
            card.clicked.connect(self._on_scene_clicked)
            scene_layout.addWidget(card)
            self._cards[mk] = card

        layout.addWidget(scene_group)

        # Default: Mode A active
        self._cards[MODE_A].set_active(True)

        # ── Device Controls ───────────────────────────────────────
        ctrl_group = QGroupBox("统一设备控制")
        ctrl_layout = QVBoxLayout(ctrl_group)

        # Input source
        src_label = QLabel("视频输入源:")
        src_label.setStyleSheet("color: #8b949e; font-size: 10px; border:none;")
        ctrl_layout.addWidget(src_label)

        self._src_group = QButtonGroup(self)
        src_row = QHBoxLayout()
        self._src_cam = QRadioButton("OV13855 相机")
        self._src_cam.setChecked(True)
        self._src_file = QRadioButton("本地文件")
        self._src_group.addButton(self._src_cam, 0)
        self._src_group.addButton(self._src_file, 1)
        src_row.addWidget(self._src_cam)
        src_row.addWidget(self._src_file)
        ctrl_layout.addLayout(src_row)

        # File path row with browse button
        file_row = QHBoxLayout()
        self._src_path = QLineEdit()
        self._src_path.setPlaceholderText("选择视频文件...")
        self._src_path.setEnabled(False)
        file_row.addWidget(self._src_path)

        self._btn_browse = QPushButton("📂")
        self._btn_browse.setFixedWidth(36)
        self._btn_browse.setToolTip("浏览本地视频文件")
        self._btn_browse.setEnabled(False)
        self._btn_browse.clicked.connect(self._on_browse_local)
        file_row.addWidget(self._btn_browse)
        ctrl_layout.addLayout(file_row)

        self._src_cam.toggled.connect(
            lambda v: self._src_path.setEnabled(not v))
        self._src_cam.toggled.connect(
            lambda v: self._btn_browse.setEnabled(not v))

        # Cloud toggle
        self._cloud_check = QCheckBox("自动上云 (MQTT + OSS)")
        self._cloud_check.setStyleSheet(
            "color: #c9d1d9; font-size: 11px;")
        self._cloud_check.toggled.connect(self._mw.set_cloud)
        ctrl_layout.addWidget(self._cloud_check)

        layout.addWidget(ctrl_group)

        # ── RUN / STOP ────────────────────────────────────────────
        run_group = QGroupBox("任务控制")
        run_layout = QVBoxLayout(run_group)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._btn_run = QPushButton("▶ 启动运行 (RUN)")
        self._btn_run.setObjectName("btnRun")
        self._btn_run.setCursor(Qt.PointingHandCursor)
        self._btn_run.clicked.connect(self._on_run)
        self._btn_run.setStyleSheet(
            "QPushButton#btnRun{background:#d8ff45;color:#08131d;font-weight:bold;"
            "border:none;border-radius:6px;min-height:32px;max-height:32px;padding:0px 8px;font-size:12px}"
            "QPushButton#btnRun:hover{background:#d4ff20}"
            "QPushButton#btnRun:disabled{background:#30363d;color:#484f58}")

        self._btn_stop = QPushButton("■ 停止 (STOP)")
        self._btn_stop.setObjectName("btnStop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setCursor(Qt.PointingHandCursor)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_stop.setStyleSheet(
            "QPushButton#btnStop{background:#e34b58;color:#fff;font-weight:bold;"
            "border:none;border-radius:6px;min-height:32px;max-height:32px;padding:0px 8px;font-size:12px}"
            "QPushButton#btnStop:hover{background:#f85149}"
            "QPushButton#btnStop:disabled{background:#30363d;color:#484f58}")

        btn_row.addWidget(self._btn_run)
        btn_row.addWidget(self._btn_stop)
        run_layout.addLayout(btn_row)

        self._lbl_status = QLabel("就绪 — 选择场景后点击 RUN")
        self._lbl_status.setStyleSheet("color:#8b949e;font-size:10px;")
        self._lbl_status.setWordWrap(True)
        run_layout.addWidget(self._lbl_status)

        layout.addWidget(run_group)

        # ── Quick Actions ─────────────────────────────────────────
        quick_group = QGroupBox("快捷操作")
        quick_layout = QVBoxLayout(quick_group)

        self._btn_calib = QPushButton("一键标定球场")
        self._btn_calib.setCursor(Qt.PointingHandCursor)
        self._btn_calib.clicked.connect(self._on_calibrate)
        quick_layout.addWidget(self._btn_calib)

        self._btn_roi = QPushButton("划分 ROI 区域")
        self._btn_roi.setCursor(Qt.PointingHandCursor)
        self._btn_roi.clicked.connect(self._on_roi)
        quick_layout.addWidget(self._btn_roi)

        # Camera preview toggle
        self._btn_cam = QPushButton("相机实时预览")
        self._btn_cam.setCursor(Qt.PointingHandCursor)
        self._btn_cam.clicked.connect(self._on_toggle_camera)
        quick_layout.addWidget(self._btn_cam)

        # ── Realtime AI Models ──
        ai_model_group = QGroupBox("实时AI模型")
        ai_layout = QVBoxLayout(ai_model_group)
        ai_label = QLabel("摄像头预览时叠加AI效果")
        ai_label.setStyleSheet("color:#8b949e;font-size:9px;border:none;")
        ai_layout.addWidget(ai_label)

        self._btn_yolo = QPushButton("🎯 YOLO 网球检测")
        self._btn_yolo.setCursor(Qt.PointingHandCursor)
        self._btn_yolo.setStyleSheet(
            "QPushButton{background:#1a3a1a;border:1px solid #3fb950;"
            "border-radius:4px;padding:5px 10px;font-size:11px;color:#c9d1d9;}"
            "QPushButton:hover{background:#2a5a2a;}")
        self._btn_yolo.clicked.connect(self._on_launch_yolo)
        ai_layout.addWidget(self._btn_yolo)

        self._btn_pose = QPushButton("🦴 Pose 人体骨架")
        self._btn_pose.setCursor(Qt.PointingHandCursor)
        self._btn_pose.setStyleSheet(
            "QPushButton{background:#1a1a3a;border:1px solid #a371f7;"
            "border-radius:4px;padding:5px 10px;font-size:11px;color:#c9d1d9;}"
            "QPushButton:hover{background:#2a2a5a;}")
        self._btn_pose.clicked.connect(self._on_launch_pose)
        ai_layout.addWidget(self._btn_pose)

        quick_layout.addWidget(ai_model_group)

        # Camera recording
        rec_row = QHBoxLayout()
        self._btn_rec_start = QPushButton("● 录制")
        self._btn_rec_start.setObjectName("btnStartJudgement")
        self._btn_rec_start.setCursor(Qt.PointingHandCursor)
        self._btn_rec_start.clicked.connect(self._on_start_recording)
        self._btn_rec_stop = QPushButton("■ 停止")
        self._btn_rec_stop.setObjectName("btnStop")
        self._btn_rec_stop.setEnabled(False)
        self._btn_rec_stop.setCursor(Qt.PointingHandCursor)
        self._btn_rec_stop.clicked.connect(self._on_stop_recording)
        for button in (self._btn_rec_start, self._btn_rec_stop):
            button.setStyleSheet(
                "QPushButton{min-height:30px;max-height:30px;padding:0px 8px;}")
        rec_row.addWidget(self._btn_rec_start)
        rec_row.addWidget(self._btn_rec_stop)
        quick_layout.addLayout(rec_row)

        self._lbl_rec_status = QLabel("")
        self._lbl_rec_status.setStyleSheet("color:#8b949e;font-size:10px;")
        quick_layout.addWidget(self._lbl_rec_status)

        layout.addWidget(quick_group)

        layout.addStretch()
        self._cam_preview_on = False
        self._is_recording = False
        self._rec_writer = None
        self._operation_locked = False

    def set_operation_locked(self, locked: bool, allow_live_controls: bool = False):
        """Lock incompatible controls while preserving approved live actions."""
        self._operation_locked = locked
        for card in self._cards.values():
            card.setEnabled(not locked)
        for widget in (
            self._src_cam, self._src_file, self._src_path, self._btn_browse,
            self._cloud_check,
        ):
            widget.setEnabled(not locked)
        # Calibration and ROI remain usable during a top-view task.
        self._btn_calib.setEnabled(True)
        self._btn_roi.setEnabled(True)
        self._btn_cam.setEnabled(not locked or allow_live_controls)
        self._btn_yolo.setEnabled(not locked or allow_live_controls)
        self._btn_pose.setEnabled(not locked or allow_live_controls)
        self._btn_rec_start.setEnabled(not locked or allow_live_controls)
        self._btn_rec_stop.setEnabled(
            self._is_recording and (not locked or allow_live_controls))
        self._btn_run.setEnabled(not locked)

    def _on_camera_stopped(self):
        """Called when camera preview is closed via X button."""
        self._cam_preview_on = False
        self._btn_cam.setText("相机实时预览")
        if not self._mw._running:
            self.set_operation_locked(False)

    def _on_toggle_camera(self):
        video = self._mw.center_panel.video_widget
        if getattr(video, '_camera_mode', False):
            video.stop_camera_preview()
            self._on_camera_stopped()
            self._mw._log("[相机] 预览已关闭")
        else:
            ok = video.start_camera_preview()
            if ok:
                self._btn_cam.setText("关闭预览")
                self._cam_preview_on = True
                self.set_operation_locked(True, allow_live_controls=True)
                self._btn_calib.setEnabled(False)
                self._btn_roi.setEnabled(False)

    def _next_rec_filename(self):
        i = 1
        rec_dir = os.path.join(self._project_root, "outputs", "recordings")
        os.makedirs(rec_dir, exist_ok=True)
        while True:
            path = os.path.join(rec_dir, "recording_{}.mp4".format(i))
            if not os.path.exists(path):
                return path
            i += 1

    def _start_camera_recording(self):
        """Start recording from camera (called by tennis_gui in camera mode)."""
        self._on_start_recording()

    def _on_start_recording(self):
        import cv2
        out_path = self._next_rec_filename()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._rec_writer = cv2.VideoWriter(out_path, fourcc, 30.0, (1920, 1080))
        if not self._rec_writer.isOpened():
            self._lbl_rec_status.setText("录制启动失败")
            return
        self._rec_path = out_path
        self._is_recording = True
        self._btn_rec_start.setEnabled(False)
        self._btn_rec_stop.setEnabled(True)
        self._lbl_rec_status.setText("● REC {}".format(os.path.basename(out_path)))
        self._lbl_rec_status.setStyleSheet("color:#f85149;font-size:10px;font-weight:bold;")

    def _on_frame_captured(self, frame):
        """Receive raw camera frames for recording."""
        if self._is_recording and self._rec_writer:
            self._rec_writer.write(frame)

    def _on_stop_recording(self):
        self._is_recording = False
        if self._rec_writer:
            self._rec_writer.release()
            self._rec_writer = None
        self._btn_rec_start.setEnabled(True)
        self._btn_rec_stop.setEnabled(False)
        self._lbl_rec_status.setText("已保存 → 桌面")
        self._lbl_rec_status.setStyleSheet("color:#3fb950;font-size:10px;")

    def _on_browse_local(self):
        """Open file picker to manually select a video."""
        start_dir = self._project_root
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", start_dir,
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        if path:
            self._src_path.setText(path)
            self.signal_video_source_changed.emit("file", path)

    # ── Slots ────────────────────────────────────────────────────

    def _on_scene_clicked(self, mode_key):
        for mk, card in self._cards.items():
            card.set_active(mk == mode_key)
        self._mw.set_mode(mode_key)

    def _on_run(self):
        if self._operation_locked:
            return
        if self._src_file.isChecked() and (
                not self._src_path.text().strip() or
                not os.path.exists(self._src_path.text().strip())):
            self._mw._msgbox("未选择文件", "请先选择本地视频文件再启动分析。")
            return
        info = get_mode_info(self._mw._active_mode)
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._lbl_status.setText("● 运行中 — {}".format(info["name"]))
        self._lbl_status.setStyleSheet(
            "color:#ccff00;font-size:10px;font-weight:bold;")
        if not self._mw.run_pipeline():
            self._btn_run.setEnabled(True)
            self._btn_stop.setEnabled(False)
            self._lbl_status.setText("就绪 — 等待有效输入")
            self._lbl_status.setStyleSheet("color:#8b949e;font-size:10px;")
            return
        self.set_operation_locked(
            True, allow_live_controls=self._src_cam.isChecked())
        if self._src_cam.isChecked():
            self._btn_cam.setEnabled(False)
        self._btn_stop.setEnabled(True)

        # Re-enable RUN when done
        def _on_finished(name, rc):
            mode_info = get_mode_info(self._mw._active_mode)
            if name == mode_info["task_name"]:
                self._btn_run.setEnabled(True)
                self._btn_stop.setEnabled(False)
                if rc == 0:
                    self._lbl_status.setText("✓ 完成 — {}".format(
                        mode_info["name"]))
                    self._lbl_status.setStyleSheet(
                        "color:#4caf50;font-size:10px;")
                else:
                    self._lbl_status.setText("✗ 异常 (code={})".format(rc))
                    self._lbl_status.setStyleSheet(
                        "color:#f85149;font-size:10px;")

        self._mw.process_mgr.signal_process_finished.connect(_on_finished)

    def _on_stop(self):
        buffering = self._mw.stop_pipeline()
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(False)
        if buffering:
            self.set_operation_locked(True, allow_live_controls=False)
            self._lbl_status.setText("● 录制结束，正在汇总结果...")
            self._lbl_status.setStyleSheet("color:#58a6ff;font-size:10px;font-weight:bold;")
        else:
            self.set_operation_locked(False)
            self._lbl_status.setText("■ 已停止")
            self._lbl_status.setStyleSheet("color:#d29922;font-size:10px;")

    def _on_calibrate(self):
        """Run court calibration via match_judgement infer script."""
        match_dir = os.path.join(self._project_root, "match_judgement")
        script = os.path.join(match_dir, "infer_tracknet_rknn_lite.py")
        cmd = [
            "python3", script,
            "--model", os.path.join(match_dir,
                "model_v37_360x640_b1_sigmoid_fp.rknn"),
            "--court-model", os.path.join(match_dir,
                "court_detector_fp.rknn"),
            "--video", os.path.join(match_dir, "input.mp4"),
            "--no-events", "--no-video", "--no-ipc", "--no-progress",
        ]
        self._mw.process_mgr.start_script(
            "court_calibration", cmd, cwd=match_dir)
        self._mw._log("[标定] 场地标定已启动...")
        self._show_message(
            "场地标定", "场地标定已启动。\n成功后棋盘格关键点将被锁定复用。",
            QMessageBox.Information)

    def _on_launch_yolo(self):
        """Toggle YOLO detection overlay on camera."""
        ok = self._mw.center_panel.video_widget.toggle_ai_overlay("yolo")
        if ok:
            self._btn_yolo.setText("🎯 YOLO 网球检测 ✓")
            self._btn_yolo.setStyleSheet(
                "QPushButton{background:#2a5a2a;border:2px solid #3fb950;"
                "border-radius:4px;padding:5px 10px;font-size:11px;color:#fff;}")
            self._mw._log("[YOLO] 网球检测叠加 → ON")
        else:
            self._btn_yolo.setText("🎯 YOLO 网球检测")
            self._btn_yolo.setStyleSheet(
                "QPushButton{background:#1a3a1a;border:1px solid #3fb950;"
                "border-radius:4px;padding:5px 10px;font-size:11px;color:#c9d1d9;}")
            self._mw._log("[YOLO] 网球检测叠加 → OFF")

    def _on_launch_pose(self):
        """Toggle Pose skeleton overlay."""
        ok = self._mw.center_panel.video_widget.toggle_ai_overlay("pose")
        if ok:
            self._btn_pose.setText("🦴 Pose 人体骨架 ✓")
            self._btn_pose.setStyleSheet(
                "QPushButton{background:#2a2a5a;border:2px solid #a371f7;"
                "border-radius:4px;padding:5px 10px;font-size:11px;color:#fff;}")
            self._mw._log("[Pose] 人体骨架叠加 → ON")
        else:
            self._btn_pose.setText("🦴 Pose 人体骨架")
            self._btn_pose.setStyleSheet(
                "QPushButton{background:#1a1a3a;border:1px solid #a371f7;"
                "border-radius:4px;padding:5px 10px;font-size:11px;color:#c9d1d9;}")
            self._mw._log("[Pose] 人体骨架叠加 → OFF")

    def _on_roi(self):
        """Validate and report the active normalized player ROIs."""
        roi_path = os.path.join(
            self._project_root, "match_judgement",
            "broadcast_court_only.json")
        if not os.path.exists(roi_path):
            self._show_message("ROI", "broadcast_court_only.json 不存在", QMessageBox.Warning)
            return
        try:
            with open(roi_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            crops = config.get("crops")
            if not isinstance(crops, list) or len(crops) != 2:
                raise ValueError("crops 必须是包含 2 个区域的列表")
            names = []
            for crop in crops:
                if not isinstance(crop, dict):
                    raise ValueError("每个区域必须是对象")
                name = crop.get("name", "未命名")
                values = [crop.get(key) for key in ("x1", "y1", "x2", "y2")]
                if not all(isinstance(value, (int, float)) for value in values):
                    raise ValueError("{} 的坐标必须为数字".format(name))
                x1, y1, x2, y2 = values
                if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                    raise ValueError("{} 的坐标必须满足 0≤x1<x2≤1、0≤y1<y2≤1".format(name))
                names.append(name)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self._mw._log("[ROI] 配置错误: {}".format(error))
            self._show_message("ROI 配置错误", "JSON 配置不可用：\n{}".format(error), QMessageBox.Critical)
            return
        self._mw._log("[ROI] 打开区域配置: {}".format(roi_path))
        self._show_message(
            "ROI 配置有效",
            "已加载 2 个归一化检测区域：{}。\n\n配置文件：\n{}".format("、".join(names), roi_path),
            QMessageBox.Information)

    def _show_message(self, title, text, icon):
        """Use the application's one centered, text-only prompt component."""
        return self._mw._msgbox(title, text, icon, QMessageBox.Ok)
