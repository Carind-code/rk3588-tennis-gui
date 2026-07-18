# -*- coding: utf-8 -*-
"""
Video Stream Widget — displays real-time or playback video frames.

Renders video from:
  - A local MP4 file (tracknet_rknn_output.mp4 or body_action_overlay.mp4)
  - RTSP camera stream (future)

The widget auto-refreshes via QTimer and can overlay ball trails,
skeleton overlays, and event flash animations.
"""

import os
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QSizePolicy, QPushButton


class VideoWidget(QWidget):
    """Video display widget with optional overlay painting."""

    signal_frame_changed = pyqtSignal(int)  # current frame number

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setStyleSheet(
            "background-color:transparent; border:1px solid rgba(126,231,255,190);"
            "border-radius:10px;")

        # Transparent broadcast-frame decoration. It is painted over the
        # display only; saved recordings retain the clean AI annotations.
        skin_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 "resources", "broadcast_skin_overlay.png")
        self._broadcast_skin = QPixmap(skin_path)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._video_label = QLabel()
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_label.setMinimumSize(320, 180)
        # The label fills the widget. Let the parent receive taps so the
        # top-right close control works on a touch screen as well as a mouse.
        self._video_label.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self._video_label.mousePressEvent = self._on_video_label_click
        self._video_label.setText("视频画面")
        self._video_label.setStyleSheet("color: #484f58; font-size: 24px; border: none;")
        layout.addWidget(self._video_label)

        self._close_view_btn = QPushButton("✕", self)
        self._close_view_btn.setFixedSize(40, 40)
        self._close_view_btn.setCursor(Qt.PointingHandCursor)
        self._close_view_btn.setStyleSheet(
            "QPushButton{background:rgba(0,0,0,180);color:#fff;"
            "border:1px solid #64748b;border-radius:20px;font-size:20px;font-weight:bold;}"
            "QPushButton:hover{background:#dc2626;border-color:#fecaca;}")
        self._close_view_btn.clicked.connect(self._close_active_view)
        self._close_view_btn.hide()

        # Click-to-play/pause
        self.setMouseTracking(True)
        self._paused = True   # start paused for demo videos
        self._auto_pause = False  # auto-pause on first frame for demo

        # State
        self._cap = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._fps = 0
        self._display_fps_frames = 0
        self._display_fps_started = time.perf_counter()
        self._frame_count = 0
        self._current_frame_num = 0
        self._video_path = None
        self._paused = False
        self._loaded_mode = None  # which mode loaded this video

        # Per-mode video cache: {mode_key: video_path}
        self._mode_videos = {}

        # Overlay data (updated externally)
        self._ball_trail = []
        self._skeleton_points = []
        self._event_flash = None
        self._score_top = 0
        self._score_bottom = 0

        # Placeholder pixmap
        self._placeholder = self._create_placeholder()

    # ── Public API ────────────────────────────────────────────────

    def stop(self):
        """Stop playback and release resources."""
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._current_frame_num = 0
        self._show_placeholder_pixmap()

    def seek_frame(self, frame_num: int):
        """Seek to a specific frame number."""
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self._current_frame_num = frame_num
            ret, frame = self._cap.read()
            if ret:
                self._display_frame(frame)

    def pause(self):
        self._paused = True
        self._timer.stop()
        self._show_pause_overlay = True
        self.update()

    def resume(self):
        self._paused = False
        self._show_pause_overlay = False
        self._auto_pause = False
        if self._cap is not None:
            self._timer.start(max(1, int(1000.0 / self._fps)))
        self.update()

    def mousePressEvent(self, event):
        """Click to toggle play/pause, X close, or ⛶ fullscreen."""
        if self._cap is None and not getattr(self, '_camera_mode', False):
            return
        # Scale click coords
        pm = self._video_label.pixmap()
        if pm is None or self._video_label.width() == 0:
            return
        sx = pm.width() / self._video_label.width()
        sy = pm.height() / self._video_label.height()
        px = event.pos().x() * sx
        py = event.pos().y() * sy

        # X button — stop camera + restore everything
        xr = getattr(self, '_x_btn_rect', None)
        if xr and xr[0] <= px <= xr[0]+xr[2] and xr[1] <= py <= xr[1]+xr[3]:
            if getattr(self, '_camera_mode', False):
                self._fs_active = False
                self.stop_camera_preview()
                self._restore_all_panels()
                self.signal_camera_stopped.emit()
            else:
                self.clear_video()
            return

        # Toggle play/pause
        if self._paused:
            self.resume()
        else:
            self.pause()

    def _on_video_label_click(self, event):
        """Reliable playback control for the full-size QLabel touch surface."""
        if self._cap is not None:
            if self._paused:
                self.resume()
            else:
                self.pause()
        event.accept()

    def load_video_paused(self, path, mode_key=None):
        """Load video but keep it paused on first frame. Saves per-mode."""
        if mode_key:
            self._loaded_mode = mode_key
            self._mode_videos[mode_key] = path
        self._auto_pause = True
        self.load_video(path)

    def clear_video(self):
        """Close current video, return to placeholder."""
        if self._loaded_mode:
            self._mode_videos.pop(self._loaded_mode, None)
        self._loaded_mode = None
        self.stop()
        self._close_view_btn.hide()
        self.show_placeholder("")
        self._show_pause_overlay = False

    def set_mode(self, mode_key):
        """Switch to this mode — restore its video if cached."""
        if mode_key in self._mode_videos:
            path = self._mode_videos[mode_key]
            self._loaded_mode = mode_key
            self._auto_pause = True
            self.load_video(path)
        else:
            # No video for this mode — show placeholder
            if self._loaded_mode is not None:
                # Save current before switching away
                if self._video_path:
                    self._mode_videos[self._loaded_mode] = self._video_path
            self._loaded_mode = mode_key
            self.stop()
            self.show_placeholder("")
            self._show_pause_overlay = False

    def load_video(self, path):
        """Load and start playing a video file."""
        if not os.path.exists(path):
            self._video_label.setText("视频文件不存在:\n{}".format(path))
            return

        self.stop()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._video_label.setText("无法打开视频:\n{}".format(path))
            self._cap = None
            return

        self._video_path = path
        self._close_view_btn.show()
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._display_fps_frames = 0
        self._display_fps_started = time.perf_counter()
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Read first frame to show preview
        ret, frame = self._cap.read()
        if ret:
            self._current_frame_num = 0
            self._display_frame(frame)

        if self._auto_pause:
            self._paused = True
            self._show_pause_overlay = True
            self._auto_pause = False
        else:
            interval_ms = max(1, int(1000.0 / self._fps))
            self._timer.start(interval_ms)
        self.update()
        self._video_label.update()

    def is_playing(self) -> bool:
        return self._cap is not None and self._timer.isActive()

    # ── Live camera preview ──────────────────────────────────────

    signal_frame_captured = pyqtSignal(object)
    signal_camera_stopped = pyqtSignal()
    signal_camera_status = pyqtSignal(bool, str)
    signal_camera_fps = pyqtSignal(float)
    signal_display_fps = pyqtSignal(float)

    def toggle_ai_overlay(self, name):
        """Enable/disable an AI model overlay on the camera feed."""
        if not hasattr(self, '_ai_overlay') or self._ai_overlay is None:
            from gui.widgets.ai_overlay import AIOverlay
            self._ai_overlay = AIOverlay()
            self._ai_overlay._log = lambda msg: print("[AI] " + msg)
        return self._ai_overlay.toggle(name)

    def start_camera_preview(self, device="/dev/video12"):
        """Start live camera feed directly in the video widget."""
        self.stop()

        # Start rkaiq 3A engine for auto exposure/white balance
        import subprocess
        if subprocess.run(["pgrep", "-f", "rkaiq"],
                          capture_output=True).returncode != 0:
            iqfile = "/etc/iqfiles/ov13855_CMK-OT2016-FV1_default.json"
            subprocess.Popen(["rkaiq_3A_server", "-c", iqfile],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            time.sleep(0.8)  # Wait for 3A init

        self._cam_cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cam_cap.isOpened():
            self._video_label.setText("摄像头不可用:\n{}".format(device))
            return False
        self._cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self._cam_cap.set(cv2.CAP_PROP_FPS, 30)
        self._cam_w = int(self._cam_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._cam_h = int(self._cam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._camera_mode = True
        self._close_view_btn.show()
        self._cam_count = 0
        self._cam_fps_frames = 0
        self._cam_fps_started = time.perf_counter()
        self._cam_fullscreen = False
        self._paused = False
        self._show_pause_overlay = False
        try:
            self._timer.timeout.disconnect()
        except Exception:
            pass
        self._timer.timeout.connect(self._next_camera_frame)
        self._timer.start(33)
        self.signal_camera_status.emit(True, "{}×{}".format(self._cam_w, self._cam_h))
        return True

    def stop_camera_preview(self):
        """Stop the live source and reset every preview state deterministically."""
        self._timer.stop()
        self._camera_mode = False
        self._close_fullscreen()
        # Restore bottom bar
        win = self.window()
        if win and hasattr(win, 'center_panel'):
            win.center_panel.show_bottom_bar()
        if hasattr(self, '_cam_cap') and self._cam_cap:
            self._cam_cap.release()
            self._cam_cap = None
        self._paused = False
        self._current_frame_num = 0
        self._video_path = None
        self._close_view_btn.hide()
        self._show_placeholder_pixmap()
        self.signal_camera_status.emit(False, "")

    def _close_active_view(self):
        """Close camera or playback from a real touch-capable Qt button."""
        owner = self.window()
        if getattr(owner, '_running', False):
            # During a recording/analysis task STOP is the only safe exit;
            # closing the camera here would corrupt the active capture.
            return
        if getattr(self, '_camera_mode', False):
            self.stop_camera_preview()
            self.signal_camera_stopped.emit()
        else:
            self.clear_video()

    def _toggle_fullscreen(self):
        """Simple toggle: hide/show panels around video."""
        self._fs_active = not getattr(self, '_fs_active', False)
        win = self.window()
        if not win:
            return
        if self._fs_active:
            if hasattr(win, '_dock_left'):
                win._dock_left.close()
            if hasattr(win, '_dock_right'):
                win._dock_right.close()
            if hasattr(win, '_dock_log'):
                win._dock_log.close()
            if hasattr(win, 'top_bar'):
                win.top_bar.hide()
            if hasattr(win, 'center_panel'):
                win.center_panel.hide_bottom_bar()
        else:
            if hasattr(win, 'top_bar'):
                win.top_bar.show()
            if hasattr(win, 'center_panel'):
                win.center_panel.show_bottom_bar()

    def _restore_all_panels(self):
        self._fs_active = False
        win = self.window()
        if win:
            if hasattr(win, 'top_bar'):
                win.top_bar.show()
            if hasattr(win, 'center_panel'):
                win.center_panel.show_bottom_bar()

    def _close_fullscreen(self):
        self._restore_all_panels()

    def _next_camera_frame(self):
        if not getattr(self, '_camera_mode', False):
            return
        cap = getattr(self, '_cam_cap', None)
        if cap is None:
            return
        ret, frame = cap.read()
        if not ret:
            return
        self._cam_count += 1
        self._cam_fps_frames += 1
        self._current_frame_num = self._cam_count

        # Run inference and draw onto the original camera resolution first.
        # The recorder receives this annotated frame, so a selected ball trail
        # or pose skeleton is preserved in the saved MP4 as well as on screen.
        annotated = frame
        ai = getattr(self, '_ai_overlay', None)
        if ai and ai.active_names:
            annotated = ai.process(frame)
        self.signal_frame_captured.emit(annotated)

        # Keep the source aspect ratio. _display_frame performs a letterboxed
        # fit for the QLabel; resizing to its raw width/height here would
        # stretch the 16:9 OV13855 image on tall preview panels.
        display = annotated.copy()

        # Overlay info
        h, w = display.shape[:2]

        # FPS counter (bottom-left)
        elapsed = time.perf_counter() - self._cam_fps_started
        if elapsed >= 1.0:
            self._cam_fps = self._cam_fps_frames / elapsed
            self._cam_fps_frames = 0
            self._cam_fps_started = time.perf_counter()
            self.signal_camera_fps.emit(self._cam_fps)
            self.signal_display_fps.emit(self._cam_fps)
        fps_text = "OV13855 {}x{}  {:.1f} fps".format(
            self._cam_w, self._cam_h,
            getattr(self, '_cam_fps', 0))
        cv2.putText(display, fps_text, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        self._display_frame(display)

    def is_playing(self) -> bool:
        return self._cap is not None and self._timer.isActive()


    def set_overlay_data(self, ball_trail=None, skeleton=None, event=None, score_top=0, score_bottom=0):
        """Update overlay annotations from external data sources."""
        if ball_trail is not None:
            self._ball_trail = ball_trail
        if skeleton is not None:
            self._skeleton_points = skeleton
        if event is not None:
            self._event_flash = event
        self._score_top = score_top
        self._score_bottom = score_bottom

    # ── Internal ──────────────────────────────────────────────────

    def _next_frame(self):
        if self._cap is None or self._paused:
            return

        ret, frame = self._cap.read()
        if not ret:
            # Loop or stop
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._current_frame_num = 0
            return

        self._current_frame_num = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._display_fps_frames += 1
        elapsed = time.perf_counter() - self._display_fps_started
        if elapsed >= 1.0:
            self.signal_display_fps.emit(self._display_fps_frames / elapsed)
            self._display_fps_frames = 0
            self._display_fps_started = time.perf_counter()
        self.signal_frame_changed.emit(self._current_frame_num)
        self._display_frame(frame)

    def _display_frame(self, frame: np.ndarray):
        """Convert OpenCV BGR frame to QPixmap and paint overlays."""
        if frame is None:
            return

        # Resize to fit widget while keeping aspect ratio
        h, w = frame.shape[:2]
        target_w = self._video_label.width()
        target_h = self._video_label.height()

        if target_w < 10 or target_h < 10:
            return

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        if new_w > 0 and new_h > 0:
            frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)

        # Paint overlays
        if self._ball_trail or self._skeleton_points or self._event_flash:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)

            # Ball trail
            if self._ball_trail:
                pen = QPen(QColor(255, 255, 0, 180), 2)
                painter.setPen(pen)
                scaled_trail = [(int(x * scale), int(y * scale)) for (x, y) in self._ball_trail]
                for i in range(len(scaled_trail) - 1):
                    painter.drawLine(
                        scaled_trail[i][0], scaled_trail[i][1],
                        scaled_trail[i + 1][0], scaled_trail[i + 1][1],
                    )

            # Skeleton
            if self._skeleton_points:
                pen = QPen(QColor(0, 255, 128, 200), 2)
                painter.setPen(pen)
                for pt in self._skeleton_points:
                    sx, sy = int(pt[0] * scale), int(pt[1] * scale)
                    painter.drawEllipse(sx - 2, sy - 2, 4, 4)

            # Event flash
            if self._event_flash is not None:
                event_type, timestamp = self._event_flash
                elapsed = cv2.getTickCount() / cv2.getTickFrequency() - timestamp
                if elapsed < 2.0:
                    alpha = max(0, int(255 * (1.0 - elapsed / 2.0)))
                    color = QColor(0, 180, 0, alpha) if event_type == "IN" else QColor(233, 69, 96, alpha)
                    painter.fillRect(QRect(0, 0, w, h), color)

            # Score overlay (top-left)
            font = QFont("DejaVu Sans", 18, QFont.Bold)
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(10, 30, "Top {} - {} Bottom".format(self._score_top, self._score_bottom))

            painter.end()

        # Draw X close button when video/camera is active
        has_content = self._video_path or getattr(self, '_camera_mode', False)
        if has_content:
            btn_painter = QPainter(pixmap)
            btn_painter.setRenderHint(QPainter.Antialiasing)
            btn_w = 28; btn_h = 28
            x_x = pixmap.width() - 36; btn_y = 8
            btn_painter.setBrush(QColor(0, 0, 0, 160))
            btn_painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
            btn_painter.drawRoundedRect(x_x, btn_y, btn_w, btn_h, 6, 6)
            btn_painter.setPen(QPen(QColor(255, 255, 255, 220), 2.5))
            btn_painter.drawLine(x_x+8, btn_y+8, x_x+20, btn_y+20)
            btn_painter.drawLine(x_x+20, btn_y+8, x_x+8, btn_y+20)
            self._x_btn_rect = (x_x, btn_y, btn_w, btn_h)
            btn_painter.end()

        # Pause overlay
        if self._show_pause_overlay:
            p_painter = QPainter(pixmap)
            p_painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 120))
            p_painter.setRenderHint(QPainter.Antialiasing)
            p_painter.setBrush(QColor(255, 255, 255, 200))
            p_painter.setPen(Qt.NoPen)
            cx, cy = pixmap.width() // 2, pixmap.height() // 2
            bar_w, bar_h = 12, 50
            gap = 16
            p_painter.drawRoundedRect(cx - gap - bar_w, cy - bar_h // 2, bar_w, bar_h, 4, 4)
            p_painter.drawRoundedRect(cx + gap, cy - bar_h // 2, bar_w, bar_h, 4, 4)
            font = QFont("DejaVu Sans", 16, QFont.Bold)
            p_painter.setFont(font)
            p_painter.setPen(QColor(255, 255, 255, 220))
            p_painter.drawText(pixmap.rect().adjusted(0, 60, 0, 0),
                               Qt.AlignHCenter | Qt.AlignTop, "点击播放")
            p_painter.end()

        self._video_label.setPixmap(pixmap)

    def _paint_broadcast_skin(self, pixmap):
        """Apply the transparent sports-broadcast frame to displayed frames."""
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        if not self._broadcast_skin.isNull():
            painter.setOpacity(.82)
            painter.drawPixmap(pixmap.rect(), self._broadcast_skin)
            painter.setOpacity(1.0)
        # High-contrast HUD frame. This remains legible on the board's small
        # panel even after the 640px source is scaled down to the screen.
        width, height = pixmap.width(), pixmap.height()
        top_h = max(26, height // 13)
        painter.fillRect(QRect(4, 4, max(1, width - 8), top_h),
                         QColor(3, 15, 29, 205))
        painter.setPen(QPen(QColor(216, 255, 69, 230), 2))
        painter.drawLine(7, top_h + 4, width - 7, top_h + 4)
        title_font = QFont("DejaVu Sans", max(8, height // 32), QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(216, 255, 69, 255))
        painter.drawText(QRect(14, 5, width // 2, top_h - 2),
                         Qt.AlignVCenter | Qt.AlignLeft, "QIUWU  //  LIVE ANALYTICS")
        painter.setPen(QColor(126, 231, 255, 235))
        painter.drawText(QRect(width // 2, 5, width // 2 - 14, top_h - 2),
                         Qt.AlignVCenter | Qt.AlignRight, "TRACKING ONLINE")
        painter.setPen(QPen(QColor(126, 231, 255, 230), 3))
        corner = max(24, min(width, height) // 11)
        for x, y, dx, dy in ((8, 8, 1, 1), (width - 8, 8, -1, 1),
                             (8, height - 8, 1, -1), (width - 8, height - 8, -1, -1)):
            painter.drawLine(x, y, x + dx * corner, y)
            painter.drawLine(x, y, x, y + dy * corner)
        bottom_h = max(20, height // 18)
        painter.fillRect(QRect(4, height - bottom_h - 4, max(1, width - 8), bottom_h),
                         QColor(3, 15, 29, 190))
        painter.setFont(QFont("DejaVu Sans", max(7, height // 38), QFont.Bold))
        painter.setPen(QColor(126, 231, 255, 245))
        painter.drawText(QRect(14, height - bottom_h - 2, width - 28, bottom_h - 4),
                         Qt.AlignVCenter | Qt.AlignLeft, "●  LIVE VISION   •   RK3588 EDGE AI")
        painter.end()

    def show_placeholder(self, mode_text=""):
        """Show the translucent standby surface used before a video loads."""
        pixmap = QPixmap(640, 360)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        # The standby surface is deliberately translucent: it reads as frosted
        # glass over the selected skin instead of as a second dark video.
        painter.setPen(QPen(QColor(126, 231, 255, 150), 2))
        painter.setBrush(QColor(7, 20, 34, 28))
        painter.drawRoundedRect(QRect(2, 2, 636, 356), 10, 10)
        # Grid lines for monitor feel
        painter.setPen(QPen(QColor(126, 231, 255, 42), 1))
        for i in range(0, 640, 40):
            painter.drawLine(i, 0, i, 360)
        for i in range(0, 360, 40):
            painter.drawLine(0, i, 640, i)
        # Border
        painter.setPen(QPen(QColor(126, 231, 255, 145), 2))
        painter.drawRect(2, 2, 636, 356)
        # Text
        painter.setPen(QColor(208, 237, 247, 205))
        font = QFont("DejaVu Sans", 14)
        painter.setFont(font)
        painter.drawText(QRect(0, 60, 640, 80), Qt.AlignCenter,
                         "● REC STANDBY")
        font2 = QFont("DejaVu Sans", 11)
        painter.setFont(font2)
        painter.setPen(QColor(172, 222, 239, 190))
        if mode_text:
            painter.drawText(QRect(0, 130, 640, 40), Qt.AlignCenter, mode_text)
        painter.drawText(QRect(0, 180, 640, 40), Qt.AlignCenter,
                         "点击 ▶ RUN 启动推理流水线")
        painter.drawText(QRect(0, 210, 640, 40), Qt.AlignCenter,
                         "结果视频将在此回放")
        # Corner timestamp
        import time
        painter.setPen(QColor(172, 222, 239, 170))
        font3 = QFont("DejaVu Sans Mono", 9)
        painter.setFont(font3)
        painter.drawText(QRect(10, 330, 200, 20), Qt.AlignLeft,
                         time.strftime("%Y-%m-%d %H:%M:%S"))
        painter.drawText(QRect(430, 330, 200, 20), Qt.AlignRight,
                         "RK3588 NPU")
        painter.end()
        self._placeholder = pixmap
        self._show_placeholder_pixmap()

    def _show_placeholder_pixmap(self):
        """Let the glass standby surface use the entire available preview."""
        target = self._video_label.size()
        if target.width() > 1 and target.height() > 1:
            display = self._placeholder.scaled(
                target, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        else:
            display = self._placeholder
        self._video_label.setPixmap(display)

    def _create_placeholder(self) -> QPixmap:
        """Legacy — use show_placeholder instead."""
        return QPixmap(640, 360)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._close_view_btn.move(max(4, self.width() - 48), 8)
        # Re-display current frame at new size
        if self._cap is not None and hasattr(self, '_current_frame_num'):
            pass  # next frame will auto-resize
        elif hasattr(self, '_placeholder') and not getattr(self, '_camera_mode', False):
            self._show_placeholder_pixmap()
