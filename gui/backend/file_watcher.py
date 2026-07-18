# -*- coding: utf-8 -*-
"""
File Watcher — monitor output files for changes and emit parsed data signals.

Watches:
  - judgement.json       → signal_judgement_updated
  - bounce_events.csv    → signal_bounce_events_updated
  - tracknet_rknn_output.csv → signal_trajectory_updated
"""

import os

from PyQt5.QtCore import QFileSystemWatcher, QObject, pyqtSignal


class FileWatcher(QObject):
    """Thin wrapper around QFileSystemWatcher with parsed-data signals."""

    signal_judgement_updated = pyqtSignal(object)   # dict
    signal_bounce_events_updated = pyqtSignal(object)  # list[dict]
    signal_trajectory_updated = pyqtSignal(object)  # list[dict]
    signal_action_prediction_updated = pyqtSignal(object)  # list[str]

    def __init__(self, project_root: str, parent=None):
        super().__init__(parent)
        self._project_root = project_root
        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._on_directory_changed)
        self._watcher.fileChanged.connect(self._on_file_changed)

        # Paths being watched
        self._watch_paths: dict = {}  # normalized path → type string

        self._active = False

    # ── Public API ────────────────────────────────────────────────

    def start_watching(self, task_name: str):
        """Begin watching output files for the given task.

        Args:
            task_name: One of "match_judgement", "body_action", or "court_calibration".
        """
        if self._active:
            self.stop_watching()

        self._active = True

        # Map task names to their output file paths
        match_base = os.path.join(self._project_root, "match_judgement")
        body_base = os.path.join(self._project_root, "body_action")

        files_to_watch = {}

        if task_name in ("match_judgement", "court_calibration"):
            files_to_watch[os.path.join(match_base, "judgement.json")] = "judgement"
            files_to_watch[os.path.join(match_base, "bounce_events.csv")] = "bounce_events"
            files_to_watch[os.path.join(match_base, "tracknet_rknn_output.csv")] = "trajectory"

        if task_name == "body_action":
            files_to_watch[os.path.join(body_base, "outputs", "body_action_prediction.txt")] = "action_prediction"

        if task_name == "side_training":
            side_base = os.path.join(self._project_root, "side_training")
            files_to_watch[os.path.join(side_base, "side_tracknet_output.csv")] = "trajectory"
            files_to_watch[os.path.join(side_base, "body_action_prediction.txt")] = "action_prediction"

        for path, ftype in files_to_watch.items():
            abs_path = os.path.normpath(os.path.abspath(path))
            self._watch_paths[abs_path] = ftype
            # Watch both the file and its parent directory
            if os.path.exists(abs_path):
                self._watcher.addPath(abs_path)
            else:
                # File doesn't exist yet — watch the directory for creation
                parent_dir = os.path.dirname(abs_path)
                if os.path.isdir(parent_dir) and parent_dir not in self._watcher.directories():
                    self._watcher.addPath(parent_dir)

    def stop_watching(self):
        """Remove all watched paths."""
        paths = self._watcher.files() + self._watcher.directories()
        if paths:
            self._watcher.removePaths(paths)
        self._watch_paths.clear()
        self._active = False

    # ── Slots ─────────────────────────────────────────────────────

    def _on_file_changed(self, path: str):
        """Called by QFileSystemWatcher when a watched file is modified."""
        self._process_file(path)

    def _on_directory_changed(self, dir_path: str):
        """Called when a watched directory changes (e.g. output file created).

        If the file we're waiting for appears, add it to the watcher and read it.
        """
        for file_path, ftype in list(self._watch_paths.items()):
            if os.path.dirname(file_path) == dir_path and os.path.exists(file_path):
                if file_path not in self._watcher.files():
                    self._watcher.addPath(file_path)
                    self._process_file(file_path)

    # ── Helpers ───────────────────────────────────────────────────

    def _process_file(self, path: str):
        """Parse a changed file and emit the appropriate signal."""
        normalized = os.path.normpath(path)
        ftype = self._watch_paths.get(normalized)
        if ftype is None:
            return

        # Avoid reading while file is still being written (brief debounce)
        try:
            if os.path.getsize(normalized) == 0:
                return
        except OSError:
            return

        from gui.backend.data_parser import DataParser

        parser = DataParser(self._project_root)

        try:
            if ftype == "judgement":
                data = parser.parse_judgement_json(normalized)
                if data is not None:
                    self.signal_judgement_updated.emit(data)
            elif ftype == "bounce_events":
                data = parser.parse_bounce_events(normalized)
                if data is not None:
                    self.signal_bounce_events_updated.emit(data)
            elif ftype == "trajectory":
                data = parser.parse_trajectory_csv(normalized)
                if data is not None:
                    self.signal_trajectory_updated.emit(data)
            elif ftype == "action_prediction":
                data = parser.parse_action_prediction(normalized)
                if data:
                    self.signal_action_prediction_updated.emit(data)
        except Exception:
            # File may be mid-write; ignore parse errors, next change
            # event will retry.
            pass
