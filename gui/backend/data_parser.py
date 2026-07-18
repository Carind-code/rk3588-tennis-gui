# -*- coding: utf-8 -*-
"""
Data Parser — read and parse output files from backend scripts.

Supported formats:
  - judgement.json       → dict with score, verdict, point_summary
  - bounce_events.csv    → list of per-bounce dicts
  - tracknet_rknn_output.csv → list of per-frame dicts
  - body_action_prediction.txt → list of action labels
"""

import csv
import json
import os


class DataParser:
    """Parse output artifacts from the tennis analysis backend.

    All file I/O uses explicit ``encoding='utf-8'`` to avoid garbled text.
    """

    def __init__(self, project_root: str = None):
        self._project_root = project_root or ""

    # ── Public parsers ────────────────────────────────────────────

    def parse_judgement_json(self, path: str = None) -> dict:
        """Parse judgement.json into a dictionary.

        Returns a dict with keys like::

            {
                "verdict": "IN" | "OUT",
                "score_top": int,
                "score_bottom": int,
                "last_bounce": {...},
                "point_summary": [...],
                ...
            }

        Returns an empty dict on error.
        """
        path = self._resolve(path, "match_judgement", "judgement.json")
        return self._read_json(path)

    def parse_bounce_events(self, path: str = None) -> list:
        """Parse bounce_events.csv into a list of per-bounce dicts.

        Each dict has keys: frame, court_x, court_y, confidence, verdict, ...
        """
        path = self._resolve(path, "match_judgement", "bounce_events.csv")
        return self._read_csv(path)

    def parse_trajectory_csv(self, path: str = None) -> list:
        """Parse tracknet_rknn_output.csv into a list of per-frame dicts.

        Each dict has keys: frame, img_x, img_y, court_x, court_y, confidence, ...
        """
        path = self._resolve(path, "match_judgement", "tracknet_rknn_output.csv")
        return self._read_csv(path)

    def parse_action_prediction(self, path: str = None) -> list:
        """Parse body_action_prediction.txt into a list of action label strings.

        Example: ["serve", "serve", "forehand", "background", ...]
        """
        path = self._resolve(
            path, "body_action", "outputs", "body_action_prediction.txt"
        )
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except Exception:
            return []

        # The file format is:
        #   ### Frame level recognition: ###
        #   serve serve forehand forehand background ...
        lines = text.strip().splitlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                return line.split()
        return []

    # ── Internal helpers ──────────────────────────────────────────

    def _resolve(self, path, *parts):
        """Resolve a path — absolute, or relative to project_root."""
        if path and os.path.isabs(path):
            return os.path.normpath(path)
        if path:
            return os.path.normpath(os.path.join(self._project_root, path))
        return os.path.normpath(os.path.join(self._project_root, *parts))

    @staticmethod
    def _read_json(path: str) -> dict:
        """Safely read a JSON file. Returns empty dict on any error."""
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
            return {"_raw": data}
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _read_csv(path: str) -> list:
        """Safely read a CSV file into a list of dicts (first row = header)."""
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = []
                for row in reader:
                    # Attempt numeric conversion for common fields
                    cleaned = {}
                    for k, v in row.items():
                        k = k.strip()
                        v = v.strip()
                        try:
                            cleaned[k] = float(v) if "." in v or "e" in v.lower() else int(v)
                        except (ValueError, TypeError):
                            cleaned[k] = v
                    rows.append(cleaned)
                return rows
        except Exception:
            return []
