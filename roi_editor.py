#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Editor — visual drag-to-select player detection regions.

Usage: python3 roi_editor.py [--image path/to/screenshot.jpg]

Shows the camera/image with two draggable rectangles:
  Green = top player area
  Blue  = bottom player area

Drag corners to resize, drag center to move.
Saves to broadcast_court_only.json as 0-1 fractions.
"""

import json
import os
import sys
import tkinter as tk

import cv2
from PIL import Image, ImageTk

CONFIG_PATH = os.path.expanduser(
    "~/rk3588_tennis_system/match_judgement/broadcast_court_only.json")

# Default regions (fractions 0-1)
DEFAULT_CROPS = {
    "top_player":    {"x1": 0.12, "y1": 0.10, "x2": 0.88, "y2": 0.56},
    "bottom_player": {"x1": 0.10, "y1": 0.38, "x2": 0.90, "y2": 0.98},
}

COLORS = {
    "top_player":    (57, 255, 20),   # green
    "bottom_player": (58, 166, 255),  # blue
}


class ROIEditor:
    def __init__(self, image_path=None):
        self.root = tk.Tk()
        self.root.title("ROI 区域编辑器 — 拖动绿框/蓝框调整范围")
        self.root.configure(bg="#0a0e14")

        # Load image
        if image_path and os.path.exists(image_path):
            self.frame = cv2.imread(image_path)
        else:
            # Grab one frame from camera
            cap = cv2.VideoCapture("/dev/video11", cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            ret, self.frame = cap.read()
            cap.release()
            if not ret:
                self.frame = np.zeros((600, 800, 3), dtype=np.uint8)

        self.h, self.w = self.frame.shape[:2]
        self.scale = min(800 / self.w, 500 / self.h)
        self.disp_w = int(self.w * self.scale)
        self.disp_h = int(self.h * self.scale)

        # Load existing or default config
        self.crops = DEFAULT_CROPS.copy()
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH) as f:
                    saved = json.load(f)
                for c in saved.get("crops", []):
                    if c["name"] in self.crops:
                        self.crops[c["name"]] = {
                            "x1": c["x1"], "y1": c["y1"],
                            "x2": c["x2"], "y2": c["y2"],
                        }
            except Exception:
                pass

        # Canvas
        self.canvas = tk.Canvas(
            self.root, width=self.disp_w, height=self.disp_h,
            bg="black", highlightthickness=0)
        self.canvas.pack(padx=4, pady=(4, 0))

        # Buttons
        bar = tk.Frame(self.root, bg="#0a0e14")
        bar.pack(fill=tk.X, padx=8, pady=6)

        tk.Button(bar, text="💾 保存配置", bg="#1a7f37", fg="white",
                  font=("", 11, "bold"), relief=tk.FLAT,
                  command=self._save).pack(side=tk.LEFT, padx=2)

        self.lbl = tk.Label(bar, text="拖动矩形框调整球员检测区域",
                            bg="#0a0e14", fg="#8b949e", font=("", 9))
        self.lbl.pack(side=tk.LEFT, padx=12)

        tk.Button(bar, text="✕ 退出", bg="#da3633", fg="white",
                  font=("", 10), relief=tk.FLAT,
                  command=self.root.destroy).pack(side=tk.RIGHT, padx=2)

        # Mouse state
        self._drag = None   # (crop_name, corner) or (crop_name, "move")
        self._active_crop = "top_player"

        # Bindings
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._switch_crop)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self._redraw()

    # ── Drawing ──────────────────────────────────────────────────

    def _redraw(self):
        self.canvas.delete("all")

        # Draw image
        img = cv2.resize(self.frame, (self.disp_w, self.disp_h))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

        # Draw rectangles
        for name in ["top_player", "bottom_player"]:
            c = self.crops[name]
            x1 = int(c["x1"] * self.disp_w)
            y1 = int(c["y1"] * self.disp_h)
            x2 = int(c["x2"] * self.disp_w)
            y2 = int(c["y2"] * self.disp_h)
            color = "#{:02x}{:02x}{:02x}".format(*COLORS[name])
            active = name == self._active_crop

            # Fill
            alpha = 40 if active else 20
            self.canvas.create_rectangle(x1, y1, x2, y2,
                                         fill="", outline=color,
                                         width=3 if active else 2,
                                         tags=(name,))

            # Corner handles
            for cx, cy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                sz = 8 if active else 5
                self.canvas.create_rectangle(
                    cx-sz, cy-sz, cx+sz, cy+sz,
                    fill=color, outline="", tags=(name, "corner"))

            # Label
            label = "上半场" if name == "top_player" else "下半场"
            self.canvas.create_text(
                (x1+x2)//2, (y1+y2)//2, text=label,
                fill=color, font=("", 14 if active else 10, "bold"),
                tags=(name,))

    # ── Mouse ────────────────────────────────────────────────────

    def _switch_crop(self, event):
        """Right-click to switch which rectangle is active."""
        self._active_crop = (
            "bottom_player" if self._active_crop == "top_player"
            else "top_player")
        self.lbl.configure(text="当前编辑: {} (右键切换)".format(
            "上半场绿框" if self._active_crop == "top_player" else "下半场蓝框"))
        self._redraw()

    def _on_press(self, event):
        ex, ey = event.x, event.y
        c = self.crops[self._active_crop]
        corners = {
            "nw": (c["x1"], c["y1"]), "ne": (c["x2"], c["y1"]),
            "sw": (c["x1"], c["y2"]), "se": (c["x2"], c["y2"]),
        }
        for name, (fx, fy) in corners.items():
            cx = int(fx * self.disp_w)
            cy = int(fy * self.disp_h)
            if abs(ex - cx) < 12 and abs(ey - cy) < 12:
                self._drag = (name,)
                return
        # Check if inside rect
        x1 = int(c["x1"] * self.disp_w)
        y1 = int(c["y1"] * self.disp_h)
        x2 = int(c["x2"] * self.disp_w)
        y2 = int(c["y2"] * self.disp_h)
        if x1 <= ex <= x2 and y1 <= ey <= y2:
            self._drag = ("move", ex - x1, ey - y1)
            return

    def _on_drag(self, event):
        if not self._drag:
            return
        c = self.crops[self._active_crop]
        fx = max(0, min(1, event.x / self.disp_w))
        fy = max(0, min(1, event.y / self.disp_h))

        if self._drag[0] == "move":
            dw = c["x2"] - c["x1"]
            dh = c["y2"] - c["y1"]
            ox, oy = self._drag[1], self._drag[2]
            c["x1"] = max(0, min(1-dw, fx - ox/self.disp_w))
            c["y1"] = max(0, min(1-dh, fy - oy/self.disp_h))
            c["x2"] = c["x1"] + dw
            c["y2"] = c["y1"] + dh
        elif len(self._drag) == 1:  # corner name
            corner = self._drag[0]
            if "n" in corner:
                c["y1"] = min(fy, c["y2"] - 0.02)
            if "s" in corner:
                c["y2"] = max(fy, c["y1"] + 0.02)
            if "w" in corner:
                c["x1"] = min(fx, c["x2"] - 0.02)
            if "e" in corner:
                c["x2"] = max(fx, c["x1"] + 0.02)
        self._redraw()

    def _on_release(self, event):
        self._drag = None

    # ── Save ─────────────────────────────────────────────────────

    def _save(self):
        config = {
            "description": "Player detection ROI — set via visual editor",
            "crops": [
                {"name": k, **v} for k, v in self.crops.items()
            ],
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        self.lbl.configure(text="✅ 已保存到 broadcast_court_only.json")
        self.root.after(2000, lambda: self.lbl.configure(
            text="拖动矩形框调整球员检测区域"))


if __name__ == "__main__":
    import numpy as np
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    editor = ROIEditor(img_path)
    editor.root.mainloop()
