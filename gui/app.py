# -*- coding: utf-8 -*-
"""
RK3588 Tennis Cloud Analysis System — Application Bootstrap
Handles QApplication creation, global stylesheet loading, and font setup.

Supports both PyQt5 (primary, for RK3588 board) and PySide6 (fallback).
"""

import os
import sys

# ── Qt binding compatibility ─────────────────────────────────────
_HAS_PYQT5 = False
_HAS_PYSIDE6 = False

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont, QFontDatabase
    from PyQt5.QtWidgets import QApplication
    _HAS_PYQT5 = True
except ImportError:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont, QFontDatabase
        from PySide6.QtWidgets import QApplication
        _HAS_PYSIDE6 = True
    except ImportError:
        raise ImportError(
            "Neither PyQt5 nor PySide6 is installed. "
            "Install with: pip install pyqt5  or  apt install python3-pyqt5"
        )

# ── Path helpers ─────────────────────────────────────────────────

def _gui_dir():
    """Absolute path to the gui/ package directory."""
    return os.path.dirname(os.path.abspath(__file__))

def _resource_path(relative):
    """Get absolute path to a resource file (works both dev and frozen)."""
    return os.path.join(_gui_dir(), "resources", relative)

# ── Stylesheet loading ───────────────────────────────────────────

def _load_stylesheet():
    """Load the dark theme QSS from resources/style.qss."""
    qss_path = _resource_path("style.qss")
    if not os.path.exists(qss_path):
        return ""
    with open(qss_path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    marker = 'DARK_THEME_QSS = r"""'
    start = raw.find(marker)
    if start >= 0:
        start += len(marker)
        end = raw.rfind('"""')
        if end >= 0:
            return raw[start:end]
    return raw

# ── Font setup for CJK ───────────────────────────────────────────

def _setup_fonts(app: QApplication):
    """Detect and set the best available CJK-capable font."""
    preferred_families = [
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Noto Sans SC",
        "Source Han Sans SC",
        "AR PL UMing CN",
        "SimHei",
        "DejaVu Sans",
    ]
    available = QFontDatabase().families()
    chosen = None
    for family in preferred_families:
        if family in available:
            chosen = family
            break
    if chosen is None:
        return
    default_font = QFont(chosen, 10)
    app.setFont(default_font)

# ── Application factory ──────────────────────────────────────────

def create_application(argv=None):
    """Bootstrap the QApplication with the dark theme.

    Returns a configured QApplication instance ready for window creation.
    """
    if argv is None:
        argv = sys.argv

    # High-DPI support
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(argv)
    app.setApplicationName("RK3588 Tennis Cloud Analysis System")
    app.setOrganizationName("TennisCloud")
    app.setApplicationVersion("3.0.0")

    # Apply dark theme
    qss = _load_stylesheet()
    if qss:
        app.setStyleSheet(qss)

    # Setup CJK fonts
    _setup_fonts(app)

    return app


def using_pyqt5():
    """Check if we're running on PyQt5."""
    return _HAS_PYQT5
