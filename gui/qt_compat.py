# -*- coding: utf-8 -*-
"""
Qt binding compatibility layer.

Provides unified imports that work with both PyQt5 and PySide6.
To use in other modules::

    from gui.qt_compat import QtCore, QtGui, QtWidgets
    from gui.qt_compat import Signal, pyqtSlot, Property
"""

# ── Core ─────────────────────────────────────────────────────────

_HAS_PYQT5 = False
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import (
        Qt, QTimer, QRect, QRectF, QPointF, QSize, QEvent,
        QObject, QFileSystemWatcher, QTimer, QThread,
        pyqtSignal as Signal,
        pyqtSlot as Slot,
        Property,
    )
    _HAS_PYQT5 = True
except ImportError:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        from PySide6.QtCore import (
            Qt, QTimer, QRect, QRectF, QPointF, QSize, QEvent,
            QObject, QFileSystemWatcher, QTimer, QThread,
            Signal,
            Slot,
            Property,
        )
    except ImportError:
        raise ImportError(
            "Neither PyQt5 nor PySide6 is installed.\n"
            "Install: pip install pyqt5  or  apt install python3-pyqt5"
        )

# ── Re-export commonly used names ────────────────────────────────

__all__ = [
    "QtCore", "QtGui", "QtWidgets",
    "Qt", "QTimer", "QRect", "QRectF", "QPointF", "QSize", "QEvent",
    "QObject", "QFileSystemWatcher", "QThread",
    "Signal", "Slot", "Property",
    "using_pyqt5",
]

def using_pyqt5():
    return _HAS_PYQT5
