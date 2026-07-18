#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# RK3588 Tennis Cloud Analysis System — GUI Launcher
# Place this script on the RK3588 board desktop for one-click launch.
#
# Usage:
#   bash run_gui.sh              # windowed mode (1280x800)
#   bash run_gui.sh --fullscreen # fullscreen mode
#   bash run_gui.sh --kiosk      # frameless fullscreen kiosk mode

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GUI_DIR="${SCRIPT_DIR}/gui"

# Check GUI directory exists
if [ ! -d "$GUI_DIR" ]; then
    echo "ERROR: gui/ directory not found at ${GUI_DIR}"
    exit 1
fi

# Check Python dependencies
echo "Checking Python environment..."
python3 -c "import sys; sys.stdout.reconfigure(encoding='utf-8')" 2>/dev/null || true
python3 -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')" 2>/dev/null || {
    echo "ERROR: PyQt5 is not installed."
    echo "Install with: sudo apt install python3-pyqt5"
    exit 1
}

# Kill any existing GUI instance
if pgrep -f "tennis_gui.py" > /dev/null 2>&1; then
    echo "Stopping existing GUI instance..."
    pkill -f "tennis_gui.py" 2>/dev/null || true
    sleep 1
fi

# Parse arguments
FULLSCREEN=""
KIOSK=""
for arg in "$@"; do
    case "$arg" in
        --fullscreen|-f) FULLSCREEN="--fullscreen" ;;
        --kiosk|-k) KIOSK="--kiosk" ;;
    esac
done

# Set display environment — RK3588 board runs Wayland
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland}"
export QT_SCALE_FACTOR="${QT_SCALE_FACTOR:-1}"

# Fix OpenCV Qt plugin conflict: OpenCV bundles a broken libqxcb.so that
# overrides the system Qt plugins. Force system plugin path.
export QT_PLUGIN_PATH="${QT_PLUGIN_PATH:-/usr/lib/aarch64-linux-gnu/qt5/plugins}"

# For EGLFS (direct rendering without compositor), uncomment:
# export QT_QPA_PLATFORM=eglfs
# export QT_QPA_EGLFS_PHYSICAL_WIDTH=1920
# export QT_QPA_EGLFS_PHYSICAL_HEIGHT=1080

echo "============================================"
echo "  RK3588 Tennis Cloud Analysis System"
echo "  Starting GUI..."
echo "  Display: ${DISPLAY:-(Wayland)}"
echo "  Platform: ${QT_QPA_PLATFORM}"
echo "============================================"

cd "$GUI_DIR"
exec python3 tennis_gui.py $FULLSCREEN $KIOSK
