#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# RDDRONE-FMUK66 — Sudo-required setup steps
# Run this ONCE in a real terminal (needs password input):
#   bash sudo_setup.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing system packages ==="
sudo apt-get update -qq
sudo apt-get install -y \
    gcc-arm-none-eabi binutils-arm-none-eabi \
    screen minicom \
    fuse libfuse2 \
    python3-serial \
    tmux

echo "=== Adding $USER to dialout group ==="
sudo usermod -aG dialout "$USER"
echo "  [OK] Added — you must log out and back in for this to take effect"
echo "       Or run: newgrp dialout"

echo "=== Installing udev rules for FMUK66 ==="
sudo cp "$SCRIPT_DIR/udev/99-fmuk66.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
echo "  [OK] udev rules installed → /etc/udev/rules.d/99-fmuk66.rules"

echo "=== Installing GeographicLib datasets (MAVROS) ==="
if [ ! -f /usr/share/GeographicLib/geoids/egm96-5.pgm ]; then
    sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
    echo "  [OK] GeographicLib datasets installed"
else
    echo "  [OK] Already installed"
fi

echo ""
echo "=== sudo_setup.sh complete ==="
echo "  → Log out and back in (dialout group)"
echo "  → Then run: bash install.sh (for Python packages + QGC)"
echo ""
