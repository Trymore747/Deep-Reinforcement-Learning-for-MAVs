#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# RUN ME FIRST — One-time sudo setup for RDDRONE-FMUK66 HITL
# Open a terminal and run:   bash "Hardware in the Loop/RDDRONE-FMUK66/RUN_ME_FIRST.sh"
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

echo ""
echo "══════════════════════════════════════════════════════"
echo "  RDDRONE-FMUK66 HITL — One-time sudo setup"
echo "══════════════════════════════════════════════════════"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. System packages
echo "[1/4] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    gcc-arm-none-eabi \
    binutils-arm-none-eabi \
    screen \
    tmux \
    libfuse2 \
    minicom
echo "  OK"

# 2. dialout group (serial port access)
echo "[2/4] Adding $USER to dialout group..."
sudo usermod -aG dialout "$USER"
echo "  OK — will take effect after logout/login"

# 3. udev rules
echo "[3/4] Installing udev rules for FMUK66..."
sudo cp "$SCRIPT_DIR/udev/99-fmuk66.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
echo "  OK — /etc/udev/rules.d/99-fmuk66.rules installed"

# 4. GeographicLib datasets (MAVROS)
echo "[4/4] Checking GeographicLib datasets..."
if [ ! -f /usr/share/GeographicLib/geoids/egm96-5.pgm ]; then
    sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh
    echo "  OK — datasets installed"
else
    echo "  OK — already installed"
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  IMPORTANT: Log out and back in for dialout group."
echo "  Then plug in the FMUK66 and run:"
echo "    bash \"$SCRIPT_DIR/scripts/hitl_launch.sh\""
echo "══════════════════════════════════════════════════════"
echo ""
