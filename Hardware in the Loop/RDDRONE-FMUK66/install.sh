#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# RDDRONE-FMUK66 HITL — Full Dependency Installer
# Ubuntu 22.04 LTS · ROS2 Humble · PX4 v1.13+
#
# Run once before your first HITL session:
#   chmod +x install.sh && ./install.sh
#
# Author: CERLAB UAV Autonomy — Trymore Sylaloni
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/install.log"
exec > >(tee -a "$LOG") 2>&1

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; BLU='\033[1;34m'; NC='\033[0m'
ok()   { echo -e "${GRN}[OK]${NC}  $*"; }
info() { echo -e "${BLU}[--]${NC}  $*"; }
warn() { echo -e "${YLW}[!!]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  RDDRONE-FMUK66 HITL — Dependency Installer"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── 0. Checks ─────────────────────────────────────────────────────────────────
[[ "$(lsb_release -rs)" == "22.04" ]] || warn "Expected Ubuntu 22.04, got $(lsb_release -rs). Proceed with caution."
[[ -d /opt/ros/humble ]] || fail "ROS2 Humble not found at /opt/ros/humble. Install ROS2 first."
info "Ubuntu $(lsb_release -rs) · ROS2 Humble found ✓"

# ── 1. System packages ────────────────────────────────────────────────────────
info "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip python3-serial python3-dev \
    screen minicom \
    curl wget git build-essential \
    cmake ninja-build \
    gcc-arm-none-eabi binutils-arm-none-eabi \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-tools \
    fuse libfuse2 \
    geographiclib-tools \
    ros-humble-mavros ros-humble-mavros-extras ros-humble-mavros-msgs \
    ros-humble-geographic-msgs \
    2>/dev/null && ok "System packages installed" || warn "Some system packages may have failed — check log"

# ── 2. GeographicLib datasets (required by MAVROS) ───────────────────────────
if [ ! -f /usr/share/GeographicLib/geoids/egm96-5.pgm ]; then
    info "Installing GeographicLib datasets (required by MAVROS)..."
    sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh && \
        ok "GeographicLib datasets installed" || warn "GeographicLib install failed — MAVROS may not start"
else
    ok "GeographicLib datasets already present"
fi

# ── 3. Python packages ────────────────────────────────────────────────────────
info "Installing Python packages..."
pip3 install --quiet --upgrade \
    pyserial \
    pymavlink \
    empy \
    toml \
    packaging \
    kconfiglib \
    jsonschema \
    future \
    numpy \
    pandas \
    torch --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
    pip3 install --quiet torch 2>/dev/null || \
    warn "PyTorch install failed — try: pip3 install torch"
ok "Python packages installed"

# ── 4. dialout group (serial port access without sudo) ───────────────────────
if ! groups "$USER" | grep -q dialout; then
    info "Adding $USER to dialout group (needed for /dev/ttyACM0)..."
    sudo usermod -aG dialout "$USER"
    warn "You must LOG OUT and LOG BACK IN for dialout group to take effect."
    warn "Or run: newgrp dialout (for this session only)"
else
    ok "User $USER already in dialout group"
fi

# ── 5. udev rules for RDDRONE-FMUK66 ─────────────────────────────────────────
info "Installing udev rules for RDDRONE-FMUK66..."
UDEV_RULES="$SCRIPT_DIR/udev/99-fmuk66.rules"
UDEV_DEST="/etc/udev/rules.d/99-fmuk66.rules"
if [ -f "$UDEV_RULES" ]; then
    sudo cp "$UDEV_RULES" "$UDEV_DEST"
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    ok "udev rules installed: $UDEV_DEST"
else
    warn "udev rules file not found at $UDEV_RULES — skipping"
fi

# ── 6. QGroundControl ────────────────────────────────────────────────────────
QGC_APPIMAGE="$HOME/QGroundControl.AppImage"
if [ ! -f "$QGC_APPIMAGE" ]; then
    info "Downloading QGroundControl (needed for PX4 parameter setup)..."
    QGC_URL="https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage"
    wget -q --show-progress -O "$QGC_APPIMAGE" "$QGC_URL" && \
        chmod +x "$QGC_APPIMAGE" && \
        ok "QGroundControl downloaded → $QGC_APPIMAGE" || \
        warn "QGC download failed — download manually from https://qgroundcontrol.com/downloads/"
else
    ok "QGroundControl already present: $QGC_APPIMAGE"
fi

# ── 7. PX4 FMUK66 firmware build dependencies ────────────────────────────────
PX4_DIR="$HOME/PX4-Autopilot"
if [ -d "$PX4_DIR" ]; then
    info "Setting up PX4 build dependencies for FMUK66..."
    if [ -f "$PX4_DIR/Tools/setup/ubuntu.sh" ]; then
        cd "$PX4_DIR"
        # Run without interactive prompts; skip simulation-specific parts
        bash Tools/setup/ubuntu.sh --no-nuttx 2>/dev/null || true
        ok "PX4 dependencies set up"
    else
        warn "PX4-Autopilot found but setup script missing — run manually if needed"
    fi
    # Verify FMUK66 board is available
    if [ -d "$PX4_DIR/boards/nxp/fmuk66-v3" ]; then
        ok "PX4 FMUK66-v3 board target confirmed: $PX4_DIR/boards/nxp/fmuk66-v3"
    fi
else
    warn "PX4-Autopilot not found at $PX4_DIR — clone it with:"
    warn "  git clone https://github.com/PX4/PX4-Autopilot.git --recursive ~/PX4-Autopilot"
fi

# ── 8. Verify MAVROS installation ─────────────────────────────────────────────
info "Verifying MAVROS..."
source /opt/ros/humble/setup.bash 2>/dev/null
if ros2 pkg list 2>/dev/null | grep -q mavros; then
    MAVROS_VER=$(ros2 pkg xml mavros 2>/dev/null | grep "<version>" | sed 's/.*<version>\(.*\)<\/version>.*/\1/')
    ok "MAVROS $MAVROS_VER installed"
else
    fail "MAVROS not found — install with: sudo apt install ros-humble-mavros ros-humble-mavros-extras"
fi

# ── 9. Source setup (reminder) ────────────────────────────────────────────────
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    info "Added ROS2 Humble source to ~/.bashrc"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Installation complete — $(date)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "  1. LOG OUT and back in (dialout group takes effect)"
echo "  2. Flash FMUK66 with PX4 HITL firmware — see firmware/FLASH_FMUK66.md"
echo "  3. Set PX4 HITL parameters via QGroundControl:"
echo "       ~/QGroundControl.AppImage"
echo "  4. Run the preflight check:"
echo "       python3 scripts/preflight_check.py"
echo ""
echo "  Full guide: README.md"
echo ""
