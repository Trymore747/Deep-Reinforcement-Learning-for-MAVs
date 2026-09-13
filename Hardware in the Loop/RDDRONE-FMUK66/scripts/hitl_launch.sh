#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# RDDRONE-FMUK66 HITL — One-shot launcher
# Starts Gazebo (real-world tunnel) + MAVROS in a single terminal using tmux.
# Opens separate panes for Gazebo, MAVROS, and a status monitor.
#
# Usage:
#   chmod +x scripts/hitl_launch.sh
#   ./scripts/hitl_launch.sh [--port /dev/ttyACM0] [--baud 57600] [--no-gui]
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

PORT="/dev/ttyACM0"
BAUD="57600"
GUI="true"
WORLD="tunnel_100m_realworld.world"

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)  PORT="$2"; shift 2 ;;
        --baud)  BAUD="$2"; shift 2 ;;
        --no-gui) GUI="false"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
HITL_DIR="$PROJECT_ROOT/Hardware in the Loop"
FMUK_DIR="$HITL_DIR/RDDRONE-FMUK66"

# ── Source ROS2 ───────────────────────────────────────────────────────────────
source /opt/ros/humble/setup.bash
if [ -f "$PROJECT_ROOT/install/setup.bash" ]; then
    source "$PROJECT_ROOT/install/setup.bash"
fi

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  RDDRONE-FMUK66 HITL Launcher"
echo "  FC port : $PORT @ $BAUD baud"
echo "  World   : $WORLD"
echo "  GUI     : $GUI"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Check serial port ─────────────────────────────────────────────────────────
if [ ! -e "$PORT" ]; then
    echo "[!!] Serial port $PORT not found."
    echo "     Connect the FMUK66 via USB, then re-run."
    echo "     Available ports: $(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo 'none')"
    exit 1
fi
echo "[OK] Serial port $PORT present"

# ── Check GeographicLib datasets (MAVROS requirement) ────────────────────────
if [ ! -f /usr/share/GeographicLib/geoids/egm96-5.pgm ]; then
    echo "[!!] GeographicLib datasets missing. Run:"
    echo "     sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh"
    exit 1
fi
echo "[OK] GeographicLib datasets present"

# ── Launch: prefer tmux, fall back to background processes ───────────────────
if command -v tmux &>/dev/null; then
    SESSION="hitl_fmuk66"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" -x 220 -y 50

    # Pane 0: Gazebo + MAVROS launch
    tmux send-keys -t "$SESSION:0" "
source /opt/ros/humble/setup.bash
[ -f '$PROJECT_ROOT/install/setup.bash' ] && source '$PROJECT_ROOT/install/setup.bash'
ros2 launch '$HITL_DIR/launch/hitl.launch.py' \
    world:=$WORLD \
    fcu_port:=$PORT \
    fcu_baud:=$BAUD \
    gui:=$GUI
" Enter

    # Pane 1: status monitor
    tmux split-window -h -t "$SESSION:0"
    tmux send-keys -t "$SESSION:0.1" "
sleep 5
source /opt/ros/humble/setup.bash
echo '--- ROS2 topic list ---'
ros2 topic list
echo ''
echo '--- Waiting for MAVROS state ---'
ros2 topic echo /mavros/state --once
" Enter

    tmux attach -t "$SESSION"
else
    # No tmux — launch Gazebo+MAVROS in background, print instructions
    echo "[--] tmux not found — launching in background (install tmux for split-pane view)"
    ros2 launch "$HITL_DIR/launch/hitl.launch.py" \
        world:=$WORLD \
        fcu_port:=$PORT \
        fcu_baud:=$BAUD \
        gui:=$GUI &
    LAUNCH_PID=$!
    echo "[OK] Launch PID: $LAUNCH_PID"
    echo "     Stop with: kill $LAUNCH_PID"
    echo ""
    echo "     Next: run the connection test in another terminal:"
    echo "       python3 '$FMUK_DIR/scripts/fmuk66_connect_test.py'"
    wait $LAUNCH_PID
fi
