# RDDRONE-FMUK66 Hardware-in-the-Loop (HITL) Testing
## R54 TD3 Autonomous Tunnel Navigation Policy

**Author:** Trymore Sylaloni — CERLAB UAV Autonomy  
**Date:** 2026-09-10  
**Target:** NXP RDDRONE-FMUK66 v3 running PX4 v1.13+

---

## What is HITL?

Hardware-in-the-Loop (HITL) places a **real flight controller** running its full
firmware stack into a simulation loop. Gazebo provides physics, sensors, and
actuator feedback — but every flight-critical computation (EKF2 state estimation,
motor mixing, PID loops, failsafe logic) runs on the actual FMUK66 hardware.

```
┌─────────────────────────────┐    MAVLink (USB/UART)
│  Gazebo 11                  │ ◄────────────────────► RDDRONE-FMUK66
│  • Physics simulation       │                        • PX4 firmware
│  • Depth camera (RealSense) │                        • EKF2 (real)
│  • IMU / GPS / Baro (sim)   │                        • Motor mixing
│  • tunnel_100m_realworld    │                        • PID loops
└─────────────────────────────┘                        • Failsafes
             │
             │ ROS2 topics
             ▼
┌─────────────────────────────┐
│  MAVROS (ROS2 Humble)       │
│  /mavros/state              │
│  /mavros/local_position/odom│
│  /mavros/setpoint_position/ │
└─────────────────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  R54 HITL Inference Script  │
│  run_hitl_r54.py            │
│  • Reads depth camera       │
│  • Runs TD3 actor (21D→3)   │
│  • Sends position setpoints │
└─────────────────────────────┘
```

---

## About the RDDRONE-FMUK66

| Property | Detail |
|----------|--------|
| MCU | NXP Kinetis K66 (ARM Cortex-M4 @ 180 MHz) |
| IMU | NXP FXOS8700 + FXAS21002C (accel/gyro/mag) |
| Baro | NXP MPL3115A2 |
| USB | USB-CDC ACM (appears as `/dev/ttyACM0`) |
| PX4 build target | `nxp_fmuk66-v3_default` |
| MAVLink baud | 57600 (USB default) |
| Firmware | PX4 v1.13+ |

---

## Folder Structure

```
RDDRONE-FMUK66/
├── README.md                    ← this file
├── install.sh                   ← one-shot dependency installer
├── config/
│   ├── mavros_fmuk66.yaml       ← MAVROS connection + plugin config
│   └── fmuk66_hitl_params.txt  ← PX4 parameters to set in QGroundControl
├── firmware/
│   └── FLASH_FMUK66.md         ← How to flash PX4 HITL firmware
├── scripts/
│   ├── hitl_launch.sh           ← One-shot Gazebo + MAVROS launcher
│   └── fmuk66_connect_test.py   ← Verifies FC is connected and responsive
├── udev/
│   └── 99-fmuk66.rules         ← USB access rules (no sudo needed)
└── install.log                  ← Created by install.sh
```

The actual episode runner and preflight checker are in the parent folder:
```
Hardware in the Loop/
├── scripts/
│   ├── run_hitl_r54.py          ← Run 10 HITL test episodes
│   ├── preflight_check.py       ← 8-point safety check
│   └── emergency_stop.py        ← Instant disarm
└── launch/
    └── hitl.launch.py           ← ROS2 launch (Gazebo + MAVROS)
```

---

## Step 0 — Install all dependencies (one time only)

```bash
cd "Hardware in the Loop/RDDRONE-FMUK66"
chmod +x install.sh
./install.sh
```

This installs:
- `gcc-arm-none-eabi` — ARM cross-compiler (for PX4 firmware build)
- `ros-humble-mavros`, `ros-humble-mavros-extras` — MAVROS ROS2 packages
- GeographicLib datasets — required by MAVROS at startup
- `pyserial`, `pymavlink` — Python serial utilities
- `screen`, `minicom` — serial terminal tools
- QGroundControl AppImage — for PX4 parameter setup
- udev rules for `/dev/ttyACM0` access without sudo

**After install:** log out and back in so the `dialout` group takes effect.

---

## Step 1 — Flash PX4 HITL firmware onto FMUK66

> Skip if firmware is already flashed. Check with `screen /dev/ttyACM0 57600` → type `ver all`.

**Easiest method — QGroundControl:**
```bash
~/QGroundControl.AppImage
```
Connect FMUK66 via USB → Vehicle Setup → Firmware → PX4 Pro Stable → OK.

**Build from source:**
```bash
cd ~/PX4-Autopilot
make nxp_fmuk66-v3_default        # build
make nxp_fmuk66-v3_default upload  # flash (board in bootloader mode)
```

Full flashing guide: `firmware/FLASH_FMUK66.md`

---

## Step 2 — Set PX4 HITL parameters in QGroundControl

```bash
~/QGroundControl.AppImage
```

Connect FMUK66 → Vehicle Setup → Parameters → search each name:

| Parameter | Value | Why |
|-----------|-------|-----|
| `HITL_ENABLE` | **1** | Enable HITL mode — **REBOOT after this** |
| `SIM_GPS_USED` | 1 | Use Gazebo GPS, not real GPS |
| `CBRK_IO_SAFETY` | 22027 | No IO board in HITL |
| `CBRK_USB_CHK` | 197848 | Allow arming via USB |
| `CBRK_FLIGHTTERM` | 121212 | Disable flight termination |
| `COM_RCL_EXCEPT` | 4 | Ignore RC loss in OFFBOARD mode |
| `COM_RC_IN_MODE` | 1 | No RC transmitter required |
| `GF_ACTION` | 0 | No geofence action |
| `MPC_XY_CRUISE` | 4.5 | Match R54 policy speed cap |

**Reboot the FC** after setting `HITL_ENABLE=1` (power cycle or QGC Reboot button).

Full parameter list with explanations: `config/fmuk66_hitl_params.txt`

---

## Step 3 — Check serial port permissions

```bash
ls -la /dev/ttyACM*
# Should show crw-rw---- with group dialout
# If not: sudo chmod a+rw /dev/ttyACM0
```

Stable symlink after udev rules are installed:
```bash
ls -la /dev/fmuk66   # → /dev/ttyACM0
```

---

## Step 4 — Run HITL test (4 terminals)

Source ROS2 in every terminal:
```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
```

### Terminal 1 — Launch Gazebo + MAVROS
```bash
# Option A: one-shot launcher (uses tmux if available)
bash "Hardware in the Loop/RDDRONE-FMUK66/scripts/hitl_launch.sh" \
    --port /dev/ttyACM0 --baud 57600

# Option B: manual ROS2 launch
ros2 launch "Hardware in the Loop/launch/hitl.launch.py" \
    world:=tunnel_100m_realworld.world \
    fcu_port:=/dev/ttyACM0 \
    fcu_baud:=57600 \
    gui:=true
```
Wait until you see: `FCU: [INFO] HITL mode` in the MAVROS output.

### Terminal 2 — Verify FMUK66 connection
```bash
python3 "Hardware in the Loop/RDDRONE-FMUK66/scripts/fmuk66_connect_test.py"
```
All 5 checks must show `[OK]`.

### Terminal 3 — Full preflight safety check
```bash
python3 "Hardware in the Loop/scripts/preflight_check.py"
```
All 8 checks must pass before proceeding.

### Terminal 4 — Run 10 test episodes
```bash
python3 "Hardware in the Loop/scripts/run_hitl_r54.py" --episodes 10 --max-speed 4.5
```

### Terminal 5 — Keep emergency stop ready at all times
```bash
# Do NOT run yet — paste and press Enter only if something goes wrong:
python3 "Hardware in the Loop/scripts/emergency_stop.py"
```

---

## What happens during a HITL episode

```
1. Script streams position setpoints (pre-arm OFFBOARD requirement)
2. MAVROS sends MAV_CMD_COMPONENT_ARM_DISARM → FMUK66 arms
3. Set mode: OFFBOARD
4. Climb to SAFE_ALT = 1.5 m
5. R54 actor runs at ~10 Hz:
      depth_image → 21D state → TD3 actor → (dx, dy, dz) setpoint
6. MAVROS publishes /mavros/setpoint_position/local
7. FMUK66 EKF2 estimates position from Gazebo sensors
8. FMUK66 MPC executes position control → Gazebo physics updates
9. Episode ends: SUCCESS / STUCK / TIMEOUT / CRASH_ALT / DISARMED
10. FMUK66 disarmed → Gazebo model reset → wait 30-60 s → next episode
```

---

## Results

Saved automatically to `Hardware in the Loop/results/`:
```
hitl_r54_attempt{N}_{timestamp}.csv          ← step-by-step telemetry
hitl_r54_{N}ep_summary_{timestamp}.json      ← aggregate stats
```

Directly comparable to the Gazebo-only real-world results in:
```
Final Results/testing/R54_RealWorld_TunnelTile5_20260905/
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `[FAIL] MAVROS ↔ FMUK66 heartbeat` | Wrong port / baud, no Gazebo running | Check cable, run `hitl.launch.py` first |
| `[FAIL] EKF local position` | `HITL_ENABLE` not set | Set in QGC, reboot FC |
| Drone won't arm | FC still in MANUAL, not enough setpoints | Let launch script stream 2 s of setpoints first |
| MAVROS `GeographicLib` error | Datasets not installed | Run `sudo /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh` |
| Permission denied `/dev/ttyACM0` | Not in dialout group | `sudo chmod a+rw /dev/ttyACM0` (temporary) or log out/in (permanent) |
| `CBRK_USB_CHK` arming fail | Param not set | Set `CBRK_USB_CHK=197848` in QGC |
| Drone oscillates badly | MPC gains not tuned for HITL | Lower `MPC_XY_VEL_P_ACC` or increase `MPC_XY_P` |

---

## Key differences: FMUK66 HITL vs Gazebo-only (SITL)

| Aspect | Gazebo-only SITL | FMUK66 HITL |
|--------|-----------------|-------------|
| State estimation | Gazebo ground truth | FMUK66 EKF2 on real silicon |
| Motor mixing | Simulated | Real PX4 mixer on K66 |
| Stabilisation | Simulated PID | Real PX4 PID on K66 |
| Arming | Auto via topic | MAVROS → MAVLink → FMUK66 |
| Position topic | `/CERLAB/quadcopter/pose_raw` | `/mavros/local_position/odom` |
| Setpoint topic | `/CERLAB/quadcopter/setpoint_pose` | `/mavros/setpoint_position/local` |
| Depth camera | Gazebo plugin (unchanged) | Gazebo plugin (unchanged) |
| R54 policy | Unchanged | Unchanged |
| Failsafes | None | Full PX4 failsafe stack |

---

## Emergency procedure

If anything goes wrong at any point:

```bash
python3 "Hardware in the Loop/scripts/emergency_stop.py"
```

This sends `STABILIZED` mode + disarm via MAVROS in under 500 ms.

**Hardware kill:** disconnect the FMUK66 USB cable. MAVROS loses heartbeat →
PX4 triggers `COM_OF_LOSS_T` failsafe within 5 s.
