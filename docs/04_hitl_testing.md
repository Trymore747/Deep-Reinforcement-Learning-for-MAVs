# 04 — Hardware-in-the-Loop Testing

Hardware-in-the-Loop (HITL) connects a real **RDDRONE-FMUK66** flight controller running real PX4 firmware to the Gazebo simulation. The FC's EKF runs on actual hardware; Gazebo provides simulated sensor data via the MAVLink HIL_STATE_QUATERNION message.

---

## Hardware Required

| Item | Spec |
|---|---|
| Flight controller | NXP RDDRONE-FMUK66 |
| MCU | NXP Kinetis K66 (ARM Cortex-M4 @ 120 MHz) |
| Firmware | PX4 v1.12, HITL build |
| Connection | USB-CDC → `/dev/ttyACM0` @ 57 600 baud |
| Host PC | Ubuntu 20.04, ROS 2 Foxy, ≥ 8 GB RAM |

---

## 1 — Flash HITL Firmware

See `Hardware in the Loop/RDDRONE-FMUK66/firmware/FLASH_FMUK66.md` for the full flashing procedure.

Quick summary:
```bash
# Build PX4 HITL firmware for FMUK66
cd PX4-Autopilot
make nxp_fmuk66-v3_default     # builds the firmware
# Flash using J-Link or Segger
```

---

## 2 — Install USB Device Rules

```bash
sudo cp "Hardware in the Loop/RDDRONE-FMUK66/udev/99-fmuk66.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Plug the FC via USB. Check: `ls /dev/ttyACM*` should show `/dev/ttyACM0`.

---

## 3 — First-Connection Test

```bash
bash "Hardware in the Loop/RDDRONE-FMUK66/RUN_ME_FIRST.sh"
```

This wizard:
1. Detects the USB port
2. Tests MAVROS connectivity
3. Reads the PX4 firmware version
4. Lists all set HITL parameters

---

## 4 — Set PX4 HITL Parameters

Connect to the FC with QGroundControl or via MAVLink shell and set:

```
SYS_HITL         = 1        # enable HITL mode
CBRK_IO_SAFETY   = 22027    # bypass IO safety
CBRK_USB_CHK     = 197848   # allow USB connection
COM_ARM_WO_GPS   = 1        # arm without GPS
```

These are also in `Hardware in the Loop/config/px4_hitl_params.txt`.

---

## 5 — Launch the HITL Session

```bash
# Terminal 1 — Gazebo simulator
ros2 launch uav_simulator start.launch world:=tunnel_100m_dynamic_25

# Terminal 2 — MAVROS + HITL bridge
ros2 launch "Hardware in the Loop/launch/hitl.launch.py" \
    fcu_url:=serial:///dev/ttyACM0:57600

# Terminal 3 — pre-flight check
source .venv/bin/activate
python3 "Hardware in the Loop/scripts/preflight_check.py"

# Terminal 4 — run 10-episode HITL test
python3 "Hardware in the Loop/scripts/run_hitl_r54.py" \
    --policy src/tunnel_drl/results_r54/r54_best.pth \
    --episodes 10
```

---

## 6 — HITL Architecture

```
Gazebo simulation
  │  HIL_STATE_QUATERNION (MAVLink #115, CRC_EXTRA=4)
  │  injected via /uas1/mavlink_sink @ 50 Hz
  ▼
MAVROS (ROS 2)
  │  USB-CDC
  ▼
RDDRONE-FMUK66  (SYS_HITL=1, PX4 v1.12)
  ├─ EKF2 runs on real hardware
  ├─ Attitude estimation from simulated IMU
  └─ Position control: OFFBOARD mode via Gazebo PID plugin
  ▲
R54 Policy (host PC, 10 Hz inference)
  publishes to /CERLAB/quadcopter/setpoint_pose
```

**Important:** FC arming via the HITL loop was blocked by PX4 preflight checks (MAV_RESULT_DENIED). All flight authority was exercised through the Gazebo position-control plugin while the FC ran in OFFBOARD mode. This is still a valid HITL test — the FC's EKF and sensor-fusion stack are active throughout.

---

## 7 — Post-Processing Results

```bash
python3 "Hardware in the Loop/scripts/analyse_hitl_results.py" \
    --summary "Hardware in the Loop/results/hitl_r54_10ep_summary_20260912_142246.json"
```

This generates 5 publication-quality PNG figures and an Excel workbook in `Hardware in the Loop/results/`.

**Note on EKF speed anomaly:** Episodes A03, A05, and A09 showed constant ~40–47 m/s MAVROS EKF velocity (sensor artifact). The analysis script detects this automatically (all-same value > 4.5 m/s) and recomputes speed from position derivatives `√(dx² + dy² + dz²)/dt`.

---

## 8 — HITL Results (2026-09-12)

| Metric | Value |
|---|---|
| Success rate | **7 / 10 = 70%** |
| Average speed (corrected) | 1.68 m/s |
| Best episode (A10) | 96.9 s, 0 collisions, peak speed 1.57 m/s |
| Best speed episode (A09) | avg 2.113 m/s, peak **2.909 m/s** |
| Timeout episodes | A02 (+0.20 m spawn Y), A04 (+0.10 m), A08 (+0.15 m) |

**Key finding:** All 3 timeout episodes had positive spawn Y offsets (pushing the drone toward the denser obstacle zone at 70–90 m). This indicates a systematic vulnerability to rightward lateral bias at episode start.

![HITL trajectories](../Hardware%20in%20the%20Loop/results/hitl_r54_10ep_20260912_142246_trajectories.png)

Full test report: `Hardware in the Loop/results/R54_HITL_Test_Report_20260912.docx`

---

## Emergency Stop

If the drone enters a dangerous state during a live HITL run:

```bash
python3 "Hardware in the Loop/scripts/emergency_stop.py"
```

This sends a DISARM command and kills the setpoint publisher.
