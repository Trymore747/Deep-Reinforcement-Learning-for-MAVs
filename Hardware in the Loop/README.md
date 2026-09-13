# Hardware in the Loop (HITL) — R54 TD3 Policy Testing

Hardware-in-the-Loop (HITL) replaces PX4 SITL software with a **real Pixhawk
flight controller** running actual PX4 firmware. Gazebo still provides the physics
simulation and simulated sensors (depth camera, GPS, barometer, IMU) via MAVLink —
but the motor mixing, EKF, and flight control loops run on the real hardware.

```
┌──────────────┐     MAVLink      ┌─────────────────────────┐
│  Gazebo      │ ←──────────────→ │  Pixhawk (real PX4)     │
│  (physics +  │    HITL msgs     │  firmware — EKF, mixing  │
│   sensors)   │                  └───────────┬─────────────┘
└──────────────┘                              │ USB serial
                                   ┌──────────▼─────────────┐
                                   │  MAVROS (ROS 2)         │
                                   │  /mavros/state          │
                                   │  /mavros/local_position │
                                   │  /mavros/setpoint_*     │
                                   └──────────┬─────────────┘
                                              │
                                   ┌──────────▼─────────────┐
                                   │  R54 HITL script        │
                                   │  (R54 actor → setpts)   │
                                   └────────────────────────┘
```

---

## Folder structure

```
Hardware in the Loop/
├── README.md                    ← this file
├── config/
│   ├── mavros_hitl.yaml         ← MAVROS params (FCU URL, plugins)
│   └── px4_hitl_params.txt      ← PX4 parameters to set via QGroundControl
├── launch/
│   └── hitl.launch.py           ← Starts Gazebo + MAVROS (no PX4 SITL)
├── scripts/
│   ├── fcu_connect_test.py      ← Step 1: verify FC is connected and responsive
│   ├── preflight_check.py       ← Step 2: all safety checks before episodes
│   ├── run_hitl_r54.py          ← Step 3: run N test episodes (default 10)
│   └── emergency_stop.py        ← Any time: immediately disarm FC
└── results/                     ← Created automatically when tests run
    ├── hitl_r54_attempt*.csv
    └── hitl_r54_*ep_summary_*.json
```

---

## Hardware requirements

| Item | Requirement |
|------|-------------|
| Flight controller | Pixhawk 4, Pixhawk 6C, or compatible PX4 board |
| Connection | USB cable (ttyACM0) or UART (ttyUSB0 at 921600 baud) |
| Firmware | PX4 v1.13+ with HITL support |
| Computer | Ubuntu 20.04/22.04 with ROS2 Humble, MAVROS, Gazebo 11 |

---

## Setup (one-time)

### 1. Set PX4 HITL parameters via QGroundControl

Open QGroundControl → Vehicle Setup → Parameters → search for each:

```
HITL_ENABLE     = 1      ← enables HITL mode (REBOOT REQUIRED after this)
SIM_GPS_USED    = 1
COM_RCL_EXCEPT  = 4
CBRK_IO_SAFETY  = 22027
CBRK_USB_CHK    = 197848
MPC_XY_CRUISE   = 4.5
GF_ACTION       = 0
```

Full parameter list: `config/px4_hitl_params.txt`

**Reboot the FC after setting HITL_ENABLE=1.**

### 2. Confirm serial port

```bash
ls /dev/ttyACM* /dev/ttyUSB*      # find your port
sudo chmod a+rw /dev/ttyACM0      # grant permission (or add user to dialout group)
```

### 3. Edit MAVROS config if needed

Edit `config/mavros_hitl.yaml`:
```yaml
fcu_url: "serial:///dev/ttyACM0:57600"   # change port/baud to match your FC
```

---

## Running HITL test episodes

Open **4 terminals**, all sourced:
```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
```

**Terminal 1 — Launch world + MAVROS:**
```bash
ros2 launch "Hardware in the Loop/launch/hitl.launch.py" \
    world:=tunnel_100m_realworld.world \
    fcu_port:=/dev/ttyACM0 \
    fcu_baud:=57600 \
    gui:=true
```

**Terminal 2 — Verify FC connection:**
```bash
python3 "Hardware in the Loop/scripts/fcu_connect_test.py"
```
All checks must show `[OK]` before proceeding.

**Terminal 3 — Pre-flight check:**
```bash
python3 "Hardware in the Loop/scripts/preflight_check.py"
```
All checks must pass.

**Terminal 4 — Run episodes (keep emergency_stop.py ready):**
```bash
# Keep this in a 5th terminal, ready to run instantly:
# python3 "Hardware in the Loop/scripts/emergency_stop.py"

python3 "Hardware in the Loop/scripts/run_hitl_r54.py" --episodes 10 --max-speed 4.5
```

---

## Results

Results are saved to `Hardware in the Loop/results/`:
- `hitl_r54_attempt{N}_{ts}.csv` — step-by-step log per episode
- `hitl_r54_{N}ep_summary_{ts}.json` — aggregate stats (SR, avg speed, collisions)

The JSON includes an `episodes` list with per-attempt outcome, duration, speed, and
collision count — directly comparable to the Gazebo-only real-world test results
in `Final Results/testing/test_r54_realworld_10ep_summary_20260905_111452.json`.

---

## Key difference vs Gazebo-only test

| Aspect | Gazebo-only (SITL) | HITL |
|---|---|---|
| EKF | Simulated (Gazebo plugin) | Real PX4 EKF2 on FC |
| Motor mixing | Simulated | Real firmware |
| Stabilisation | Simulated | Real PX4 inner loop |
| Arming | Auto via topic | MAVROS service → FC |
| Position data | Gazebo ground truth | FC EKF estimate via MAVROS |
| Depth camera | Gazebo plugin | Gazebo plugin (unchanged) |
| Policy | R54 actor (unchanged) | R54 actor (unchanged) |

HITL is the closest test to real-world deployment without leaving the lab.

---

## Emergency procedure

If the drone behaves dangerously at any point:

```bash
python3 "Hardware in the Loop/scripts/emergency_stop.py"
```

This sends a DISARM command and switches the FC to STABILIZED mode via MAVROS.
In true emergency, manually disconnect the FC USB — this kills MAVROS heartbeat
and the FC will failsafe within `COM_OF_LOSS_T` seconds.
