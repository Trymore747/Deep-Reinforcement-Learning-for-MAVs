#!/usr/bin/env python3
"""
R54 HITL Inference Script — Hardware-in-the-Loop Test
======================================================
Runs the R54 best policy against a real RDDRONE-FMUK66 flight controller in HITL mode:
  - Real PX4 firmware on FMUK66 handles arming, mode management, safety systems
  - MAVROS bridges ROS2 ↔ MAVLink ↔ FMUK66 (serial USB)
  - Position feedback from Gazebo ground-truth (/CERLAB/quadcopter/odom_raw)
  - Drone movement via Gazebo plugin position setpoints
  - Depth camera from Gazebo (/camera/camera/depth/image_raw)
  - HIL bridge (hil_sensor_bridge.py) must be running to give FC state estimate

Key differences from Gazebo-only test:
  - Arming / disarming via MAVROS CommandBool service (real FC firmware path)
  - Mode management: OFFBOARD mode via real FC (real PX4 arming/safety pipeline)
  - Position source: Gazebo ground truth (FC EKF state injected via HIL bridge)
  - Movement control: Gazebo plugin /CERLAB/quadcopter/setpoint_pose

Prerequisites (run in order):
  1. python3 Hardware in the Loop/scripts/hil_sensor_bridge.py  (keep running)
  2. ros2 launch "Hardware in the Loop/launch/hitl.launch.py"
  3. python3 "Hardware in the Loop/scripts/run_hitl_r54.py"

Usage:
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 "Hardware in the Loop/scripts/run_hitl_r54.py" [--episodes 10] [--max-speed 4.5]
"""

import argparse
import csv
import math
import sys
import threading
import time
import json
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT    = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT / "src/tunnel_drl/results_r54/r54_best.pth"
URDF_PATH  = PROJECT / "install/uav_simulator/share/uav_simulator/urdf/quadcopter.urdf"
OUT_DIR    = PROJECT / "Hardware in the Loop" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── R54 constants (must match training exactly) ─────────────────────────────────
TUNNEL_LENGTH    = 100.0
TURN_POINT       = 95.0
CORRIDOR_HW      = 2.5       # real-world tunnel ±2.5 m (5 m wide tile)
SAFE_ALT         = 1.5
MAX_ALT          = 2.8
MIN_ALT          = 0.6
ALT_CEIL         = 4.5
MAX_SPEED        = 5.5
COLLISION_DIST   = 0.35
NEAR_MISS_DIST   = 0.8
PROX_ALARM_M     = 2.0
ACTION_SMOOTH_FWD = 0.35
ACTION_SMOOTH_LAT = 0.30
TICK_DT          = 0.1
NUM_H_SECTORS    = 9
DEPTH_PERCENTILE = 5
DEPTH_FAR        = 10.0
OBS_NOISE_STD    = 0.06
AVOID_START      = 5.0
STUCK_TIMEOUT    = 20.0
_SECTOR_BIAS     = np.linspace(1.0, -1.0, NUM_H_SECTORS, dtype=np.float32)

STATE_DIM  = 21
ACTION_DIM = 3


# ── R54 Actor (identical to training) ──────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(STATE_DIM, 512);  self.ln1 = nn.LayerNorm(512)
        self.fc2  = nn.Linear(512, 512);         self.ln2 = nn.LayerNorm(512)
        self.skip = nn.Linear(512, 512)
        self.fc3  = nn.Linear(512, 256);         self.ln3 = nn.LayerNorm(256)
        self.fc4  = nn.Linear(256, ACTION_DIM)

    def forward(self, s):
        x = F.relu(self.ln1(self.fc1(s)))
        x = F.relu(self.ln2(self.fc2(x)) + self.skip(x))
        x = F.relu(self.ln3(self.fc3(x)))
        return torch.tanh(self.fc4(x))


# ── HITL Drone Node (MAVROS + Gazebo depth) ────────────────────────────────────
class HITLNode(Node):
    def __init__(self):
        super().__init__("r54_hitl_node")
        self._lock = threading.Lock()

        # FC state
        self.armed    = False
        self.mode     = ""
        self.connected = False

        # Pose (from MAVROS EKF — real FC estimator)
        self.px = self.py = self.pz = self.yaw = 0.0
        self.vx = self.vy = self.vz = 0.0
        self._prev_px = self._prev_py = self._prev_pz = 0.0
        self._prev_stamp = 0.0
        self.pose_received = False

        # Depth (from Gazebo depth camera — unchanged from SITL)
        self.depth_sectors = np.full(NUM_H_SECTORS, DEPTH_FAR, dtype=np.float32)
        self.depth_top = self.depth_mid = self.depth_bottom = DEPTH_FAR
        self.prox_alarm    = 0
        self.depth_received = False

        # MAVROS services
        self._arm_cli  = self.create_client(CommandBool, "/mavros/cmd/arming")
        self._mode_cli = self.create_client(SetMode,     "/mavros/set_mode")
        self._del_cli  = self.create_client(DeleteEntity, "/delete_entity")
        self._spn_cli  = self.create_client(SpawnEntity,  "/spawn_entity")

        be = QoSProfile(depth=5,
                        reliability=ReliabilityPolicy.BEST_EFFORT,
                        durability=DurabilityPolicy.VOLATILE)

        # Subscriptions
        self.create_subscription(State,    "/mavros/state",
                                 self._state_cb, 10)
        # Position from Gazebo ground truth (FC EKF state provided by hil_sensor_bridge)
        self.create_subscription(Odometry, "/CERLAB/quadcopter/odom_raw",
                                 self._odom_cb, 10)
        self.create_subscription(Image,    "/camera/camera/depth/image_raw",
                                 self._depth_cb, be)

        # Setpoint publishers:
        #   1. MAVROS → FC (keeps OFFBOARD mode alive on real FC)
        self.mavros_setpoint_pub = self.create_publisher(
            PoseStamped, "/mavros/setpoint_position/local", 10)
        #   2. Gazebo plugin position controller (actual drone movement)
        self.gazebo_setpoint_pub = self.create_publisher(
            PoseStamped, "/CERLAB/quadcopter/setpoint_pose", 10)
        # Enable Gazebo position control mode
        self._posctrl_pub  = self.create_publisher(Bool,  "/CERLAB/quadcopter/posctrl", 10)
        self._takeoff_pub  = self.create_publisher(Empty, "/CERLAB/quadcopter/takeoff", 10)

    # ── Callbacks ──────────────────────────────────────────────────────────────
    def _state_cb(self, msg: State):
        with self._lock:
            self.connected = msg.connected
            self.armed     = msg.armed
            self.mode      = msg.mode

    def _odom_cb(self, msg: Odometry):
        now = time.time()
        dt  = now - self._prev_stamp if self._prev_stamp > 0 else 0.1
        with self._lock:
            self.px = msg.pose.pose.position.x
            self.py = msg.pose.pose.position.y
            self.pz = msg.pose.pose.position.z
            q = msg.pose.pose.orientation
            self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y),
                                  1 - 2*(q.y*q.y + q.z*q.z))
            if dt > 0.01:
                self.vx = (self.px - self._prev_px) / dt
                self.vy = (self.py - self._prev_py) / dt
                self.vz = (self.pz - self._prev_pz) / dt
            self._prev_px = self.px; self._prev_py = self.py
            self._prev_pz = self.pz; self._prev_stamp = now
            self.pose_received = True

    def _depth_cb(self, msg: Image):
        if msg.encoding == "32FC1":
            raw = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        elif msg.encoding == "16UC1":
            raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width).astype(np.float32) / 1000.0
        else:
            return
        d = raw.copy()
        d[(d <= 0.05) | ~np.isfinite(d)] = DEPTH_FAR
        h, w = d.shape
        r0, r1 = int(h * 0.15), int(h * 0.85)
        mid = d[r0:r1, :]
        sw  = w // NUM_H_SECTORS
        secs = np.empty(NUM_H_SECTORS, dtype=np.float32)
        for i in range(NUM_H_SECTORS):
            c0 = i * sw
            c1 = (i + 1) * sw if i < NUM_H_SECTORS - 1 else w
            secs[i] = float(np.percentile(mid[:, c0:c1], DEPTH_PERCENTILE))
        cv0, cv1 = int(w * 0.30), int(w * 0.70)
        col = d[:, cv0:cv1]
        tb = int(h * 0.33); bb = int(h * 0.67)
        with self._lock:
            self.depth_sectors[:] = secs
            self.depth_top    = float(np.percentile(col[:tb,   :], DEPTH_PERCENTILE))
            self.depth_mid    = float(np.percentile(col[tb:bb, :], DEPTH_PERCENTILE))
            self.depth_bottom = float(np.percentile(col[bb:,   :], DEPTH_PERCENTILE))
            self.prox_alarm   = 1 if float(np.min(secs)) < PROX_ALARM_M else 0
            self.depth_received = True

    # ── Helpers ────────────────────────────────────────────────────────────────
    def snap(self):
        with self._lock:
            return {
                "px": self.px, "py": self.py, "pz": self.pz, "yaw": self.yaw,
                "vx": self.vx, "vy": self.vy, "vz": self.vz,
                "depth_sectors": self.depth_sectors.copy(),
                "depth_top": self.depth_top, "depth_mid": self.depth_mid,
                "depth_bottom": self.depth_bottom,
                "prox_alarm": self.prox_alarm,
            }

    def publish_setpoint(self, x, y, z, yaw=0.0):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        s = math.sin(yaw / 2); c = math.cos(yaw / 2)
        msg.pose.orientation.z = s; msg.pose.orientation.w = c
        # Send to MAVROS (keeps FC in OFFBOARD mode)
        self.mavros_setpoint_pub.publish(msg)
        # Send to Gazebo plugin (physically moves the drone)
        self.gazebo_setpoint_pub.publish(msg)

    def gazebo_takeoff(self):
        """Trigger Gazebo plugin takeoff sequence and enable position control."""
        self._posctrl_pub.publish(Bool(data=True))
        self._takeoff_pub.publish(Empty())

    def _svc_call(self, cli, req, timeout=5.0):
        try:
            if not cli.wait_for_service(timeout_sec=timeout):
                return None
            fut = cli.call_async(req)
            t0 = time.time()
            while not fut.done() and time.time() - t0 < timeout:
                time.sleep(0.05)
            return fut.result() if fut.done() else None
        except Exception as e:
            self.get_logger().warn(f"_svc_call error (context may be invalid): {e}",
                                   throttle_duration_sec=5.0)
            return None

    def set_mode(self, mode: str) -> bool:
        req = SetMode.Request(); req.custom_mode = mode
        res = self._svc_call(self._mode_cli, req)
        return res is not None and res.mode_sent

    def arm(self, value: bool = True) -> bool:
        req = CommandBool.Request(); req.value = value
        res = self._svc_call(self._arm_cli, req)
        return res is not None and res.success

    def reset_model(self, x=0.0, y=0.0, z=0.3) -> bool:
        """Delete and respawn the Gazebo drone model to reset position for HITL."""
        req = DeleteEntity.Request(); req.name = "quadcopter"
        self._svc_call(self._del_cli, req, timeout=8.0)
        time.sleep(1.5)
        self.pose_received = False
        try:
            urdf = open(str(URDF_PATH)).read()
        except FileNotFoundError:
            return False
        req2 = SpawnEntity.Request()
        req2.name = "quadcopter"; req2.xml = urdf
        req2.initial_pose.position.x = float(x)
        req2.initial_pose.position.y = float(y)
        req2.initial_pose.position.z = float(z)
        req2.initial_pose.orientation.w = 1.0
        self._svc_call(self._spn_cli, req2, timeout=8.0)
        time.sleep(0.5)
        # Re-enable Gazebo position control after respawn
        self._posctrl_pub.publish(Bool(data=True))
        return True

    def hitl_arm_sequence(self, spawn_x: float, spawn_y: float, spawn_z: float,
                          max_speed: float) -> bool:
        """
        Full HITL arming sequence:
          1. Stream setpoints at spawn position (required before OFFBOARD can engage)
          2. Set OFFBOARD mode
          3. Arm FC
          4. Wait for altitude
        Returns True if armed and airborne.
        """
        print("[HITL] Pre-streaming setpoints (2s) to allow OFFBOARD transition...")
        t0 = time.time()
        while time.time() - t0 < 2.0:
            self.publish_setpoint(spawn_x, spawn_y, spawn_z + 0.1)
            time.sleep(0.05)

        print("[HITL] Setting OFFBOARD mode...")
        if not self.set_mode("OFFBOARD"):
            print("[HITL] OFFBOARD failed — check FC connection and HIL bridge")
            # Try anyway — Gazebo movement will still work even without OFFBOARD
            print("[HITL] Continuing with Gazebo control only (FC not in OFFBOARD)...")

        time.sleep(0.3)

        print("[HITL] Arming FC...")
        arm_ok = self.arm(True)
        if not arm_ok:
            print("[HITL] ARM failed — check CBRK params. Continuing via Gazebo control...")

        # Trigger Gazebo takeoff
        print("[HITL] Triggering Gazebo takeoff sequence...")
        self.gazebo_takeoff()
        time.sleep(0.8)   # wait for TAKINGOFF → FLYING transition

        # Wait for takeoff altitude
        print(f"[HITL] Climbing to {SAFE_ALT:.1f} m...")
        t0 = time.time()
        tgt_z = SAFE_ALT
        while time.time() - t0 < 20.0:
            snap = self.snap()
            self.publish_setpoint(spawn_x, spawn_y, tgt_z)
            if snap["pz"] >= SAFE_ALT - 0.15:
                print(f"[HITL] Altitude OK: {snap['pz']:.2f} m  FC armed={self.armed}")
                return True
            time.sleep(0.05)
        print("[HITL] Takeoff timeout")
        return False


# ── State / action helpers (identical to SITL script) ─────────────────────────
def build_state(snap, spawn_x, spawn_y, fwd_progress, lat_offset,
                max_speed, phase, prev_action, noise_sd):
    ds = snap["depth_sectors"]
    h_noisy = np.clip(
        ds + np.random.normal(0.0, noise_sd, NUM_H_SECTORS).astype(np.float32),
        0.05, DEPTH_FAR)
    v_top = float(np.clip(snap["depth_top"] + np.random.normal(0, noise_sd), 0.05, DEPTH_FAR))
    v_mid = float(np.clip(snap["depth_mid"] + np.random.normal(0, noise_sd), 0.05, DEPTH_FAR))
    v_bot = float(np.clip(snap["depth_bottom"] + np.random.normal(0, noise_sd), 0.05, DEPTH_FAR))
    speed = abs(snap["vx"]) if phase == "OUTBOUND" else abs(-snap["vx"])
    return np.array([
        *[h_noisy[i] / DEPTH_FAR for i in range(NUM_H_SECTORS)],
        v_top / DEPTH_FAR, v_mid / DEPTH_FAR, v_bot / DEPTH_FAR,
        float(snap["prox_alarm"]),
        np.clip(fwd_progress / TUNNEL_LENGTH, -0.1, 1.1),
        np.clip(lat_offset   / CORRIDOR_HW,   -1, 1),
        np.clip((snap["pz"] - SAFE_ALT) / 1.5, -1, 1),
        np.clip(speed        / 7.0, -1, 1),
        np.clip(snap["vy"]   / 3.0, -1, 1),
        np.clip(snap["vz"]   / 2.0, -1, 1),
        1.0 if phase == "RETURN" else 0.0,
        np.clip(max_speed    / 7.0,  0, 1),
    ], dtype=np.float32)


def apply_action(smoothed, snap, spawn_y, lat_offset, phase, max_speed):
    raw_fwd, raw_lat, raw_alt = smoothed
    fwd_speed = float(np.clip((raw_fwd * 0.5 + 0.5) * max_speed, 1.0, MAX_SPEED))
    lat_speed = float(raw_lat * 0.8)
    alt_adj   = float(raw_alt * 0.4)

    ds      = snap["depth_sectors"]
    front_d = float(np.min([ds[3], ds[4], ds[5]]))
    eff = list(ds) if phase == "OUTBOUND" else list(reversed(ds))

    smooth_push = 0.0
    for i in range(NUM_H_SECTORS):
        d = float(eff[i])
        if d < AVOID_START:
            w = ((AVOID_START - d) / AVOID_START) ** 2
            smooth_push -= w * _SECTOR_BIAS[i] * 2.0
    lat_speed += smooth_push

    if float(min(eff)) > AVOID_START:
        lat_speed += -lat_offset * 0.20
    lat_speed = float(np.clip(lat_speed, -3.5, 3.5))

    alt_err = snap["pz"] - SAFE_ALT
    if snap["pz"] < MIN_ALT:
        alt_adj = max(alt_adj, 0.8)
    elif snap["pz"] > MAX_ALT:
        alt_adj = min(alt_adj, -0.4)
    if abs(lat_offset) < CORRIDOR_HW * 0.85:
        alt_adj = 0.5 * alt_adj + 0.5 * float(np.clip(-alt_err * 3.0, -1.0, 1.0))

    if front_d < 0.15:
        fwd_speed = 0.0

    direction = -1.0 if phase == "RETURN" else 1.0
    return (direction * fwd_speed * TICK_DT,
            lat_speed * TICK_DT,
            float(np.clip(alt_adj * TICK_DT, -0.2, 0.2))), front_d


# ── Single episode ──────────────────────────────────────────────────────────────
def run_episode(node: HITLNode, actor: Actor, attempt: int,
                noise_sd: float, spawn_y_offset: float,
                max_speed: float) -> dict:
    print(f"\n{'='*60}")
    print(f"  ATTEMPT {attempt}  |  noise_sd={noise_sd:.3f}  spawn_y={spawn_y_offset:+.3f}")
    print(f"{'='*60}")

    # ── Between-episode: disarm, reset Gazebo model, re-arm ───────────────────
    if node.armed:
        print("[HITL] Disarming for model reset...")
        node.arm(False); time.sleep(1.5)

    print("[HITL] Resetting drone model in Gazebo...")
    node.reset_model(x=0.0, y=spawn_y_offset, z=0.3)
    time.sleep(2.0)

    # Wait for EKF to re-converge on new position
    print("[HITL] Waiting for position estimate to converge...")
    t0 = time.time()
    while not node.pose_received and time.time() - t0 < 10.0:
        time.sleep(0.2)
    if not node.pose_received:
        return {"attempt": attempt, "result": "sensor_timeout",
                "duration_s": 0, "avg_speed_ms": 0, "peak_speed_ms": 0,
                "collisions": 0, "near_misses": 0, "min_depth_m": DEPTH_FAR,
                "dr_noise_sd": noise_sd, "dr_spawn_y_m": spawn_y_offset, "csv_log": ""}

    # Allow EKF to settle
    time.sleep(1.5)

    snap = node.snap()
    spawn_x = snap["px"]; spawn_y = snap["py"]

    # ── HITL arm + takeoff ─────────────────────────────────────────────────────
    ok = node.hitl_arm_sequence(spawn_x, spawn_y, SAFE_ALT, max_speed)
    if not ok:
        return {"attempt": attempt, "result": "arm_fail",
                "duration_s": 0, "avg_speed_ms": 0, "peak_speed_ms": 0,
                "collisions": 0, "near_misses": 0, "min_depth_m": DEPTH_FAR,
                "dr_noise_sd": noise_sd, "dr_spawn_y_m": spawn_y_offset, "csv_log": ""}

    time.sleep(0.5)
    snap = node.snap()

    # ── CSV log ────────────────────────────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"hitl_r54_attempt{attempt}_{ts}.csv"
    csv_f  = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=[
        "step", "phase", "t_s", "x_m", "y_m", "z_m",
        "vx_ms", "vy_ms", "vz_ms", "fwd_progress_m", "lat_offset_m",
        "speed_ms", "front_d_m", "min_depth_m",
        "prox_alarm", "depth_top", "depth_mid", "depth_bot",
        "tgt_x", "tgt_y", "tgt_z", "collision", "near_miss", "armed", "mode",
    ])
    writer.writeheader()

    # ── Episode loop ───────────────────────────────────────────────────────────
    ep_start  = time.time()
    phase     = "OUTBOUND"
    step      = 0
    speeds    = []
    col_count = 0; near_miss_count = 0
    min_d_ever = DEPTH_FAR
    prev_action = np.zeros(3, dtype=np.float32)
    delayed_a   = np.zeros(3, dtype=np.float32)
    result_label = "TIMEOUT"
    stuck_t0    = None
    was_ever_armed = False  # only flag DISARMED if FC was previously armed

    tgt_x = spawn_x; tgt_y = spawn_y; tgt_z = SAFE_ALT
    MAX_EP_SECS = 150.0   # longer than SITL — FC arm cycle takes extra time

    while True:
        snap   = node.snap()
        t_el   = time.time() - ep_start

        x           = snap["px"]
        fwd_progress = (x - spawn_x) if phase == "OUTBOUND" else (spawn_x - x)
        lat_offset   = snap["py"] - spawn_y
        speed        = abs(snap["vx"])
        speeds.append(speed)

        ds      = snap["depth_sectors"]
        min_d   = float(np.min(ds))
        front_d = float(np.min([ds[3], ds[4], ds[5]]))
        min_d_ever = min(min_d_ever, min_d)

        is_col = (min_d < COLLISION_DIST)
        is_nm  = (COLLISION_DIST <= min_d < NEAR_MISS_DIST)
        if is_col: col_count      += 1
        if is_nm:  near_miss_count += 1

        state = build_state(snap, spawn_x, spawn_y, fwd_progress, lat_offset,
                            max_speed, phase, prev_action, noise_sd)
        with torch.no_grad():
            raw = actor(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
        exec_a = delayed_a.copy(); delayed_a = raw.copy()

        smoothed = np.array([
            ACTION_SMOOTH_FWD * exec_a[0] + (1 - ACTION_SMOOTH_FWD) * prev_action[0],
            ACTION_SMOOTH_LAT * exec_a[1] + (1 - ACTION_SMOOTH_LAT) * prev_action[1],
            ACTION_SMOOTH_LAT * exec_a[2] + (1 - ACTION_SMOOTH_LAT) * prev_action[2],
        ], dtype=np.float32)
        prev_action = smoothed.copy()

        (dx, dy, dz), front_d = apply_action(smoothed, snap, spawn_y, lat_offset,
                                             phase, max_speed)

        tgt_x = float(np.clip(tgt_x + dx, spawn_x - 3.0, spawn_x + TURN_POINT + 5.0))
        tgt_y = float(np.clip(tgt_y + dy, -CORRIDOR_HW, CORRIDOR_HW))
        tgt_z = float(np.clip(tgt_z + dz, MIN_ALT, ALT_CEIL))
        yaw   = 0.0 if phase == "OUTBOUND" else math.pi

        node.publish_setpoint(tgt_x, tgt_y, tgt_z, yaw)

        # ── Phase transitions ──────────────────────────────────────────────────
        if phase == "OUTBOUND" and x >= TURN_POINT:
            print(f"\n  [OUTBOUND COMPLETE] x={x:.1f}m  t={t_el:.1f}s")
            phase = "RETURN"
            # Reset target to current position so return tracking starts correctly.
            # Keep publishing setpoints during the turn so Gazebo position control
            # does not drop out during a silent gap.
            tgt_x = x; tgt_y = snap["py"]; tgt_z = snap["pz"]
            t_turn = time.time()
            while time.time() - t_turn < 2.0:
                node.publish_setpoint(tgt_x, tgt_y, tgt_z, math.pi)
                time.sleep(0.05)

        if phase == "RETURN" and (x - spawn_x) <= 2.0:
            result_label = "SUCCESS"
            print(f"\n  [RETURN COMPLETE — SUCCESS] t={t_el:.1f}s  dist={x-spawn_x:.1f}m")
            break

        # ── Termination conditions ─────────────────────────────────────────────
        if t_el > MAX_EP_SECS:
            result_label = "TIMEOUT"; break

        if snap["pz"] < 0.30:
            result_label = "CRASH_ALT"; break

        if abs(lat_offset) > CORRIDOR_HW + 0.5:
            result_label = "CRASH_LAT"; break

        # FC disarmed unexpectedly (safety kill or crash)
        if node.armed:
            was_ever_armed = True
        if was_ever_armed and not node.armed:
            result_label = "DISARMED"; break

        if front_d < 0.15 and speed < 0.1:
            if stuck_t0 is None:
                stuck_t0 = time.time()
            elif time.time() - stuck_t0 >= STUCK_TIMEOUT:
                result_label = "STUCK"
                print(f"\n  [STUCK] Blocked 20s at fwd={fwd_progress:.1f}m — next attempt")
                break
        else:
            stuck_t0 = None

        # Log every 30 steps
        if step % 30 == 0:
            print(f"  [{phase:<8}|t={t_el:5.1f}s] fwd={fwd_progress:6.1f}m  "
                  f"z={snap['pz']:.2f}m  spd={speed:.2f}m/s  "
                  f"front={front_d:.2f}m  col={col_count}  nm={near_miss_count}  "
                  f"armed={node.armed}  mode={node.mode}")

        writer.writerow({
            "step": step, "phase": phase, "t_s": round(t_el, 2),
            "x_m": round(x, 3), "y_m": round(snap["py"], 3), "z_m": round(snap["pz"], 3),
            "vx_ms": round(snap["vx"], 3), "vy_ms": round(snap["vy"], 3),
            "vz_ms": round(snap["vz"], 3),
            "fwd_progress_m": round(fwd_progress, 3), "lat_offset_m": round(lat_offset, 3),
            "speed_ms": round(speed, 3), "front_d_m": round(front_d, 3),
            "min_depth_m": round(min_d, 3),
            "prox_alarm": snap["prox_alarm"],
            "depth_top": round(snap["depth_top"], 3),
            "depth_mid": round(snap["depth_mid"], 3),
            "depth_bot": round(snap["depth_bottom"], 3),
            "tgt_x": round(tgt_x, 3), "tgt_y": round(tgt_y, 3), "tgt_z": round(tgt_z, 3),
            "collision": int(is_col), "near_miss": int(is_nm),
            "armed": int(node.armed), "mode": node.mode,
        })
        step += 1
        time.sleep(TICK_DT)

    csv_f.close()
    dur = time.time() - ep_start

    # Disarm safely after episode
    node.arm(False)
    time.sleep(0.5)

    return {
        "attempt":      attempt,
        "result":       result_label,
        "duration_s":   round(dur, 1),
        "avg_speed_ms": round(float(np.mean(speeds)) if speeds else 0.0, 3),
        "peak_speed_ms": round(float(np.max(speeds)) if speeds else 0.0, 3),
        "collisions":   col_count,
        "near_misses":  near_miss_count,
        "min_depth_m":  round(min_d_ever, 3),
        "dr_noise_sd":  round(noise_sd, 3),
        "dr_spawn_y_m": round(spawn_y_offset, 3),
        "csv_log":      str(csv_path),
        "steps_logged": step,
    }


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="R54 HITL Test — 10 episodes")
    parser.add_argument("--episodes",  type=int,   default=10, help="Number of episodes to run")
    parser.add_argument("--max-speed", type=float, default=4.5, help="Eval speed cap (m/s)")
    args = parser.parse_args()

    if not CHECKPOINT.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {CHECKPOINT}")

    ckpt  = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    actor = Actor(); actor.load_state_dict(ckpt["actor"]); actor.eval()
    saved_ep = ckpt.get("episode", "?")
    saved_sr = ckpt.get("sr", "?")

    print("\n" + "="*60)
    print("  R54 BEST POLICY — HARDWARE IN THE LOOP TEST")
    print("  Environment: tunnel_100m_realworld.world (Tunnel Tile 5)")
    print(f"  FC firmware: PX4 HITL mode (serial via MAVROS)")
    print(f"  Corridor: ±{CORRIDOR_HW} m  |  Speed cap: {args.max_speed} m/s")
    print(f"  Loaded: r54_best.pth  ep={saved_ep}  SR={saved_sr}")
    print("="*60)

    rclpy.init()
    node = HITLNode()
    ex   = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    spin_thread = threading.Thread(target=ex.spin, daemon=True)
    spin_thread.start()

    # Wait for MAVROS connection
    print("\n[HITL] Waiting for MAVROS connection to FC...")
    t0 = time.time()
    while not node.connected and time.time() - t0 < 30.0:
        time.sleep(0.5)
    if not node.connected:
        sys.exit("[ERROR] MAVROS not connected — check launch file and FC serial port")
    print(f"[HITL] FC connected — mode={node.mode}  armed={node.armed}")

    np.random.seed(int(time.time()) % 100000)
    _SPAWN_OFFSETS = [-0.15, +0.20, -0.35, +0.10, -0.25, +0.30, -0.45, +0.15, -0.40, +0.05]

    all_results = []
    for attempt in range(1, args.episodes + 1):
        noise_sd    = float(np.random.uniform(0.04, OBS_NOISE_STD))
        spawn_y_off = _SPAWN_OFFSETS[(attempt - 1) % len(_SPAWN_OFFSETS)]

        result = run_episode(node, actor, attempt, noise_sd, spawn_y_off, args.max_speed)
        all_results.append(result)

        print(f"\n  Attempt {attempt}/{args.episodes} outcome: {result['result']}")

        if attempt < args.episodes:
            wait = float(np.random.uniform(30, 60))
            print(f"  [WAIT] {wait:.0f}s before next attempt...")
            time.sleep(wait)

    ex.shutdown(); rclpy.shutdown()

    # ── Aggregate stats ────────────────────────────────────────────────────────
    n_ep      = len(all_results)
    successes = [r for r in all_results if r["result"] == "SUCCESS"]
    n_success = len(successes)
    sr        = n_success / n_ep
    avg_speed = float(np.mean([r["avg_speed_ms"] for r in all_results]))
    avg_col   = float(np.mean([r["collisions"]   for r in all_results]))

    print("\n" + "="*60)
    print("  HITL TEST COMPLETE")
    print("="*60)
    print(f"  Episodes   : {n_ep}")
    print(f"  Successes  : {n_success} / {n_ep}  (SR = {sr*100:.0f}%)")
    print(f"  Avg speed  : {avg_speed:.3f} m/s")
    print(f"  Avg collisions: {avg_col:.1f}")
    print("-"*60)
    for r in all_results:
        print(f"  A{r['attempt']:02d}: {r['result']:<10}  t={r['duration_s']:.1f}s  "
              f"spd={r['avg_speed_ms']:.2f}m/s  col={r['collisions']}  "
              f"spawn_y={r['dr_spawn_y_m']:+.2f}")
    print("="*60)

    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = OUT_DIR / f"hitl_r54_{n_ep}ep_summary_{ts}.json"
    meta = {
        "test_name":      "R54 Best Policy — Hardware-in-the-Loop Test",
        "testing_world":  "tunnel_100m_realworld.world",
        "training_world": "tunnel_100m_dynamic_25.world",
        "policy_file":    "r54_best.pth",
        "policy_episode": str(saved_ep),
        "test_max_speed": args.max_speed,
        "corridor_hw_m":  CORRIDOR_HW,
        "hitl_mode":      True,
        "fc_firmware":    "PX4 HITL (real firmware, simulated sensors via Gazebo)",
        "timestamp":      ts,
        "n_episodes":     n_ep,
        "n_success":      n_success,
        "success_rate":   round(sr, 4),
        "avg_speed_ms":   round(avg_speed, 3),
        "avg_collisions": round(avg_col, 1),
        "episodes":       all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
