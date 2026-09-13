#!/usr/bin/env python3
"""
R54 Domain-Randomisation Test — Real-World Environment (Tunnel Tile 5)
=======================================================================
Loads the best R54 policy (r54_best.pth) and runs a single episode
in the DARPA SubT Tunnel Tile 5 environment (tunnel_100m_realworld.world).

Key differences from testing world:
  - Corridor width: 5 m (±2.5 m) vs 7 m training — NARROWER, more challenging
  - Geometry: SubT mine-tile mesh (chainlink, wood, rough walls, strip lights)
  - Static obstacles: 10 mine-realistic objects (debris, crates, barrels)
  - Dynamic obstacles: 10 walkers — SAME waypoints as testing world
  - CORRIDOR_HW updated to 2.5 m to match physical tunnel

Pre-requisite: Gazebo must already be running with the real-world tile world
  ros2 launch uav_simulator tunnel_drl.launch.py gui:=true \
      world:=.../tunnel_100m_realworld.world

Usage:
  cd /home/makhosazana/Project/CERLAB-UAV-Autonomy
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 "Final Results/testing/run_test_r54_realworld.py"
"""

import csv, math, os, sys, threading, time, json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
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
OUT_DIR    = Path(__file__).resolve().parent   # Final Results/testing/
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── R54 constants (must match training exactly) ─────────────────────────────────
TUNNEL_LENGTH   = 100.0
TURN_POINT      = 95.0
CORRIDOR_HW     = 2.5    # Real-world tunnel is 5 m wide (±2.5 m) vs 7 m training
SAFE_ALT        = 1.5
MAX_ALT         = 2.8
MIN_ALT         = 0.6
ALT_FLOOR       = 0.3
ALT_CEIL        = 4.5
MAX_SPEED       = 5.5
COLLISION_DIST  = 0.35
NEAR_MISS_DIST  = 0.8
PROX_ALARM_M    = 2.0
DANGER_DIST     = 2.5
ACTION_SMOOTH_FWD = 0.35
ACTION_SMOOTH_LAT = 0.30
TICK_DT         = 0.1
NUM_H_SECTORS   = 9
DEPTH_PERCENTILE = 5
DEPTH_FAR       = 10.0
OBS_NOISE_STD   = 0.06
SPAWN_Y_JITTER  = 0.20
_SECTOR_BIAS    = np.linspace(1.0, -1.0, NUM_H_SECTORS, dtype=np.float32)
AVOID_START     = 5.0
STATE_DIM       = 21
ACTION_DIM      = 3
TEST_MAX_SPEED  = 4.5     # m/s — conservative for unseen environment
MAX_ATTEMPTS    = 16      # retry until success
STUCK_TIMEOUT   = 20.0    # abort attempt if pressed against obstacle this long (prevents ROS crash at ~76s)

# ── R54 Actor (exactly matching training) ──────────────────────────────────────
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


# ── ROS2 Drone Node ─────────────────────────────────────────────────────────────
class DroneNode(Node):
    def __init__(self):
        super().__init__("r54_test_node")
        self._lock = threading.Lock()

        # Sensor state
        self.px = self.py = self.pz = self.yaw = 0.0
        self.vx = self.vy = self.vz = 0.0
        self._prev_px = self._prev_py = self._prev_pz = 0.0
        self._prev_stamp = 0.0
        self.pose_received = False

        self.depth_sectors  = np.full(NUM_H_SECTORS, DEPTH_FAR, dtype=np.float32)
        self.depth_top      = DEPTH_FAR
        self.depth_mid      = DEPTH_FAR
        self.depth_bottom   = DEPTH_FAR
        self.depth_received = False
        self.prox_alarm     = 0

        # Publishers
        self.setpoint_pub = self.create_publisher(PoseStamped,
            "/CERLAB/quadcopter/setpoint_pose", 10)
        self.takeoff_pub  = self.create_publisher(Empty,
            "/CERLAB/quadcopter/takeoff", 10)
        self.posctrl_pub  = self.create_publisher(Bool,
            "/CERLAB/quadcopter/posctrl", 10)
        self._del_cli = self.create_client(DeleteEntity, "/delete_entity")
        self._spn_cli = self.create_client(SpawnEntity,  "/spawn_entity")

        be = QoSProfile(depth=5,
                        reliability=ReliabilityPolicy.BEST_EFFORT,
                        durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(PoseStamped,
            "/CERLAB/quadcopter/pose_raw", self._pose_cb, 10)
        self.create_subscription(Image,
            "/camera/camera/depth/image_raw", self._depth_cb, be)

    def _pose_cb(self, msg):
        now = time.time()
        dt  = now - self._prev_stamp if self._prev_stamp > 0 else 0.1
        with self._lock:
            self.px  = msg.pose.position.x
            self.py  = msg.pose.position.y
            self.pz  = msg.pose.position.z
            q = msg.pose.orientation
            self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y),
                                  1 - 2*(q.y*q.y + q.z*q.z))
            if dt > 0.01:
                self.vx = (self.px - self._prev_px) / dt
                self.vy = (self.py - self._prev_py) / dt
                self.vz = (self.pz - self._prev_pz) / dt
            self._prev_px = self.px; self._prev_py = self.py
            self._prev_pz = self.pz; self._prev_stamp = now
            self.pose_received = True

    def _depth_cb(self, msg):
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

        # 9 horizontal sectors (5th percentile of middle row band)
        sw = w // NUM_H_SECTORS
        secs = np.empty(NUM_H_SECTORS, dtype=np.float32)
        for i in range(NUM_H_SECTORS):
            c0 = i * sw
            c1 = (i + 1) * sw if i < NUM_H_SECTORS - 1 else w
            secs[i] = float(np.percentile(mid[:, c0:c1], DEPTH_PERCENTILE))

        # 3 vertical zones (centre 40% width)
        cv0, cv1 = int(w * 0.30), int(w * 0.70)
        col = d[:, cv0:cv1]
        tb  = int(h * 0.33); bb = int(h * 0.67)

        with self._lock:
            self.depth_sectors[:] = secs
            self.depth_top    = float(np.percentile(col[:tb,   :], DEPTH_PERCENTILE))
            self.depth_mid    = float(np.percentile(col[tb:bb, :], DEPTH_PERCENTILE))
            self.depth_bottom = float(np.percentile(col[bb:,   :], DEPTH_PERCENTILE))
            self.prox_alarm   = 1 if float(np.min(secs)) < PROX_ALARM_M else 0
            self.depth_received = True

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
        msg.header.frame_id = "world"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        s = math.sin(yaw / 2); c = math.cos(yaw / 2)
        msg.pose.orientation.x = 0.0; msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = s;   msg.pose.orientation.w = c
        self.setpoint_pub.publish(msg)

    def takeoff(self):
        b = Bool(); b.data = True
        for _ in range(5):
            self.posctrl_pub.publish(b); time.sleep(0.05)
        self.takeoff_pub.publish(Empty())

    def teleport(self, x=0.0, y=0.0, z=0.2):
        helper = Node("_tp_helper")
        ex = rclpy.executors.SingleThreadedExecutor(); ex.add_node(helper)
        dc = helper.create_client(DeleteEntity, "/delete_entity")
        sc = helper.create_client(SpawnEntity,  "/spawn_entity")
        if dc.wait_for_service(timeout_sec=5.0):
            req = DeleteEntity.Request(); req.name = "quadcopter"
            fut = dc.call_async(req); t0 = time.time()
            while not fut.done() and time.time() - t0 < 8.0:
                ex.spin_once(timeout_sec=0.05)
        time.sleep(1.5); self.pose_received = False
        try:
            urdf = open(str(URDF_PATH)).read()
        except FileNotFoundError:
            helper.destroy_node(); return False
        if sc.wait_for_service(timeout_sec=5.0):
            req = SpawnEntity.Request(); req.name = "quadcopter"; req.xml = urdf
            req.initial_pose.position.x = float(x)
            req.initial_pose.position.y = float(y)
            req.initial_pose.position.z = float(z)
            req.initial_pose.orientation.w = 1.0
            fut = sc.call_async(req); t0 = time.time()
            while not fut.done() and time.time() - t0 < 8.0:
                ex.spin_once(timeout_sec=0.05)
        helper.destroy_node(); return True


# ── State builder (mirrors R54 get_state) ──────────────────────────────────────
def build_state(snap, spawn_x, spawn_y, fwd_progress, lat_offset,
                max_speed, phase, prev_action, noise_sd):
    ds = snap["depth_sectors"]    # 9 values — true readings
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


# ── Action → delta position (mirrors R54 apply_action) ─────────────────────────
def apply_action(smoothed, snap, spawn_y, lat_offset, phase, max_speed):
    raw_fwd, raw_lat, raw_alt = smoothed
    fwd_speed = float(np.clip((raw_fwd * 0.5 + 0.5) * max_speed, 1.0, MAX_SPEED))
    lat_speed = float(raw_lat * 0.8)
    alt_adj   = float(raw_alt * 0.4)

    ds = snap["depth_sectors"]
    front_d = float(np.min([ds[3], ds[4], ds[5]]))
    left_d  = float(np.min([ds[0], ds[1]]))
    right_d = float(np.min([ds[7], ds[8]]))
    if phase == "RETURN":
        left_d, right_d = right_d, left_d

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

    # Altitude correction
    alt_err = snap["pz"] - SAFE_ALT
    if snap["pz"] < MIN_ALT:
        alt_adj = max(alt_adj, 0.8)
    elif snap["pz"] > MAX_ALT:
        alt_adj = min(alt_adj, -0.4)
    if abs(lat_offset) < CORRIDOR_HW * 0.85:
        alt_adj = 0.5 * alt_adj + 0.5 * float(np.clip(-alt_err * 3.0, -1.0, 1.0))

    # Emergency forward stop
    if front_d < 0.15:
        fwd_speed = 0.0

    direction = -1.0 if phase == "RETURN" else 1.0
    return (direction * fwd_speed * TICK_DT,
            lat_speed * TICK_DT,
            float(np.clip(alt_adj * TICK_DT, -0.2, 0.2))), front_d


# ── Single episode ──────────────────────────────────────────────────────────────
def run_episode(node: DroneNode, actor: Actor, attempt: int, noise_sd: float,
                spawn_y_offset: float) -> dict:
    print(f"\n{'='*60}")
    print(f"  ATTEMPT {attempt}  |  noise_sd={noise_sd:.3f}  spawn_y={spawn_y_offset:+.3f}")
    print(f"{'='*60}")

    # Wait for sensors — if sensors absent after 10s, try to respawn the drone
    print("[TEST] Waiting for sensors...")
    t0 = time.time()
    _respawned = False
    while not (node.pose_received and node.depth_received):
        time.sleep(0.2)
        elapsed = time.time() - t0
        if not _respawned and elapsed > 10.0:
            print("[TEST] Sensors absent — attempting emergency respawn...")
            node.teleport(x=0.0, y=spawn_y_offset, z=0.5)
            time.sleep(2.0)
            _respawned = True
            t0 = time.time()  # reset timer after respawn
        elif _respawned and elapsed > 20.0:
            return {"result": "sensor_timeout"}

    # Teleport to start
    print(f"[TEST] Teleporting to start (y_offset={spawn_y_offset:+.3f})...")
    node.teleport(x=0.0, y=spawn_y_offset, z=0.2)
    time.sleep(2.5)

    t0 = time.time()
    while not node.pose_received and time.time() - t0 < 10.0:
        time.sleep(0.1)

    snap = node.snap()
    spawn_x = snap["px"]; spawn_y = snap["py"]
    tgt_x = spawn_x; tgt_y = spawn_y; tgt_z = SAFE_ALT
    prev_action = np.zeros(3, dtype=np.float32)

    # Takeoff
    print("[TEST] Taking off...")
    node.takeoff()
    t0 = time.time()
    while True:
        snap = node.snap()
        if snap["pz"] >= SAFE_ALT - 0.15:
            print(f"[TEST] Altitude OK: {snap['pz']:.2f} m"); break
        node.publish_setpoint(spawn_x, spawn_y, SAFE_ALT)
        time.sleep(0.1)
        if time.time() - t0 > 20.0:
            return {"result": "takeoff_timeout"}
    time.sleep(0.8)

    # CSV log
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"test_r54_realworld_attempt{attempt}_{ts}.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=[
        "step", "phase", "t_s", "x_m", "y_m", "z_m",
        "vx_ms", "vy_ms", "vz_ms", "fwd_progress_m", "lat_offset_m",
        "speed_ms", "front_d_m", "min_depth_m",
        "prox_alarm", "depth_top", "depth_mid", "depth_bot",
        "tgt_x", "tgt_y", "tgt_z", "collision", "near_miss",
    ])
    writer.writeheader()

    # Episode loop
    ep_start  = time.time()
    phase     = "OUTBOUND"
    step      = 0
    speeds    = []
    col_count = 0; near_miss_count = 0
    min_d_ever = DEPTH_FAR
    _return_start_step = 0
    delayed_a = np.zeros(3, dtype=np.float32)
    result_label = "TIMEOUT"
    stuck_t0  = None   # wall-time when stuck condition began

    while True:
        snap = node.snap()
        t_el = time.time() - ep_start

        # Phase-dependent progress
        ds      = snap["depth_sectors"]
        x       = snap["px"]
        fwd_progress = (x - spawn_x) if phase == "OUTBOUND" else (spawn_x - x)
        lat_offset   = snap["py"] - spawn_y
        speed        = abs(snap["vx"])
        speeds.append(speed)

        min_d = float(np.min(ds))
        front_d = float(np.min([ds[3], ds[4], ds[5]]))
        min_d_ever = min(min_d_ever, min_d)

        # Collision / near-miss detection
        is_col = (min_d < COLLISION_DIST)
        is_nm  = (COLLISION_DIST <= min_d < NEAR_MISS_DIST)
        if is_col:  col_count      += 1
        if is_nm:   near_miss_count += 1

        # Build state + infer (1-step action delay matching training)
        state = build_state(snap, spawn_x, spawn_y, fwd_progress, lat_offset,
                            TEST_MAX_SPEED, phase, prev_action, noise_sd)
        with torch.no_grad():
            raw = actor(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
        exec_a = delayed_a.copy(); delayed_a = raw.copy()

        # Smooth action
        smoothed = np.array([
            ACTION_SMOOTH_FWD * exec_a[0] + (1 - ACTION_SMOOTH_FWD) * prev_action[0],
            ACTION_SMOOTH_LAT * exec_a[1] + (1 - ACTION_SMOOTH_LAT) * prev_action[1],
            ACTION_SMOOTH_LAT * exec_a[2] + (1 - ACTION_SMOOTH_LAT) * prev_action[2],
        ], dtype=np.float32)
        prev_action = smoothed.copy()

        (dx, dy, dz), front_d = apply_action(smoothed, snap, spawn_y, lat_offset,
                                              phase, TEST_MAX_SPEED)

        tgt_x = float(np.clip(tgt_x + dx, spawn_x - 3.0, spawn_x + TURN_POINT + 5.0))
        tgt_y = float(np.clip(tgt_y + dy, -CORRIDOR_HW, CORRIDOR_HW))
        tgt_z = float(np.clip(tgt_z + dz, MIN_ALT, ALT_CEIL))
        yaw   = 0.0 if phase == "OUTBOUND" else math.pi

        node.publish_setpoint(tgt_x, tgt_y, tgt_z, yaw)

        # Log
        writer.writerow({
            "step": step, "phase": phase,
            "t_s": round(t_el, 3),
            "x_m": round(snap["px"], 3), "y_m": round(snap["py"], 3),
            "z_m": round(snap["pz"], 3),
            "vx_ms": round(snap["vx"], 3), "vy_ms": round(snap["vy"], 3),
            "vz_ms": round(snap["vz"], 3),
            "fwd_progress_m": round(fwd_progress, 2),
            "lat_offset_m":   round(lat_offset, 3),
            "speed_ms": round(speed, 3),
            "front_d_m": round(front_d, 3), "min_depth_m": round(min_d, 3),
            "prox_alarm": snap["prox_alarm"],
            "depth_top": round(snap["depth_top"], 3),
            "depth_mid": round(snap["depth_mid"], 3),
            "depth_bot": round(snap["depth_bottom"], 3),
            "tgt_x": round(tgt_x, 3), "tgt_y": round(tgt_y, 3), "tgt_z": round(tgt_z, 3),
            "collision": int(is_col), "near_miss": int(is_nm),
        })

        if step % 30 == 0:
            print(f"  [{phase:8s}|t={t_el:5.1f}s] "
                  f"fwd={fwd_progress:5.1f}m  z={snap['pz']:.2f}m  "
                  f"spd={speed:.2f}m/s  front={front_d:.2f}m  "
                  f"col={col_count}  nm={near_miss_count}")

        # Hold setpoint for tick duration
        t_tick = time.time()
        while time.time() - t_tick < TICK_DT:
            node.publish_setpoint(tgt_x, tgt_y, tgt_z, yaw)
            time.sleep(0.02)
        step += 1

        # Phase transitions
        if phase == "OUTBOUND" and (snap["px"] - spawn_x) >= TURN_POINT:
            phase = "RETURN"
            _return_start_step = step
            tgt_x = spawn_x + TURN_POINT   # don't overshoot
            print(f"\n  [OUTBOUND COMPLETE] x={snap['px'] - spawn_x:.1f}m  t={t_el:.1f}s")
            time.sleep(2.0)  # stabilise before turn — lets Person10 clear X=95m zone

        if phase == "RETURN" and (snap["px"] - spawn_x) <= 2.0:
            result_label = "SUCCESS"
            print(f"\n  [RETURN COMPLETE — SUCCESS] t={t_el:.1f}s  "
                  f"dist={snap['px'] - spawn_x:.1f}m"); break

        # Stuck detection — exit before ROS executor crash at ~76s
        if front_d < 0.15 and speed < 0.1:
            if stuck_t0 is None:
                stuck_t0 = time.time()
            elif time.time() - stuck_t0 > STUCK_TIMEOUT:
                result_label = "STUCK"
                print(f"\n  [STUCK] Blocked {STUCK_TIMEOUT:.0f}s at fwd={fwd_progress:.1f}m — next attempt"); break
        else:
            stuck_t0 = None

        # Terminal conditions
        if t_el >= 120.0:
            result_label = "TIMEOUT"
            print(f"\n  [TIMEOUT] fwd={fwd_progress:.1f}m"); break
        if snap["pz"] < ALT_FLOOR:
            result_label = "CRASH_ALT"
            print(f"\n  [CRASH — altitude] z={snap['pz']:.2f}m"); break
        if abs(snap["py"] - spawn_y) > CORRIDOR_HW + 0.5:
            result_label = "CRASH_LAT"
            print(f"\n  [CRASH — lateral] y_off={snap['py']-spawn_y:.2f}m"); break

    csv_f.close()
    duration = time.time() - ep_start
    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    peak_speed = float(np.max(speeds)) if speeds else 0.0

    return {
        "attempt":         attempt,
        "result":          result_label,
        "duration_s":      round(duration, 1),
        "avg_speed_ms":    round(avg_speed, 3),
        "peak_speed_ms":   round(peak_speed, 3),
        "collisions":      col_count,
        "near_misses":     near_miss_count,
        "min_depth_m":     round(min_d_ever, 3),
        "dr_noise_sd":     round(noise_sd, 3),
        "dr_spawn_y_m":    round(spawn_y_offset, 3),
        "csv_log":         str(csv_path),
        "steps_logged":    step,
    }


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  R54 BEST POLICY — REAL-WORLD ENVIRONMENT TEST")
    print("  Testing environment: tunnel_100m_realworld.world (Tunnel Tile 5)")
    print("  Corridor: 5 m wide (±2.5 m) — narrower than training (±3.5 m)")
    print("  Policy: r54_best.pth")
    print("="*60)

    # Load checkpoint
    if not CHECKPOINT.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {CHECKPOINT}")
    ckpt  = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    actor = Actor(); actor.load_state_dict(ckpt["actor"]); actor.eval()
    saved_ep  = ckpt.get("episode", "?")
    saved_sr  = ckpt.get("sr", "?")
    print(f"\n  Loaded: r54_best.pth  |  saved_episode={saved_ep}  SR={saved_sr}")
    print(f"  Architecture: 21D → 512 → 512 → 256 → 3  (LN + residual)")
    print(f"  Eval speed cap: {TEST_MAX_SPEED} m/s  (training max: {MAX_SPEED} m/s)")

    # ROS2 init
    rclpy.init()
    node = DroneNode()
    ex   = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    spin_thread = threading.Thread(target=ex.spin, daemon=True)
    spin_thread.start()

    np.random.seed(int(time.time()) % 100000)

    # Deterministic spawn offsets — 10 values covering both sides of narrow tunnel
    _SPAWN_OFFSETS = [-0.15, +0.20, -0.35, +0.10, -0.25, +0.30, -0.45, +0.15, -0.40, +0.05]

    N_EPISODES = 10   # run all 10 regardless of outcome
    all_results = []

    for attempt in range(1, N_EPISODES + 1):
        noise_sd    = float(np.random.uniform(0.04, OBS_NOISE_STD))
        spawn_y_off = _SPAWN_OFFSETS[(attempt - 1) % len(_SPAWN_OFFSETS)]

        result = run_episode(node, actor, attempt, noise_sd, spawn_y_off)
        all_results.append(result)

        outcome = result["result"]
        print(f"\n  Attempt {attempt}/{N_EPISODES} outcome: {outcome}")

        if attempt < N_EPISODES:
            # Wait between episodes to reset walkers and break phase lock
            wait = float(np.random.uniform(30, 60))
            print(f"  [WAIT] {wait:.0f}s before next attempt...")
            time.sleep(wait)

    ex.shutdown(); rclpy.shutdown()

    # ── Aggregate stats ────────────────────────────────────────────────────────
    successes  = [r for r in all_results if r["result"] == "SUCCESS"]
    n_success  = len(successes)
    sr         = n_success / N_EPISODES
    avg_speed  = float(np.mean([r["avg_speed_ms"]  for r in all_results]))
    avg_col    = float(np.mean([r["collisions"]     for r in all_results]))
    avg_dur    = float(np.mean([r["duration_s"]     for r in all_results]))

    print("\n" + "="*60)
    print("  10-EPISODE TEST COMPLETE — REAL-WORLD ENVIRONMENT")
    print("="*60)
    print(f"  Episodes       : {N_EPISODES}")
    print(f"  Successes      : {n_success} / {N_EPISODES}  (SR = {sr*100:.0f}%)")
    print(f"  Avg speed      : {avg_speed:.3f} m/s")
    print(f"  Avg collisions : {avg_col:.1f}")
    print(f"  Avg duration   : {avg_dur:.1f} s")
    print("-"*60)
    for r in all_results:
        print(f"  A{r['attempt']:02d}: {r['result']:<8}  t={r['duration_s']:.1f}s  "
              f"spd={r['avg_speed_ms']:.2f}m/s  col={r['collisions']}  "
              f"spawn_y={r['dr_spawn_y_m']:+.2f}")
    print("="*60)

    # ── Save JSON summary ──────────────────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = OUT_DIR / f"test_r54_realworld_10ep_summary_{ts}.json"
    meta = {
        "test_name":         "R54 Best Policy — Real-World Domain Randomisation Test (Tunnel Tile 5, 10 Episodes)",
        "testing_world":     "tunnel_100m_realworld.world",
        "training_world":    "tunnel_100m_dynamic_25.world",
        "policy_file":       "r54_best.pth",
        "policy_episode":    str(saved_ep),
        "policy_SR":         str(saved_sr),
        "test_max_speed":    TEST_MAX_SPEED,
        "corridor_hw_m":     CORRIDOR_HW,
        "state_dim":         STATE_DIM,
        "num_h_sectors":     NUM_H_SECTORS,
        "depth_percentile":  DEPTH_PERCENTILE,
        "obs_noise_std":     OBS_NOISE_STD,
        "spawn_y_jitter":    SPAWN_Y_JITTER,
        "timestamp":         ts,
        "n_episodes":        N_EPISODES,
        "n_success":         n_success,
        "success_rate":      round(sr, 4),
        "avg_speed_ms":      round(avg_speed, 3),
        "avg_collisions":    round(avg_col, 1),
        "avg_duration_s":    round(avg_dur, 1),
        "episodes":          all_results,
    }
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Summary saved : {summary_path}")


if __name__ == "__main__":
    main()
