#!/usr/bin/env python3
"""
R54 Cross-Validation — Native Drone in Training Environment
===========================================================
Runs the r54_best.pth policy in tunnel_100m_dynamic_25.world (the training
world) under five observation conditions:

  L0  Full 21D observation      (baseline)
  L1  No vertical depth         dims 9–11  → 0  (top/mid/bottom depth)
  L2  No velocity feedback      dims 16–18 → 0  (vx, vy, vz)
  L3  Depth sensors only        dims 12–20 → 0  (only 9 horiz. sectors kept)
  L4  No depth sensors          dims 0–12  → 0  (navigation/velocity only)

One episode per condition.  All metrics, step-CSVs, and a master JSON are
saved to  Final Results/testing/cross_validation/.
A Figure-6.11 grade plot is generated at the end.

Pre-requisite:
  Gazebo must be running with tunnel_100m_dynamic_25.world
  ros2 launch uav_simulator tunnel_drl.launch.py gui:=true \
      world:=.../tunnel_100m_dynamic_25.world

Usage:
  cd /home/makhosazana/Project/CERLAB-UAV-Autonomy
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 "Final Results/testing/run_crossval_r54.py"
"""

import csv, json, math, os, sys, threading, time
from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
OUT_DIR    = Path(__file__).resolve().parent / "cross_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── R54 constants (must match training exactly) ────────────────────────────────
TUNNEL_LENGTH   = 100.0
TURN_POINT      = 95.0
CORRIDOR_HW     = 3.5
SAFE_ALT        = 1.5
MAX_ALT         = 2.8
MIN_ALT         = 0.6
ALT_FLOOR       = 0.3
ALT_CEIL        = 4.5
MAX_SPEED       = 5.5
COLLISION_DIST  = 0.35
NEAR_MISS_DIST  = 0.8
PROX_ALARM_M    = 2.0
AVOID_START     = 5.0
ACTION_SMOOTH_FWD = 0.35
ACTION_SMOOTH_LAT = 0.30
TICK_DT         = 0.1
NUM_H_SECTORS   = 9
DEPTH_PERCENTILE = 5
DEPTH_FAR       = 10.0
OBS_NOISE_STD   = 0.06
SPAWN_Y_JITTER  = 0.20
_SECTOR_BIAS    = np.linspace(1.0, -1.0, NUM_H_SECTORS, dtype=np.float32)
STATE_DIM       = 21
ACTION_DIM      = 3
TEST_MAX_SPEED  = 4.5

# ── Cross-validation observation conditions ────────────────────────────────────
# Each entry: (label, short_name, description, list_of_dims_to_zero)
CONDITIONS = [
    ("L0", "Full observation",
     "All 21 state dimensions active — training-identical input",
     []),
    ("L1", "No vertical depth",
     "dims 9–11 → 0: top/middle/bottom vertical depth zones removed",
     [9, 10, 11]),
    ("L2", "No velocity feedback",
     "dims 16–18 → 0: forward, lateral and vertical velocity zeroed",
     [16, 17, 18]),
    ("L3", "Depth sectors only",
     "dims 12–20 → 0: navigation, velocity and flags zeroed; only 9 horiz. depth sectors remain",
     list(range(12, 21))),
    ("L4", "No depth sensors",
     "dims 0–12 → 0: all depth information removed; navigation and velocity only",
     list(range(0, 13))),
]

# ── R54 Actor ──────────────────────────────────────────────────────────────────
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


# ── ROS2 Drone Node ────────────────────────────────────────────────────────────
class DroneNode(Node):
    def __init__(self):
        super().__init__("r54_crossval_node")
        self._lock = threading.Lock()
        self.px = self.py = self.pz = self.yaw = 0.0
        self.vx = self.vy = self.vz = 0.0
        self._prev_px = self._prev_py = self._prev_pz = 0.0
        self._prev_stamp = 0.0
        self.pose_received = False
        self.depth_sectors = np.full(NUM_H_SECTORS, DEPTH_FAR, dtype=np.float32)
        self.depth_top = self.depth_mid = self.depth_bottom = DEPTH_FAR
        self.depth_received = False
        self.prox_alarm = 0

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
            self.px = msg.pose.position.x
            self.py = msg.pose.position.y
            self.pz = msg.pose.position.z
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
        sw = w // NUM_H_SECTORS
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

    def snap(self):
        with self._lock:
            return {
                "px": self.px, "py": self.py, "pz": self.pz, "yaw": self.yaw,
                "vx": self.vx, "vy": self.vy, "vz": self.vz,
                "depth_sectors": self.depth_sectors.copy(),
                "depth_top": self.depth_top, "depth_mid": self.depth_mid,
                "depth_bottom": self.depth_bottom, "prox_alarm": self.prox_alarm,
            }

    def publish_setpoint(self, x, y, z, yaw=0.0):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        s = math.sin(yaw / 2); c = math.cos(yaw / 2)
        msg.pose.orientation.z = s; msg.pose.orientation.w = c
        self.setpoint_pub.publish(msg)

    def takeoff(self):
        b = Bool(); b.data = True
        for _ in range(5):
            self.posctrl_pub.publish(b); time.sleep(0.05)
        self.takeoff_pub.publish(Empty())

    def teleport(self, x=0.0, y=0.0, z=0.2):
        helper = Node("_tp_cv")
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


# ── State builder with optional masking ────────────────────────────────────────
def build_state(snap, spawn_x, spawn_y, fwd_progress, lat_offset,
                max_speed, phase, noise_sd, mask_dims: List[int]) -> np.ndarray:
    ds = snap["depth_sectors"]
    h_noisy = np.clip(
        ds + np.random.normal(0.0, noise_sd, NUM_H_SECTORS).astype(np.float32),
        0.05, DEPTH_FAR)
    v_top = float(np.clip(snap["depth_top"]    + np.random.normal(0, noise_sd), 0.05, DEPTH_FAR))
    v_mid = float(np.clip(snap["depth_mid"]    + np.random.normal(0, noise_sd), 0.05, DEPTH_FAR))
    v_bot = float(np.clip(snap["depth_bottom"] + np.random.normal(0, noise_sd), 0.05, DEPTH_FAR))
    speed = abs(snap["vx"]) if phase == "OUTBOUND" else abs(-snap["vx"])

    state = np.array([
        *[h_noisy[i] / DEPTH_FAR for i in range(NUM_H_SECTORS)],  # 0–8
        v_top / DEPTH_FAR,                                          # 9
        v_mid / DEPTH_FAR,                                          # 10
        v_bot / DEPTH_FAR,                                          # 11
        float(snap["prox_alarm"]),                                  # 12
        np.clip(fwd_progress / TUNNEL_LENGTH, -0.1, 1.1),          # 13
        np.clip(lat_offset   / CORRIDOR_HW,   -1, 1),              # 14
        np.clip((snap["pz"] - SAFE_ALT) / 1.5, -1, 1),            # 15
        np.clip(speed        / 7.0, -1, 1),                        # 16
        np.clip(snap["vy"]   / 3.0, -1, 1),                        # 17
        np.clip(snap["vz"]   / 2.0, -1, 1),                        # 18
        1.0 if phase == "RETURN" else 0.0,                          # 19
        np.clip(max_speed    / 7.0,  0, 1),                        # 20
    ], dtype=np.float32)

    # Apply observation mask
    for d in mask_dims:
        state[d] = 0.0

    return state


# ── Action → delta position ─────────────────────────────────────────────────────
def apply_action(smoothed, snap, spawn_y, lat_offset, phase, max_speed):
    raw_fwd, raw_lat, raw_alt = smoothed
    fwd_speed = float(np.clip((raw_fwd * 0.5 + 0.5) * max_speed, 1.0, MAX_SPEED))
    lat_speed = float(raw_lat * 0.8)
    alt_adj   = float(raw_alt * 0.4)

    ds = snap["depth_sectors"]
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


# ── Run one episode under a given observation mask ─────────────────────────────
def run_episode(node, actor, cond_label, cond_name, mask_dims, noise_sd,
                spawn_y_offset, episode_idx) -> dict:
    print(f"\n{'='*65}")
    print(f"  {cond_label} — {cond_name}")
    print(f"  mask={mask_dims}  noise_sd={noise_sd:.3f}  spawn_y={spawn_y_offset:+.3f}")
    print(f"{'='*65}")

    # Wait for sensors
    t0 = time.time()
    while not (node.pose_received and node.depth_received):
        time.sleep(0.2)
        if time.time() - t0 > 30.0:
            return {"result": "sensor_timeout"}

    # Teleport to start
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
    node.takeoff()
    t0 = time.time()
    while True:
        snap = node.snap()
        if snap["pz"] >= SAFE_ALT - 0.15:
            break
        node.publish_setpoint(spawn_x, spawn_y, SAFE_ALT)
        time.sleep(0.1)
        if time.time() - t0 > 20.0:
            return {"result": "takeoff_timeout"}
    time.sleep(0.8)

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"cv_{cond_label}_ep{episode_idx}_{ts}.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=[
        "step", "phase", "t_s", "x_m", "y_m", "z_m",
        "vx_ms", "vy_ms", "vz_ms", "fwd_progress_m", "lat_offset_m",
        "speed_ms", "front_d_m", "min_depth_m",
        "prox_alarm", "collision", "near_miss",
    ])
    writer.writeheader()

    ep_start  = time.time()
    phase     = "OUTBOUND"
    step      = 0
    speeds    = []
    col_count = 0; nm_count = 0
    min_d_ever = DEPTH_FAR
    delayed_a  = np.zeros(3, dtype=np.float32)
    result_label = "TIMEOUT"

    while True:
        snap  = node.snap()
        t_el  = time.time() - ep_start
        ds    = snap["depth_sectors"]
        fwd_progress = (snap["px"] - spawn_x) if phase == "OUTBOUND" else (spawn_x - snap["px"])
        lat_offset   = snap["py"] - spawn_y
        speed        = abs(snap["vx"])
        speeds.append(speed)
        min_d  = float(np.min(ds))
        front_d = float(np.min([ds[3], ds[4], ds[5]]))
        min_d_ever = min(min_d_ever, min_d)

        is_col = (min_d < COLLISION_DIST)
        is_nm  = (COLLISION_DIST <= min_d < NEAR_MISS_DIST)
        if is_col: col_count += 1
        if is_nm:  nm_count  += 1

        state = build_state(snap, spawn_x, spawn_y, fwd_progress, lat_offset,
                            TEST_MAX_SPEED, phase, noise_sd, mask_dims)
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
                                              phase, TEST_MAX_SPEED)
        tgt_x = float(np.clip(tgt_x + dx, spawn_x - 3.0, spawn_x + TURN_POINT + 5.0))
        tgt_y = float(np.clip(tgt_y + dy, -CORRIDOR_HW, CORRIDOR_HW))
        tgt_z = float(np.clip(tgt_z + dz, MIN_ALT, ALT_CEIL))
        yaw   = 0.0 if phase == "OUTBOUND" else math.pi
        node.publish_setpoint(tgt_x, tgt_y, tgt_z, yaw)

        writer.writerow({
            "step": step, "phase": phase, "t_s": round(t_el, 3),
            "x_m": round(snap["px"], 3), "y_m": round(snap["py"], 3),
            "z_m": round(snap["pz"], 3),
            "vx_ms": round(snap["vx"], 3), "vy_ms": round(snap["vy"], 3),
            "vz_ms": round(snap["vz"], 3),
            "fwd_progress_m": round(fwd_progress, 2),
            "lat_offset_m":   round(lat_offset, 3),
            "speed_ms": round(speed, 3),
            "front_d_m": round(front_d, 3), "min_depth_m": round(min_d, 3),
            "prox_alarm": snap["prox_alarm"],
            "collision": int(is_col), "near_miss": int(is_nm),
        })

        if step % 30 == 0:
            print(f"  [{phase:8s}|t={t_el:5.1f}s] "
                  f"fwd={fwd_progress:5.1f}m  z={snap['pz']:.2f}m  "
                  f"spd={speed:.2f}  front={front_d:.2f}  "
                  f"col={col_count} nm={nm_count}")

        t_tick = time.time()
        while time.time() - t_tick < TICK_DT:
            node.publish_setpoint(tgt_x, tgt_y, tgt_z, yaw)
            time.sleep(0.02)
        step += 1

        if phase == "OUTBOUND" and (snap["px"] - spawn_x) >= TURN_POINT:
            phase = "RETURN"
            tgt_x = spawn_x + TURN_POINT
            print(f"\n  [OUTBOUND DONE] x={snap['px']-spawn_x:.1f}m  t={t_el:.1f}s")
            time.sleep(0.3)

        if phase == "RETURN" and (snap["px"] - spawn_x) <= 2.0:
            result_label = "SUCCESS"
            print(f"\n  [SUCCESS] t={t_el:.1f}s"); break

        if t_el >= 200.0:
            result_label = "TIMEOUT"
            print(f"\n  [TIMEOUT] fwd={fwd_progress:.1f}m"); break
        if snap["pz"] < ALT_FLOOR:
            result_label = "CRASH_ALT"
            print(f"\n  [CRASH_ALT] z={snap['pz']:.2f}m"); break
        if abs(snap["py"] - spawn_y) > CORRIDOR_HW + 0.5:
            result_label = "CRASH_LAT"
            print(f"\n  [CRASH_LAT] y_off={snap['py']-spawn_y:.2f}m"); break

    csv_f.close()
    duration  = time.time() - ep_start
    return {
        "condition":    cond_label,
        "cond_name":    cond_name,
        "masked_dims":  mask_dims,
        "n_masked":     len(mask_dims),
        "episode":      episode_idx,
        "result":       result_label,
        "duration_s":   round(duration, 1),
        "avg_speed_ms": round(float(np.mean(speeds)), 3) if speeds else 0.0,
        "peak_speed_ms":round(float(np.max(speeds)),  3) if speeds else 0.0,
        "collisions":   col_count,
        "near_misses":  nm_count,
        "min_depth_m":  round(min_d_ever, 3),
        "noise_sd":     round(noise_sd, 3),
        "spawn_y_m":    round(spawn_y_offset, 3),
        "csv_log":      str(csv_path),
        "steps":        step,
    }


# ── Plot (Figure 6.11 equivalent) ─────────────────────────────────────────────
def make_plots(results: list):
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    mpl.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "#f5f5f5",
        "grid.color":         "white",
        "grid.linewidth":     1.2,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.labelsize":     13,
        "axes.labelweight":   "bold",
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    10,
        "lines.linewidth":    2.2,
        "font.family":        "DejaVu Sans",
    })

    labels    = [r["condition"] for r in results]
    names     = [r["cond_name"] for r in results]
    successes = [1 if r["result"] == "SUCCESS" else 0 for r in results]
    avg_spds  = [r["avg_speed_ms"]  for r in results]
    min_deps  = [r["min_depth_m"]   for r in results]
    n_masked  = [r["n_masked"]      for r in results]
    durations = [r["duration_s"]    for r in results]
    incidents = [r["collisions"] + r["near_misses"] for r in results]

    # colour: green = success, red = failure
    bar_colors = ["#2E7D32" if s else "#C62828" for s in successes]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5.5))
    fig.suptitle(
        "Figure 6.11 — Cross-Validation: Observation Withholding Robustness\n"
        "R54 Best Policy | Native CERLAB Drone | Training World (25 obstacles)",
        fontsize=13, fontweight="bold", y=1.03,
    )

    x = np.arange(len(results))
    short = [f"{r['condition']}\n({r['n_masked']}D off)" for r in results]

    # ① Success / Fail bar
    ax = axes[0]
    bars = ax.bar(x, successes, color=bar_colors, width=0.55, zorder=3,
                  edgecolor="white")
    ax.set_title("Mission Success")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["FAIL", "PASS"])
    for bar, s in zip(bars, successes):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.04,
                "✓" if s else "✗",
                ha="center", fontsize=14, fontweight="bold",
                color="#2E7D32" if s else "#C62828")

    # ② Avg speed
    ax = axes[1]
    ax.bar(x, avg_spds, color=bar_colors, width=0.55, zorder=3, edgecolor="white")
    ax.set_title("Avg Speed  (m/s)")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
    ax.set_ylim(0, max(avg_spds) * 1.25 + 0.2)
    for xi, v in enumerate(avg_spds):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold", color="#333")

    # ③ Min obstacle distance
    ax = axes[2]
    ax.axhspan(0, COLLISION_DIST, color="#fde8e8", alpha=0.6, zorder=1)
    ax.axhline(COLLISION_DIST, color="#C62828", linewidth=1.2,
               linestyle="--", alpha=0.8, zorder=2, label=f"Collision ≤{COLLISION_DIST}m")
    ax.bar(x, min_deps, color=bar_colors, width=0.55, zorder=3, edgecolor="white")
    ax.set_title("Min Obstacle Dist.  (m)")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
    ax.set_ylim(0, max(min_deps) * 1.3 + 0.1)
    for xi, v in enumerate(min_deps):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9,
                fontweight="bold", color="#333")
    ax.legend(loc="upper right", fontsize=9)

    # ④ Safety incidents
    ax = axes[3]
    cols_v = [r["collisions"]  for r in results]
    nm_v   = [r["near_misses"] for r in results]
    ax.bar(x, cols_v, color=["#C62828"]*len(results), width=0.55,
           label="Collisions", zorder=3, edgecolor="white")
    ax.bar(x, nm_v, bottom=cols_v,
           color=["#EF9A9A"]*len(results), width=0.55,
           label="Near-misses", zorder=3, edgecolor="white")
    ax.set_title("Safety Incidents")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
    ax.set_ylim(0, max(max(c+n for c,n in zip(cols_v,nm_v)), 1) * 1.3 + 1)
    for xi, (c, n) in enumerate(zip(cols_v, nm_v)):
        t = c + n
        if t == 0:
            ax.text(xi, 0.15, "CLEAN", ha="center", fontsize=8,
                    fontweight="bold", color="#2E7D32")
        else:
            ax.text(xi, t + 0.1, str(t), ha="center", fontsize=9,
                    fontweight="bold", color="#333")
    ax.legend(loc="upper right", fontsize=9)

    # shared x-label strip below
    fig.text(0.5, -0.03,
             "Observation condition (number of state dimensions zeroed)",
             ha="center", fontsize=11, color="#555")

    plt.tight_layout()
    out = OUT_DIR / "fig6_11_crossval_robustness.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Plot saved: {out}")

    # ── condition detail table figure ──────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(14, 3.5))
    fig2.patch.set_facecolor("white")
    ax2.axis("off")
    col_hdr = ["Condition", "Description", "Dims\nzero'd", "Result",
               "Duration\n(s)", "Avg spd\n(m/s)", "Peak spd\n(m/s)",
               "Min depth\n(m)", "Col.", "N.M."]
    rows = []
    for r in results:
        rows.append([
            r["condition"],
            r["cond_name"],
            str(r["n_masked"]),
            r["result"],
            str(r["duration_s"]),
            f"{r['avg_speed_ms']:.3f}",
            f"{r['peak_speed_ms']:.3f}",
            f"{r['min_depth_m']:.3f}",
            str(r["collisions"]),
            str(r["near_misses"]),
        ])
    tbl = ax2.table(cellText=rows, colLabels=col_hdr,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    # colour header
    for j in range(len(col_hdr)):
        tbl[0, j].set_facecolor("#1A237E")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # colour data rows by result
    result_fill = {"SUCCESS": "#D4EDDA", "TIMEOUT": "#FFF3CD",
                   "CRASH_ALT": "#FDEBD0", "CRASH_LAT": "#FDEBD0"}
    for i, r in enumerate(results, 1):
        fc = result_fill.get(r["result"], "#FFFFFF")
        for j in range(len(col_hdr)):
            tbl[i, j].set_facecolor(fc)
    ax2.set_title("Cross-Validation Condition Summary — R54 / Training World (25 obstacles)",
                  fontsize=12, fontweight="bold", pad=12)
    out2 = OUT_DIR / "fig6_11_crossval_table.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"  Table saved: {out2}")


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  R54 CROSS-VALIDATION — TRAINING WORLD (25 OBSTACLES)")
    print("  Policy: r54_best.pth  |  Agent: native CERLAB drone")
    print("  Conditions: 5 observation-masking levels")
    print("="*65)

    if not CHECKPOINT.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {CHECKPOINT}")
    ckpt  = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    actor = Actor(); actor.load_state_dict(ckpt["actor"]); actor.eval()
    saved_ep = ckpt.get("episode", "?")
    saved_sr = ckpt.get("sr", "?")
    print(f"\n  Loaded r54_best.pth  |  saved_ep={saved_ep}  SR={saved_sr}")

    rclpy.init()
    node = DroneNode()
    ex   = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    spin_thread = threading.Thread(target=ex.spin, daemon=True)
    spin_thread.start()

    np.random.seed(42)          # fixed seed → reproducible spawn jitter
    noise_sd = 0.055            # mid-range DR noise, same as training-world PX4 run

    all_results = []

    for idx, (label, name, desc, mask) in enumerate(CONDITIONS):
        # fixed but condition-varied spawn jitter so environments differ slightly
        spawn_y = float(np.random.uniform(-SPAWN_Y_JITTER, SPAWN_Y_JITTER))
        result  = run_episode(node, actor, label, name, mask,
                              noise_sd, spawn_y, episode_idx=idx + 1)
        result["description"] = desc
        all_results.append(result)
        print(f"\n  [{label}] → {result['result']}  "
              f"{result.get('duration_s','?')}s  "
              f"avg={result.get('avg_speed_ms','?')}m/s  "
              f"col={result.get('collisions','?')}  nm={result.get('near_misses','?')}")

        if idx < len(CONDITIONS) - 1:
            print("\n  Pausing 4 s before next condition …")
            time.sleep(4.0)

    ex.shutdown(); rclpy.shutdown()

    # ── Save master JSON ───────────────────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    master = {
        "test_name":    "R54 Cross-Validation — Observation Withholding Robustness",
        "world":        "tunnel_100m_dynamic_25.world (TRAINING ENVIRONMENT, 25 obstacles)",
        "agent":        "CERLAB native sim drone",
        "policy":       "r54_best.pth",
        "policy_ep":    str(saved_ep),
        "policy_sr":    str(saved_sr),
        "noise_sd":     noise_sd,
        "test_max_spd": TEST_MAX_SPEED,
        "date":         ts,
        "conditions":   all_results,
    }
    json_path = OUT_DIR / f"crossval_summary_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(master, f, indent=2)
    print(f"\n  Master JSON: {json_path}")

    # ── Print table ────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"  {'Cond':<5} {'Name':<28} {'Result':<12} {'Dur':>6} "
          f"{'AvgSpd':>7} {'MinDep':>7} {'Col':>4} {'NM':>4}")
    print("  " + "-"*63)
    for r in all_results:
        print(f"  {r['condition']:<5} {r['cond_name']:<28} "
              f"{r['result']:<12} {r.get('duration_s',0):>6.1f}s "
              f"{r.get('avg_speed_ms',0):>6.3f} {r.get('min_depth_m',0):>7.3f} "
              f"{r.get('collisions',0):>4} {r.get('near_misses',0):>4}")
    print("="*65)

    # ── Generate plots ─────────────────────────────────────────────────────────
    valid = [r for r in all_results if "duration_s" in r]
    if valid:
        make_plots(valid)

    print(f"\n  All files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
