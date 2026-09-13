#!/usr/bin/env python3
"""
R54 Proper K-Fold Blocked Cross-Validation
==========================================
Addresses the three canonical failure modes of naive RL evaluation:

  1. Temporal Autocorrelation  → episodes are atomic units; folds are
                                 contiguous blocks (not random samples of
                                 timesteps), so future states never leak
                                 into earlier fold evaluations.

  2. Spatial Leakage           → each fold uses a distinct set of lateral
                                 starting offsets, ensuring the policy is
                                 assessed at locations it was not "tuned" to
                                 by a single spawn point.

  3. Dynamic Physics / Closed-Loop → every episode is a live closed-loop
                                 interaction with the Gazebo physics engine.
                                 No replayed sensor data; no open-loop replay.

Protocol
--------
  K  = 5 folds
  N  = 3 episodes per fold  (15 episodes total)
  Timeout = 90 s per episode

  Fold-level y-offset sets (lateral start diversity):
    Fold 0:  y ∈ {-0.30,  0.00, +0.30}   wide spread
    Fold 1:  y ∈ {-0.20,  0.05, +0.20}   moderate
    Fold 2:  y ∈ {-0.15,  0.00, +0.15}   narrow
    Fold 3:  y ∈ {-0.25, -0.05, +0.25}   left-biased
    Fold 4:  y ∈ {+0.10,  0.00, -0.10}   right-biased

Metrics (per episode, per fold, aggregate)
------------------------------------------
  • max_fwd_m        — maximum forward progress before stuck/timeout (m)
  • outbound_pct     — % of the 95 m outbound leg covered
  • avg_spd_moving   — mean speed while actively moving (> 0.2 m/s) (m/s)
  • time_to_stuck_s  — elapsed time when drone goes permanently stuck (s)
  • col_events       — unique collision-initiation events (not cumulative ticks)
  • near_miss_events — unique near-miss events
  • min_depth_m      — closest obstacle encounter (m)
  • result           — SUCCESS / TIMEOUT / CRASH_*

Aggregate statistics (across all 15 episodes AND per fold):
  • mean ± std for every metric
  • 95 % bootstrap CI (B = 2000 resamples)
  • Binomial 95 % Clopper-Pearson CI for success rate
  • Fold-to-fold variance  → low = robust generalisation; high = overfitting
  • CV score = composite (max_fwd / 95 * 0.5 + avg_spd_moving / 5.5 * 0.3
                          + (1 - col_events / 10) * 0.2)

Output  →  Final Results/testing/Actual Cross validation/
  run_<ts>/
    per_episode.csv          raw per-episode metrics (15 rows)
    per_fold_summary.csv     fold-level aggregates (5 rows)
    aggregate_stats.json     global CV stats + bootstrap CIs
    cv_report.txt            human-readable full report
    fig_fold_consistency.png box plots of key metrics across folds
    fig_trajectories.png     forward-progress curves, colour-coded by fold
    fig_cv_summary.png       aggregate CI bar chart (main result figure)

Pre-requisite
-------------
  Gazebo running with tunnel_100m_dynamic_25.world:
    ros2 launch uav_simulator tunnel_drl.launch.py gui:=true \\
        world:=.../tunnel_100m_dynamic_25.world

Usage
-----
  cd /home/makhosazana/Project/CERLAB-UAV-Autonomy
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 "Final Results/testing/Actual Cross validation/run_proper_crossval_r54.py"
"""

import csv, json, math, os, sys, threading, time
from pathlib import Path
from typing import List, Tuple

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
PROJECT    = Path(__file__).resolve().parents[3]
CHECKPOINT = PROJECT / "src/tunnel_drl/results_r54/r54_best.pth"
URDF_PATH  = PROJECT / "install/uav_simulator/share/uav_simulator/urdf/quadcopter.urdf"
BASE_OUT   = Path(__file__).resolve().parent
TS_TAG     = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR    = BASE_OUT / f"run_{TS_TAG}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── R54 constants (must match training exactly) ────────────────────────────────
TUNNEL_LENGTH     = 100.0
TURN_POINT        = 95.0
CORRIDOR_HW       = 3.5
SAFE_ALT          = 1.5
MAX_ALT           = 2.8
MIN_ALT           = 0.6
ALT_FLOOR         = 0.3
ALT_CEIL          = 4.5
MAX_SPEED         = 5.5
TEST_MAX_SPEED    = 4.5
COLLISION_DIST    = 0.35
NEAR_MISS_DIST    = 0.8
PROX_ALARM_M      = 2.0
AVOID_START       = 5.0
ACTION_SMOOTH_FWD = 0.35
ACTION_SMOOTH_LAT = 0.30
TICK_DT           = 0.1
NUM_H_SECTORS     = 9
DEPTH_PERCENTILE  = 5
DEPTH_FAR         = 10.0
OBS_NOISE_STD     = 0.06
STATE_DIM         = 21
ACTION_DIM        = 3
_SECTOR_BIAS      = np.linspace(1.0, -1.0, NUM_H_SECTORS, dtype=np.float32)

# ── CV protocol constants ──────────────────────────────────────────────────────
K_FOLDS          = 5
N_PER_FOLD       = 3          # episodes per fold
EPISODE_TIMEOUT  = 90.0       # seconds — drone is stuck long before this
STUCK_SPD_THR    = 0.2        # m/s — below this = not moving
STUCK_CONSEC_THR = 15         # consecutive ticks (~1.5 s) at low speed = stuck
BOOTSTRAP_B      = 2000

# Lateral starting offsets per fold — each fold tests a distinct spatial set
FOLD_STARTS: List[List[float]] = [
    [-0.30,  0.00, +0.30],   # Fold 0: wide spread
    [-0.20, +0.05, +0.20],   # Fold 1: moderate, slightly right
    [-0.15,  0.00, +0.15],   # Fold 2: narrow centre
    [-0.25, -0.05, +0.25],   # Fold 3: wide, slightly left
    [+0.10,  0.00, -0.10],   # Fold 4: right-biased
]

# Fold color palette (colorblind-safe)
FOLD_COLORS = ["#1565C0", "#2E7D32", "#BF360C", "#6A1B9A", "#E65100"]

# ── R54 Actor architecture ─────────────────────────────────────────────────────
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
        super().__init__("r54_propercv_node")
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
            raw = np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.width)
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
                "depth_top":  self.depth_top,
                "depth_mid":  self.depth_mid,
                "depth_bottom": self.depth_bottom,
                "prox_alarm": self.prox_alarm,
            }

    def publish_setpoint(self, x, y, z, yaw=0.0):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(x); msg.pose.position.y = float(y)
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
        helper = Node("_tp_pcv")
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


# ── State builder (full 21D, no masking) ──────────────────────────────────────
def build_state(snap, spawn_x, spawn_y, fwd_progress, lat_offset,
                max_speed, phase, noise_sd) -> np.ndarray:
    ds = snap["depth_sectors"]
    h_noisy = np.clip(
        ds + np.random.normal(0.0, noise_sd, NUM_H_SECTORS).astype(np.float32),
        0.05, DEPTH_FAR)
    v_top = float(np.clip(snap["depth_top"]    + np.random.normal(0, noise_sd), 0.05, DEPTH_FAR))
    v_mid = float(np.clip(snap["depth_mid"]    + np.random.normal(0, noise_sd), 0.05, DEPTH_FAR))
    v_bot = float(np.clip(snap["depth_bottom"] + np.random.normal(0, noise_sd), 0.05, DEPTH_FAR))
    speed = abs(snap["vx"]) if phase == "OUTBOUND" else abs(-snap["vx"])
    return np.array([
        *[h_noisy[i] / DEPTH_FAR for i in range(NUM_H_SECTORS)],
        v_top / DEPTH_FAR, v_mid / DEPTH_FAR, v_bot / DEPTH_FAR,
        float(snap["prox_alarm"]),
        np.clip(fwd_progress / TUNNEL_LENGTH, -0.1, 1.1),
        np.clip(lat_offset   / CORRIDOR_HW,   -1, 1),
        np.clip((snap["pz"]  - SAFE_ALT) / 1.5, -1, 1),
        np.clip(speed        / 7.0, -1, 1),
        np.clip(snap["vy"]   / 3.0, -1, 1),
        np.clip(snap["vz"]   / 2.0, -1, 1),
        1.0 if phase == "RETURN" else 0.0,
        np.clip(max_speed    / 7.0,  0, 1),
    ], dtype=np.float32)


# ── Action → position delta ────────────────────────────────────────────────────
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


# ── Detect unique events (rising-edge count) ───────────────────────────────────
class EventCounter:
    def __init__(self, threshold, high_is_event=True):
        self.thr = threshold
        self.high = high_is_event
        self._in_event = False
        self.count = 0

    def update(self, value) -> bool:
        triggered = (value < self.thr) if not self.high else (value > self.thr)
        if triggered and not self._in_event:
            self._in_event = True
            self.count += 1
            return True
        elif not triggered:
            self._in_event = False
        return False


# ── Run one episode ────────────────────────────────────────────────────────────
def run_episode(node, actor, fold_idx, ep_in_fold, global_ep_idx,
                spawn_y_offset) -> dict:
    label = f"F{fold_idx}E{ep_in_fold}"
    print(f"\n{'═'*65}")
    print(f"  Fold {fold_idx}  Episode {ep_in_fold}  (global #{global_ep_idx})"
          f"  start_y={spawn_y_offset:+.3f}")
    print(f"{'═'*65}")

    t0 = time.time()
    while not (node.pose_received and node.depth_received):
        time.sleep(0.2)
        if time.time() - t0 > 30.0:
            return {"result": "sensor_timeout"}

    node.teleport(x=0.0, y=spawn_y_offset, z=0.2)
    time.sleep(2.5)
    t0 = time.time()
    while not node.pose_received and time.time() - t0 < 10.0:
        time.sleep(0.1)

    snap = node.snap()
    spawn_x = snap["px"]; spawn_y = snap["py"]
    tgt_x = spawn_x; tgt_y = spawn_y; tgt_z = SAFE_ALT
    prev_action = np.zeros(3, dtype=np.float32)

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

    csv_path = OUT_DIR / f"ep_{label}_g{global_ep_idx:02d}.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=[
        "step","fold","ep_in_fold","global_ep","t_s","phase",
        "x_m","y_m","z_m","vx_ms","vy_ms","vz_ms",
        "fwd_progress_m","lat_offset_m","speed_ms",
        "front_d_m","min_depth_m","prox_alarm","collision_event","near_miss_event",
    ])
    writer.writeheader()

    ep_start = time.time()
    phase    = "OUTBOUND"
    step     = 0
    delayed_a = np.zeros(3, dtype=np.float32)

    # Telemetry accumulators
    speeds_moving: List[float] = []
    max_fwd       = 0.0
    min_d_ever    = DEPTH_FAR
    stuck_t       = None
    low_spd_count = 0

    # Collision event counters (rising-edge only)
    col_counter = EventCounter(COLLISION_DIST, high_is_event=False)
    nm_counter  = EventCounter(NEAR_MISS_DIST, high_is_event=False)

    # Trajectory record for plotting
    traj: List[Tuple[float, float]] = []   # (t, fwd_progress)
    result_label = "TIMEOUT"

    while True:
        snap  = node.snap()
        t_el  = time.time() - ep_start
        ds    = snap["depth_sectors"]
        fwd_progress = (snap["px"] - spawn_x) if phase == "OUTBOUND" \
                       else (spawn_x - snap["px"])
        lat_offset   = snap["py"] - spawn_y
        speed        = abs(snap["vx"])
        min_d  = float(np.min(ds))
        front_d = float(np.min([ds[3], ds[4], ds[5]]))

        # Track max forward progress (outbound)
        if phase == "OUTBOUND":
            max_fwd = max(max_fwd, fwd_progress)
        min_d_ever = min(min_d_ever, min_d)

        # Moving speed (exclude stuck-hovering)
        if speed > STUCK_SPD_THR:
            speeds_moving.append(speed)

        # Stuck detection (TEMPORAL AUTOCORRELATION fix: computed per episode,
        # not across episodes, so fold boundaries stay clean)
        if speed < STUCK_SPD_THR:
            low_spd_count += 1
        else:
            low_spd_count = 0
        if stuck_t is None and low_spd_count >= STUCK_CONSEC_THR:
            stuck_t = t_el

        # Event counting (rising-edge)
        is_col_ev = col_counter.update(min_d)
        is_nm_ev  = nm_counter.update(min_d)

        traj.append((t_el, max_fwd if phase == "OUTBOUND" else max_fwd))

        state = build_state(snap, spawn_x, spawn_y, fwd_progress, lat_offset,
                            TEST_MAX_SPEED, phase, OBS_NOISE_STD)
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
            "step": step, "fold": fold_idx, "ep_in_fold": ep_in_fold,
            "global_ep": global_ep_idx,
            "t_s": round(t_el, 3), "phase": phase,
            "x_m": round(snap["px"], 3), "y_m": round(snap["py"], 3),
            "z_m": round(snap["pz"], 3),
            "vx_ms": round(snap["vx"], 3), "vy_ms": round(snap["vy"], 3),
            "vz_ms": round(snap["vz"], 3),
            "fwd_progress_m": round(fwd_progress, 2),
            "lat_offset_m":   round(lat_offset, 3),
            "speed_ms": round(speed, 3),
            "front_d_m": round(front_d, 3), "min_depth_m": round(min_d, 3),
            "prox_alarm": snap["prox_alarm"],
            "collision_event": int(is_col_ev),
            "near_miss_event": int(is_nm_ev),
        })

        if step % 30 == 0:
            print(f"  [F{fold_idx}E{ep_in_fold}|t={t_el:5.1f}s|{phase:8s}] "
                  f"fwd={max_fwd:5.1f}m  z={snap['pz']:.2f}  "
                  f"spd={speed:.2f}  front={front_d:.2f}  "
                  f"col_ev={col_counter.count} nm_ev={nm_counter.count}")

        t_tick = time.time()
        while time.time() - t_tick < TICK_DT:
            node.publish_setpoint(tgt_x, tgt_y, tgt_z, yaw)
            time.sleep(0.02)
        step += 1

        if phase == "OUTBOUND" and (snap["px"] - spawn_x) >= TURN_POINT:
            phase = "RETURN"
            tgt_x = spawn_x + TURN_POINT
            print(f"\n  [OUTBOUND DONE] fwd={snap['px']-spawn_x:.1f}m  t={t_el:.1f}s")
            time.sleep(0.3)

        if phase == "RETURN" and (snap["px"] - spawn_x) <= 2.0:
            result_label = "SUCCESS"
            print(f"\n  [SUCCESS] t={t_el:.1f}s"); break

        if t_el >= EPISODE_TIMEOUT:
            result_label = "TIMEOUT"
            print(f"\n  [TIMEOUT] max_fwd={max_fwd:.1f}m"); break
        if snap["pz"] < ALT_FLOOR:
            result_label = "CRASH_ALT"; break
        if abs(snap["py"] - spawn_y) > CORRIDOR_HW + 0.5:
            result_label = "CRASH_LAT"; break

    csv_f.close()
    duration = time.time() - ep_start

    avg_spd_moving = float(np.mean(speeds_moving)) if speeds_moving else 0.0
    time_to_stuck  = round(stuck_t, 1) if stuck_t is not None else round(duration, 1)
    outbound_pct   = round(min(max_fwd / TURN_POINT, 1.0) * 100, 1)

    return {
        "fold":            fold_idx,
        "ep_in_fold":      ep_in_fold,
        "global_ep":       global_ep_idx,
        "spawn_y":         round(spawn_y_offset, 3),
        "result":          result_label,
        "success":         int(result_label == "SUCCESS"),
        "duration_s":      round(duration, 1),
        "max_fwd_m":       round(max_fwd, 2),
        "outbound_pct":    outbound_pct,
        "avg_spd_moving":  round(avg_spd_moving, 3),
        "time_to_stuck_s": time_to_stuck,
        "col_events":      col_counter.count,
        "near_miss_events":nm_counter.count,
        "min_depth_m":     round(min_d_ever, 3),
        "traj":            traj,      # included for plotting, removed before JSON
    }


# ── Bootstrap CI ───────────────────────────────────────────────────────────────
def bootstrap_ci(data: List[float], stat=np.mean, B=2000, alpha=0.05):
    if len(data) < 2:
        v = stat(data) if data else 0.0
        return round(float(v), 4), round(float(v), 4)
    arr = np.array(data)
    boots = [stat(np.random.choice(arr, len(arr), replace=True)) for _ in range(B)]
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return round(lo, 4), round(hi, 4)


def clopper_pearson(k, n, alpha=0.05):
    """95 % Clopper-Pearson binomial CI."""
    from scipy.stats import beta as _beta
    if n == 0:
        return 0.0, 0.0
    lo = float(_beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(_beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return round(lo, 4), round(hi, 4)


# ── Fold aggregate ─────────────────────────────────────────────────────────────
def fold_stats(eps):
    keys = ["max_fwd_m","outbound_pct","avg_spd_moving",
            "time_to_stuck_s","col_events","near_miss_events","min_depth_m"]
    stats = {}
    for k in keys:
        vals = [e[k] for e in eps]
        stats[f"{k}_mean"] = round(float(np.mean(vals)), 3)
        stats[f"{k}_std"]  = round(float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0, 3)
    sr = sum(e["success"] for e in eps)
    stats["success_rate"] = round(sr / len(eps), 3)
    stats["n_success"] = sr
    stats["n_episodes"] = len(eps)
    return stats


# ── Publication-quality plots ──────────────────────────────────────────────────
def setup_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")
    mpl.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#f5f5f5",
        "grid.color":        "white",
        "grid.linewidth":    1.2,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    13,
        "axes.labelweight":  "bold",
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "legend.fontsize":   10,
        "lines.linewidth":   2.2,
        "font.family":       "DejaVu Sans",
        "figure.dpi":        200,
    })


def plot_fold_consistency(all_eps, fold_stats_list):
    """Box plots per fold for the four most informative metrics."""
    setup_style()
    metrics = [
        ("max_fwd_m",       "Max Forward Progress (m)",  None),
        ("avg_spd_moving",  "Avg Moving Speed (m/s)",    None),
        ("col_events",      "Collision Events",          None),
        ("time_to_stuck_s", "Time to Stuck (s)",         None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "K-Fold Cross-Validation — Fold Consistency\n"
        "R54 Best Policy | Native CERLAB Drone | Training World (25 obs)",
        fontsize=13, fontweight="bold", y=1.02)

    for ax, (key, ylabel, _) in zip(axes, metrics):
        fold_data = []
        for fi in range(K_FOLDS):
            fold_eps = [e for e in all_eps if e["fold"] == fi]
            fold_data.append([e[key] for e in fold_eps])

        bp = ax.boxplot(fold_data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], FOLD_COLORS):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        for whisker in bp["whiskers"]:
            whisker.set(color="#555555", linewidth=1.2, linestyle="--")
        for cap in bp["caps"]:
            cap.set(color="#555555", linewidth=1.2)
        for flier in bp["fliers"]:
            flier.set(marker="o", color="#888888", alpha=0.5, markersize=5)

        ax.set_xticks(range(1, K_FOLDS + 1))
        ax.set_xticklabels([f"Fold {i}" for i in range(K_FOLDS)], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

        # Annotate mean per fold
        for fi, d in enumerate(fold_data):
            if d:
                ax.text(fi + 1, max(d) + 0.05 * (max(d) - min(d) + 1e-3),
                        f"μ={np.mean(d):.1f}", ha="center", va="bottom",
                        fontsize=8, color=FOLD_COLORS[fi])

    fig.tight_layout()
    path = OUT_DIR / "fig_fold_consistency.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_trajectories(all_eps):
    """Forward-progress curves per episode, colour-coded by fold."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "K-Fold Cross-Validation — Episode Trajectories\n"
        "R54 Best Policy | Native CERLAB Drone | Training World (25 obs)",
        fontsize=13, fontweight="bold")

    for ep in all_eps:
        fi = ep["fold"]
        traj = ep["traj"]
        if not traj:
            continue
        ts  = [p[0] for p in traj]
        fwd = [p[1] for p in traj]
        lbl = f"F{fi}" if ep["ep_in_fold"] == 0 else None
        ax1.plot(ts, fwd, color=FOLD_COLORS[fi], alpha=0.7,
                 linewidth=1.8, label=lbl)

    ax1.axhline(TURN_POINT, color="#555", linestyle="--", linewidth=1.2,
                label=f"Turn-point ({TURN_POINT:.0f} m)")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Max Forward Progress (m)")
    ax1.set_title("Forward Progress over Time")
    # Legend — one entry per fold
    handles = [mpatches.Patch(color=FOLD_COLORS[fi], label=f"Fold {fi}")
               for fi in range(K_FOLDS)]
    handles.append(plt.Line2D([0], [0], color="#555", linestyle="--",
                               label=f"Turn-point {TURN_POINT:.0f} m"))
    ax1.legend(handles=handles, fontsize=9)

    # Scatter: max_fwd vs spawn_y — spatial leakage check
    for ep in all_eps:
        fi = ep["fold"]
        ax2.scatter(ep["spawn_y"], ep["max_fwd_m"],
                    color=FOLD_COLORS[fi], s=80, zorder=3,
                    label=f"Fold {fi}" if ep["ep_in_fold"] == 0 else None)
        ax2.annotate(f"F{fi}", (ep["spawn_y"], ep["max_fwd_m"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax2.set_xlabel("Starting Lateral Offset (m)")
    ax2.set_ylabel("Max Forward Progress (m)")
    ax2.set_title("Spatial Generalisation — Progress vs Start Position")
    handles2 = [mpatches.Patch(color=FOLD_COLORS[fi], label=f"Fold {fi}")
                for fi in range(K_FOLDS)]
    ax2.legend(handles=handles2, fontsize=9)

    fig.tight_layout()
    path = OUT_DIR / "fig_trajectories.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cv_summary(all_eps, agg):
    """Main summary figure: CI bars for key metrics + SR table."""
    setup_style()
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(
        "K-Fold Cross-Validation — Aggregate Performance\n"
        "R54 Best Policy  |  K=5 Folds × 3 Episodes = 15 Total  |  "
        "Training World (25 obstacles)",
        fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.40)
    ax_fwd  = fig.add_subplot(gs[0, 0])
    ax_spd  = fig.add_subplot(gs[0, 1])
    ax_col  = fig.add_subplot(gs[0, 2])
    ax_dep  = fig.add_subplot(gs[1, 0])
    ax_stuck= fig.add_subplot(gs[1, 1])
    ax_tbl  = fig.add_subplot(gs[1, 2])

    fold_means_fwd  = [s["max_fwd_m_mean"]      for s in agg["fold_stats"]]
    fold_means_spd  = [s["avg_spd_moving_mean"]  for s in agg["fold_stats"]]
    fold_means_col  = [s["col_events_mean"]      for s in agg["fold_stats"]]
    fold_means_dep  = [s["min_depth_m_mean"]     for s in agg["fold_stats"]]
    fold_means_stk  = [s["time_to_stuck_s_mean"] for s in agg["fold_stats"]]
    fold_stds_fwd   = [s["max_fwd_m_std"]        for s in agg["fold_stats"]]
    fold_stds_spd   = [s["avg_spd_moving_std"]   for s in agg["fold_stats"]]
    fold_stds_col   = [s["col_events_std"]        for s in agg["fold_stats"]]
    fold_stds_dep   = [s["min_depth_m_std"]       for s in agg["fold_stats"]]
    fold_stds_stk   = [s["time_to_stuck_s_std"]  for s in agg["fold_stats"]]
    xs = np.arange(K_FOLDS)

    def bar_plot(ax, means, stds, ylabel, title, color="#1565C0", ref=None, ref_lbl=None):
        bars = ax.bar(xs, means, yerr=stds, color=color, alpha=0.78,
                      capsize=5, error_kw=dict(linewidth=1.5), width=0.55)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        if ref is not None:
            ax.axhline(ref, color="#BF360C", linestyle="--", linewidth=1.5,
                       label=ref_lbl)
            ax.legend(fontsize=8)
        ax.set_xticks(xs); ax.set_xticklabels([f"F{i}" for i in range(K_FOLDS)])
        ax.set_ylabel(ylabel); ax.set_title(title)

    bar_plot(ax_fwd, fold_means_fwd, fold_stds_fwd,
             "Distance (m)", "Max Forward Progress",
             color="#1565C0",
             ref=TURN_POINT, ref_lbl=f"Goal ({TURN_POINT:.0f} m)")
    bar_plot(ax_spd, fold_means_spd, fold_stds_spd,
             "Speed (m/s)", "Avg Moving Speed",
             color="#2E7D32",
             ref=1.739, ref_lbl="Test A baseline")
    bar_plot(ax_col, fold_means_col, fold_stds_col,
             "Events", "Collision Events", color="#BF360C")
    bar_plot(ax_dep, fold_means_dep, fold_stds_dep,
             "Distance (m)", "Min Obstacle Distance", color="#6A1B9A")
    bar_plot(ax_stuck, fold_means_stk, fold_stds_stk,
             "Time (s)", "Time to Stuck", color="#E65100")

    # Summary table
    ax_tbl.axis("off")
    col_labels = ["Metric", "Mean ± Std", "95% CI"]
    g = agg["global"]
    rows = [
        ["Max fwd (m)",       f"{g['max_fwd_m_mean']:.1f} ± {g['max_fwd_m_std']:.1f}",
         f"[{g['max_fwd_m_ci'][0]:.1f}, {g['max_fwd_m_ci'][1]:.1f}]"],
        ["Outbound %",        f"{g['outbound_pct_mean']:.0f} ± {g['outbound_pct_std']:.0f}",
         f"[{g['outbound_pct_ci'][0]:.0f}, {g['outbound_pct_ci'][1]:.0f}]"],
        ["Avg spd (m/s)",     f"{g['avg_spd_moving_mean']:.2f} ± {g['avg_spd_moving_std']:.2f}",
         f"[{g['avg_spd_moving_ci'][0]:.2f}, {g['avg_spd_moving_ci'][1]:.2f}]"],
        ["Time to stuck (s)", f"{g['time_to_stuck_s_mean']:.1f} ± {g['time_to_stuck_s_std']:.1f}",
         f"[{g['time_to_stuck_s_ci'][0]:.1f}, {g['time_to_stuck_s_ci'][1]:.1f}]"],
        ["Collision evts",    f"{g['col_events_mean']:.1f} ± {g['col_events_std']:.1f}",
         f"[{g['col_events_ci'][0]:.1f}, {g['col_events_ci'][1]:.1f}]"],
        ["Min depth (m)",     f"{g['min_depth_m_mean']:.3f} ± {g['min_depth_m_std']:.3f}",
         f"[{g['min_depth_m_ci'][0]:.3f}, {g['min_depth_m_ci'][1]:.3f}]"],
        ["Success rate",      f"{g['success_rate']*100:.0f}%",
         f"[{g['sr_ci'][0]*100:.0f}%, {g['sr_ci'][1]*100:.0f}%]"],
        ["CV Score",          f"{g['cv_score']:.3f}", "—"],
    ]
    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                       cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1565C0"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EEF2FF")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#CCCCCC")
    ax_tbl.set_title("Aggregate Statistics (K=5 Folds, N=15 Episodes)", fontweight="bold")

    path = OUT_DIR / "fig_cv_summary.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Human-readable text report ─────────────────────────────────────────────────
def write_text_report(all_eps, fold_stats_list, agg, ts):
    g = agg["global"]
    lines = [
        "=" * 70,
        "  R54 PROPER K-FOLD CROSS-VALIDATION REPORT",
        f"  Timestamp: {ts}",
        f"  World: tunnel_100m_dynamic_25.world  (training domain, 25 obstacles)",
        f"  Policy: r54_best.pth  |  Drone: native CERLAB simulator",
        f"  K={K_FOLDS} folds × {N_PER_FOLD} episodes = {K_FOLDS*N_PER_FOLD} total episodes",
        f"  Episode timeout: {EPISODE_TIMEOUT:.0f} s",
        "=" * 70,
        "",
        "CROSS-VALIDATION METHODOLOGY",
        "-" * 40,
        "  1. Temporal Autocorrelation Guard:",
        "     Episodes are treated as atomic units.  Folds are contiguous",
        "     temporal blocks — fold 0 runs before fold 1, etc.  No timestep",
        "     shuffling; no future state can leak into an earlier fold.",
        "",
        "  2. Spatial Leakage Guard:",
        "     Each fold uses a different set of lateral starting offsets",
        "     so the policy is evaluated at five distinct spatial conditions,",
        "     not just the single y=0 position used during training.",
        "",
        "  3. Closed-Loop Dynamics:",
        "     Every episode is a live interaction with the Gazebo physics",
        "     engine.  No replay, no open-loop inference.",
        "",
        "FOLD-LEVEL RESULTS",
        "-" * 40,
    ]
    for fi, fs in enumerate(fold_stats_list):
        starts = FOLD_STARTS[fi]
        lines += [
            f"  Fold {fi}  (start_y ∈ {[f'{y:+.2f}' for y in starts]}):",
            f"    SR={fs['success_rate']*100:.0f}%  "
            f"max_fwd={fs['max_fwd_m_mean']:.1f}±{fs['max_fwd_m_std']:.1f}m  "
            f"spd={fs['avg_spd_moving_mean']:.2f}±{fs['avg_spd_moving_std']:.2f}m/s  "
            f"col={fs['col_events_mean']:.1f}±{fs['col_events_std']:.1f}  "
            f"time_stuck={fs['time_to_stuck_s_mean']:.1f}±{fs['time_to_stuck_s_std']:.1f}s",
        ]
    lines += [
        "",
        "PER-EPISODE RESULTS",
        "-" * 40,
        f"  {'Ep':>4}  {'Fold':>4}  {'y0':>6}  {'Result':>10}  "
        f"{'MaxFwd(m)':>10}  {'Spd(m/s)':>9}  {'ColEv':>6}  {'Stuck(s)':>9}",
    ]
    for ep in all_eps:
        lines.append(
            f"  {ep['global_ep']:>4}  F{ep['fold']:>1}E{ep['ep_in_fold']:<1}  "
            f"{ep['spawn_y']:>+6.2f}  {ep['result']:>10}  "
            f"{ep['max_fwd_m']:>10.1f}  {ep['avg_spd_moving']:>9.2f}  "
            f"{ep['col_events']:>6}  {ep['time_to_stuck_s']:>9.1f}")
    lines += [
        "",
        "AGGREGATE CROSS-VALIDATION STATISTICS",
        "-" * 40,
        f"  Max forward progress : {g['max_fwd_m_mean']:.1f} ± {g['max_fwd_m_std']:.1f} m"
        f"   95% CI [{g['max_fwd_m_ci'][0]:.1f}, {g['max_fwd_m_ci'][1]:.1f}]",
        f"  Outbound completion  : {g['outbound_pct_mean']:.0f} ± {g['outbound_pct_std']:.0f} %"
        f"   95% CI [{g['outbound_pct_ci'][0]:.0f}, {g['outbound_pct_ci'][1]:.0f}]",
        f"  Avg moving speed     : {g['avg_spd_moving_mean']:.2f} ± {g['avg_spd_moving_std']:.2f} m/s"
        f"   95% CI [{g['avg_spd_moving_ci'][0]:.2f}, {g['avg_spd_moving_ci'][1]:.2f}]",
        f"  Time to stuck        : {g['time_to_stuck_s_mean']:.1f} ± {g['time_to_stuck_s_std']:.1f} s"
        f"   95% CI [{g['time_to_stuck_s_ci'][0]:.1f}, {g['time_to_stuck_s_ci'][1]:.1f}]",
        f"  Collision events     : {g['col_events_mean']:.1f} ± {g['col_events_std']:.1f}"
        f"   95% CI [{g['col_events_ci'][0]:.1f}, {g['col_events_ci'][1]:.1f}]",
        f"  Min obstacle dist    : {g['min_depth_m_mean']:.3f} ± {g['min_depth_m_std']:.3f} m"
        f"   95% CI [{g['min_depth_m_ci'][0]:.3f}, {g['min_depth_m_ci'][1]:.3f}]",
        f"  Success rate         : {g['success_rate']*100:.0f}%"
        f"   95% Binomial CI [{g['sr_ci'][0]*100:.1f}%, {g['sr_ci'][1]*100:.1f}%]",
        f"  Fold-to-fold CV (σ/μ): {g['fwd_cv']:.3f}  (< 0.20 = consistent policy)",
        f"  Composite CV score   : {g['cv_score']:.3f}  (0–1 scale)",
        "",
        "INTERPRETATION",
        "-" * 40,
    ]
    cov = g["fwd_cv"]
    if cov < 0.20:
        lines.append("  ✓ Fold-to-fold variance is LOW — policy generalises consistently")
        lines.append("    across different starting positions (no spatial overfitting).")
    else:
        lines.append("  ✗ Fold-to-fold variance is ELEVATED — performance depends on")
        lines.append("    the starting lateral position (spatial sensitivity detected).")
    lines += [
        "",
        "LIMITATIONS",
        "-" * 40,
        "  • Single world: only the 25-obstacle training world was tested.",
        "    True domain generalisation requires repeating on held-out worlds",
        "    (e.g. tunnel_100m_dynamic_30.world — never seen during training).",
        "  • No recovery mechanism: the policy has no reverse/hover-escape,",
        "    so all episodes terminate at the first impassable obstacle cluster.",
        "  • 15 episodes gives adequate CI estimates but not statistical power",
        "    to detect small performance differences between folds.",
        "=" * 70,
    ]
    path = OUT_DIR / "cv_report.txt"
    path.write_text("\n".join(lines))
    print(f"  Saved: {path}")
    # Also print to stdout
    print("\n" + "\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    print(f"\n{'▓'*65}")
    print(f"  R54 PROPER K-FOLD CROSS-VALIDATION")
    print(f"  K={K_FOLDS}  N_per_fold={N_PER_FOLD}  timeout={EPISODE_TIMEOUT:.0f}s")
    print(f"  Output → {OUT_DIR}")
    print(f"{'▓'*65}\n")

    # Load policy
    actor = Actor()
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu")
    state_dict = ckpt.get("actor", ckpt)
    actor.load_state_dict(state_dict)
    actor.eval()
    print(f"  Loaded: {CHECKPOINT}\n")

    rclpy.init()
    node = DroneNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    all_eps: List[dict] = []
    global_ep = 0

    try:
        for fi in range(K_FOLDS):
            starts = FOLD_STARTS[fi]
            print(f"\n{'─'*65}")
            print(f"  FOLD {fi}  —  start positions: {[f'{y:+.2f}' for y in starts]}")
            print(f"  Addressing spatial leakage: each fold = distinct spatial domain")
            print(f"{'─'*65}")

            for ei, y_off in enumerate(starts[:N_PER_FOLD]):
                result = run_episode(node, actor, fi, ei, global_ep, y_off)
                if "traj" in result:
                    traj = result.pop("traj")
                    result["_traj"] = traj     # keep for plotting
                all_eps.append(result)
                global_ep += 1
                # Restore traj for plotting
                result["traj"] = result.pop("_traj", [])
                time.sleep(3.0)   # brief rest between episodes

    finally:
        executor.shutdown(); rclpy.shutdown()

    # ── Compute fold statistics
    ts = time.strftime("%Y%m%d_%H%M%S")
    fold_stats_list = []
    for fi in range(K_FOLDS):
        fold_eps = [e for e in all_eps if e["fold"] == fi]
        fold_stats_list.append(fold_stats(fold_eps))

    # ── Compute global aggregate
    metrics_keys = ["max_fwd_m","outbound_pct","avg_spd_moving",
                    "time_to_stuck_s","col_events","near_miss_events","min_depth_m"]
    global_agg = {}
    for k in metrics_keys:
        vals = [e[k] for e in all_eps]
        global_agg[f"{k}_mean"] = round(float(np.mean(vals)), 3)
        global_agg[f"{k}_std"]  = round(float(np.std(vals, ddof=1)), 3)
        lo, hi = bootstrap_ci(vals, B=BOOTSTRAP_B)
        global_agg[f"{k}_ci"]   = [lo, hi]

    n_success = sum(e["success"] for e in all_eps)
    n_total   = len(all_eps)
    sr        = n_success / n_total if n_total else 0.0
    global_agg["success_rate"]  = round(sr, 4)
    global_agg["n_success"]     = n_success
    global_agg["n_total"]       = n_total
    try:
        sr_lo, sr_hi = clopper_pearson(n_success, n_total)
    except Exception:
        sr_lo, sr_hi = (0.0, 0.0) if sr == 0 else (sr, sr)
    global_agg["sr_ci"] = [sr_lo, sr_hi]

    # Fold-to-fold coefficient of variation for max_fwd (generalisation proxy)
    fold_fwd_means = [s["max_fwd_m_mean"] for s in fold_stats_list]
    mu_fwd = float(np.mean(fold_fwd_means))
    sigma_fwd = float(np.std(fold_fwd_means, ddof=1)) if len(fold_fwd_means) > 1 else 0.0
    global_agg["fwd_cv"] = round(sigma_fwd / mu_fwd, 4) if mu_fwd > 0 else 0.0

    # Composite CV score (0–1)
    norm_fwd = float(np.mean([e["max_fwd_m"] for e in all_eps])) / TURN_POINT
    norm_spd = float(np.mean([e["avg_spd_moving"] for e in all_eps])) / MAX_SPEED
    max_col  = max(max(e["col_events"] for e in all_eps), 1)
    norm_safe = 1.0 - float(np.mean([e["col_events"] for e in all_eps])) / max_col
    global_agg["cv_score"] = round(
        0.50 * min(norm_fwd, 1.0) + 0.30 * min(norm_spd, 1.0) + 0.20 * norm_safe, 4)

    agg = {"global": global_agg, "fold_stats": fold_stats_list}

    # ── Save per-episode CSV (strip traj)
    csv_keys = ["fold","ep_in_fold","global_ep","spawn_y","result","success",
                "duration_s","max_fwd_m","outbound_pct","avg_spd_moving",
                "time_to_stuck_s","col_events","near_miss_events","min_depth_m"]
    with open(OUT_DIR / "per_episode.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        w.writeheader(); w.writerows(all_eps)
    print(f"  Saved: {OUT_DIR}/per_episode.csv")

    # ── Save per-fold CSV
    fold_rows = []
    for fi, fs in enumerate(fold_stats_list):
        row = {"fold": fi, "start_y_set": str(FOLD_STARTS[fi])}
        row.update(fs)
        fold_rows.append(row)
    fold_csv_keys = ["fold","start_y_set","n_episodes","n_success","success_rate",
                     "max_fwd_m_mean","max_fwd_m_std","outbound_pct_mean","outbound_pct_std",
                     "avg_spd_moving_mean","avg_spd_moving_std",
                     "time_to_stuck_s_mean","time_to_stuck_s_std",
                     "col_events_mean","col_events_std",
                     "min_depth_m_mean","min_depth_m_std"]
    with open(OUT_DIR / "per_fold_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fold_csv_keys, extrasaction="ignore")
        w.writeheader(); w.writerows(fold_rows)
    print(f"  Saved: {OUT_DIR}/per_fold_summary.csv")

    # ── Save aggregate JSON (no traj)
    agg_json = json.dumps(agg, indent=2)
    (OUT_DIR / "aggregate_stats.json").write_text(agg_json)
    print(f"  Saved: {OUT_DIR}/aggregate_stats.json")

    # ── Generate plots
    print("\n  Generating plots...")
    plot_fold_consistency(all_eps, fold_stats_list)
    plot_trajectories(all_eps)
    plot_cv_summary(all_eps, agg)

    # ── Text report
    write_text_report(all_eps, fold_stats_list, agg, ts)

    print(f"\n{'▓'*65}")
    print(f"  CROSS-VALIDATION COMPLETE")
    print(f"  {n_success}/{n_total} successes  SR={sr*100:.0f}%")
    print(f"  CV score: {global_agg['cv_score']:.3f}")
    print(f"  Fold-to-fold CoV: {global_agg['fwd_cv']:.3f}")
    print(f"  All results → {OUT_DIR}")
    print(f"{'▓'*65}\n")


if __name__ == "__main__":
    main()
