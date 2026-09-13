#!/usr/bin/env python3
"""
R54 Demonstration World — 10 Test Flights (native CERLAB drone)
===============================================================
Runs exactly 10 episodes in tunnel_100m_demonstration.world using the
best R54 policy (r54_best.pth).  Records per-episode metrics and writes
a consolidated JSON + CSV summary.

Pre-requisite (in a separate terminal):
  cd /home/makhosazana/Project/CERLAB-UAV-Autonomy
  source /opt/ros/humble/setup.bash && source install/setup.bash
  ros2 launch uav_simulator tunnel_drl.launch.py gui:=true \
      world:=$(pwd)/src/uav_simulator/worlds/tunnel/tunnel_100m_demonstration.world

Usage:
  python3 "Final Results/testing/run_demo_10flights.py"
"""

import csv, json, math, os, sys, threading, time
from pathlib import Path

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
OUT_DIR    = Path(__file__).resolve().parent / "demonstration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── R54 constants (identical to training) ──────────────────────────────────────
TUNNEL_LENGTH    = 100.0
TURN_POINT       = 95.0
CORRIDOR_HW      = 3.5
SAFE_ALT         = 1.5
MAX_ALT          = 2.8
MIN_ALT          = 0.6
ALT_FLOOR        = 0.3
ALT_CEIL         = 4.5
MAX_SPEED        = 5.5
TEST_MAX_SPEED   = 4.5     # conservative cap for evaluation
COLLISION_DIST   = 0.35
NEAR_MISS_DIST   = 0.8
PROX_ALARM_M     = 2.0
AVOID_START      = 5.0
ACTION_SMOOTH_FWD = 0.35
ACTION_SMOOTH_LAT = 0.30
TICK_DT          = 0.1
NUM_H_SECTORS    = 9
DEPTH_PERCENTILE = 5
DEPTH_FAR        = 10.0
OBS_NOISE_STD    = 0.06
SPAWN_Y_JITTER   = 0.25     # slightly wider than Test A to stress-test
STATE_DIM        = 21
ACTION_DIM       = 3
N_FLIGHTS        = 10
EP_TIMEOUT       = 90.0     # 90 s max per episode (matches K-fold CV)

_SECTOR_BIAS = np.linspace(1.0, -1.0, NUM_H_SECTORS, dtype=np.float32)


# ── Rising-edge collision counter (avoids inflation when drone is stuck) ───────
class EventCounter:
    """Counts unique collision-entry events (rising-edge only)."""
    def __init__(self, threshold):
        self.thr = threshold
        self._in_event = False
        self.count = 0

    def update(self, value) -> bool:
        triggered = (value < self.thr)
        if triggered and not self._in_event:
            self._in_event = True; self.count += 1; return True
        elif not triggered:
            self._in_event = False
        return False

    def reset(self):
        self._in_event = False; self.count = 0


# ── R54 Actor (must match training exactly) ────────────────────────────────────
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


# ── ROS2 node ─────────────────────────────────────────────────────────────────
class DroneNode(Node):
    def __init__(self):
        super().__init__("demo_flight_node")
        self._lock = threading.Lock()
        self.px = self.py = self.pz = self.yaw = 0.0
        self.vx = self.vy = self.vz = 0.0
        self._prev_px = self._prev_py = self._prev_pz = 0.0
        self._prev_stamp = 0.0
        self.pose_received  = False
        self.depth_sectors  = np.full(NUM_H_SECTORS, DEPTH_FAR, dtype=np.float32)
        self.depth_top      = DEPTH_FAR
        self.depth_mid      = DEPTH_FAR
        self.depth_bottom   = DEPTH_FAR
        self.depth_received = False
        self.prox_alarm     = 0

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
        sw = w // NUM_H_SECTORS
        secs = np.empty(NUM_H_SECTORS, dtype=np.float32)
        for i in range(NUM_H_SECTORS):
            c0 = i * sw
            c1 = (i+1)*sw if i < NUM_H_SECTORS-1 else w
            secs[i] = float(np.percentile(mid[:, c0:c1], DEPTH_PERCENTILE))
        cv0, cv1 = int(w*0.30), int(w*0.70)
        col = d[:, cv0:cv1]
        tb = int(h*0.33); bb = int(h*0.67)
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
        s = math.sin(yaw/2); c = math.cos(yaw/2)
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
        if dc.wait_for_service(timeout_sec=6.0):
            req = DeleteEntity.Request(); req.name = "quadcopter"
            fut = dc.call_async(req); t0 = time.time()
            while not fut.done() and time.time()-t0 < 8.0:
                ex.spin_once(timeout_sec=0.05)
        time.sleep(1.5); self.pose_received = False
        try:
            urdf = open(str(URDF_PATH)).read()
        except FileNotFoundError:
            print(f"[ERROR] URDF not found: {URDF_PATH}")
            helper.destroy_node(); return False
        if sc.wait_for_service(timeout_sec=6.0):
            req = SpawnEntity.Request(); req.name = "quadcopter"; req.xml = urdf
            req.initial_pose.position.x = float(x)
            req.initial_pose.position.y = float(y)
            req.initial_pose.position.z = float(z)
            req.initial_pose.orientation.w = 1.0
            fut = sc.call_async(req); t0 = time.time()
            while not fut.done() and time.time()-t0 < 8.0:
                ex.spin_once(timeout_sec=0.05)
        helper.destroy_node(); return True


# ── State builder (matches R54 training) ──────────────────────────────────────
def build_state(snap, spawn_y, fwd_progress, lat_offset, phase, noise_sd):
    ds = snap["depth_sectors"]
    h_noisy = np.clip(
        ds + np.random.normal(0.0, noise_sd, NUM_H_SECTORS).astype(np.float32),
        0.05, DEPTH_FAR)
    v_top = float(np.clip(snap["depth_top"]    + np.random.normal(0, noise_sd), 0.05, DEPTH_FAR))
    v_mid = float(np.clip(snap["depth_mid"]    + np.random.normal(0, noise_sd), 0.05, DEPTH_FAR))
    v_bot = float(np.clip(snap["depth_bottom"] + np.random.normal(0, noise_sd), 0.05, DEPTH_FAR))
    speed = abs(snap["vx"]) if phase == "OUTBOUND" else abs(-snap["vx"])
    return np.array([
        *[h_noisy[i]/DEPTH_FAR for i in range(NUM_H_SECTORS)],
        v_top/DEPTH_FAR, v_mid/DEPTH_FAR, v_bot/DEPTH_FAR,
        float(snap["prox_alarm"]),
        np.clip(fwd_progress/TUNNEL_LENGTH, -0.1, 1.1),
        np.clip(lat_offset/CORRIDOR_HW, -1, 1),
        np.clip((snap["pz"]-SAFE_ALT)/1.5, -1, 1),
        np.clip(speed/7.0, -1, 1),
        np.clip(snap["vy"]/3.0, -1, 1),
        np.clip(snap["vz"]/2.0, -1, 1),
        1.0 if phase == "RETURN" else 0.0,
        np.clip(TEST_MAX_SPEED/7.0, 0, 1),
    ], dtype=np.float32)


# ── Action → delta position ────────────────────────────────────────────────────
def apply_action(smoothed, snap, spawn_y, lat_offset, phase):
    raw_fwd, raw_lat, raw_alt = smoothed
    fwd_speed = float(np.clip((raw_fwd*0.5+0.5)*TEST_MAX_SPEED, 1.0, MAX_SPEED))
    lat_speed = float(raw_lat*0.8)
    alt_adj   = float(raw_alt*0.4)

    ds = snap["depth_sectors"]
    front_d = float(np.min([ds[3], ds[4], ds[5]]))
    eff = list(ds) if phase == "OUTBOUND" else list(reversed(ds))

    # Smooth proportional avoidance
    smooth_push = 0.0
    for i in range(NUM_H_SECTORS):
        d_i = float(eff[i])
        if d_i < AVOID_START:
            w = ((AVOID_START-d_i)/AVOID_START)**2
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
    if abs(lat_offset) < CORRIDOR_HW*0.85:
        alt_adj = 0.5*alt_adj + 0.5*float(np.clip(-alt_err*3.0, -1.0, 1.0))

    if front_d < 0.15:
        fwd_speed = 0.0

    direction = -1.0 if phase == "RETURN" else 1.0
    return (direction*fwd_speed*TICK_DT, lat_speed*TICK_DT,
            float(np.clip(alt_adj*TICK_DT, -0.2, 0.2))), front_d


# ── Single episode ──────────────────────────────────────────────────────────────
def run_episode(node: DroneNode, actor: Actor, flight_n: int,
                noise_sd: float, spawn_y_offset: float) -> dict:
    print(f"\n{'='*65}")
    print(f"  FLIGHT {flight_n:2d}/{N_FLIGHTS}  │  "
          f"noise_sd={noise_sd:.3f}  spawn_y={spawn_y_offset:+.3f} m")
    print(f"{'='*65}")

    # Wait for sensors
    print("[DEMO] Waiting for sensors...")
    t0 = time.time()
    while not (node.pose_received and node.depth_received):
        time.sleep(0.2)
        if time.time()-t0 > 35.0:
            return {"flight": flight_n, "result": "SENSOR_TIMEOUT"}

    # Teleport to tunnel entrance
    print(f"[DEMO] Teleporting to X=0, Y={spawn_y_offset:+.3f} ...")
    node.teleport(x=0.0, y=spawn_y_offset, z=0.2)
    time.sleep(2.5)

    t0 = time.time()
    while not node.pose_received and time.time()-t0 < 10.0:
        time.sleep(0.1)

    snap = node.snap()
    spawn_x = snap["px"]; spawn_y = snap["py"]
    tgt_x = spawn_x; tgt_y = spawn_y; tgt_z = SAFE_ALT
    prev_action = np.zeros(3, dtype=np.float32)
    delayed_a   = np.zeros(3, dtype=np.float32)

    # Takeoff
    print("[DEMO] Taking off...")
    node.takeoff()
    t0 = time.time()
    while True:
        snap = node.snap()
        if snap["pz"] >= SAFE_ALT - 0.15:
            print(f"[DEMO] Alt OK: {snap['pz']:.2f} m"); break
        node.publish_setpoint(spawn_x, spawn_y, SAFE_ALT)
        time.sleep(0.1)
        if time.time()-t0 > 20.0:
            return {"flight": flight_n, "result": "TAKEOFF_TIMEOUT"}
    time.sleep(0.8)

    # Per-episode CSV
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"demo_flight{flight_n:02d}_{ts}.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=[
        "step","phase","t_s","x_m","y_m","z_m",
        "vx_ms","vy_ms","vz_ms","fwd_m","lat_m",
        "speed_ms","front_d_m","min_depth_m",
        "prox_alarm","depth_top","depth_mid","depth_bot",
        "tgt_x","tgt_y","tgt_z","collision","near_miss",
    ])
    writer.writeheader()

    # Episode loop
    ep_start   = time.time()
    phase      = "OUTBOUND"
    step       = 0
    speeds     = []
    col_ctr    = EventCounter(COLLISION_DIST)
    nm_ctr     = EventCounter(NEAR_MISS_DIST)
    min_d_ever = DEPTH_FAR
    max_fwd    = 0.0
    result_label = "TIMEOUT"
    _stuck_steps = 0
    _kick_sign   = 1.0

    while True:
        snap = node.snap()
        t_el = time.time() - ep_start

        ds       = snap["depth_sectors"]
        fwd      = (snap["px"]-spawn_x) if phase == "OUTBOUND" else (spawn_x-snap["px"])
        lat_off  = snap["py"] - spawn_y
        speed    = abs(snap["vx"])
        speeds.append(speed)
        if phase == "OUTBOUND":
            max_fwd = max(max_fwd, fwd)

        min_d   = float(np.min(ds))
        front_d = float(np.min([ds[3], ds[4], ds[5]]))
        min_d_ever = min(min_d_ever, min_d)
        is_col = col_ctr.update(min_d)
        is_nm  = nm_ctr.update(min_d) and not is_col
        col_count = col_ctr.count; nm_count = nm_ctr.count

        # Infer (1-step delay)
        state = build_state(snap, spawn_y, fwd, lat_off, phase, noise_sd)
        with torch.no_grad():
            raw = actor(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
        exec_a  = delayed_a.copy(); delayed_a = raw.copy()
        smoothed = np.array([
            ACTION_SMOOTH_FWD*exec_a[0] + (1-ACTION_SMOOTH_FWD)*prev_action[0],
            ACTION_SMOOTH_LAT*exec_a[1] + (1-ACTION_SMOOTH_LAT)*prev_action[1],
            ACTION_SMOOTH_LAT*exec_a[2] + (1-ACTION_SMOOTH_LAT)*prev_action[2],
        ], dtype=np.float32)
        prev_action = smoothed.copy()

        (dx, dy, dz), front_d = apply_action(smoothed, snap, spawn_y, lat_off, phase)

        # ── Stuck recovery: teleport drone 2 m back when pinned ─────────────────
        if front_d < 0.25 and speed < 0.05:
            _stuck_steps += 1
        else:
            _stuck_steps = 0
        if _stuck_steps >= 20:
            _kick_sign  *= -1.0
            _stuck_steps = 0
            _dir  = 1.0 if phase == "OUTBOUND" else -1.0
            rec_x = float(np.clip(snap["px"] - _dir * 2.5,
                                  spawn_x - 1.0, spawn_x + TURN_POINT))
            rec_y = float(np.clip(snap["py"] + _kick_sign * 1.2,
                                  -CORRIDOR_HW + 0.4, CORRIDOR_HW - 0.4))
            print(f"  [RECOVERY] teleport to x={rec_x:.1f} y={rec_y:.2f}")
            csv_f.flush()
            node.teleport(x=rec_x, y=rec_y, z=0.3)
            time.sleep(2.5)
            node.takeoff()
            t_rec = time.time()
            while time.time() - t_rec < 12.0:
                s2 = node.snap()
                if s2["pz"] >= SAFE_ALT - 0.2:
                    break
                node.publish_setpoint(rec_x, rec_y, SAFE_ALT)
                time.sleep(0.1)
            tgt_x = rec_x; tgt_y = rec_y; tgt_z = SAFE_ALT
            prev_action[:] = 0.0; delayed_a[:] = 0.0
            continue
        # ─────────────────────────────────────────────────────────────────────────

        tgt_x = float(np.clip(tgt_x+dx, spawn_x-3.0, spawn_x+TURN_POINT+5.0))
        tgt_y = float(np.clip(tgt_y+dy, -CORRIDOR_HW, CORRIDOR_HW))
        tgt_z = float(np.clip(tgt_z+dz, MIN_ALT, ALT_CEIL))
        yaw   = 0.0 if phase == "OUTBOUND" else math.pi

        node.publish_setpoint(tgt_x, tgt_y, tgt_z, yaw)

        writer.writerow({
            "step":step, "phase":phase,
            "t_s":round(t_el,3),
            "x_m":round(snap["px"],3), "y_m":round(snap["py"],3),
            "z_m":round(snap["pz"],3),
            "vx_ms":round(snap["vx"],3), "vy_ms":round(snap["vy"],3),
            "vz_ms":round(snap["vz"],3),
            "fwd_m":round(fwd,2), "lat_m":round(lat_off,3),
            "speed_ms":round(speed,3),
            "front_d_m":round(front_d,3), "min_depth_m":round(min_d,3),
            "prox_alarm":snap["prox_alarm"],
            "depth_top":round(snap["depth_top"],3),
            "depth_mid":round(snap["depth_mid"],3),
            "depth_bot":round(snap["depth_bottom"],3),
            "tgt_x":round(tgt_x,3), "tgt_y":round(tgt_y,3),
            "tgt_z":round(tgt_z,3),
            "collision":int(is_col), "near_miss":int(is_nm),
        })

        if step % 30 == 0:
            print(f"  [{phase:8s}|{t_el:5.1f}s] "
                  f"fwd={fwd:5.1f}m  z={snap['pz']:.2f}m  "
                  f"spd={speed:.2f}m/s  front={front_d:.2f}m  "
                  f"col={col_count}  nm={nm_count}")

        # Hold setpoint for tick
        t_tick = time.time()
        while time.time()-t_tick < TICK_DT:
            node.publish_setpoint(tgt_x, tgt_y, tgt_z, yaw)
            time.sleep(0.02)
        step += 1

        # Phase transitions
        if phase == "OUTBOUND" and (snap["px"]-spawn_x) >= TURN_POINT:
            phase = "RETURN"
            tgt_x = spawn_x + TURN_POINT
            print(f"\n  ↩ OUTBOUND COMPLETE — x={snap['px']-spawn_x:.1f}m  t={t_el:.1f}s")
            time.sleep(0.3)

        if phase == "RETURN" and (snap["px"]-spawn_x) <= 2.0:
            result_label = "SUCCESS"
            print(f"\n  ✓ ROUND-TRIP SUCCESS — t={t_el:.1f}s"); break

        if t_el >= EP_TIMEOUT:
            result_label = "TIMEOUT"
            print(f"\n  ✗ TIMEOUT — best_fwd={max_fwd:.1f}m"); break
        if snap["pz"] < ALT_FLOOR:
            result_label = "CRASH_ALT"
            print(f"\n  ✗ CRASH (altitude) — z={snap['pz']:.2f}m"); break
        if abs(snap["py"]-spawn_y) > CORRIDOR_HW+0.5:
            result_label = "CRASH_LAT"
            print(f"\n  ✗ CRASH (lateral) — y_off={snap['py']-spawn_y:.2f}m"); break

    csv_f.close()
    duration   = time.time() - ep_start
    avg_speed  = float(np.mean(speeds)) if speeds else 0.0
    peak_speed = float(np.max(speeds)) if speeds else 0.0

    return {
        "flight":         flight_n,
        "result":         result_label,
        "duration_s":     round(duration, 1),
        "max_fwd_m":      round(max_fwd, 2),
        "avg_speed_ms":   round(avg_speed, 3),
        "peak_speed_ms":  round(peak_speed, 3),
        "collisions":     col_count,
        "near_misses":    nm_count,
        "min_depth_m":    round(min_d_ever, 3),
        "dr_noise_sd":    round(noise_sd, 4),
        "dr_spawn_y_m":   round(spawn_y_offset, 4),
        "csv_log":        str(csv_path),
        "steps":          step,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*65)
    print("  R54 BEST POLICY — DEMONSTRATION WORLD — 10 FLIGHTS")
    print("  World : tunnel_100m_demonstration.world")
    print("  Drone : CERLAB quadcopter (native sim, non-PX4)")
    print("  Policy: r54_best.pth  (21D state, 512→512→256 + LN + skip)")
    print("="*65)

    if not CHECKPOINT.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {CHECKPOINT}")
    ckpt  = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    actor = Actor(); actor.load_state_dict(ckpt["actor"]); actor.eval()
    saved_ep = ckpt.get("episode", "?")
    saved_sr = ckpt.get("sr", "?")
    print(f"\n  Loaded: r54_best.pth  │  episode={saved_ep}  SR={saved_sr}")
    print(f"  Test speed cap: {TEST_MAX_SPEED} m/s")

    rclpy.init()
    node = DroneNode()
    ex   = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    spin_thread = threading.Thread(target=ex.spin, daemon=True)
    spin_thread.start()

    np.random.seed(int(time.time()) % 100000)

    all_results = []
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    for fi in range(1, N_FLIGHTS + 1):
        noise_sd    = float(np.random.uniform(0.03, OBS_NOISE_STD))
        spawn_y_off = float(np.random.uniform(-SPAWN_Y_JITTER, SPAWN_Y_JITTER))
        r = run_episode(node, actor, fi, noise_sd, spawn_y_off)
        all_results.append(r)

        icon = "✓" if r["result"] == "SUCCESS" else "✗"
        print(f"\n  {icon} Flight {fi:2d}: {r['result']:12s}  "
              f"fwd={r.get('max_fwd_m',0):.1f}m  "
              f"spd={r.get('avg_speed_ms',0):.2f}m/s  "
              f"col={r.get('collisions',0)}  nm={r.get('near_misses',0)}")

        if fi < N_FLIGHTS:
            print(f"\n  [Pausing 4 s before next flight...]\n")
            time.sleep(4.0)

    ex.shutdown(); rclpy.shutdown()

    # ── Aggregate statistics ───────────────────────────────────────────────────
    successes   = [r for r in all_results if r.get("result") == "SUCCESS"]
    sr          = len(successes) / N_FLIGHTS
    fwds        = [r.get("max_fwd_m", 0) for r in all_results]
    spds        = [r.get("avg_speed_ms", 0) for r in all_results]
    cols        = [r.get("collisions", 0) for r in all_results]
    nms         = [r.get("near_misses", 0) for r in all_results]
    min_depths  = [r.get("min_depth_m", 10) for r in all_results]
    durations   = [r.get("duration_s", 0) for r in all_results if r.get("result") == "SUCCESS"]

    agg = {
        "world":            "tunnel_100m_demonstration",
        "drone":            "CERLAB quadcopter (non-PX4)",
        "policy":           "r54_best.pth",
        "policy_episode":   saved_ep,
        "policy_sr":        saved_sr,
        "run_timestamp":    run_ts,
        "n_flights":        N_FLIGHTS,
        "success_count":    len(successes),
        "success_rate":     round(sr, 3),
        "avg_fwd_progress_m": round(float(np.mean(fwds)), 2),
        "std_fwd_progress_m": round(float(np.std(fwds)), 2),
        "avg_speed_ms":     round(float(np.mean(spds)), 3),
        "peak_speed_ms":    round(float(np.max([r.get("peak_speed_ms",0) for r in all_results])), 3),
        "avg_collisions":   round(float(np.mean(cols)), 2),
        "total_near_misses":int(np.sum(nms)),
        "min_depth_ever_m": round(float(np.min(min_depths)), 3),
        "avg_duration_s_success": round(float(np.mean(durations)), 1) if durations else None,
        "flights":          all_results,
    }

    # Save JSON
    json_path = OUT_DIR / f"demo_10flights_summary_{run_ts}.json"
    with open(json_path, "w") as f:
        json.dump(agg, f, indent=2)

    # Save flat CSV summary
    flat_csv = OUT_DIR / f"demo_10flights_flat_{run_ts}.csv"
    with open(flat_csv, "w", newline="") as f:
        keys = [k for k in all_results[0].keys() if k != "csv_log"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k, "") for k in keys})

    # ── Final printout ──────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  DEMONSTRATION — 10-FLIGHT SUMMARY")
    print("="*65)
    print(f"  World  : tunnel_100m_demonstration (20 obstacles, novel visuals)")
    print(f"  Drone  : CERLAB quadcopter (non-PX4)")
    print(f"  Policy : r54_best.pth  (ep={saved_ep}, train SR={saved_sr})")
    print(f"  {'─'*58}")
    print(f"  {'Metric':<30}  {'Value'}")
    print(f"  {'─'*58}")
    print(f"  {'Success rate':<30}  {len(successes)}/{N_FLIGHTS}  ({sr*100:.0f}%)")
    print(f"  {'Avg fwd progress':<30}  {np.mean(fwds):.1f} ± {np.std(fwds):.1f} m")
    print(f"  {'Avg moving speed':<30}  {np.mean(spds):.3f} m/s")
    print(f"  {'Peak speed (any flight)':<30}  {max(r.get('peak_speed_ms',0) for r in all_results):.3f} m/s")
    print(f"  {'Avg collisions/flight':<30}  {np.mean(cols):.2f}")
    print(f"  {'Total near-misses':<30}  {int(np.sum(nms))}")
    print(f"  {'Min obstacle distance':<30}  {np.min(min_depths):.3f} m")
    if durations:
        print(f"  {'Avg successful duration':<30}  {np.mean(durations):.1f} s")
    print(f"  {'─'*58}")
    print(f"\n  Per-flight results:")
    for r in all_results:
        icon = "✓" if r["result"] == "SUCCESS" else "✗"
        print(f"   {icon} F{r['flight']:02d}: {r['result']:12s}  "
              f"fwd={r.get('max_fwd_m',0):5.1f}m  "
              f"spd={r.get('avg_speed_ms',0):.2f}  "
              f"col={r.get('collisions',0)}  nm={r.get('near_misses',0)}  "
              f"min_d={r.get('min_depth_m',10):.2f}m")
    print(f"\n  Results saved to: {OUT_DIR}")
    print(f"  JSON summary : {json_path.name}")
    print(f"  Flat CSV     : {flat_csv.name}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
