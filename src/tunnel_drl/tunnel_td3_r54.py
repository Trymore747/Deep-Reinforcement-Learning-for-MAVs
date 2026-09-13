#!/usr/bin/env python3
"""
Tunnel TD3 R54 — Improved Sensor & Architecture
================================================
Based on v5r (54.3% SR baseline) with targeted improvements:

KEY CHANGES vs v5r:
  1. 9 horizontal depth sectors (was 5)  → ~10°/sector resolution
  2. 5th-percentile depth (was 10th)     → captures closest pixel in sector
  3. 3 vertical zones top/mid/bottom     → was top/bottom only
  4. Proximity alarm flag in state       → hard signal when any d < 1.5m
  5. STATE_DIM = 21 (was 14)
  6. Actor: 512→512→256 + LayerNorm + residual skip (was 256→256→128)
  7. Critic: 512→512→256 + LayerNorm (was 256→256→128)
  8. Exponential proximity penalty in reward (was linear step-based)
  9. Buffer 100k (was 50k), LR_ACTOR 3e-4 (was 1e-4)

World: tunnel_100m_dynamic_20_cerlab.world (8 dyn + 12 static = 20 obstacles)
Platform: CERLAB Quadcopter (non-PX4), position-setpoint control
"""

from __future__ import annotations

import argparse, json, math, os, signal, sys, time, threading
from collections import deque
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty
from gazebo_msgs.srv import DeleteEntity, SpawnEntity

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

TUNNEL_LENGTH   = 100.0
TURN_POINT      = 95.0
CORRIDOR_HW     = 3.5
LAT_LIMIT       = 3.8

SAFE_ALT        = 1.5
MAX_ALT         = 2.8
MIN_ALT         = 0.6
ALT_FLOOR       = 0.3
ALT_CEIL        = 4.5

COLLISION_DIST  = 0.35
NEAR_MISS_DIST  = 0.8
DANGER_DIST     = 2.5        # wider danger zone (was 1.5)
SAFE_DIST       = 3.0

MAX_SPEED       = 5.5
SPEED_STAGES    = [2.0, 3.0, 4.0, 4.5, 5.0]
SR_THRESHOLDS   = [0.55, 0.65, 0.72, 0.80, 0.85]  # SR needed to advance FROM each stage
MIN_MISSIONS_STAGE = 20                              # minimum missions at stage before advancing

ACTION_SMOOTH_FWD = 0.35
ACTION_SMOOTH_LAT = 0.30     # faster lateral response (was 0.15 — too sluggish)

COLLISION_PENALTY_CAP = 999  # effectively no cap — every collision always penalized

# ── Generalization / domain randomization ────────────────────────────────────
OBS_NOISE_STD  = 0.06   # Gaussian noise (m) on depth obs — sensor robustness (conservative)
SPAWN_Y_JITTER = 0.20   # ±m lateral spawn offset per episode — breaks path memorization
SPEED_JITTER   = 0.15   # ±m/s speed variation per episode — prevents stage-specific habits

# ── R54-specific depth parameters ──
NUM_H_SECTORS   = 9          # was 5 — finer resolution
DEPTH_PERCENTILE = 5         # was 10 — more sensitive
PROX_ALARM_M    = 2.0        # wider proximity alarm (was 1.5)
PROX_EXP_K      = 1.0        # wider penalty decay field (was 0.5)
PROX_EXP_SCALE  = 150.0      # stronger proximity penalty (was 80.0)

# Sector lateral bias: index 0 = world +Y (left outbound), index 8 = world -Y
_SECTOR_BIAS = np.linspace(1.0, -1.0, NUM_H_SECTORS, dtype=np.float32)

GAMMA           = 0.99
TAU             = 0.005
LR_ACTOR        = 3e-4       # was 1e-4
LR_CRITIC       = 3e-4
BUFFER_SIZE     = 100_000    # was 50k
BATCH_SIZE      = 256
POLICY_DELAY    = 2
EXPLORE_NOISE_INIT = 0.20
EXPLORE_NOISE_MIN  = 0.05
TARGET_NOISE       = 0.2
TARGET_CLIP        = 0.5
WARMUP_STEPS       = 3000
UPDATES_PER_STEP   = 4

TICK_DT         = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch device: {DEVICE}")

STATE_DIM  = 21   # 9H + 3V + 1alarm + 3pos + 3vel + phase + speed
ACTION_DIM = 3

# ═══════════════════════════════════════════════════════════════════════════
# NEURAL NETWORKS — R54 improved architecture
# ═══════════════════════════════════════════════════════════════════════════

class Actor(nn.Module):
    """512→512→256, LayerNorm in actor + residual skip between layers 1 and 2."""
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.fc1   = nn.Linear(state_dim, 512)
        self.ln1   = nn.LayerNorm(512)
        self.fc2   = nn.Linear(512, 512)
        self.ln2   = nn.LayerNorm(512)
        self.skip  = nn.Linear(512, 512)   # residual projection
        self.fc3   = nn.Linear(512, 256)
        self.ln3   = nn.LayerNorm(256)
        self.fc4   = nn.Linear(256, action_dim)
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc4.bias,   -3e-3, 3e-3)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x  = F.relu(self.ln1(self.fc1(s)))
        x  = F.relu(self.ln2(self.fc2(x)) + self.skip(x))   # residual
        x  = F.relu(self.ln3(self.fc3(x)))
        return torch.tanh(self.fc4(x))


class Critic(nn.Module):
    """Twin critic — 512→512→256, LayerNorm each layer."""
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        inp = state_dim + action_dim

        def make_q():
            return nn.ModuleList([
                nn.Linear(inp, 512), nn.LayerNorm(512),
                nn.Linear(512, 512), nn.LayerNorm(512),
                nn.Linear(512, 256), nn.LayerNorm(256),
                nn.Linear(256, 1),
            ])
        self.q1_layers = make_q()
        self.q2_layers = make_q()

    def _fwd(self, layers, x):
        fc1, ln1, fc2, ln2, fc3, ln3, out = layers
        x = F.relu(ln1(fc1(x)))
        x = F.relu(ln2(fc2(x)))
        x = F.relu(ln3(fc3(x)))
        return out(x)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self._fwd(self.q1_layers, sa), self._fwd(self.q2_layers, sa)

    def q1_only(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self._fwd(self.q1_layers, sa)


# ═══════════════════════════════════════════════════════════════════════════
# TD3 AGENT
# ═══════════════════════════════════════════════════════════════════════════

class TD3Agent:
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.actor      = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_tgt  = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_tgt.load_state_dict(self.actor.state_dict())

        self.critic     = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_tgt = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.iter = 0

        total_params = sum(p.numel() for p in self.actor.parameters()) + \
                       sum(p.numel() for p in self.critic.parameters())
        print(f"TD3-R54 Agent — {total_params:,} parameters  ({DEVICE})")
        print(f"  Actor : 512→512→256 + LayerNorm + residual")
        print(f"  Critic: 512→512→256 + LayerNorm (twin)")

    @torch.no_grad()
    def select_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        a = self.actor(s).cpu().numpy().flatten()
        if noise_scale > 0:
            a = np.clip(a + np.random.normal(0, noise_scale, self.action_dim), -1.0, 1.0)
        return a

    @torch.no_grad()
    def q_value(self, state: np.ndarray, action: np.ndarray) -> float:
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        a = torch.FloatTensor(action).unsqueeze(0).to(DEVICE)
        return self.critic.q1_only(s, a).item()

    def update(self, buffer: deque, batch_size: int = BATCH_SIZE):
        if len(buffer) < batch_size:
            return 0.0, 0.0

        indices = np.random.choice(len(buffer), batch_size, replace=False)
        batch   = [buffer[i] for i in indices]

        states  = torch.FloatTensor(np.array([b[0] for b in batch])).to(DEVICE)
        actions = torch.FloatTensor(np.array([b[1] for b in batch])).to(DEVICE)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1).to(DEVICE)
        next_st = torch.FloatTensor(np.array([b[3] for b in batch])).to(DEVICE)
        dones   = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            noise  = (torch.randn_like(actions) * TARGET_NOISE).clamp(-TARGET_CLIP, TARGET_CLIP)
            next_a = (self.actor_tgt(next_st) + noise).clamp(-1, 1)
            tq1, tq2 = self.critic_tgt(next_st, next_a)
            target_q = rewards + GAMMA * (1 - dones) * torch.min(tq1, tq2)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        actor_loss_val = 0.0
        self.iter += 1
        if self.iter % POLICY_DELAY == 0:
            actor_loss = -self.critic.q1_only(states, self.actor(states)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()
            actor_loss_val = actor_loss.item()

            for p, t in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                t.data.copy_(TAU * p.data + (1 - TAU) * t.data)
            for p, t in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                t.data.copy_(TAU * p.data + (1 - TAU) * t.data)

        return critic_loss.item(), actor_loss_val

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_tgt': self.actor_tgt.state_dict(),
            'critic_tgt': self.critic_tgt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'iter': self.iter,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, path)
        print(f"Checkpoint saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        saved_sd = ckpt.get('state_dim', None)
        # Attempt partial transfer for different state dims (e.g. loading v5r 14D)
        def merge(model_sd, saved_dict):
            new_sd = {}
            for k, v in model_sd.items():
                if k in saved_dict:
                    sv = saved_dict[k]
                    if sv.shape == v.shape:
                        new_sd[k] = sv
                    else:
                        new = v.clone()
                        if len(sv.shape) == len(v.shape):
                            sl = tuple(slice(0, min(sv.shape[i], v.shape[i])) for i in range(len(v.shape)))
                            new[sl] = sv[sl]
                        new_sd[k] = new
                        print(f"  partial-copied {k}: {tuple(sv.shape)} → {tuple(v.shape)}")
                else:
                    new_sd[k] = v
            return new_sd

        try:
            self.actor.load_state_dict(merge(self.actor.state_dict(), ckpt.get('actor', {})))
            self.critic.load_state_dict(merge(self.critic.state_dict(), ckpt.get('critic', {})))
            self.actor_tgt.load_state_dict(self.actor.state_dict())
            self.critic_tgt.load_state_dict(self.critic.state_dict())
            self.iter = ckpt.get('iter', 0)
            print(f"Loaded checkpoint: {path}  (saved_state_dim={saved_sd})")
        except Exception as e:
            print(f"Load warning: {e}  — starting fresh")


# ═══════════════════════════════════════════════════════════════════════════
# ROS2 DRONE INTERFACE — enhanced depth processing for R54
# ═══════════════════════════════════════════════════════════════════════════

def quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

def yaw_from_quat(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class CERLABDroneInterface(Node):
    def __init__(self):
        super().__init__("tunnel_drl_r54")

        self.px = 0.0; self.py = 0.0; self.pz = 0.0; self.yaw = 0.0
        self.vx = 0.0; self.vy = 0.0; self.vz = 0.0
        self.pose_received = False
        self._prev_px = 0.0; self._prev_py = 0.0; self._prev_pz = 0.0
        self._prev_stamp = 0.0

        # R54: 9 horizontal + 3 vertical depth sectors
        self.depth_sectors = np.full(NUM_H_SECTORS, 10.0, dtype=np.float32)
        self.depth_top    = 10.0
        self.depth_mid    = 10.0
        self.depth_bottom = 10.0
        self.prox_alarm   = 0       # 1 if any sector < PROX_ALARM_M

        self.setpoint_pub = self.create_publisher(PoseStamped, "/CERLAB/quadcopter/setpoint_pose", 10)
        self.takeoff_pub  = self.create_publisher(Empty, "/CERLAB/quadcopter/takeoff", 10)
        self.land_pub     = self.create_publisher(Empty, "/CERLAB/quadcopter/land", 10)
        self.posctrl_pub  = self.create_publisher(Bool, "/CERLAB/quadcopter/posctrl", 10)

        self._delete_cli = self.create_client(DeleteEntity, '/delete_entity')
        self._spawn_cli  = self.create_client(SpawnEntity,  '/spawn_entity')

        self.create_subscription(PoseStamped, "/CERLAB/quadcopter/pose_raw", self._pose_cb, 10)
        self.create_subscription(PoseStamped, "/CERLAB/quadcopter/pose",     self._pose_cb, 10)

        depth_qos = QoSProfile(depth=5,
                               reliability=ReliabilityPolicy.BEST_EFFORT,
                               durability=DurabilityPolicy.VOLATILE)
        for t in ["/camera/camera/depth/image_raw", "/camera/depth/image_raw",
                  "/camera/color/depth/image_raw"]:
            self.create_subscription(Image, t, lambda msg, _t=t: self._depth_cb(msg, _t), depth_qos)

        self.get_logger().info(f"CERLABDroneInterface R54 — {NUM_H_SECTORS}H + 3V sectors, {DEPTH_PERCENTILE}th percentile")

    def _pose_cb(self, msg: PoseStamped):
        now = time.time()
        dt  = now - self._prev_stamp if self._prev_stamp > 0 else 0.1
        self.px  = msg.pose.position.x
        self.py  = msg.pose.position.y
        self.pz  = msg.pose.position.z
        self.yaw = yaw_from_quat(msg.pose.orientation)
        if dt > 0.01:
            self.vx = (self.px - self._prev_px) / dt
            self.vy = (self.py - self._prev_py) / dt
            self.vz = (self.pz - self._prev_pz) / dt
        self._prev_px = self.px; self._prev_py = self.py; self._prev_pz = self.pz
        self._prev_stamp = now
        self.pose_received = True

    def _depth_cb(self, msg: Image, topic=None):
        if topic is not None and not hasattr(self, '_depth_topic'):
            self._depth_topic = topic
            self.get_logger().info(f"Depth on: {topic}")

        if msg.encoding == "32FC1":
            raw = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        elif msg.encoding == "16UC1":
            raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
        else:
            return

        d = raw.copy()
        far = 10.0
        d[(d <= 0.05) | ~np.isfinite(d)] = far
        h, w = d.shape

        # ── 9 horizontal sectors (full-height rows 20–80%) ──
        r0, r1 = int(h * 0.20), int(h * 0.80)
        mid = d[r0:r1, :]
        sw  = w // NUM_H_SECTORS
        for i in range(NUM_H_SECTORS):
            c0 = i * sw
            c1 = (i + 1) * sw if i < NUM_H_SECTORS - 1 else w
            self.depth_sectors[i] = float(np.percentile(mid[:, c0:c1], DEPTH_PERCENTILE))

        # ── 3 vertical zones (centre columns 25–75%) ──
        cv0, cv1 = int(w * 0.25), int(w * 0.75)
        col = d[:, cv0:cv1]
        t_bound = h // 4
        b_bound = 3 * h // 4
        self.depth_top    = float(np.percentile(col[:t_bound],  DEPTH_PERCENTILE))
        self.depth_mid    = float(np.percentile(col[t_bound:b_bound], DEPTH_PERCENTILE))
        self.depth_bottom = float(np.percentile(col[b_bound:],  DEPTH_PERCENTILE))

        # ── Proximity alarm ──
        min_d = float(np.min(self.depth_sectors))
        self.prox_alarm = 1 if min_d < PROX_ALARM_M else 0

    def publish_setpoint(self, x: float, y: float, z: float, yaw: float = 0.0):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        qx, qy, qz, qw = quat_from_yaw(yaw)
        msg.pose.orientation.x = qx; msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz; msg.pose.orientation.w = qw
        try:
            self.setpoint_pub.publish(msg)
        except Exception:
            pass   # absorb transient RCL errors; drone holds last setpoint

    def takeoff(self):
        b = Bool(); b.data = True
        for _ in range(5):
            self.posctrl_pub.publish(b); time.sleep(0.05)
        self.takeoff_pub.publish(Empty())
        self.get_logger().info("Takeoff + posctrl sent")

    def land(self):
        self.land_pub.publish(Empty())

    def start_spinning(self):
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        time.sleep(1.0)

    def stop_spinning(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown()

    def spin_once(self, timeout_sec: float = 0.0):
        time.sleep(max(timeout_sec, 0.001))

    _URDF_PATH = "/home/makhosazana/Project/CERLAB-UAV-Autonomy/install/uav_simulator/share/uav_simulator/urdf/quadcopter.urdf"

    def teleport_to_spawn(self, x: float = 0.0, y: float = 0.0, z: float = 0.2) -> bool:
        print("  Deleting quadcopter …")
        del_req = DeleteEntity.Request(); del_req.name = "quadcopter"
        if self._delete_cli.wait_for_service(timeout_sec=5.0):
            future = self._delete_cli.call_async(del_req)
            t0 = time.time()
            while not future.done() and time.time() - t0 < 10.0:
                time.sleep(0.05)
            if future.done():
                print(f"  Delete: {future.result().success}")
            else:
                print("  Delete timed out")
        else:
            print("  /delete_entity not available")
            return False

        time.sleep(3.0)
        self.pose_received = False; self.px = x; self.py = y; self.pz = z

        print(f"  Spawning at ({x:.1f}, {y:.1f}, {z:.1f}) …")
        try:
            with open(self._URDF_PATH) as f: urdf_xml = f.read()
        except FileNotFoundError:
            print(f"  URDF not found: {self._URDF_PATH}"); return False

        spawn_req = SpawnEntity.Request()
        spawn_req.name = "quadcopter"; spawn_req.xml = urdf_xml
        spawn_req.initial_pose.position.x = float(x)
        spawn_req.initial_pose.position.y = float(y)
        spawn_req.initial_pose.position.z = float(z)
        spawn_req.initial_pose.orientation.w = 1.0

        if self._spawn_cli.wait_for_service(timeout_sec=5.0):
            future = self._spawn_cli.call_async(spawn_req)
            t0 = time.time()
            while not future.done() and time.time() - t0 < 15.0: time.sleep(0.05)
            if future.done():
                res = future.result()
                print(f"  Spawn: {res.success}")
                if not res.success: return False
            else:
                print("  Spawn timed out"); return False
        else:
            print("  /spawn_entity not available"); return False

        print("  Waiting for pose …")
        t0 = time.time(); self.pose_received = False
        while time.time() - t0 < 15.0:
            time.sleep(0.1)
            if self.pose_received: break
        if not self.pose_received:
            print("  No pose after respawn"); return False
        print(f"  Ready at ({self.px:.2f}, {self.py:.2f}, {self.pz:.2f})")
        return True


# ═══════════════════════════════════════════════════════════════════════════
# MISSION — full round-trip, collision ≠ death, R54 reward shaping
# ═══════════════════════════════════════════════════════════════════════════

class Mission:
    def __init__(self, drone: CERLABDroneInterface, max_speed: float):
        self.drone = drone
        self.max_speed = min(max_speed, MAX_SPEED)
        self.spawn_x = 0.0; self.spawn_y = 0.0
        self.tgt_x = 0.0;   self.tgt_y = 0.0;   self.tgt_z = SAFE_ALT
        self.phase = "OUTBOUND"
        self.steps = 0; self.max_steps = 3000
        self.fwd_progress = 0.0; self.lat_offset = 0.0
        self.speed = 0.0; self.max_vel = 0.0; self.max_dist = 0.0
        self.collision_count = 0; self.near_miss_count = 0
        self.total_obstacle_encounters = 0; self.obstacles_avoided = 0
        self._in_danger_zone = False; self._collision_cooldown = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._prev_vy = 0.0; self._stuck_counter = 0
        self._last_progress_x = 0.0; self._return_start_step = 0
        self._prev_prox_alarm = 0

    def start(self):
        self.spawn_x = self.drone.px; self.spawn_y = self.drone.py
        self.tgt_x = self.spawn_x;   self.tgt_y = self.spawn_y;   self.tgt_z = SAFE_ALT
        self.phase = "OUTBOUND"; self.steps = 0
        self.max_vel = 0.0; self.max_dist = 0.0
        self.collision_count = 0; self.near_miss_count = 0
        self.total_obstacle_encounters = 0; self.obstacles_avoided = 0
        self._in_danger_zone = False; self._collision_cooldown = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._prev_vy = 0.0; self._stuck_counter = 0
        self._last_progress_x = self.spawn_x; self._return_start_step = 0
        self._prev_prox_alarm = 0
        self._refresh()

    def _refresh(self):
        self.fwd_progress = self.drone.px - self.spawn_x
        self.lat_offset   = self.drone.py - self.spawn_y
        self.speed        = self.drone.vx
        self.max_vel      = max(self.max_vel, min(abs(self.speed), MAX_SPEED))
        self.max_dist     = max(self.max_dist, self.fwd_progress)

    # ── 21-D state vector ──────────────────────────────────────────────────
    def get_state(self) -> np.ndarray:
        self._refresh()
        ds = self.drone.depth_sectors   # 9 values — TRUE readings (used by avoidance/reward)

        # Observation noise injected ONLY into actor input — avoidance/reward use true ds
        h_noisy = np.clip(
            np.array([ds[i] for i in range(NUM_H_SECTORS)], dtype=np.float32)
            + np.random.normal(0.0, OBS_NOISE_STD, NUM_H_SECTORS).astype(np.float32),
            0.05, 10.0)
        v_top = float(np.clip(self.drone.depth_top    + np.random.normal(0, OBS_NOISE_STD), 0.05, 10.0))
        v_mid = float(np.clip(self.drone.depth_mid    + np.random.normal(0, OBS_NOISE_STD), 0.05, 10.0))
        v_bot = float(np.clip(self.drone.depth_bottom + np.random.normal(0, OBS_NOISE_STD), 0.05, 10.0))

        return np.array([
            # 9 horizontal depth sectors (noisy, normalized 0–1)
            *[h_noisy[i] / 10.0 for i in range(NUM_H_SECTORS)],
            # 3 vertical zones (noisy)
            v_top / 10.0, v_mid / 10.0, v_bot / 10.0,
            # Proximity alarm (from true reading — hard safety signal, no noise)
            float(self.drone.prox_alarm),
            # Position (3)
            np.clip(self.fwd_progress / TUNNEL_LENGTH, -0.1, 1.1),
            np.clip(self.lat_offset   / CORRIDOR_HW,   -1,   1),
            np.clip((self.drone.pz - SAFE_ALT) / 1.5,  -1,   1),
            # Velocity (3)
            np.clip(self.speed    / 7.0, -1, 1),
            np.clip(self.drone.vy / 3.0, -1, 1),
            np.clip(self.drone.vz / 2.0, -1, 1),
            # Phase & curriculum (2)
            1.0 if self.phase == "RETURN" else 0.0,
            np.clip(self.max_speed / 7.0, 0, 1),
        ], dtype=np.float32)

    def _smooth_action(self, raw: np.ndarray) -> np.ndarray:
        s = np.array([
            ACTION_SMOOTH_FWD * raw[0] + (1 - ACTION_SMOOTH_FWD) * self._prev_action[0],
            ACTION_SMOOTH_LAT * raw[1] + (1 - ACTION_SMOOTH_LAT) * self._prev_action[1],
            ACTION_SMOOTH_LAT * raw[2] + (1 - ACTION_SMOOTH_LAT) * self._prev_action[2],
        ], dtype=np.float32)
        self._prev_action = s.copy()
        return s

    def apply_action(self, action: np.ndarray) -> Tuple[float, float, float]:
        smoothed  = self._smooth_action(action)
        raw_fwd   = smoothed[0]; raw_lat = smoothed[1]; raw_alt = smoothed[2]

        fwd_speed = (raw_fwd * 0.5 + 0.5) * self.max_speed
        fwd_speed = max(fwd_speed, 1.0)
        fwd_speed = min(fwd_speed, MAX_SPEED)

        lat_speed = raw_lat * 0.8
        alt_adj   = raw_alt * 0.4

        ds = self.drone.depth_sectors
        front_d = float(np.min([ds[3], ds[4], ds[5]]))   # 3 central sectors
        left_d  = float(np.min([ds[0], ds[1]]))           # world +Y outbound
        right_d = float(np.min([ds[7], ds[8]]))           # world -Y outbound

        # World-frame swap for RETURN (camera faces -X)
        if self.phase == "RETURN":
            left_d, right_d = right_d, left_d

        # Effective sector ordering: eff[0]=world+Y, eff[8]=world-Y (phase-corrected)
        eff = list(ds) if self.phase == "OUTBOUND" else list(reversed(ds))

        # ── 1. SMOOTH PROPORTIONAL LATERAL AVOIDANCE (starts at 5m) ───────────
        # Quadratic weight: zero at 5m, maximum at 0m → gentle far, strong near
        AVOID_START = 5.0
        smooth_push = 0.0
        for i in range(NUM_H_SECTORS):
            d = float(eff[i])
            if d < AVOID_START:
                w = ((AVOID_START - d) / AVOID_START) ** 2   # quadratic weight
                smooth_push -= w * _SECTOR_BIAS[i] * 2.0     # push away from obstacle
        lat_speed += smooth_push
        lat_speed = np.clip(lat_speed, -3.5, 3.5)

        # Centre-seeking only when all sectors completely clear
        if float(min(eff)) > AVOID_START:
            lat_speed += -self.lat_offset * 0.20

        # ── 2. PROPORTIONAL BRAKING — front obstacle within 2.8m ──────────────
        # Tighter zone + softer curve → agent maintains higher sustained speed
        if front_d < 2.8:
            brake_ratio = (front_d / 2.8) ** 0.4   # softer exponent: mild far, firm near
            fwd_speed = fwd_speed * max(0.25, brake_ratio)

        # ── 3. EMERGENCY CLOSE-RANGE OVERRIDE (<1.5m) — hard override ─────────
        self._reverse_escape = False

        if front_d < 1.5:
            if left_d > right_d + 0.2:
                lat_speed = max(lat_speed, 2.5)
            elif right_d > left_d + 0.2:
                lat_speed = min(lat_speed, -2.5)
            elif left_d > right_d:
                lat_speed = max(lat_speed, 1.5)
            else:
                lat_speed = min(lat_speed, -1.5)

            # Fully surrounded → reverse escape
            if front_d < 1.2 and max(left_d, right_d) < 1.0:
                self._reverse_escape = True
                fwd_speed = 3.0
                if not hasattr(self, '_escape_dir'):
                    self._escape_dir = 1.0 if np.random.random() > 0.5 else -1.0
                lat_speed = self._escape_dir * 2.5
                alt_adj   = 0.6
            elif front_d < 1.2 and max(left_d, right_d) < 2.0:
                if left_d > right_d: lat_speed = max(lat_speed, 2.5)
                else:                lat_speed = min(lat_speed, -2.5)
                alt_adj = max(alt_adj, 0.4)
                fwd_speed = min(fwd_speed, 0.5)
            else:
                if hasattr(self, '_escape_dir'): del self._escape_dir
                if max(left_d, right_d) < 1.0:
                    if self.drone.depth_top > self.drone.depth_bottom + 0.3:
                        alt_adj = max(alt_adj, 0.6)
                    elif self.drone.depth_bottom > self.drone.depth_top + 0.3:
                        alt_adj = min(alt_adj, -0.6)

        # Wall push
        y = self.lat_offset
        if abs(y) > CORRIDOR_HW * 0.85:
            lat_speed = -np.sign(y) * 1.0

        # Braking near turn point
        if self.phase == "OUTBOUND":
            dist_to_turn = TURN_POINT - self.fwd_progress
            brake_dist   = max(4.0, self.max_speed * 1.5)
            if 0 < dist_to_turn < brake_dist:
                frac = dist_to_turn / brake_dist
                fwd_speed = min(fwd_speed, max(0.5, frac * self.max_speed))
            elif self.fwd_progress >= TURN_POINT:
                fwd_speed = min(fwd_speed, 0.3)

        # Return: maintain minimum speed
        if self.phase == "RETURN":
            steps_since_turn = self.steps - getattr(self, '_return_start_step', 0)
            if steps_since_turn < 50:
                fwd_speed = max(fwd_speed, 2.0)
            elif not self._reverse_escape and front_d > 1.0:
                fwd_speed = max(fwd_speed, 1.5)   # don't enforce min speed when front is blocked
            if self.fwd_progress < 3.0:
                frac = max(0.1, self.fwd_progress / 3.0)
                fwd_speed = min(fwd_speed, max(0.3, frac * self.max_speed))

        # Altitude PID correction
        alt_err = self.drone.pz - SAFE_ALT
        if abs(alt_err) > 0.5:
            correction = -alt_err * 1.0
            alt_adj = 0.5 * alt_adj + 0.5 * correction
        # Emergency altitude hold — override everything when dangerously low
        if self.drone.pz < 0.6:
            alt_adj = max(alt_adj, 4.0)     # extreme emergency climb
            self.tgt_z = max(self.tgt_z, SAFE_ALT)   # snap target up immediately
        elif self.drone.pz < 0.9:
            alt_adj = max(alt_adj, 2.0)     # force strong climb
            self.tgt_z = max(self.tgt_z, 1.2)
        elif self.drone.pz < 1.2:
            alt_adj = max(alt_adj, 1.0)     # moderate climb
        new_z   = self.tgt_z + alt_adj * TICK_DT
        new_z   = np.clip(new_z, MIN_ALT, MAX_ALT)
        alt_adj = (new_z - self.tgt_z) / TICK_DT

        direction = -1.0 if self.phase == "RETURN" else 1.0
        if self._reverse_escape: direction = -direction

        return direction * fwd_speed * TICK_DT, lat_speed * TICK_DT, alt_adj * TICK_DT

    def _track_obstacles(self):
        min_d = float(np.min(self.drone.depth_sectors))
        if self._collision_cooldown > 0: self._collision_cooldown -= 1
        if min_d < DANGER_DIST and not self._in_danger_zone:
            self._in_danger_zone = True; self.total_obstacle_encounters += 1
        if min_d < COLLISION_DIST and self._in_danger_zone and self._collision_cooldown == 0:
            self.collision_count += 1; self._collision_cooldown = 10
        if min_d < NEAR_MISS_DIST and min_d >= COLLISION_DIST and self._in_danger_zone:
            self.near_miss_count += 1
        if min_d > SAFE_DIST and self._in_danger_zone:
            self.obstacles_avoided += 1; self._in_danger_zone = False

    def _check_stuck(self) -> bool:
        if self.phase == "RETURN" and self.steps - self._return_start_step < 80:
            return False
        if self.steps % 30 == 0 and self.steps > 0:
            if self.phase == "OUTBOUND":
                progress = self.fwd_progress - self._last_progress_x
            else:
                progress = self._last_progress_x - self.fwd_progress
            if progress < 0.3: self._stuck_counter += 1
            else:              self._stuck_counter = max(0, self._stuck_counter - 1)
            self._last_progress_x = self.fwd_progress
        # Early outbound (<25m) needs patience for dynamic persons to clear the path
        if self.phase == "OUTBOUND":
            threshold = 4 if self.fwd_progress < 25.0 else 2
        else:
            threshold = 4   # return: escape maneuvers look like no-progress
        return self._stuck_counter >= threshold

    def step(self, action: np.ndarray):
        prev_x = self.fwd_progress
        curr_prox_alarm = self.drone.prox_alarm

        dx, dy, dz = self.apply_action(action)
        self.tgt_x += dx; self.tgt_y += dy; self.tgt_z += dz
        self.tgt_x = np.clip(self.tgt_x, self.spawn_x - 3.0, self.spawn_x + TURN_POINT + 3.0)
        self.tgt_y = np.clip(self.tgt_y, -CORRIDOR_HW, CORRIDOR_HW)   # world coords — safe regardless of spawn jitter
        self.tgt_z = np.clip(self.tgt_z, MIN_ALT, MAX_ALT)

        yaw = 0.0 if self.phase == "OUTBOUND" else math.pi
        self.drone.publish_setpoint(self.tgt_x, self.tgt_y, self.tgt_z, yaw)
        t0 = time.time()
        while time.time() - t0 < TICK_DT:
            time.sleep(0.02)
            self.drone.publish_setpoint(self.tgt_x, self.tgt_y, self.tgt_z, yaw)

        self.steps += 1
        self._refresh()
        self._track_obstacles()

        reward, done, info = self._reward(prev_x, self._prev_prox_alarm, curr_prox_alarm)
        self._prev_prox_alarm = curr_prox_alarm
        return self.get_state(), reward, done, info

    # ── REWARD — v5r baseline + R54 exponential proximity penalty ──────────
    def _reward(self, prev_x: float, prev_alarm: int, curr_alarm: int):
        x       = self.fwd_progress
        y       = self.lat_offset
        alt     = self.drone.pz
        ds      = self.drone.depth_sectors
        min_d   = float(np.min(ds))
        front_d = float(np.min([ds[3], ds[4], ds[5]]))   # 3 central sectors
        left_d  = float(np.min([ds[0], ds[1]]))
        right_d = float(np.min([ds[7], ds[8]]))

        reward = 0.0

        # Hard episode enders
        if alt < ALT_FLOOR:  return -150.0, True, {'event': 'ground_crash'}
        if alt > ALT_CEIL:   return -150.0, True, {'event': 'ceiling_crash'}
        if abs(self.drone.py) > LAT_LIMIT: return -150.0, True, {'event': 'left_tunnel'}  # world coords
        if x > TUNNEL_LENGTH + 5.0: return -100.0, True, {'event': 'overshot'}
        if x < -5.0:         return -100.0, True, {'event': 'flew_backward'}
        if self.steps >= self.max_steps:
            credit = self.max_dist / TURN_POINT * 30.0
            return -30.0 + credit, True, {'event': 'timeout'}
        if self._check_stuck(): return -50.0, True, {'event': 'stuck'}

        # Collision cap (unchanged from v5r)
        if min_d < COLLISION_DIST:
            if self.collision_count <= COLLISION_PENALTY_CAP:
                reward -= 60.0

        # Progress (speed-coupled)
        progress  = x - prev_x
        spd       = abs(self.speed)
        speed_mult = 1.0 + min(spd / self.max_speed, 1.0)
        if self.phase == "OUTBOUND": reward += progress * 120.0 * speed_mult
        else:                        reward -= progress * 120.0 * speed_mult

        # Straight-line bonus
        if abs(progress) > 0.01:
            if abs(y) < 0.5:  reward += 4.0
            elif abs(y) < 1.0: reward += 2.0

        reward += 1.0   # alive bonus

        # Direct per-step speed incentive — creates constant gradient toward full throttle
        reward += spd * 1.2

        # Speed tier bonuses (coarse shaping on top of continuous incentive)
        if spd > 4.5:    reward += 25.0
        elif spd > 4.0:  reward += 18.0
        elif spd > 3.0:  reward += 10.0
        elif spd > 2.0:  reward += 4.0
        else:            reward -= 8.0

        # ── R54+: Approach penalty — punish flying toward an obstacle ──────────
        # Encourages the agent to predict and dodge before getting close
        prev_fd = getattr(self, '_prev_front_d', front_d)
        if front_d < 3.5 and front_d < prev_fd - 0.08:
            approach_speed = (prev_fd - front_d) / TICK_DT
            reward -= min(approach_speed * (3.5 - front_d) * 3.0, 30.0)
        self._prev_front_d = front_d

        # ── R54+: Exponential proximity penalty (wider and stronger than v5r) ──
        if min_d < SAFE_DIST:
            exp_penalty = PROX_EXP_SCALE * math.exp(-min_d / PROX_EXP_K)
            reward -= exp_penalty
        else:
            reward += 8.0   # stronger safe-clearance bonus (was 3.0)

        # ── R54: Alarm escape bonus — reward actively opening distance ──
        if prev_alarm == 1 and curr_alarm == 0:
            reward += 80.0   # stronger escape reward (was 60.0)

        # Clear-path bonus
        if front_d < 2.5:
            best_side = max(left_d, right_d)
            if best_side > 3.0:   reward += 5.0
            elif best_side > 1.5: reward += 2.5
            if left_d > right_d  and self._prev_action[1] > 0.1: reward += 3.0
            elif right_d > left_d and self._prev_action[1] < -0.1: reward += 3.0

        # Smoothness (relaxed lateral penalty — allows fast dodging at high speed)
        reward -= abs(self.drone.vy) * 1.5
        reward -= abs(self.drone.vz) * 0.8
        vy_change = abs(self.drone.vy - self._prev_vy)
        reward -= vy_change * 1.0
        self._prev_vy = self.drone.vy

        # Centre-lane and altitude
        reward -= abs(y)   * 1.5
        reward -= abs(alt - SAFE_ALT) * 2.0

        # Phase transition
        if self.phase == "OUTBOUND" and x >= TURN_POINT:
            self.phase = "RETURN"
            self._return_start_step = self.steps
            self._stuck_counter = 0
            self._last_progress_x = self.fwd_progress
            col_bonus = max(0, 1000 - self.collision_count * 50)
            reward += 3000.0 + col_bonus
            print(f"\n  OUTBOUND COMPLETE  x={x:.1f}m  "
                  f"avoided={self.obstacles_avoided}  col={self.collision_count}  "
                  f"col_bonus={col_bonus}\n")
            return reward, False, {'event': 'turn'}

        # Success
        if self.phase == "RETURN" and x <= 2.0:
            reward += 3000.0
            col = self.collision_count
            if col == 0:     reward += 2500.0
            elif col <= 3:   reward += 1500.0
            elif col <= 8:   reward += 800.0
            elif col <= 15:  reward += 300.0
            if self.max_vel > 5.0: reward += 1200.0
            elif self.max_vel > 4.5: reward += 900.0
            elif self.max_vel > 4.0: reward += 600.0
            elif self.max_vel > 3.0: reward += 300.0
            elif self.max_vel > 2.0: reward += 100.0
            print(f"\n  MISSION SUCCESS  round-trip!  "
                  f"spd={self.max_vel:.2f} m/s  "
                  f"avoided={self.obstacles_avoided}  col={col}  steps={self.steps}\n")
            return reward, True, {'event': 'success'}

        return reward, False, {}


# ═══════════════════════════════════════════════════════════════════════════
# LIVE PLOT (updated every 10 missions)
# ═══════════════════════════════════════════════════════════════════════════

def plot_live(m_ok, m_dist, m_spd, m_collisions, m_avoided, save_path):
    if not HAS_MPL or len(m_ok) < 2:
        return
    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)
    eps = range(1, len(m_ok) + 1)
    window = min(20, len(m_ok))

    def rolling_sr(data, w):
        return [np.mean(data[max(0, i-w):i+1]) * 100 for i in range(len(data))]

    a = fig.add_subplot(gs[0, 0])
    sr_roll = rolling_sr(m_ok, window)
    a.plot(eps, sr_roll, color='#2196F3', lw=1.5)
    a.axhline(55, color='g', ls='--', alpha=0.5, label='55% target')
    a.set(xlabel='Mission', ylabel='SR %', title=f'Success Rate (rolling {window})')
    a.legend(fontsize=7); a.grid(alpha=0.3)

    a = fig.add_subplot(gs[0, 1])
    a.bar(eps, m_dist, color='purple', alpha=0.7)
    a.axhline(TURN_POINT, color='g', ls='--')
    a.set(xlabel='Mission', ylabel='m', title='Max Distance Reached')
    a.grid(alpha=0.3)

    a = fig.add_subplot(gs[0, 2])
    a.plot(eps, m_spd, color='#FF9800', lw=1.0, marker='o', ms=2)
    a.axhline(2.0, color='r', ls='--', alpha=0.5)
    a.set(xlabel='Mission', ylabel='m/s', title='Avg Peak Speed')
    a.grid(alpha=0.3)

    a = fig.add_subplot(gs[1, 0])
    a.bar(eps, m_collisions, color='#F44336', alpha=0.8, label='Collisions')
    a.bar(eps, m_avoided,   color='#4CAF50', alpha=0.6,
          bottom=m_collisions, label='Avoided')
    a.set(xlabel='Mission', ylabel='Count', title='Obstacle Encounters')
    a.legend(fontsize=7); a.grid(alpha=0.3)

    a = fig.add_subplot(gs[1, 1])
    avoid_rate = [(av/(en)*100 if en > 0 else 100)
                  for av, en in zip(m_avoided,
                                    [c+av for c,av in zip(m_collisions, m_avoided)])]
    a.bar(eps, avoid_rate, color='#2196F3', alpha=0.8)
    a.axhline(90, color='g', ls='--', alpha=0.5, label='90% target')
    a.set(xlabel='Mission', ylabel='%', title='Avoidance Rate')
    a.legend(fontsize=7); a.grid(alpha=0.3)

    # Summary panel
    a = fig.add_subplot(gs[1, 2])
    a.axis('off')
    n    = len(m_ok)
    sr   = np.mean(m_ok) * 100
    aspd = np.mean(m_spd)
    total_col = sum(m_collisions)
    total_av  = sum(m_avoided)
    oar = total_av / max(total_col + total_av, 1) * 100
    txt = (f"TUNNEL TD3 R54\n{'─'*28}\n"
           f"Missions    : {n}\n"
           f"SR all-time : {sr:.1f}%\n"
           f"SR last-20  : {np.mean(m_ok[-20:])*100:.1f}%\n"
           f"Avg speed   : {aspd:.2f} m/s\n"
           f"Avoid rate  : {oar:.1f}%\n"
           f"{'─'*28}\n"
           f"State={STATE_DIM}  Act={ACTION_DIM}\n"
           f"Sectors: {NUM_H_SECTORS}H + 3V\n"
           f"Percentile: {DEPTH_PERCENTILE}th\n"
           f"Device: {DEVICE}\n")
    a.text(0.05, 0.95, txt, transform=a.transAxes, fontsize=9,
           family='monospace', va='top',
           bbox=dict(boxstyle='round,pad=0.4', fc='#E8F5E9', alpha=0.85))

    fig.suptitle('Tunnel TD3 R54 — 100m Round-Trip (20 Obstacles)',
                 fontsize=14, fontweight='bold', color='#1B5E20')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Live plot saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# EPISODE RESET
# ═══════════════════════════════════════════════════════════════════════════

def episode_reset(drone, spawn_x, spawn_y):
    print("\n  EPISODE RESET …")
    ok = drone.teleport_to_spawn(spawn_x, spawn_y, 0.2)
    if not ok:
        print("  Reset failed!"); return False
    drone.takeoff()
    t0 = time.time()
    while time.time() - t0 < 6.0:
        drone.publish_setpoint(spawn_x, spawn_y, SAFE_ALT, 0.0)
        time.sleep(0.08)
    for _ in range(30):
        drone.publish_setpoint(spawn_x, spawn_y, SAFE_ALT, 0.0)
        time.sleep(0.08)
    print(f"  Ready  alt={drone.pz:.2f}  pos=({drone.px:.1f},{drone.py:.1f})")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_training(args):
    rclpy.init()
    drone = CERLABDroneInterface()
    drone.start_spinning()

    print("Waiting for pose …")
    for _ in range(200):
        time.sleep(0.1)
        if drone.pose_received: break
    if not drone.pose_received:
        print("No pose — attempting spawn via /spawn_entity")
        ok = drone.teleport_to_spawn(0.0, 0.0, 0.2)
        if not ok:
            print("Could not spawn quadcopter; aborting"); return

    print(f"Pose: ({drone.px:.2f}, {drone.py:.2f}, {drone.pz:.2f})")

    if abs(drone.px) > 1.0 or abs(drone.py) > 1.0:
        drone.teleport_to_spawn(0.0, 0.0, 0.2)
        for _ in range(80):
            drone.publish_setpoint(0.0, 0.0, SAFE_ALT, 0.0)
            time.sleep(0.08)

    SPAWN_X = drone.px; SPAWN_Y = drone.py

    print("Takeoff …")
    drone.takeoff()
    t0 = time.time()
    while time.time() - t0 < 6.0:
        drone.publish_setpoint(SPAWN_X, SPAWN_Y, SAFE_ALT, 0.0)
        time.sleep(0.08)
    for _ in range(30):
        drone.publish_setpoint(SPAWN_X, SPAWN_Y, SAFE_ALT, 0.0)
        time.sleep(0.08)
    print(f"Spawn: ({SPAWN_X:.3f},{SPAWN_Y:.3f})  alt={drone.pz:.2f}")

    agent  = TD3Agent()
    buffer = deque(maxlen=BUFFER_SIZE)

    speed_idx   = max(0, min(args.speed_stage, len(SPEED_STAGES) - 1))
    stage_start = 0

    if args.load:
        agent.load(args.load)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  TUNNEL TD3 R54 — 100m ROUND-TRIP (20 Obstacles, 9H+3V sectors)")
    print(f"{'='*72}")
    print(f"  Missions         : {args.missions}")
    print(f"  State dim        : {STATE_DIM}  (9H+3V+alarm+3pos+3vel+phase+speed)")
    print(f"  Depth sectors    : {NUM_H_SECTORS} horizontal, 5th-percentile")
    print(f"  Architecture     : 512→512→256 + LayerNorm + residual (Actor)")
    print(f"                     512→512→256 + LayerNorm (Twin Critic)")
    print(f"  Speed stages     : {SPEED_STAGES}")
    print(f"  Buffer           : {BUFFER_SIZE:,}  Batch: {BATCH_SIZE}")
    print(f"  Output           : {out_dir}")
    print(f"{'='*72}\n")

    # Tracking
    m_ok = []; m_dist = []; m_spd = []; m_collisions = []; m_avoided = []
    m_q_mean = []; m_critic_loss = []; m_actor_loss = []; m_noise = []
    all_rewards = []
    best_reward = -float('inf')
    t_global    = 0.0

    def _save_all():
        # Save checkpoint
        agent.save(os.path.join(out_dir, f"r54_ep{len(m_ok)}.pth"))
        # Save JSON
        data = dict(
            missions=len(m_ok),
            sr_all=np.mean(m_ok) if m_ok else 0,
            sr_last20=np.mean(m_ok[-20:]) if len(m_ok) >= 20 else np.mean(m_ok) if m_ok else 0,
            avg_peak_speed=np.mean(m_spd) if m_spd else 0,
            total_collisions=int(sum(m_collisions)),
            total_avoided=int(sum(m_avoided)),
            avoidance_rate=sum(m_avoided)/max(sum(m_collisions)+sum(m_avoided),1)*100,
            best_dist=float(max(m_dist)) if m_dist else 0,
            m_ok=m_ok, m_dist=m_dist, m_spd=m_spd,
            m_collisions=m_collisions, m_avoided=m_avoided,
        )
        with open(os.path.join(out_dir, "training_history_r54.json"), "w") as f:
            json.dump(data, f, indent=2)
        print(f"  JSON saved  SR={data['sr_all']*100:.1f}%  avd={data['avoidance_rate']:.1f}%")

    def _on_sigint(sig, frame):
        print("\n  Interrupt — saving …")
        _save_all()
        plot_live(m_ok, m_dist, m_spd, m_collisions, m_avoided,
                  os.path.join(out_dir, "r54_live.png"))
        drone.stop_spinning(); rclpy.shutdown(); sys.exit(0)

    signal.signal(signal.SIGINT, _on_sigint)

    log_path = os.path.join(out_dir, "training_history_r54.log")

    for mi in range(args.missions):
        drone.spin_once(timeout_sec=0.1)

        # ── Per-episode domain randomization (generalization) ─────────────────
        ep_jitter_y  = float(np.random.uniform(-SPAWN_Y_JITTER, SPAWN_Y_JITTER))
        ep_spawn_y   = float(np.clip(SPAWN_Y + ep_jitter_y, -1.5, 1.5))  # stay well inside tunnel
        ep_spd_jitter = float(np.random.uniform(-SPEED_JITTER, SPEED_JITTER))
        ep_max_speed  = float(np.clip(SPEED_STAGES[speed_idx] + ep_spd_jitter,
                                      SPEED_STAGES[speed_idx] - SPEED_JITTER,
                                      MAX_SPEED))

        if mi > 0:
            dist = math.hypot(drone.px - SPAWN_X, drone.py - SPAWN_Y)
            if dist > 2.0 or drone.pz < 0.5:
                ok = episode_reset(drone, SPAWN_X, ep_spawn_y)
                if not ok:
                    print("  Reset failed — aborting"); break
            else:
                for _ in range(25):
                    drone.publish_setpoint(SPAWN_X, ep_spawn_y, SAFE_ALT, 0.0)
                    time.sleep(0.08)

        mission = Mission(drone, ep_max_speed)
        mission.start()

        t_wait = time.time()
        while time.time() - t_wait < 15.0:
            if float(np.min(drone.depth_sectors)) > 1.0: break
            drone.publish_setpoint(SPAWN_X, ep_spawn_y, SAFE_ALT, 0.0)
            time.sleep(0.1)

        state = mission.get_state()
        frac  = mi / max(args.missions - 1, 1)
        noise = EXPLORE_NOISE_INIT - frac * (EXPLORE_NOISE_INIT - EXPLORE_NOISE_MIN)
        # SR-adaptive noise: when agent is performing well, exploit rather than explore
        if len(m_ok) >= 20:
            sr_recent = np.mean(m_ok[-20:])
            # At SR=70%: factor≈1.0 (no change); at SR=90%: factor≈0.5; at SR=99%: factor≈0.25
            sr_factor = max(0.25, 1.0 - max(0.0, sr_recent - 0.70) * 3.5)
            noise *= sr_factor
        noise = max(noise, EXPLORE_NOISE_MIN)

        ep_reward = 0.0; ep_start = time.time(); done = False; info = {}
        ep_q_vals = []; ep_critic_losses = []; ep_actor_losses = []

        print(f"\n{'─'*72}")
        print(f"  MISSION {mi+1}/{args.missions}   "
              f"speed={ep_max_speed:.2f}m/s  noise={noise:.3f}  "
              f"SR={np.mean(m_ok)*100:.1f}%  buf={len(buffer)}  "
              f"spawn_y={ep_spawn_y:+.2f}m")
        print(f"{'─'*72}")

        step_count = 0
        while not done:
            action = agent.select_action(state, noise)
            q      = agent.q_value(state, action)
            ep_q_vals.append(q)
            next_state, rew, done, info = mission.step(action)
            buffer.append((state, action, rew, next_state, float(done)))
            ep_reward += rew; step_count += 1

            if len(buffer) >= WARMUP_STEPS:
                for _ in range(UPDATES_PER_STEP):
                    cl, al = agent.update(buffer)
                    ep_critic_losses.append(cl)
                    if al != 0.0:   # actor only updates every POLICY_DELAY steps (loss is negative)
                        ep_actor_losses.append(al)

            state = next_state

        q_mean    = float(np.mean(ep_q_vals))      if ep_q_vals      else 0.0
        c_loss    = float(np.mean(ep_critic_losses)) if ep_critic_losses else 0.0
        a_loss    = float(np.mean(ep_actor_losses))  if ep_actor_losses  else 0.0

        ep_dur  = time.time() - ep_start
        t_global += ep_dur
        event   = info.get('event', 'unknown')
        success = (event == 'success')
        dist    = mission.max_dist
        spd     = mission.max_vel

        m_ok.append(int(success))
        m_dist.append(dist)
        m_spd.append(spd)
        m_collisions.append(mission.collision_count)
        m_avoided.append(mission.obstacles_avoided)
        all_rewards.append(ep_reward)

        sr_all  = np.mean(m_ok) * 100
        sr20    = np.mean(m_ok[-20:]) * 100 if len(m_ok) >= 20 else sr_all

        m_q_mean.append(q_mean)
        m_critic_loss.append(c_loss)
        m_actor_loss.append(a_loss)
        m_noise.append(noise)

        log_line = (
            f"{'OK' if success else 'FAIL'} M{mi+1:05d} "
            f"rew={ep_reward:+.0f} dist={dist:.1f}m spd={spd:.2f}m/s "
            f"col={mission.collision_count} avd={mission.obstacles_avoided} "
            f"dur={ep_dur:.0f}s sr={sr_all:.1f}% sr20={sr20:.1f}% "
            f"stage={SPEED_STAGES[speed_idx]}m/s event={event} "
            f"critic_loss={c_loss:.4f} actor_loss={a_loss:.4f} "
            f"q_mean={q_mean:.2f} noise={noise:.4f}"
        )
        print(f"\n  {log_line}\n")
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        # Save best
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(out_dir, "r54_best.pth"))

        # Curriculum — advance/regress based on missions AT current stage, not total
        missions_at_stage = mi + 1 - stage_start
        if missions_at_stage >= MIN_MISSIONS_STAGE:
            recent_sr = np.mean(m_ok[-MIN_MISSIONS_STAGE:])
            advance_thr = SR_THRESHOLDS[speed_idx]
            regress_thr = SR_THRESHOLDS[max(speed_idx - 1, 0)] * 0.65 if speed_idx > 0 else 0.0
            if recent_sr >= advance_thr and speed_idx < len(SPEED_STAGES) - 1:
                speed_idx += 1; stage_start = mi + 1
                print(f"\n  CURRICULUM ADVANCE → {SPEED_STAGES[speed_idx]} m/s "
                      f"(SR-{MIN_MISSIONS_STAGE}={recent_sr*100:.1f}%, "
                      f"{missions_at_stage} missions at prev stage)\n")
            elif recent_sr < regress_thr and speed_idx > 0:
                speed_idx -= 1; stage_start = mi + 1
                print(f"\n  CURRICULUM REGRESS → {SPEED_STAGES[speed_idx]} m/s "
                      f"(SR-{MIN_MISSIONS_STAGE}={recent_sr*100:.1f}%)\n")

        # Save every 10 missions
        if (mi + 1) % 10 == 0:
            _save_all()
            plot_live(m_ok, m_dist, m_spd, m_collisions, m_avoided,
                      os.path.join(out_dir, "r54_live.png"))
            # Also regenerate full dashboard from log
            import subprocess
            subprocess.Popen(
                [sys.executable, os.path.join(os.path.dirname(__file__), "plot_r54.py")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

    print("\n  Training complete!")
    _save_all()
    plot_live(m_ok, m_dist, m_spd, m_collisions, m_avoided,
              os.path.join(out_dir, "r54_final.png"))
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "plot_r54.py")])
    drone.stop_spinning()
    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--missions",    type=int, default=5000)
    parser.add_argument("--load",        type=str, default="")
    parser.add_argument("--output-dir",  type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "results_r54"))
    parser.add_argument("--speed-stage", type=int, default=0,
                        help="Curriculum speed stage index to start at (0=2m/s, 1=3m/s, 2=4m/s, 3=4.5m/s)")
    args = parser.parse_args()
    run_training(args)

if __name__ == "__main__":
    main()
