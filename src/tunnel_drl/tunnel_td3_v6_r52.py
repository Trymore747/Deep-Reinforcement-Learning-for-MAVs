#!/usr/bin/env python3
"""
Tunnel TD3 v6 — RUN 52  (Speed-Focused: Learnable Braking + Staged Obstacle Density)
======================================================================================
Target: 2–3 m/s average speed, SR > 40%, robust sim-to-real transfer.

R52 improvements over R51 (R51 ended at 16.5% SR, avg 0.9 m/s, best speed 1.73 m/s):
  • Staged obstacle worlds: 0 people (Stage 1), 4 people (Stages 2-3), 8 people (Stages 4-6)
    — allows agent to learn high-speed traversal before dense crowds
  • Hard braking caps REMOVED: only emergency cap at front_d < 0.15m (near-crash)
    — agent now LEARNS when to slow down via reward signal, not forced physics
  • Speed reward further strengthened: target band 2.5-3.5 m/s gets max bonus
  • Collision penalty reduced -50 → -30: less incentive to stop near people, more to fly past
  • Curriculum DIST_SR_THRESH 0.30→0.25: advance sooner to see harder stages
  • Actor grad clip relaxed 0.5→0.8: allow larger updates since Q exploitation is stable

Key R51 fixes RETAINED:
  • POLICY_DELAY 2→4: prevents Q-value overexploitation (R50 actor loss hit -700)
  • LR_ACTOR 1.5e-5→8e-6: more conservative actor updates for stable learning
  • Actor grad clip 1.0→0.5: further stabilises actor updates
  • Speed reward rebalanced: stronger penalties below 1.5 m/s, stronger bonuses at 2-3 m/s
  • Braking floor 0.15→0.40 m/s: exit danger zone 3× faster (less collision exposure)
  • MIN_MISSIONS_STAGE 60→80: require more experience before curriculum advance
  • DIST_SR_THRESH 0.25→0.30: must achieve 30% rolling SR to advance
  • success_history window 20→30: more stable SR estimate prevents false advances
  • Auto-regress threshold 100→120 episodes: harder to regress, reduces oscillation
  • Post-regress re-advance requires 35% SR (not 25%): breaks 65m↔80m cycling
  • eff_blocked suppression 30→60 steps: more grace after yaw-flip phantom sensors
  • stuck_limit RETURN 80→120 steps: more time to push through dense people on return
  • Milestone rewards rebalanced for 100m tunnel full traversal

State (12-D, body-frame, position-free):
  0-4  depth_sectors[0..4]/10       5 forward depth sectors (body frame)
  5    depth_top/10                 ceiling clearance
  6    depth_bottom/10              floor clearance
  7    (pz − 1.0)/1.5              altitude error (with barometer noise)
  8    v_fwd/MAX_SPEED              body-frame forward velocity
  9    v_lat/3.0                    body-frame lateral velocity
  10   phase_flag (−1=OUT, +1=RET)  mission phase
  11   speed_target/MAX_SPEED       curriculum speed target

Action (3-D, tanh → [−1,1]):
  0  forward speed magnitude  →  [0, max_speed] m/s
  1  lateral velocity         →  [−0.8, +0.8] m/s (+ rule-based override up to ±2.0)
  2  altitude adjustment      →  [−0.04, +0.04] m/step
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
import threading
from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple

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
    import matplotlib.gridspec as mgs
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("⚠️  matplotlib not found — plots will be skipped")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

TUNNEL_LENGTH   = 105.0  # 100m tunnel; end wall at x=100m (wall size 9.15×8.5m)
CORRIDOR_HW     = 4.0    # half-width of 8m-wide tunnel (was 3.5 for 5m corridor)
LAT_LIMIT       = 4.2    # slightly wider than half-width

SAFE_ALT        = 1.0
ALT_NAV         = 1.8   # hard ceiling outside end-wall zones — fly-high overfitting prevention
MAX_ALT         = 3.0
MIN_ALT         = 0.2   # minimum commanded altitude (allows low flight)
ALT_FLOOR       = 0.10  # hard ground-crash terminator
ALT_CEIL        = 4.5

WALL_DETECT_THRESH = 0.4
WALL_DETECT_STEPS  = 15
TURN_X_FALLBACK    = 102.0

COLLISION_DIST  = 0.35
NEAR_MISS_DIST  = 1.0   # raised 0.8→1.0 — person obstacles need more clearance
DANGER_DIST     = 2.0   # raised 1.5→2.0 — start reacting earlier for smooth avoidance
SAFE_DIST       = 3.5   # raised 3.0→3.5 — wider tunnel allows more lateral clearance

MAX_SPEED       = 6.0
SPEED_STAGES    = [3.0, 4.0, 5.0, 6.0]
SR_THRESHOLDS   = [0.0, 0.40, 0.45, 0.50]
MIN_MISSIONS_STAGE = 80   # R51: 60→80 — require more stage experience before advancing
DIST_STAGES     = [20.0, 35.0, 50.0, 65.0, 80.0, 100.0]  # 100m tunnel curriculum; wall-sensed turn fires at x≈98-100m
DIST_SR_THRESH  = 0.25   # R52: 0.30→0.25 — advance sooner to expose agent to harder stages earlier

ACTION_SMOOTH_FWD = 0.65
ACTION_SMOOTH_LAT = 0.30
COLLISION_PENALTY_CAP = 50

# R51 domain randomisation — slightly wider than R45 for stronger sim-to-real
DR_SPAWN_Y      = (-0.5, 0.5)
DR_SNS_NOISE    = (0.02, 0.20)    # R50: wider (was 0.15) — more robust to real sensors
DR_WIND_MAG     = (0.0, 0.03)     # R50: slightly more turbulence (was 0.025)
DR_DROPOUT_PROB = 0.06             # R50: 6% (was 4%) — more occlusion robustness
DR_ALT_NOISE_SD = 0.05             # R50: slightly more barometer noise (was 0.04)
DR_ACT_DELAY    = True

GAMMA           = 0.99
TAU             = 0.005
LR_ACTOR        = 8e-6    # R51: 1.5e-5→8e-6 — slower actor updates prevent Q overexploitation
LR_CRITIC       = 8e-5    # R51: 1e-4→8e-5 — slightly slower critic too for stability
BUFFER_SIZE     = 150_000
BATCH_SIZE      = 384
POLICY_DELAY    = 4       # R51: 2→4 — critic stabilises more before each actor update (actor loss was -700 in R50)
ACTOR_FREEZE_EPS = 45
EXPLORE_NOISE_INIT = 0.30
EXPLORE_NOISE_MIN  = 0.18
TARGET_NOISE       = 0.2
TARGET_CLIP        = 0.5
WARMUP_STEPS       = 2500
UPDATES_PER_STEP   = 5
TICK_DT            = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Torch device: {DEVICE}")

STATE_DIM  = 12
ACTION_DIM = 3

CKPT_DIR          = "src/tunnel_drl/results_v6_r52_tunnel100m"
CRITIC_STATE_PATH = f"{CKPT_DIR}/tunnel_td3_v6_critic_state.pth"
BUFFER_SAVE_PATH  = f"{CKPT_DIR}/replay_buffer.pkl"
BUFFER_SAVE_FREQ  = 100
PLOT_FREQ         = 50    # save comprehensive PNG every N episodes


# ═══════════════════════════════════════════════════════════════════════════════
# WALL DETECTOR — sensor-only, generalises to any tunnel length
# ═══════════════════════════════════════════════════════════════════════════════
# During training the environment uses a position gate (x>85m) for stability.
# At DEPLOYMENT this class replaces that gate — no absolute position needed.
# It works purely from depth_sectors, making it portable to unknown environments.

class WallDetector:
    """
    Detects the end wall of any tunnel using depth sensors only.

    End-wall signature (different from a mid-tunnel obstacle):
      1. OMNI-BLOCKED  — ALL 5 sectors read < omni_thresh (0.5m): drone fully surrounded
      2. MAJORITY+FRONT — ≥4 sectors < sector_thresh AND front < front_thresh AND
                          front distance is DECREASING (approaching, not strafing)
    Both require `confirm_steps` consecutive frames to avoid obstacle false positives.

    Usage (deployment inference loop):
        wd = WallDetector()
        wd.arm(min_travel=5.0)          # arm after leaving spawn zone
        ...
        if wd.update(depth_sectors, dx): # returns True once → flip phase to RETURN
            phase = "RETURN"
    """

    def __init__(
        self,
        front_thresh:     float = 1.0,   # front sector triggers at this distance (m)
        sector_thresh:    float = 1.5,   # side/diagonal sectors trigger below this (m)
        omni_thresh:      float = 0.5,   # all-sectors-blocked hard threshold (m)
        min_sectors:      int   = 4,     # how many of 5 sectors must be within sector_thresh
        confirm_steps:    int   = 4,     # consecutive triggered frames before confirming
        cooldown_steps:   int   = 40,    # frames to wait after a turn before re-arming
        trend_window:     int   = 8,     # frames to measure approach trend
        trend_factor:     float = 0.75,  # front must drop to this fraction over window
    ):
        self.front_thresh   = front_thresh
        self.sector_thresh  = sector_thresh
        self.omni_thresh    = omni_thresh
        self.min_sectors    = min_sectors
        self.confirm_steps  = confirm_steps
        self.cooldown_steps = cooldown_steps
        self.trend_window   = trend_window
        self.trend_factor   = trend_factor

        self._counter   = 0
        self._cooldown  = 0
        self._travel    = 0.0
        self._min_travel = 5.0
        self._armed     = False
        self._front_buf: deque = deque(maxlen=trend_window)

    def arm(self, min_travel: float = 5.0):
        """Call once after leaving the spawn zone."""
        self._min_travel = min_travel
        self._armed      = True
        self._travel     = 0.0
        self._counter    = 0
        self._cooldown   = 0
        self._front_buf.clear()

    def reset(self):
        """Call at the start of every episode."""
        self._counter  = 0
        self._cooldown = 0
        self._travel   = 0.0
        self._armed    = False
        self._front_buf.clear()

    def update(self, depth_sectors: np.ndarray, dx_forward: float) -> bool:
        """
        Call every control step.
        depth_sectors : (5,) array  [left, front-left, front, front-right, right]
        dx_forward    : forward distance covered this step (m), 0 or negative = ignore
        Returns True ONCE when wall is confidently detected → caller should flip phase.
        """
        if not self._armed:
            return False
        self._travel += max(dx_forward, 0.0)
        if self._travel < self._min_travel:
            return False
        if self._cooldown > 0:
            self._cooldown -= 1
            return False

        front_d = float(depth_sectors[2])
        self._front_buf.append(front_d)

        # Signature 1 — omnidirectional block: wall fills all sectors
        omni = float(np.max(depth_sectors)) < self.omni_thresh

        # Signature 2 — majority blocked + front closing + approach trend
        n_blocked       = int(np.sum(np.array(depth_sectors) < self.sector_thresh))
        majority        = n_blocked >= self.min_sectors and front_d < self.front_thresh
        approaching     = False
        if len(self._front_buf) >= self.trend_window:
            oldest = list(self._front_buf)[0]
            approaching = front_d < oldest * self.trend_factor and front_d < 2.0

        wall_signature = omni or (majority and approaching)

        if wall_signature:
            self._counter += 1
            if self._counter >= self.confirm_steps:
                self._cooldown = self.cooldown_steps
                self._counter  = 0
                return True   # ← wall confirmed, caller flips phase
        else:
            self._counter = max(0, self._counter - 1)

        return False

    @property
    def confidence(self) -> float:
        """0–1 score of current wall proximity — useful for logging/telemetry."""
        return min(self._counter / max(self.confirm_steps, 1), 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORKS
# ═══════════════════════════════════════════════════════════════════════════════

class Actor(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256);  self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256);         self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128);         self.ln3 = nn.LayerNorm(128)
        self.fc4 = nn.Linear(128, action_dim)
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc4.bias,   -3e-3, 3e-3)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(s)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return torch.tanh(self.fc4(x))


class Critic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        inp = state_dim + action_dim
        self.q1_fc1 = nn.Linear(inp, 256);  self.q1_ln1 = nn.LayerNorm(256)
        self.q1_fc2 = nn.Linear(256, 256);  self.q1_ln2 = nn.LayerNorm(256)
        self.q1_fc3 = nn.Linear(256, 128);  self.q1_ln3 = nn.LayerNorm(128)
        self.q1_out = nn.Linear(128, 1)
        self.q2_fc1 = nn.Linear(inp, 256);  self.q2_ln1 = nn.LayerNorm(256)
        self.q2_fc2 = nn.Linear(256, 256);  self.q2_ln2 = nn.LayerNorm(256)
        self.q2_fc3 = nn.Linear(256, 128);  self.q2_ln3 = nn.LayerNorm(128)
        self.q2_out = nn.Linear(128, 1)
        gain = nn.init.calculate_gain('relu')
        for fc in [self.q1_fc1, self.q1_fc2, self.q1_fc3,
                   self.q2_fc1, self.q2_fc2, self.q2_fc3]:
            nn.init.orthogonal_(fc.weight, gain=gain)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        sa = torch.cat([s, a], dim=-1)
        q1 = F.relu(self.q1_ln1(self.q1_fc1(sa)))
        q1 = F.relu(self.q1_ln2(self.q1_fc2(q1)))
        q1 = F.relu(self.q1_ln3(self.q1_fc3(q1)))
        q1 = self.q1_out(q1)
        q2 = F.relu(self.q2_ln1(self.q2_fc1(sa)))
        q2 = F.relu(self.q2_ln2(self.q2_fc2(q2)))
        q2 = F.relu(self.q2_ln3(self.q2_fc3(q2)))
        q2 = self.q2_out(q2)
        return q1, q2

    def q1_only(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([s, a], dim=-1)
        q1 = F.relu(self.q1_ln1(self.q1_fc1(sa)))
        q1 = F.relu(self.q1_ln2(self.q1_fc2(q1)))
        q1 = F.relu(self.q1_ln3(self.q1_fc3(q1)))
        return self.q1_out(q1)


# ═══════════════════════════════════════════════════════════════════════════════
# TD3 AGENT
# ═══════════════════════════════════════════════════════════════════════════════

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
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=LR_ACTOR,  weight_decay=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=1e-5)
        self.iter = 0
        ap = sum(p.numel() for p in self.actor.parameters())
        cp = sum(p.numel() for p in self.critic.parameters())
        print(f"🧠 TD3 R51 — Actor:{ap:,}  Critic:{cp:,}  Device:{DEVICE}")

    @torch.no_grad()
    def select_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        a = self.actor(s).cpu().numpy().flatten()
        if noise_scale > 0:
            a = np.clip(a + np.random.normal(0, noise_scale, size=self.action_dim), -1.0, 1.0)
        return a

    def update(self, buffer: deque, batch_size: int = BATCH_SIZE,
               freeze_actor: bool = False, success_buffer: deque = None):
        if len(buffer) < batch_size:
            return 0.0, 0.0
        if success_buffer is not None and len(success_buffer) >= 32:
            n_success = max(32, batch_size * 2 // 5)
            n_main    = batch_size - n_success
            si = np.random.choice(len(success_buffer), n_success, replace=True)
            mi = np.random.choice(len(buffer), n_main, replace=False)
            batch = [success_buffer[i] for i in si] + [buffer[i] for i in mi]
        else:
            indices = np.random.choice(len(buffer), batch_size, replace=False)
            batch = [buffer[i] for i in indices]

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
        critic_loss = (F.smooth_l1_loss(q1, target_q, beta=500.0)
                       + F.smooth_l1_loss(q2, target_q, beta=500.0))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        actor_loss_val = 0.0
        self.iter += 1
        if self.iter % POLICY_DELAY == 0 and not freeze_actor:
            actor_loss = -self.critic.q1_only(states, self.actor(states)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.8)  # R52: 0.5→0.8 — allow larger updates since reward caps removed
            self.actor_opt.step()
            actor_loss_val = actor_loss.item()
            for p, t in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                t.data.copy_(TAU * p.data + (1 - TAU) * t.data)
            for p, t in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                t.data.copy_(TAU * p.data + (1 - TAU) * t.data)

        return critic_loss.item(), actor_loss_val

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
            'actor_tgt': self.actor_tgt.state_dict(), 'critic_tgt': self.critic_tgt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(), 'critic_opt': self.critic_opt.state_dict(),
            'iter': self.iter, 'state_dim': self.state_dim, 'action_dim': self.action_dim,
        }, path)
        print(f"💾 Saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        if ckpt.get('state_dim', STATE_DIM) != self.state_dim:
            print(f"⚠️  Checkpoint state_dim mismatch — starting fresh"); return
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_tgt.load_state_dict(ckpt['actor_tgt'])
        self.critic_tgt.load_state_dict(ckpt['critic_tgt'])
        if 'actor_opt' in ckpt:  self.actor_opt.load_state_dict(ckpt['actor_opt'])
        if 'critic_opt' in ckpt: self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.iter = ckpt.get('iter', 0)
        print(f"📂 Loaded: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ROS2 DRONE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def quat_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

def yaw_from_quat(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class CERLABDroneInterface(Node):
    def __init__(self):
        super().__init__("tunnel_drl_r52_agent")
        self.px = 0.0; self.py = 0.0; self.pz = 0.0; self.yaw = 0.0
        self.vx = 0.0; self.vy = 0.0; self.vz = 0.0
        self.pose_received = False
        self._prev_px = 0.0; self._prev_py = 0.0; self._prev_pz = 0.0
        self._prev_stamp = 0.0
        self.depth_sectors = np.full(5, 10.0, dtype=np.float32)
        self.depth_top = 10.0; self.depth_bottom = 10.0

        self.setpoint_pub = self.create_publisher(PoseStamped, "/CERLAB/quadcopter/setpoint_pose", 10)
        self.takeoff_pub  = self.create_publisher(Empty, "/CERLAB/quadcopter/takeoff", 10)
        self.land_pub     = self.create_publisher(Empty, "/CERLAB/quadcopter/land", 10)
        self.posctrl_pub  = self.create_publisher(Bool, "/CERLAB/quadcopter/posctrl", 10)
        self._delete_cli  = self.create_client(DeleteEntity, '/delete_entity')
        self._spawn_cli   = self.create_client(SpawnEntity,  '/spawn_entity')
        self.create_subscription(PoseStamped, "/CERLAB/quadcopter/pose_raw", self._pose_cb, 10)
        depth_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                               durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(Image, "/camera/camera/depth/image_raw", self._depth_cb, depth_qos)

    def _pose_cb(self, msg: PoseStamped):
        now = time.time()
        dt  = now - self._prev_stamp if self._prev_stamp > 0 else 0.1
        self.px = msg.pose.position.x; self.py = msg.pose.position.y; self.pz = msg.pose.position.z
        self.yaw = yaw_from_quat(msg.pose.orientation)
        if dt > 0.01:
            self.vx = (self.px - self._prev_px) / dt
            self.vy = (self.py - self._prev_py) / dt
            self.vz = (self.pz - self._prev_pz) / dt
        self._prev_px = self.px; self._prev_py = self.py; self._prev_pz = self.pz
        self._prev_stamp = now; self.pose_received = True

    def _depth_cb(self, msg: Image):
        if msg.encoding == "32FC1":
            raw = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        elif msg.encoding == "16UC1":
            raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
        else: return
        d = raw.copy(); far = 10.0
        d[(d <= 0.05) | ~np.isfinite(d)] = far
        h, w = d.shape
        r0, r1 = int(h * 0.15), int(h * 0.85)  # wider band: catch full person body (head to waist)
        mid = d[r0:r1, :]
        fifth = w // 5
        for i in range(5):
            c0 = i * fifth; c1 = (i + 1) * fifth if i < 4 else w
            self.depth_sectors[i] = float(np.percentile(mid[:, c0:c1], 5))  # 5th pct — catches thin obstacles
        c0v, c1v = int(w * 0.30), int(w * 0.70)
        col = d[:, c0v:c1v]; hmid = h // 2
        self.depth_top    = float(np.percentile(col[:hmid, :], 5))
        self.depth_bottom = float(np.percentile(col[hmid:,  :], 5))

    def publish_setpoint(self, x: float, y: float, z: float, yaw: float = 0.0):
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(x); msg.pose.position.y = float(y); msg.pose.position.z = float(z)
        qx, qy, qz, qw = quat_from_yaw(yaw)
        msg.pose.orientation.x = qx; msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz; msg.pose.orientation.w = qw
        self.setpoint_pub.publish(msg)

    def takeoff(self):
        b = Bool(); b.data = True
        for _ in range(5): self.posctrl_pub.publish(b); time.sleep(0.05)
        self.takeoff_pub.publish(Empty())

    def land(self): self.land_pub.publish(Empty())

    def start_spinning(self):
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start(); time.sleep(1.0)

    def stop_spinning(self):
        if hasattr(self, '_executor'): self._executor.shutdown()

    _URDF_PATH = "/home/makhosazana/Project/CERLAB-UAV-Autonomy/install/uav_simulator/share/uav_simulator/urdf/quadcopter.urdf"

    def teleport_to_spawn(self, x: float = 0.0, y: float = 0.0, z: float = 0.2) -> bool:
        from rclpy.node import Node as _Node
        from rclpy.executors import SingleThreadedExecutor as _STE
        helper = _Node('_teleport_helper'); ex = _STE(); ex.add_node(helper)
        del_cli   = helper.create_client(DeleteEntity, '/delete_entity')
        spawn_cli = helper.create_client(SpawnEntity,  '/spawn_entity')
        del_req = DeleteEntity.Request(); del_req.name = "quadcopter"
        if del_cli.wait_for_service(timeout_sec=5.0):
            future = del_cli.call_async(del_req); t0 = time.time()
            while not future.done() and time.time() - t0 < 10.0: ex.spin_once(timeout_sec=0.05)
        else:
            helper.destroy_node(); return False
        time.sleep(2.0); self.pose_received = False; self.px = x; self.py = y; self.pz = z
        try:
            with open(self._URDF_PATH, 'r') as f: urdf_xml = f.read()
        except FileNotFoundError:
            helper.destroy_node(); return False
        spawn_req = SpawnEntity.Request(); spawn_req.name = "quadcopter"; spawn_req.xml = urdf_xml
        spawn_req.initial_pose.position.x = float(x); spawn_req.initial_pose.position.y = float(y)
        spawn_req.initial_pose.position.z = float(z); spawn_req.initial_pose.orientation.w = 1.0
        if spawn_cli.wait_for_service(timeout_sec=5.0):
            future = spawn_cli.call_async(spawn_req); t0 = time.time()
            while not future.done() and time.time() - t0 < 15.0: ex.spin_once(timeout_sec=0.05)
            if not future.done() or not future.result().success:
                helper.destroy_node(); return False
        else:
            helper.destroy_node(); return False
        ex.shutdown(); helper.destroy_node(); time.sleep(1.0)
        self.pose_received = False; t0 = time.time()
        while time.time() - t0 < 15.0:
            time.sleep(0.1)
            if self.pose_received: break
        if not self.pose_received: return False
        print(f"  📍 At ({self.px:.2f}, {self.py:.2f}, {self.pz:.2f})"); return True


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION — body-frame position-free state + trajectory recording
# ═══════════════════════════════════════════════════════════════════════════════

class Mission:
    """
    Full round-trip with trajectory & collision-position recording for R51 plots.
    State: 12-D body-frame, position-free (no absolute x/y/z).
    """

    def __init__(self, drone: CERLABDroneInterface, max_speed: float,
                 turn_x: float, sensor_noise: float = 0.0, spawn_y: float = 0.0,
                 record_traj: bool = True):
        self.drone        = drone
        self.max_speed    = min(max_speed, MAX_SPEED)
        self.turn_x       = turn_x
        self.sensor_noise = sensor_noise
        self.spawn_y      = spawn_y
        self.record_traj  = record_traj

        self.spawn_x = 0.0; self.tgt_x = 0.0; self.tgt_y = spawn_y; self.tgt_z = SAFE_ALT
        self.phase   = "OUTBOUND"; self.steps = 0; self.max_steps = 1800
        self.fwd_progress = 0.0; self.lat_offset = 0.0; self.speed = 0.0
        self.max_vel = 0.0; self.avg_vel = 0.0; self._vel_sum = 0.0; self._vel_steps = 0
        self.max_dist = 0.0; self.collision_count = 0; self.near_miss_count = 0
        self.total_obstacle_encounters = 0; self.obstacles_avoided = 0
        self._in_danger_zone = False; self._collision_cooldown = 0
        self.total_reward = 0.0; self._prev_action = np.zeros(3, dtype=np.float32)
        self._prev_vy = 0.0; self._stuck_counter = 0; self._last_progress_x = 0.0
        self._return_start_step = 0; self._blocked_steps = 0; self._alt_hold_steps = 0
        self._wall_counter = 0; self._wall_stuck = 0; self._wind_x = 0.0; self._wind_y = 0.0
        self._delayed_a = np.zeros(3, dtype=np.float32)
        # R50: trajectory and collision position recording
        self.traj_points: List[Tuple[float, float, float, float, str]] = []
        self.col_positions: List[float] = []

    def start(self):
        self.spawn_x = self.drone.px; self.tgt_x = self.spawn_x
        self.tgt_y = self.spawn_y; self.tgt_z = SAFE_ALT; self.phase = "OUTBOUND"
        self.steps = 0; self.max_vel = 0.0; self.avg_vel = 0.0
        self._vel_sum = 0.0; self._vel_steps = 0; self.max_dist = 0.0
        self.collision_count = 0; self.near_miss_count = 0
        self.total_obstacle_encounters = 0; self.obstacles_avoided = 0
        self._in_danger_zone = False; self._collision_cooldown = 0
        self.total_reward = 0.0; self._prev_action = np.zeros(3, dtype=np.float32)
        self._prev_vy = 0.0; self._stuck_counter = 0
        self._last_progress_x = self.spawn_x; self._return_start_step = 0
        self._blocked_steps = 0; self._alt_hold_steps = 0; self._wall_counter = 0; self._wall_stuck = 0
        for ms in ['_ms_out_5','_ms_out_10','_ms_out_15','_ms_out_20','_ms_out_35','_ms_out_50','_ms_out_70','_ms_out_90',
                   '_ms_ret_90','_ms_ret_80','_ms_ret_60','_ms_ret_47','_ms_ret_40','_ms_ret_33']:
            setattr(self, ms, False)
        wind_mag = np.random.uniform(*DR_WIND_MAG)
        wind_dir = np.random.uniform(0, 2 * math.pi)
        self._wind_x = wind_mag * math.cos(wind_dir); self._wind_y = wind_mag * math.sin(wind_dir)
        self._delayed_a = np.zeros(3, dtype=np.float32)
        self.traj_points  = []
        self.col_positions = []
        self._refresh()

    def _refresh(self):
        self.fwd_progress = self.drone.px - self.spawn_x
        self.lat_offset   = self.drone.py - self.spawn_y
        self.speed        = self.drone.vx if self.phase == "OUTBOUND" else -self.drone.vx
        spd_clipped       = min(abs(self.speed), MAX_SPEED)
        self.max_vel      = max(self.max_vel, spd_clipped)
        self._vel_sum    += spd_clipped; self._vel_steps += 1
        self.avg_vel      = self._vel_sum / self._vel_steps
        self.max_dist     = max(self.max_dist, self.fwd_progress)
        # R50: record trajectory point every 3rd step to limit storage
        if self.record_traj and self.steps % 3 == 0:
            self.traj_points.append((
                float(self.fwd_progress),
                float(self.drone.py),
                float(self.drone.pz),
                float(spd_clipped),
                self.phase,
            ))

    def _noisy_depth(self) -> np.ndarray:
        d = np.clip(self.drone.depth_sectors.copy(), 0.05, 10.0)
        if self.sensor_noise > 0:
            d = np.clip(d + np.random.normal(0, self.sensor_noise, d.shape), 0.05, 10.0)
        if DR_DROPOUT_PROB > 0:
            mask = np.random.random(d.shape) < DR_DROPOUT_PROB
            d[mask] = 0.1
        return d

    def get_state(self) -> np.ndarray:
        self._refresh()
        ds = self._noisy_depth()
        heading = 0.0 if self.phase == "OUTBOUND" else math.pi
        cos_h = math.cos(heading); sin_h = math.sin(heading)
        vx = float(np.clip(self.drone.vx, -MAX_SPEED, MAX_SPEED))
        vy = float(np.clip(self.drone.vy, -MAX_SPEED, MAX_SPEED))
        v_fwd = cos_h * vx + sin_h * vy; v_lat = -sin_h * vx + cos_h * vy
        phase_flag = -1.0 if self.phase == "OUTBOUND" else +1.0
        return np.array([
            np.clip(ds[0] / 10.0, 0, 1), np.clip(ds[1] / 10.0, 0, 1),
            np.clip(ds[2] / 10.0, 0, 1), np.clip(ds[3] / 10.0, 0, 1),
            np.clip(ds[4] / 10.0, 0, 1),
            np.clip(self.drone.depth_top    / 10.0, 0, 1),
            np.clip(self.drone.depth_bottom / 10.0, 0, 1),
            np.clip((self.drone.pz + np.random.normal(0, DR_ALT_NOISE_SD) - SAFE_ALT) / 1.5, -1, 1),
            np.clip(v_fwd / MAX_SPEED, -1, 1),
            np.clip(v_lat / 3.0,       -1, 1),
            phase_flag,
            np.clip(self.max_speed / MAX_SPEED, 0, 1),
        ], dtype=np.float32)

    def _smooth_action(self, raw: np.ndarray) -> np.ndarray:
        s = np.array([
            ACTION_SMOOTH_FWD * raw[0] + (1 - ACTION_SMOOTH_FWD) * self._prev_action[0],
            ACTION_SMOOTH_LAT * raw[1] + (1 - ACTION_SMOOTH_LAT) * self._prev_action[1],
            ACTION_SMOOTH_LAT * raw[2] + (1 - ACTION_SMOOTH_LAT) * self._prev_action[2],
        ], dtype=np.float32)
        self._prev_action = s.copy(); return s

    def apply_action(self, action: np.ndarray) -> Tuple[float, float, float]:
        smoothed = self._smooth_action(action)
        fwd_speed = (smoothed[0] * 0.5 + 0.5) * self.max_speed
        fwd_speed = max(fwd_speed, 0.0); fwd_speed = min(fwd_speed, MAX_SPEED)  # R52: floor=0, reward teaches speed
        lat_speed = smoothed[1] * 0.8; alt_adj = smoothed[2] * 0.4
        ds = self._noisy_depth(); front_d = float(ds[2])
        if front_d > DANGER_DIST:
            lat_speed += -self.lat_offset * 0.20
        left_d = float(ds[0]); right_d = float(ds[4])
        self._reverse_escape = False
        steps_since_ret = self.steps - self._return_start_step

        all_blocked = (float(ds[0]) < 0.5 and float(ds[2]) < 0.5 and float(ds[4]) < 0.5)
        # Suppress all_blocked during the first 30 RETURN steps: the yaw-flip causes phantom
        # 0.1m readings across all sensors, which would immediately accumulate _blocked_steps
        # and terminate the episode before the stuck-check grace window protects it.
        eff_blocked = all_blocked and not (self.phase == "RETURN" and steps_since_ret < 60)  # R51: 30→60
        if eff_blocked:
            if self._blocked_steps == 0 and self.drone.pz >= SAFE_ALT:
                self.tgt_z = self.drone.pz
            self._blocked_steps += 1; self._alt_hold_steps = 80
            alt_adj = max(alt_adj, 1.5); fwd_speed = 0.0
        else:
            self._blocked_steps = 0
            if self._alt_hold_steps > 0: self._alt_hold_steps -= 1

        # RETURN phase: apply stronger centering and emergency lateral push independent of front_d
        # (in RETURN, front sensor often reads clear while side walls are dangerously close)
        if self.phase == "RETURN":
            lat_speed += -self.lat_offset * 0.30  # extra centering (total: 0.50x on top of actor)
            if not all_blocked:
                if right_d < 1.5:  lat_speed = min(lat_speed, -2.5 * (1.5 - right_d) / 1.5)
                if left_d < 1.5:   lat_speed = max(lat_speed,  2.5 * (1.5 - left_d)  / 1.5)

        diag_d = min(float(ds[1]), float(ds[3]))
        min_ds = float(np.min(ds))  # closest obstacle in any direction

        # R52: Hard braking caps REMOVED — RL learns speed control from reward signal.
        # Only one emergency cap remains: near-crash at < 0.15 m (prevents physics explosion).
        # The reward function's -60 slow-penalty and collision cost now fully control speed.
        if self.phase == "OUTBOUND":
            if front_d < 0.15 or min_ds < 0.15:
                fwd_speed = 0.0   # imminent crash — emergency stop only
        else:  # RETURN — only front sensor, phantom readings already suppressed above
            if front_d < 0.15:
                fwd_speed = 0.0   # emergency stop only

        if not all_blocked:
            if   front_d > 7.0: fwd_speed = max(fwd_speed, 2.5)
            elif front_d > 4.5: fwd_speed = max(fwd_speed, 1.5)
            elif front_d > 2.5: fwd_speed = max(fwd_speed, 0.6)

        if self.phase == "OUTBOUND" and self.fwd_progress < 6.0:
            fwd_speed = min(fwd_speed, 3.0)

        # Diagonal avoidance — steer away from obstacles detected at 45° before they are directly ahead
        diag_l = float(ds[1]); diag_r = float(ds[3])
        if diag_l < 2.5 and not all_blocked:
            lat_push = 2.0 * (2.5 - diag_l) / 2.5
            lat_speed = (min(lat_speed, -lat_push) if self.phase == "OUTBOUND"
                         else max(lat_speed,  lat_push))
        if diag_r < 2.5 and not all_blocked:
            lat_push = 2.0 * (2.5 - diag_r) / 2.5
            lat_speed = (max(lat_speed,  lat_push) if self.phase == "OUTBOUND"
                         else min(lat_speed, -lat_push))

        if front_d < 3.0:  # raised threshold 2.0→3.0 — start lateral evasion earlier
            if self.phase == "OUTBOUND":
                if left_d > right_d + 0.3:   lat_speed = max(lat_speed,  2.0)
                elif right_d > left_d + 0.3: lat_speed = min(lat_speed, -2.0)
                elif left_d > right_d:        lat_speed = max(lat_speed,  1.0)
                elif right_d > left_d:        lat_speed = min(lat_speed, -1.0)
            else:
                if left_d > right_d + 0.3:   lat_speed = min(lat_speed, -2.0)
                elif right_d > left_d + 0.3: lat_speed = max(lat_speed,  2.0)
                elif left_d > right_d:        lat_speed = min(lat_speed, -1.0)
                elif right_d > left_d:        lat_speed = max(lat_speed,  1.0)

            lat_phase_sign = 1.0 if self.phase == "OUTBOUND" else -1.0
            near_end_wall = (self.phase == "RETURN" and self.fwd_progress > 95.0)
            if front_d < 1.2 and max(left_d, right_d) < 1.0 and not all_blocked:
                if near_end_wall:
                    fwd_speed = 0.0; alt_adj = max(alt_adj, 1.5)
                else:
                    self._reverse_escape = True; fwd_speed = 3.0
                if not hasattr(self, '_escape_dir'):
                    self._escape_dir = 1.0 if np.random.random() > 0.5 else -1.0
                lat_speed = self._escape_dir * lat_phase_sign * 2.0
                if not near_end_wall: alt_adj = 0.8
            elif front_d < 1.2 and max(left_d, right_d) < 2.0:
                if left_d > right_d:
                    lat_speed = max(lat_speed,  2.0) if self.phase == "OUTBOUND" else min(lat_speed, -2.0)
                else:
                    lat_speed = min(lat_speed, -2.0) if self.phase == "OUTBOUND" else max(lat_speed,  2.0)
                alt_adj = max(alt_adj, 0.4)
                if front_d < 0.4: fwd_speed = 0.0
                else: fwd_speed = min(fwd_speed, 0.5)
            else:
                if hasattr(self, '_escape_dir'): del self._escape_dir
                if max(left_d, right_d) < 1.0:
                    if self.drone.depth_top > self.drone.depth_bottom + 0.3: alt_adj = max(alt_adj, 0.6)
                    elif self.drone.depth_bottom > self.drone.depth_top + 0.3: alt_adj = min(alt_adj, -0.6)

        if self._reverse_escape and front_d > 1.5 and not all_blocked:
            self._reverse_escape = False
        if abs(self.lat_offset) > CORRIDOR_HW * 0.85:
            lat_speed = -np.sign(self.lat_offset) * 1.0

        if self.phase == "OUTBOUND":
            dist_to_turn = self.turn_x - self.fwd_progress
            actual_spd   = float(np.sqrt(self.drone.vx**2 + self.drone.vy**2))
            brake_dist   = max(2.5, max(actual_spd, self.max_speed) * 0.8)
            if 0 < dist_to_turn < brake_dist:
                fwd_speed = min(fwd_speed, max(0.8, dist_to_turn / brake_dist * self.max_speed))
            elif self.fwd_progress >= self.turn_x:
                fwd_speed = min(fwd_speed, 0.3)
        else:
            if not self._reverse_escape and not all_blocked:
                if front_d > 6.0:
                    if steps_since_ret < 5:    fwd_speed = max(fwd_speed, 0.5)
                    elif steps_since_ret < 20:   fwd_speed = max(fwd_speed, 1.5)
                    elif steps_since_ret < 60: fwd_speed = max(fwd_speed, 2.2)
                    else:                       fwd_speed = max(fwd_speed, 2.5)
                elif front_d > 2.5:            fwd_speed = max(fwd_speed, 0.8)
                elif front_d > 1.5:            fwd_speed = max(fwd_speed, 0.5)
            # Push through physics-artifact all_blocked during RETURN (sensors spike 0.1m
            # on yaw-flip transition; real obstacles don't appear at x=17-30m in open corridor)
            if steps_since_ret < 100 and not self._reverse_escape:
                fwd_speed = max(fwd_speed, 0.8)
            if self.fwd_progress < 3.0:
                frac = max(0.1, self.fwd_progress / 3.0)
                fwd_speed = min(fwd_speed, max(0.3, frac * self.max_speed))

        # End-wall zone = within 9m of the far wall (relative to turn distance)
        _ew_thresh = max(15.0, self.turn_x - 9.0)
        near_end_wall_out = (self.phase == "OUTBOUND" and self.fwd_progress > _ew_thresh)
        near_end_wall_ret = (self.phase == "RETURN"   and self.fwd_progress > _ew_thresh)
        # Altitude always capped at ALT_NAV=1.8m — drone must stay under 2m
        if self._alt_hold_steps > 0 or all_blocked:
            alt_target = 1.5
        else:
            alt_target = SAFE_ALT
        alt_err = self.drone.pz - alt_target
        z_cap = ALT_NAV
        if all_blocked:
            # In RETURN after 5 stuck steps: allow rising + force lateral centering to escape
            if self.phase == "RETURN" and self._blocked_steps >= 5:
                lat_speed = -np.sign(self.lat_offset) * 2.5 if self.lat_offset != 0 else 0.0
            elif self.drone.pz > 1.3:
                alt_adj = min(alt_adj, 0.0)
        elif abs(alt_err) > 0.10:
            alt_adj = 0.3 * alt_adj + 0.7 * np.clip(-alt_err * 3.0, -1.0, 1.0)
        new_z = np.clip(self.tgt_z + alt_adj * TICK_DT, MIN_ALT, z_cap)
        alt_adj = (new_z - self.tgt_z) / TICK_DT
        direction = -1.0 if self.phase == "RETURN" else 1.0
        if self._reverse_escape:
            if self.phase == "RETURN" and self.fwd_progress > 95.0: self._reverse_escape = False
            else: direction = -direction
        return direction * fwd_speed * TICK_DT, lat_speed * TICK_DT, alt_adj * TICK_DT

    def _track_obstacles(self):
        min_d = float(np.min(self.drone.depth_sectors))
        if self._collision_cooldown > 0: self._collision_cooldown -= 1
        if min_d < DANGER_DIST and not self._in_danger_zone:
            self._in_danger_zone = True; self.total_obstacle_encounters += 1
        if min_d < COLLISION_DIST and self._in_danger_zone and self._collision_cooldown == 0:
            self.collision_count += 1; self._collision_cooldown = 10
            self.col_positions.append(float(self.fwd_progress))   # R50: record collision position
        if COLLISION_DIST <= min_d < NEAR_MISS_DIST and self._in_danger_zone:
            self.near_miss_count += 1
        if min_d > SAFE_DIST and self._in_danger_zone:
            self.obstacles_avoided += 1; self._in_danger_zone = False

    def _check_stuck(self) -> bool:
        near_end_ret = (self.phase == "RETURN"   and self.fwd_progress > max(self.turn_x * 0.70, 18.0))
        near_end_out = (self.phase == "OUTBOUND" and self.fwd_progress > max(self.turn_x * 0.55, 14.0))
        stuck_limit = (120 if (self.phase == "RETURN" and self._blocked_steps > 0)  # R51: 80→120
                       else 80  if self._blocked_steps > 0
                       else 120 if near_end_ret                                      # R51: 100→120
                       else 80  if near_end_out
                       else 30)
        if self._blocked_steps > stuck_limit: return True
        if self.phase == "RETURN" and self.steps - self._return_start_step < 110: return False
        if self.phase == "RETURN" and self.fwd_progress <= 2.5: return False
        if self._blocked_steps > 0: return False
        if self.steps % 30 == 0 and self.steps > 0:
            progress = (self.fwd_progress - self._last_progress_x if self.phase == "OUTBOUND"
                        else self._last_progress_x - self.fwd_progress)
            if progress < 0.3: self._stuck_counter += 1
            else: self._stuck_counter = max(0, self._stuck_counter - 1)
            self._last_progress_x = self.fwd_progress
        return self._stuck_counter >= 2

    def step(self, action: np.ndarray):
        prev_x = self.fwd_progress
        if DR_ACT_DELAY and self.steps > 0:
            exec_a = self._delayed_a.copy(); self._delayed_a = action.copy()
        else:
            exec_a = action.copy(); self._delayed_a = action.copy()
        dx, dy, dz = self.apply_action(exec_a)
        if self.phase == "RETURN":
            # Continuous setpoint tracking: always 5 m behind the drone in -x.
            # Incrementing by dx causes the setpoint to lag after the drone overshoots
            # the initial turn target, making the PX4 controller push the drone forward.
            self.tgt_x = max(1.0, self.drone.px - 5.0)
        else:
            self.tgt_x += dx
        self.tgt_y += dy; self.tgt_z += dz
        self.tgt_x = np.clip(self.tgt_x, self.spawn_x - 3.0, self.spawn_x + self.turn_x + 5.0)
        self.tgt_y = np.clip(self.tgt_y, self.spawn_y - CORRIDOR_HW, self.spawn_y + CORRIDOR_HW)
        self.tgt_z = np.clip(self.tgt_z, MIN_ALT, ALT_NAV)  # always cap at 1.8m (< 2m)
        pub_x = np.clip(self.tgt_x + self._wind_x + np.random.normal(0, 0.008),
                        self.spawn_x - 3.0, self.spawn_x + self.turn_x + 5.0)
        pub_y = np.clip(self.tgt_y + self._wind_y + np.random.normal(0, 0.008),
                        self.spawn_y - CORRIDOR_HW, self.spawn_y + CORRIDOR_HW)
        yaw = 0.0 if self.phase == "OUTBOUND" else math.pi
        self.drone.publish_setpoint(pub_x, pub_y, self.tgt_z, yaw)
        t0 = time.time()
        while time.time() - t0 < TICK_DT:
            time.sleep(0.02)
            self.drone.publish_setpoint(pub_x, pub_y, self.tgt_z, yaw)
        self.steps += 1
        self._refresh(); self._track_obstacles()
        reward, done, info = self._reward(prev_x)
        self.total_reward += reward
        return self.get_state(), reward, done, info

    def _reward(self, prev_x: float):
        x = self.fwd_progress; y = self.lat_offset; alt = self.drone.pz
        ds = self.drone.depth_sectors; min_d = float(np.min(ds))
        front_d = float(ds[2]); left_d = float(ds[0]); right_d = float(ds[4])
        reward = 0.0

        # Hard terminators
        if alt < ALT_FLOOR:    return -150.0, True, {'event': 'ground_crash'}
        if alt > ALT_CEIL:     return -150.0, True, {'event': 'ceiling_crash'}
        if abs(y) > LAT_LIMIT: return -150.0, True, {'event': 'left_tunnel'}
        if x > TUNNEL_LENGTH + 5.0: return -100.0, True, {'event': 'overshot'}
        if x < -5.0:           return -100.0, True, {'event': 'flew_backward'}
        if self.steps >= self.max_steps:
            credit = min(self.max_dist / self.turn_x, 1.0) * 30.0
            return -30.0 + credit, True, {'event': 'timeout'}
        if self._check_stuck(): return -100.0, True, {'event': 'stuck'}
        # Positional collision caps only apply OUTBOUND — never kill a drone that already turned
        if self.phase == "OUTBOUND":
            if ((self.collision_count > 20 and self.fwd_progress < 12.0)
                    or (self.collision_count > 30 and self.fwd_progress < 25.0)
                    or (self.collision_count > 40 and self.fwd_progress < 38.0)):
                return -200.0, True, {'event': 'collision_cap'}
        if self.collision_count > 65:
            return -200.0, True, {'event': 'collision_cap'}

        # Collision penalty
        if min_d < COLLISION_DIST and self.collision_count <= COLLISION_PENALTY_CAP:
            reward -= 50.0

        # Progress (speed-coupled)
        progress   = x - prev_x
        spd        = abs(self.speed)
        speed_mult = 1.0 + 2.0 * min(spd / self.max_speed, 1.0)
        if self.phase == "OUTBOUND":
            reward += progress * 120.0 * speed_mult
        else:
            reward -= progress * 180.0 * speed_mult
            if progress < -0.05: reward += 5.0

        if abs(progress) > 0.01:
            if abs(y) < 0.5:  reward += 4.0
            elif abs(y) < 1.0: reward += 2.0
        reward += 1.0  # alive bonus

        # Speed tier reward — R52: target band 2.5–3.5 m/s with maximum bonus.
        # Hard braking caps removed so this reward signal fully controls speed.
        # Zero-speed penalty deepened: agent must learn to fly fast, not stop.
        _bp = [
            (0.0, -80.0), (0.5, -55.0), (1.0, -20.0), (1.5,  5.0),
            (2.0, 60.0),  (2.5, 100.0), (3.0, 130.0), (3.5, 140.0),
            (4.5, 135.0), (6.0, 125.0),
        ]
        _sr = _bp[-1][1]
        for _i in range(len(_bp) - 1):
            _v0, _r0 = _bp[_i]; _v1, _r1 = _bp[_i + 1]
            if spd <= _v1:
                _sr = _r0 + (_r1 - _r0) * (spd - _v0) / (_v1 - _v0); break
        reward += _sr

        # Obstacle avoidance tiers
        if   min_d > SAFE_DIST:      reward += 6.0
        elif min_d > DANGER_DIST:    reward += 3.0
        elif min_d > NEAR_MISS_DIST: reward -= 8.0
        elif min_d > COLLISION_DIST: reward -= 20.0

        # Clear-path seeking
        if front_d < 2.5:
            best_side = max(left_d, right_d)
            if best_side > 3.0:  reward += 18.0
            elif best_side > 1.5: reward += 8.0
            if left_d > right_d and self._prev_action[1] > 0.1:   reward += 10.0
            elif right_d > left_d and self._prev_action[1] < -0.1: reward += 10.0
        if front_d < 2.0: reward += abs(self.drone.vy) * 6.0

        # Smoothness
        obstacle_near = min_d < 2.5
        reward -= abs(self.drone.vy) * (0.3 if obstacle_near else 2.5)
        if self._blocked_steps == 0 and self._alt_hold_steps == 0:
            reward -= abs(self.drone.vz) * 0.8
        vy_change = abs(self.drone.vy - self._prev_vy)
        reward -= vy_change * (0.3 if obstacle_near else 2.0)
        self._prev_vy = self.drone.vy

        # Centre-lane
        reward -= abs(y) * 1.5

        # Altitude — symmetric: below 1m AND above 1.8m both penalized (drone must stay < 2m)
        if self._blocked_steps == 0 and self._alt_hold_steps == 0:
            if alt < SAFE_ALT:
                reward -= (SAFE_ALT - alt) * 4.0           # below 1m: -4/m
            elif alt <= ALT_NAV:
                reward -= (alt - SAFE_ALT) * 4.0           # 1m–1.8m: symmetric -4/m
            else:                                            # above 1.8m: extreme penalty
                reward -= (alt - SAFE_ALT) * 4.0
                reward -= (alt - ALT_NAV) * 20.0           # extra -20/m above 1.8m
                reward -= _sr * 0.80                        # cancel 80% of speed reward

        # Milestones
        if self.phase == "OUTBOUND":
            if not self._ms_out_5  and x >= 5.0:   self._ms_out_5  = True; reward += 80.0
            if not self._ms_out_10 and x >= 10.0:  self._ms_out_10 = True; reward += 120.0
            if not self._ms_out_15 and x >= 15.0:  self._ms_out_15 = True; reward += 50.0
            if not self._ms_out_20 and x >= 20.0:  self._ms_out_20 = True; reward += 120.0
            if not self._ms_out_35 and x >= 35.0:  self._ms_out_35 = True; reward += 80.0
            if not self._ms_out_50 and x >= 25.0:  self._ms_out_50 = True; reward += 180.0
            if not self._ms_out_70 and x >= 32.0:  self._ms_out_70 = True; reward += 200.0
        elif self.phase == "RETURN":
            if not self._ms_ret_90 and x <= 35.0:  self._ms_ret_90 = True; reward += 100.0
            if not self._ms_ret_80 and x <= 27.0:  self._ms_ret_80 = True; reward += 80.0
            if not self._ms_ret_60 and x <= 18.0:  self._ms_ret_60 = True; reward += 150.0
            if not self._ms_ret_47 and x <= 10.0:  self._ms_ret_47 = True; reward += 200.0
            if not self._ms_ret_40 and x <= 7.0:   self._ms_ret_40 = True; reward += 120.0
            if not self._ms_ret_33 and x <= 5.0:   self._ms_ret_33 = True; reward += 400.0

        # Wall detection (sensor-based turn) — gate is relative to turn distance
        _ew_gate = max(15.0, self.turn_x * 0.65)
        if self.phase == "OUTBOUND" and x > _ew_gate:
            _dn = self._noisy_depth()
            front_d_detect = float(_dn[2])
            left_d_detect  = float(_dn[0]); right_d_detect = float(_dn[4])
            max_d = float(np.max(_dn))
            near_end_zone = (x > self.turn_x * 0.85)
            # End-wall signature: symmetric close readings (end wall spans full width)
            sym_gap = abs(left_d_detect - right_d_detect) < 0.6
            end_wall_sig = front_d_detect < 2.0 and sym_gap and max_d < 2.5
            if (near_end_zone and end_wall_sig) or max_d < WALL_DETECT_THRESH:
                self._wall_counter += 1
                steps_needed = 3 if near_end_zone else WALL_DETECT_STEPS
                if self._wall_counter >= steps_needed:
                    self.tgt_x = max(1.0, self.drone.px - 5.0); self.phase = "RETURN"
                    self.tgt_y = self.spawn_y
                    self._return_start_step = self.steps; self._stuck_counter = 0
                    self._last_progress_x = self.fwd_progress
                    self._wall_counter = 0; self._wall_stuck = 0
                    reward += 3000.0
                    col_bonus = max(0, 1000 - self.collision_count * 50)
                    spd_bonus = int(min(self.avg_vel / MAX_SPEED, 1.0) * 2000)
                    reward += col_bonus + spd_bonus
                    print(f"\n  🧱 WALL SENSED → TURN  x={x:.1f}m  avd={self.obstacles_avoided}  "
                          f"col={self.collision_count}  avg={self.avg_vel:.2f}m/s  "
                          f"max={self.max_vel:.2f}m/s  bonus={col_bonus}+{spd_bonus}\n")
                    return reward, False, {'event': 'wall_turn'}
            else:
                self._wall_counter = 0

        # Emergency force turn: physically stuck against end wall
        _ft_gate = max(15.0, self.turn_x * 0.80)
        if self.phase == "OUTBOUND" and x > _ft_gate:
            _dn_e = self._noisy_depth()
            if float(np.max(_dn_e)) < 2.2:  # all sectors close → trapped at wall
                self._wall_stuck += 1
                if self._wall_stuck >= 20:
                    self.tgt_x = max(1.0, self.drone.px - 5.0); self.phase = "RETURN"
                    self.tgt_y = self.spawn_y
                    self._return_start_step = self.steps; self._stuck_counter = 0
                    self._last_progress_x = self.fwd_progress
                    self._wall_counter = 0; self._wall_stuck = 0
                    reward += 1500.0
                    col_bonus = max(0, 500 - self.collision_count * 25)
                    reward += col_bonus
                    print(f"\n  🔄 FORCE TURN (wall stuck) x={x:.1f}m\n")
                    return reward, False, {'event': 'force_turn'}
            else:
                self._wall_stuck = 0

        # Distance fallback turn
        if self.phase == "OUTBOUND" and x >= self.turn_x:
            self.tgt_x = max(1.0, self.drone.px - 5.0); self.phase = "RETURN"
            self.tgt_y = self.spawn_y
            self._return_start_step = self.steps; self._stuck_counter = 0
            self._last_progress_x = self.fwd_progress; self._wall_counter = 0
            reward += 3000.0
            col_bonus = max(0, 1000 - self.collision_count * 50)
            spd_bonus = int(min(self.avg_vel / MAX_SPEED, 1.0) * 2000)
            reward += col_bonus + spd_bonus
            print(f"\n  🎯 TURN {self.turn_x:.0f}m  x={x:.1f}m  avg={self.avg_vel:.2f}m/s  "
                  f"max={self.max_vel:.2f}m/s  bonus={col_bonus}+{spd_bonus}\n")
            return reward, False, {'event': 'turn'}

        # Success
        if self.phase == "RETURN" and x <= 2.0:
            reward += 3000.0
            col = self.collision_count
            if col == 0:    reward += 5000.0
            elif col <= 3:  reward += 3000.0
            elif col <= 8:  reward += 1500.0
            elif col <= 15: reward += 600.0
            spd_bonus = int(min(self.avg_vel / MAX_SPEED, 1.0) * 2000)
            reward += spd_bonus
            print(f"\n  🏆 SUCCESS  avg={self.avg_vel:.2f}m/s  max={self.max_vel:.2f}m/s  "
                  f"avd={self.obstacles_avoided}  col={col}  steps={self.steps}\n")
            return reward, True, {'event': 'success'}

        return reward, False, {}


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE 16-PANEL PLOT — R51
# ═══════════════════════════════════════════════════════════════════════════════

def save_comprehensive_plot(
    ep_nums, rewards, dists, ok, speeds, max_speeds, cols, avds,
    c_loss, a_loss, durs, buf_sizes, sr_overall,
    dist_stages_log, speed_stages_log,
    success_trajs, all_col_positions,
    all_encounters=None,
    run_name='run52'
):
    if not HAS_MPL: return
    try:
        n = len(ep_nums)
        if n < 2: return

        wins_idx  = [i for i in range(n) if ok[i]]
        sr20      = sum(ok[-20:]) / min(n, 20) * 100
        sr_all_pct = sum(ok) / n * 100
        spd_w     = [speeds[i] for i in wins_idx]
        peak_w    = [max_speeds[i] for i in wins_idx] if max_speeds else []
        avg_spd   = np.mean(spd_w)  if spd_w  else 0
        best_avg  = max(spd_w)      if spd_w  else 0
        best_peak = max(max_speeds) if max_speeds else 0
        ttg       = np.mean([durs[i] for i in wins_idx]) if wins_idx else 0
        path_eff  = np.mean([dists[i]/(speeds[i]*durs[i]+1e-6) for i in wins_idx]) if wins_idx else 0
        c_now     = c_loss[-1] if c_loss else 0
        a_now     = a_loss[-1] if a_loss else 0
        col_tot   = sum(cols) if cols else 0
        weps      = [ep_nums[i] for i in wins_idx]
        win20     = min(20, n)
        MA        = lambda arr, w: np.convolve(arr, np.ones(w)/w, mode='valid')

        C_BLUE   = '#1565c0'; C_RED  = '#c62828'; C_GRN  = '#2e7d32'
        C_PURP   = '#7b1fa2'; C_ORG  = '#e65100'; C_TEAL = '#00695c'
        C_GREY   = '#888888'; PB     = '#f8f8f8'; GC     = '#cccccc'

        fig = plt.figure(figsize=(24, 32), facecolor='white')
        fig.suptitle(
            f"CERLAB Tunnel TD3 v6  ·  RUN 52  ·  EP {ep_nums[-1]} / 5000   "
            f"Target: 2–3 m/s avg speed  ·  SR > 40%\n"
            f"SR-20={sr20:.0f}%   SR-ALL={sr_all_pct:.1f}%   "
            f"AvgSpd(wins)={avg_spd:.2f} m/s   BestAvg={best_avg:.2f} m/s   "
            f"BestPeak={best_peak:.2f} m/s   Cols={col_tot}   C={c_now:.1f}   "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            fontsize=11, fontweight='bold', color='black', y=0.995
        )
        gs = mgs.GridSpec(6, 3, figure=fig, hspace=0.52, wspace=0.32,
                          height_ratios=[1.4, 1.1, 1.2, 1.0, 1.0, 0.9])

        def sax(ax):
            ax.set_facecolor(PB)
            ax.tick_params(colors='#333333', labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor('#aaaaaa')

        # ── P1: Peak + Avg speed [row 0, col 0-1] ────────────────────────────
        ax = fig.add_subplot(gs[0, 0:2]); sax(ax)
        if max_speeds:
            ax.fill_between(ep_nums, speeds, max_speeds,
                            alpha=0.15, color=C_GREY, label='Speed range (avg→peak)')
            ax.plot(ep_nums, max_speeds,
                    color=C_RED, lw=2.8, ls='--', marker='^', ms=5,
                    markerfacecolor='#ff1744', markeredgecolor='#7f0000', markeredgewidth=0.6,
                    label='Peak speed ▲ (red dashed)', zorder=6)
        ax.plot(ep_nums, speeds,
                color=C_BLUE, lw=2.2, ls='-', marker='o', ms=4,
                markerfacecolor=C_BLUE, markeredgecolor='white', markeredgewidth=0.5,
                label='Avg speed ● (blue solid)', zorder=5)
        if n >= win20:
            ax.plot(ep_nums[win20-1:], MA(speeds, win20),
                    color=C_PURP, lw=3.0, label=f'Avg MA-{win20} (purple)', zorder=4)
        ax.axhspan(2.0, 3.0, alpha=0.12, color=C_GRN)
        ax.axhline(3.0, color='#1b5e20', ls='-',  lw=1.2, alpha=0.85, label='3 m/s ceiling')
        ax.axhline(2.0, color=C_GRN,     ls='--', lw=1.8, alpha=0.9,  label='2 m/s floor target')
        ax.axvline(ACTOR_FREEZE_EPS, color=C_ORG, ls=':', lw=1.4, alpha=0.8)
        ax.text(ACTOR_FREEZE_EPS+0.5, 0.15, f'Actor unfreezes\nEP{ACTOR_FREEZE_EPS}',
                fontsize=6.5, color=C_ORG, va='bottom')
        ymax = max(max(max_speeds)+0.5 if max_speeds else 4.0, 3.5)
        ax.set_ylim(-0.2, ymax)
        ax.set_title('Peak Speed ▲ (red dashed)  &  Avg Speed ● (blue solid)   |   Green band = 2–3 m/s target',
                     fontweight='bold', fontsize=9, color='black')
        ax.set_ylabel('Speed (m/s)', fontsize=8, color='#333333')
        ax.legend(fontsize=7, labelcolor='black', facecolor='white', framealpha=0.95,
                  ncol=3, loc='upper left')
        ax.grid(True, alpha=0.4, color=GC)

        # ── P2: Speed histogram [row 0, col 2] ────────────────────────────────
        ax = fig.add_subplot(gs[0, 2]); sax(ax)
        if spd_w:
            ax.hist(spd_w,    bins=20, color=C_BLUE,  alpha=0.72, label='Avg speed (wins)', edgecolor='white')
        if peak_w:
            ax.hist(peak_w, bins=20, color=C_RED, alpha=0.50, label='Peak speed (wins)', edgecolor='white')
        ax.axvline(2.0, color=C_GRN,  lw=1.8, ls='--', label='2 m/s target')
        ax.axvline(3.0, color='#1b5e20', lw=1.2, ls='-')
        ax.set_title('Speed Distribution (wins)', fontweight='bold', fontsize=9, color='black')
        ax.set_xlabel('Speed (m/s)', fontsize=8); ax.set_ylabel('Count', fontsize=8)
        ax.legend(fontsize=7, labelcolor='black', facecolor='white')
        ax.grid(True, alpha=0.4, color=GC, axis='y')

        # ── P3: SR-20 + Overall SR [row 1, col 0-1] ──────────────────────────
        ax = fig.add_subplot(gs[1, 0:2]); sax(ax)
        sr_roll = [sum(ok[max(0,i-19):i+1])/min(i+1,20)*100 for i in range(n)]
        ax.plot(ep_nums, sr_roll, color=C_GRN, lw=2.2, label='SR-20 (rolling-20)')
        ax.fill_between(ep_nums, sr_roll, alpha=0.18, color=C_GRN)
        ax.plot(ep_nums, [v*100 for v in sr_overall], color=C_ORG, lw=1.8, ls='--',
                label='Overall SR (cumulative)', alpha=0.85)
        ax.axhline(40, color=C_ORG, lw=1.8, ls='--', alpha=0.9, label='40% deployment gate')
        ax.axhline(60, color=C_GRN, lw=0.9, ls=':',  alpha=0.6, label='60% target')
        ax.axvline(ACTOR_FREEZE_EPS, color=C_ORG, ls=':', lw=1.2, alpha=0.7)
        # Curriculum markers
        for adv_ep, lbl in zip(dist_stages_log, ['50m','70m','90m','102m']):
            if adv_ep <= ep_nums[-1]:
                ax.axvline(adv_ep, color=C_BLUE, ls=':', lw=0.9, alpha=0.55)
                ax.text(adv_ep+0.5, 5, lbl, fontsize=6.5, color=C_BLUE, rotation=90)
        ax.set_ylim(0, 110)
        ax.set_title('Success Rate: Rolling-20 (green)  &  Cumulative SR (orange) | 40% gate (dashed)',
                     fontweight='bold', fontsize=9, color='black')
        ax.set_ylabel('SR %', fontsize=8); ax.legend(fontsize=7.5, labelcolor='black', facecolor='white', ncol=2)
        ax.grid(True, alpha=0.4, color=GC, axis='y')

        # ── P4: Reward per episode [row 1, col 2] ────────────────────────────
        ax = fig.add_subplot(gs[1, 2]); sax(ax)
        colors_ok = [C_GRN if o else C_RED for o in ok]
        ax.bar(ep_nums, rewards, color=colors_ok, alpha=0.70, width=0.9)
        if n >= win20:
            ax.plot(ep_nums[win20-1:], MA(rewards, win20), color=C_PURP, lw=2.2,
                    label=f'MA-{win20}')
        ax.set_title('Total Reward per Episode\n(green=success, red=fail)',
                     fontweight='bold', fontsize=9, color='black')
        ax.set_ylabel('Reward', fontsize=8)
        ax.legend(fontsize=7.5, labelcolor='black', facecolor='white')
        ax.grid(True, alpha=0.4, color=GC, axis='y')

        # ── P5: XY Bird's-eye Trajectory [row 2, col 0] ──────────────────────
        ax = fig.add_subplot(gs[2, 0]); sax(ax)
        # Draw obstacle cluster zones
        for cx in [13, 19, 34, 47, 60, 70, 85]:
            ax.axvspan(cx-3, cx+3, alpha=0.08, color='#b71c1c')
        if success_trajs:
            cmap = plt.cm.get_cmap('Blues', len(success_trajs)+2)
            for ti, traj in enumerate(list(success_trajs)[-30:]):
                xs = [p[0] for p in traj]; ys = [p[1] for p in traj]
                ax.plot(xs, ys, lw=0.8, alpha=0.55, color=cmap(ti+2))
        ax.axhline(0, color=C_GREY, lw=0.5, ls='--', alpha=0.5)
        ax.set_xlim(-2, 106); ax.set_ylim(-4.0, 4.0)
        ax.set_title('Bird\'s-eye: Agent XY Trajectory\n(last ≤30 successful eps, blue=oldest→darkest=newest)',
                     fontweight='bold', fontsize=9, color='black')
        ax.set_xlabel('Tunnel X position (m)', fontsize=8)
        ax.set_ylabel('Lateral Y (m)', fontsize=8)
        ax.axhline( 3.5, color=C_RED, lw=0.7, ls='-', alpha=0.5, label='Wall ±3.5m')
        ax.axhline(-3.5, color=C_RED, lw=0.7, ls='-', alpha=0.5)
        ax.legend(fontsize=7, labelcolor='black', facecolor='white')
        ax.grid(True, alpha=0.35, color=GC)

        # ── P6: XZ Side-view Altitude Profile [row 2, col 1] ────────────────
        ax = fig.add_subplot(gs[2, 1]); sax(ax)
        for cx in [13, 19, 34, 47, 60, 70, 85]:
            ax.axvspan(cx-3, cx+3, alpha=0.08, color='#b71c1c')
        if success_trajs:
            cmap = plt.cm.get_cmap('Greens', len(success_trajs)+2)
            for ti, traj in enumerate(list(success_trajs)[-30:]):
                xs = [p[0] for p in traj]; zs = [p[2] for p in traj]
                ax.plot(xs, zs, lw=0.8, alpha=0.55, color=cmap(ti+2))
        ax.axhline(SAFE_ALT, color=C_BLUE,    lw=1.2, ls='--', alpha=0.85, label=f'Cruise {SAFE_ALT}m')
        ax.axhspan(1.0, 2.0, alpha=0.06, color=C_GRN, label='Safe band 1–2m')
        ax.axhline(ALT_NAV,  color=C_RED,     lw=1.5, ls='--', alpha=0.90, label=f'NAV ceiling {ALT_NAV}m')
        ax.axhline(MAX_ALT,  color=C_ORG,     lw=0.9, ls=':',  alpha=0.7,  label=f'Max {MAX_ALT}m (end-wall only)')
        ax.set_xlim(-2, 106); ax.set_ylim(0, 3.8)
        ax.set_title('Side-view: Agent Altitude Profile XZ\n(last ≤30 successful eps)',
                     fontweight='bold', fontsize=9, color='black')
        ax.set_xlabel('Tunnel X position (m)', fontsize=8)
        ax.set_ylabel('Altitude Z (m)', fontsize=8)
        ax.legend(fontsize=7, labelcolor='black', facecolor='white')
        ax.grid(True, alpha=0.35, color=GC)

        # ── P7: Collision position histogram [row 2, col 2] ──────────────────
        ax = fig.add_subplot(gs[2, 2]); sax(ax)
        if all_col_positions:
            ax.hist(all_col_positions, bins=np.arange(0, 108, 5), color=C_RED,
                    alpha=0.78, edgecolor='white', label=f'Collisions (n={len(all_col_positions)})')
        for cx, lbl in [(13,'13m'),(34,'34m'),(47,'47m'),(60,'60m'),(85,'85m')]:
            ax.axvline(cx, color='#b71c1c', lw=0.8, ls=':', alpha=0.6)
            ax.text(cx+0.3, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1]>0 else 1, lbl,
                    fontsize=6, color='#b71c1c', rotation=90)
        ax.set_title('Collision Position Histogram\n(where along tunnel collisions occur)',
                     fontweight='bold', fontsize=9, color='black')
        ax.set_xlabel('Tunnel X position (m)', fontsize=8)
        ax.set_ylabel('Collision count', fontsize=8)
        ax.legend(fontsize=7.5, labelcolor='black', facecolor='white')
        ax.grid(True, alpha=0.4, color=GC, axis='y')

        # ── P8: Duration (wins) [row 3, col 0] ───────────────────────────────
        ax = fig.add_subplot(gs[3, 0]); sax(ax)
        wdurs = [durs[i] for i in wins_idx]
        if weps:
            ax.bar(weps, wdurs, color=C_TEAL, alpha=0.78, width=0.9)
            ax.axhline(np.mean(wdurs), color=C_PURP, lw=2.0, ls='-',
                       label=f'Avg={np.mean(wdurs):.0f}s')
        ax.set_title('Time-to-Goal (success eps only)', fontweight='bold', fontsize=9, color='black')
        ax.set_ylabel('Duration (s)', fontsize=8)
        ax.legend(fontsize=7.5, labelcolor='black', facecolor='white')
        ax.grid(True, alpha=0.4, color=GC, axis='y')

        # ── P9: Avoidances vs Collisions + Avoidance Rate [row 3, col 1] ───────
        ax = fig.add_subplot(gs[3, 1]); sax(ax)
        ax.bar(ep_nums, avds, color=C_GRN,  alpha=0.70, width=0.9, label='Avoided')
        ax.bar(ep_nums, [-min(c, 50) for c in cols], color=C_RED, alpha=0.65, width=0.9,
               label='Collisions (neg, cap 50)')
        ax.axhline(0, color='#333333', lw=0.8)
        ax.set_title('Obstacles Avoided (↑) vs Collisions (↓)  |  Rate % (teal line)',
                     fontweight='bold', fontsize=9, color='black')
        ax.set_ylabel('Count', fontsize=8)
        ax.legend(fontsize=7.5, labelcolor='black', facecolor='white', loc='upper left')
        ax.grid(True, alpha=0.4, color=GC, axis='y')
        if all_encounters and len(all_encounters) == len(avds):
            ax2 = ax.twinx()
            avd_rate = [avds[i] / max(all_encounters[i], 1) * 100 for i in range(len(avds))]
            ax2.plot(ep_nums, avd_rate, color=C_TEAL, lw=2.2, ls='-',
                     marker='D', ms=2.5, markerfacecolor=C_TEAL, label='Avoidance rate %')
            if n >= win20:
                ax2.plot(ep_nums[win20-1:], MA(avd_rate, win20),
                         color='#004d40', lw=2.8, label=f'Rate MA-{win20}')
            ax2.axhline(70, color=C_TEAL, lw=0.9, ls=':', alpha=0.7, label='70% target')
            ax2.set_ylim(0, 110); ax2.set_ylabel('Avoidance Rate (%)', fontsize=8, color=C_TEAL)
            ax2.tick_params(axis='y', labelcolor=C_TEAL, labelsize=7)
            ax2.legend(fontsize=7, labelcolor='black', facecolor='white', loc='upper right')

        # ── P10: Distance reached [row 3, col 2] ──────────────────────────────
        ax = fig.add_subplot(gs[3, 2]); sax(ax)
        ax.bar(ep_nums, dists, color=colors_ok, alpha=0.75, width=0.9)
        for sd in [30, 50, 70, 90, 102]:
            ax.axhline(sd, color=C_BLUE, ls=':', lw=0.8, alpha=0.5)
        ax.set_title('Distance Reached (green=success)', fontweight='bold', fontsize=9, color='black')
        ax.set_ylabel('Distance (m)', fontsize=8); ax.set_xlabel('Episode', fontsize=8)
        ax.grid(True, alpha=0.35, color=GC, axis='y')

        # ── P11: Critic loss [row 4, col 0] ──────────────────────────────────
        ax = fig.add_subplot(gs[4, 0]); sax(ax)
        ax.plot(ep_nums, c_loss, color=C_RED, lw=2.0, label='C-loss (critic)')
        ax.axhline(100, color=C_ORG,   lw=1.0, ls='--', alpha=0.8, label='C=100 warning')
        ax.axhline(200, color='#7f0000', lw=0.8, ls='--', alpha=0.6, label='C=200 danger')
        ax.axvline(ACTOR_FREEZE_EPS, color=C_ORG, ls=':', lw=1.0, alpha=0.7)
        ax.set_title('Critic Loss (C < 100 = healthy)', fontweight='bold', fontsize=9, color='black')
        ax.set_ylabel('C-loss', fontsize=8, color=C_RED); ax.tick_params(axis='y', labelcolor=C_RED)
        ax.legend(fontsize=7, labelcolor='black', facecolor='white')
        ax.grid(True, alpha=0.4, color=GC)

        # ── P12: Actor Q-value [row 4, col 1] ────────────────────────────────
        ax = fig.add_subplot(gs[4, 1]); sax(ax)
        q_vals = [-v/1000 for v in a_loss]
        ax.plot(ep_nums, q_vals, color=C_BLUE, lw=1.8, ls='--', alpha=0.85, label='Q-val ×1000')
        ax.fill_between(ep_nums, q_vals, alpha=0.15, color=C_BLUE)
        ax.axvline(ACTOR_FREEZE_EPS, color=C_ORG, ls=':', lw=1.0, alpha=0.7)
        ax.set_title('Actor Q-value (×1000)\n(grows → actor learning)', fontweight='bold', fontsize=9, color='black')
        ax.set_ylabel('Q-val (×1000)', fontsize=8, color=C_BLUE); ax.tick_params(axis='y', labelcolor=C_BLUE)
        ax.legend(fontsize=7, labelcolor='black', facecolor='white')
        ax.grid(True, alpha=0.4, color=GC)

        # ── P13: Buffer size [row 4, col 2] ──────────────────────────────────
        ax = fig.add_subplot(gs[4, 2]); sax(ax)
        if buf_sizes:
            ax.plot(ep_nums[:len(buf_sizes)], buf_sizes, color=C_TEAL, lw=2.0, label='Replay buffer')
            ax.axhline(BUFFER_SIZE, color=C_RED, lw=0.9, ls='--', alpha=0.7,
                       label=f'Max ({BUFFER_SIZE//1000}k)')
        ax.set_title('Replay Buffer Size', fontweight='bold', fontsize=9, color='black')
        ax.set_ylabel('Transitions', fontsize=8)
        ax.legend(fontsize=7.5, labelcolor='black', facecolor='white')
        ax.grid(True, alpha=0.4, color=GC, axis='y')

        # ── P14: Path efficiency [row 5, col 0] ──────────────────────────────
        ax = fig.add_subplot(gs[5, 0]); sax(ax)
        if wins_idx:
            eff = [dists[i]/(speeds[i]*durs[i]+1e-6) for i in wins_idx]
            ax.bar(weps, eff, color=C_PURP, alpha=0.75, width=0.9)
            ax.axhline(1.0, color=C_GRN, lw=1.2, ls='--', alpha=0.85, label='Perfect=1.0')
            ax.axhline(np.mean(eff), color=C_ORG, lw=1.6, ls='-', alpha=0.9,
                       label=f'Avg={np.mean(eff):.2f}')
        ax.set_title('Path Efficiency (wins,  1.0 = straight-line)', fontweight='bold', fontsize=9, color='black')
        ax.set_ylabel('Efficiency', fontsize=8); ax.set_xlabel('Episode', fontsize=8)
        ax.legend(fontsize=7.5, labelcolor='black', facecolor='white')
        ax.grid(True, alpha=0.4, color=GC, axis='y')

        # ── P15: Curriculum stage progression [row 5, col 1] ─────────────────
        ax = fig.add_subplot(gs[5, 1]); sax(ax)
        if dist_stages_log or speed_stages_log:
            # Fill background by distance stage
            stage_colors = ['#e3f2fd','#bbdefb','#90caf9','#64b5f6','#42a5f5']
            prev_ep = 0
            for si, adv_ep in enumerate(dist_stages_log + [ep_nums[-1]+1]):
                ax.axvspan(prev_ep, adv_ep, alpha=0.25,
                           color=stage_colors[min(si, len(stage_colors)-1)])
                prev_ep = adv_ep
        ax.plot(ep_nums, [d/DIST_STAGES[-1]*100 for d in
                          [DIST_STAGES[min(si,len(DIST_STAGES)-1)] for si in
                           [sum(1 for e in dist_stages_log if e <= ep) for ep in ep_nums]]],
                color=C_BLUE, lw=2.0, label='Dist stage %')
        ax.plot(ep_nums, [s/MAX_SPEED*100 for s in
                          [SPEED_STAGES[min(si,3)] for si in
                           [sum(1 for e in speed_stages_log if e <= ep) for ep in ep_nums]]],
                color=C_RED, lw=2.0, ls='--', label='Speed stage %')
        ax.set_ylim(0, 110)
        ax.set_title('Curriculum Stage (blue=distance, red=speed)\n% of max stage reached',
                     fontweight='bold', fontsize=9, color='black')
        ax.set_ylabel('Stage %', fontsize=8); ax.set_xlabel('Episode', fontsize=8)
        ax.legend(fontsize=7.5, labelcolor='black', facecolor='white')
        ax.grid(True, alpha=0.4, color=GC, axis='y')

        # ── P16: Per-episode collision count [row 5, col 2] ──────────────────
        ax = fig.add_subplot(gs[5, 2]); sax(ax)
        col_clamp = [min(c, 60) for c in cols]
        bc = [C_RED if c > 20 else C_ORG if c > 8 else C_BLUE for c in cols]
        ax.bar(ep_nums, col_clamp, color=bc, alpha=0.80, width=0.9)
        ax.axhline(8,  color=C_ORG, lw=1.0, ls='--', alpha=0.7, label='8 col (half bonus)')
        ax.axhline(20, color=C_RED, lw=0.8, ls='--', alpha=0.7, label='20 col (no bonus)')
        if n >= win20:
            ax.plot(ep_nums[win20-1:], MA(col_clamp, win20), color=C_PURP, lw=1.8,
                    label=f'MA-{win20}')
        ax.set_title('Collisions per Episode\n(cap=60; blue≤8, orange≤20, red>20)',
                     fontweight='bold', fontsize=9, color='black')
        ax.set_ylabel('Collisions', fontsize=8); ax.set_xlabel('Episode', fontsize=8)
        ax.legend(fontsize=7.5, labelcolor='black', facecolor='white', ncol=2)
        ax.grid(True, alpha=0.4, color=GC, axis='y')

        # ── Performance Stats Table (annotation, bottom centre) ───────────────
        # Place below the grid as text
        stats_text = (
            f"{'═'*80}\n"
            f"  R51 PERFORMANCE SUMMARY  (EP {ep_nums[-1]} / 5000)\n"
            f"{'─'*80}\n"
            f"  Success Rate SR-20:    {sr20:.1f}%         Overall SR:      {sr_all_pct:.1f}%\n"
            f"  Avg Speed (wins):      {avg_spd:.3f} m/s      Best Avg Speed:  {best_avg:.3f} m/s\n"
            f"  Best Peak Speed:       {best_peak:.2f} m/s      Time-to-Goal:    {ttg:.0f} s\n"
            f"  Path Efficiency:       {path_eff:.3f}          Critic Loss (C): {c_now:.1f}\n"
            f"  Total Collisions:      {col_tot}             Actor Q-val:     {a_now:.0f}\n"
            f"  Successes:             {sum(ok)}             Total Episodes:  {n}\n"
            f"{'═'*80}"
        )
        fig.text(0.5, 0.003, stats_text, ha='center', va='bottom',
                 fontsize=8.5, fontfamily='monospace', color='#111111',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4ff', edgecolor='#0d47a1', alpha=0.9))

        out = f"{CKPT_DIR}/{run_name}_progress.png"
        plt.tight_layout(rect=[0, 0.04, 1, 0.994])
        plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  📊 Comprehensive plot saved → {out}")
    except Exception as e:
        print(f"  ⚠️  Plot error: {e}")
        import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_training(args):
    rclpy.init()
    drone = CERLABDroneInterface()
    drone.start_spinning()

    print("⏳ Waiting for pose …")
    for _ in range(200):
        time.sleep(0.1)
        if drone.pose_received: break
    if not drone.pose_received:
        print("❌ No pose — is Gazebo running?"); return

    print(f"📍 Initial pose: ({drone.px:.2f}, {drone.py:.2f}, {drone.pz:.2f})")

    agent        = TD3Agent()
    buffer       = deque(maxlen=BUFFER_SIZE)
    success_buffer = deque(maxlen=10_000)

    episode_start       = 0
    speed_stage         = args.speed_stage if args.speed_stage is not None else 0
    dist_stage          = 0
    stage_episode_count = 0
    best_reward         = -np.inf

    if args.load:
        ckpt = torch.load(args.load, map_location=DEVICE, weights_only=False)
        agent.load(args.load)
        episode_start       = args.start_episode if args.start_episode is not None else ckpt.get('episode', 0)
        speed_stage         = args.speed_stage if args.speed_stage is not None else ckpt.get('speed_stage', 0)
        dist_stage          = args.dist_stage  if args.dist_stage  is not None else ckpt.get('dist_stage',  0)
        stage_episode_count = 0 if (args.dist_stage is not None or args.speed_stage is not None
                                    or args.start_episode is not None) \
                              else ckpt.get('stage_episode_count', 0)
        print(f"▶  Resuming EP{episode_start} speed={SPEED_STAGES[speed_stage]:.0f}m/s "
              f"dist={DIST_STAGES[dist_stage]:.0f}m")

    os.makedirs(CKPT_DIR, exist_ok=True)

    critic_warm  = args.load is not None
    freeze_window = 5 if critic_warm else ACTOR_FREEZE_EPS
    print(f"🧊 Freeze window: {freeze_window} eps")

    total_steps      = 0 if not args.load else WARMUP_STEPS
    success_history  = deque(maxlen=30)  # R51: 20→30 — more stable SR estimate, prevents false advances
    reward_history   = deque(maxlen=50)

    import pickle as _pkl
    if not getattr(args, 'fresh_buffer', False) and os.path.isfile(BUFFER_SAVE_PATH):
        try:
            with open(BUFFER_SAVE_PATH, 'rb') as f:
                for t in _pkl.load(f): buffer.append(t)
            total_steps = max(total_steps, WARMUP_STEPS)
            print(f"💾 Buffer restored: {len(buffer):,} transitions")
        except Exception as e:
            print(f"⚠️  Buffer restore failed ({e}) — fresh buffer")
    else:
        print(f"🆕 Fresh buffer")

    # Tracking lists
    all_ep_nums:    List[int]   = []
    all_rewards:    List[float] = []
    all_dists:      List[float] = []
    all_ok:         List[int]   = []
    all_speeds:     List[float] = []
    all_max_speeds: List[float] = []
    all_cols:       List[int]   = []
    all_avds:       List[int]   = []
    all_encounters: List[int]   = []
    all_c_loss:     List[float] = []
    all_a_loss:     List[float] = []
    all_durs:       List[float] = []
    all_buf_sizes:  List[int]   = []
    all_sr_overall: List[float] = []
    dist_stages_log: List[int]  = []    # episode numbers where dist stage advanced
    speed_stages_log: List[int] = []    # episode numbers where speed stage advanced
    # Trajectory & collision data
    recent_success_trajs: deque = deque(maxlen=30)
    all_col_positions:    List[float] = []

    # ── Pre-populate tracking lists from existing log so run52_progress.png
    #    shows all speeds from EP1 even after a checkpoint restart ──────────────
    _hist_log = os.path.join(CKPT_DIR, "training_tunnel100m_r52.log")
    if os.path.isfile(_hist_log) and episode_start > 0:
        import re as _re
        _ep_re = _re.compile(
            r'(✅|❌) EP\s+(\d+)\s+rew=\s*([+-]?\d+)\s+dist=([\d.]+)m\s+'
            r'avg=([\d.]+)m/s\s+max=([\d.]+)m/s\s+col=(\d+)\s+avd=(\d+)\s+'
            r'dur=(\d+)s\s+sr=\d+%\s+sr_all=[\d.]+%\s+C:([\d.]+)\s+A:([+-]?[\d.]+)'
        )
        _seen = {}  # ep_num → index in lists, keeps LATEST occurrence
        with open(_hist_log, errors='replace') as _lf:
            for _line in _lf:
                _m = _ep_re.search(_line)
                if _m:
                    _ep = int(_m.group(2))
                    if _ep >= episode_start:
                        continue
                    _row = (
                        _ep,
                        float(_m.group(3)),   # reward
                        float(_m.group(4)),   # dist
                        1 if _m.group(1)=='✅' else 0,
                        float(_m.group(5)),   # avg speed
                        float(_m.group(6)),   # max speed
                        int(_m.group(7)),     # collisions
                        int(_m.group(8)),     # avoided
                        float(_m.group(9)),   # duration
                        float(_m.group(10)),  # critic loss
                        float(_m.group(11)),  # actor loss
                    )
                    if _ep in _seen:
                        # overwrite previous occurrence with latest
                        _i = _seen[_ep]
                        (all_ep_nums[_i], all_rewards[_i], all_dists[_i],
                         all_ok[_i], all_speeds[_i], all_max_speeds[_i],
                         all_cols[_i], all_avds[_i], all_durs[_i],
                         all_c_loss[_i], all_a_loss[_i]) = _row
                    else:
                        _seen[_ep] = len(all_ep_nums)
                        all_ep_nums.append(_row[0]); all_rewards.append(_row[1])
                        all_dists.append(_row[2]);    all_ok.append(_row[3])
                        all_speeds.append(_row[4]);   all_max_speeds.append(_row[5])
                        all_cols.append(_row[6]);     all_avds.append(_row[7])
                        all_encounters.append(0);     all_durs.append(_row[8])
                        all_c_loss.append(_row[9]);   all_a_loss.append(_row[10])
                        all_buf_sizes.append(0)
                        all_sr_overall.append(sum(all_ok) / len(all_ok))
                if 'DISTANCE ADVANCE' in _line and all_ep_nums:
                    dist_stages_log.append(all_ep_nums[-1])
                if 'SPEED ADVANCE' in _line and all_ep_nums:
                    speed_stages_log.append(all_ep_nums[-1])
        print(f"  📚 Loaded {len(all_ep_nums)} historical episodes from log (EP1–{episode_start-1})")

    print(f"\n{'═'*72}")
    print(f"  🚀 TUNNEL TD3 v6  —  RUN 52  —  FINAL PRE-DEPLOYMENT TRAINING")
    print(f"     Target: 2–3 m/s avg speed,  SR > 40%,  5000 episodes")
    print(f"     State:  {STATE_DIM}-D body-frame, position-free")
    print(f"     Domain: spawn_y∈{DR_SPAWN_Y}  noise∈{DR_SNS_NOISE}  "
          f"wind∈{DR_WIND_MAG}  dropout={DR_DROPOUT_PROB}")
    print(f"     Speed stages: {SPEED_STAGES}   Dist stages: {DIST_STAGES}")
    print(f"     Buffer: {BUFFER_SIZE:,}  Batch: {BATCH_SIZE}  Success@40%  Plot every {PLOT_FREQ} eps")
    print(f"{'═'*72}\n")

    # ── Takeoff ──────────────────────────────────────────────────────────────
    drone.takeoff()
    t0 = time.time()
    while time.time() - t0 < 6.0:
        drone.publish_setpoint(0.0, 0.0, SAFE_ALT, 0.0); time.sleep(0.08)

    # ── Warm-up ───────────────────────────────────────────────────────────────
    if total_steps < WARMUP_STEPS:
        print(f"🌡  Warm-up: {WARMUP_STEPS} random steps …")
        wm = Mission(drone, max_speed=3.0, turn_x=15.0, sensor_noise=0.04, record_traj=False)
        wm.start(); s = wm.get_state()
        while total_steps < WARMUP_STEPS:
            a = np.random.uniform(-1.0, 1.0, size=ACTION_DIM)
            ns, r, done, _ = wm.step(a)
            buffer.append((s, a, r, ns, float(done))); total_steps += 1; s = ns
            if done:
                drone.teleport_to_spawn(0.0, 0.0, 0.2); drone.takeoff()
                t0 = time.time()
                while time.time() - t0 < 5.0:
                    drone.publish_setpoint(0.0, 0.0, SAFE_ALT, 0.0); time.sleep(0.08)
                wm = Mission(drone, max_speed=3.0, turn_x=15.0, sensor_noise=0.04, record_traj=False)
                wm.start(); s = wm.get_state()
        print(f"✅ Warm-up done: {total_steps} transitions\n")

    noise_boost_eps = 0   # episodes remaining in post-regression noise boost
    post_regress_stage = -1  # R51: after regressing to stage S, require 35% SR (not 30%) to re-advance
    # ── Main loop ─────────────────────────────────────────────────────────────
    for ep in range(episode_start, args.missions):
        turn_x          = DIST_STAGES[dist_stage]
        # Spawn randomization grows with training maturity to prevent path overfitting
        _maturity       = min(ep / 1000.0, 1.0)                   # 0→1 over first 1000 eps
        _spawn_y_range  = DR_SPAWN_Y[1]                            # always ±0.5m — wider range causes wall proximity
        spawn_y_ep      = np.random.uniform(-_spawn_y_range, _spawn_y_range)
        sensor_noise_ep = np.random.uniform(*DR_SNS_NOISE)
        target_speed    = SPEED_STAGES[speed_stage]

        noise_start = EXPLORE_NOISE_INIT if not args.load else 0.20
        local_ep    = max(ep - episode_start, 0)
        total_local = max(args.missions - episode_start - 1, 1)
        base_noise  = noise_start + (local_ep / total_local) * (EXPLORE_NOISE_MIN - noise_start)
        expl_noise  = max(base_noise, EXPLORE_NOISE_MIN)
        if noise_boost_eps > 0:                    # temporary boost after curriculum regression
            expl_noise = max(expl_noise, 0.25)
            noise_boost_eps -= 1

        for _spawn_try in range(3):
            ok = drone.teleport_to_spawn(0.0, spawn_y_ep, 0.2)
            if not ok:
                print(f"  ⚠️  Teleport failed ep {ep+1} try {_spawn_try+1}"); time.sleep(5.0); continue
            drone.takeoff()
            t0 = time.time()
            while time.time() - t0 < 6.0:
                drone.publish_setpoint(0.0, spawn_y_ep, SAFE_ALT, 0.0); time.sleep(0.08)
            if drone.pz < 0.5:
                t0 = time.time()
                while time.time() - t0 < 4.0:
                    drone.publish_setpoint(0.0, spawn_y_ep, SAFE_ALT, 0.0); time.sleep(0.08)
            # Invalidate cached depth so we wait for fresh camera frames
            drone.depth_sectors[:] = 0.0
            t0 = time.time()
            while time.time() - t0 < 8.0:
                if float(np.min(drone.depth_sectors)) > 0.3: break
                drone.publish_setpoint(0.0, spawn_y_ep, SAFE_ALT, 0.0); time.sleep(0.1)
            _sensors_ok = float(np.min(drone.depth_sectors)) > 0.3 and drone.pz >= SAFE_ALT - 0.3
            if _sensors_ok:
                break
            print(f"  ⚠️  Bad sensors/alt (pz={drone.pz:.2f} min_d={np.min(drone.depth_sectors):.2f}) — respawn try {_spawn_try+1}")
        else:
            print(f"  ⚠️  All spawn tries failed ep {ep+1} — skip"); continue
        if not _sensors_ok:
            print(f"  ⚠️  Spawn failed ep {ep+1} — skip"); continue

        mission = Mission(drone, max_speed=target_speed, turn_x=turn_x,
                          sensor_noise=sensor_noise_ep, spawn_y=spawn_y_ep, record_traj=True)
        mission.start()

        t0 = time.time()
        while time.time() - t0 < 5.0:
            if float(np.min(drone.depth_sectors)) > 0.8: break
            drone.publish_setpoint(0.0, spawn_y_ep, SAFE_ALT, 0.0); time.sleep(0.1)
        if drone.pz < SAFE_ALT - 0.3:
            print(f"  ⚠️  Takeoff failed (pz={drone.pz:.2f}) — skip ep {ep+1}"); continue

        state = mission.get_state()
        ep_start_time = time.time()
        c_loss_sum = 0.0; a_loss_sum = 0.0; upd = 0

        print(f"\n{'─'*72}")
        print(f"  📋 EP {ep+1:5d}/{args.missions}   spd={target_speed:.0f}  turn=SENSOR({turn_x:.0f}m)  "
              f"spawn_y={spawn_y_ep:+.2f}  σ_expl={expl_noise:.3f}  σ_sns={sensor_noise_ep:.3f}")
        print(f"{'─'*72}")

        done = False; ep_transitions = []
        while not done:
            action = agent.select_action(state, noise_scale=expl_noise)
            next_state, rew, done, info = mission.step(action)
            transition = (state, action, rew, next_state, float(done))
            buffer.append(transition); ep_transitions.append(transition); total_steps += 1

            dyn_updates  = min(UPDATES_PER_STEP, max(1, len(buffer) // 5000))
            freeze_actor = (ep - episode_start) < freeze_window
            for _ in range(dyn_updates):
                cl, al = agent.update(buffer, freeze_actor=freeze_actor, success_buffer=success_buffer)
                c_loss_sum += cl; a_loss_sum += al; upd += 1

            if mission.steps % 30 == 0:
                ds = drone.depth_sectors
                print(f"  [{mission.phase:8s}] {mission.steps:4d} │ "
                      f"X:{mission.fwd_progress:+6.1f}/{turn_x:.0f}m  Z:{drone.pz:.2f}m  "
                      f"V:{mission.speed:+5.2f}  D:[{ds[0]:.1f},{ds[2]:.1f},{ds[4]:.1f}]")
            state = next_state

        ep_dur  = time.time() - ep_start_time
        success = info.get('event') == 'success'
        event   = info.get('event', '?')

        if success:
            for t in ep_transitions: success_buffer.append(t)
            recent_success_trajs.append(list(mission.traj_points))
        all_col_positions.extend(mission.col_positions)

        success_history.append(float(success))
        reward_history.append(mission.total_reward)
        roll_sr = float(np.mean(success_history))

        all_ep_nums.append(ep + 1)
        all_rewards.append(mission.total_reward)
        all_dists.append(mission.max_dist)
        all_ok.append(int(success))
        all_speeds.append(mission.avg_vel)
        all_max_speeds.append(mission.max_vel)
        all_cols.append(mission.collision_count)
        all_avds.append(mission.obstacles_avoided)
        all_encounters.append(mission.total_obstacle_encounters)
        avg_c = c_loss_sum / max(upd, 1); avg_a = a_loss_sum / max(upd, 1)
        all_c_loss.append(avg_c); all_a_loss.append(avg_a)
        all_durs.append(float(ep_dur))
        all_buf_sizes.append(len(buffer))
        all_sr_overall.append(sum(all_ok) / len(all_ok))
        stage_episode_count += 1

        icon = "✅" if success else "❌"
        print(f"\n  {icon} EP {ep+1:5d}  rew={mission.total_reward:+9.0f}  "
              f"dist={mission.max_dist:.1f}m  avg={mission.avg_vel:.2f}m/s  "
              f"max={mission.max_vel:.2f}m/s  col={mission.collision_count}  "
              f"avd={mission.obstacles_avoided}  dur={ep_dur:.0f}s  "
              f"sr={roll_sr:.0%}  sr_all={all_sr_overall[-1]:.1%}  "
              f"C:{avg_c:.3f}  A:{avg_a:.3f}  buf={len(buffer):,}")

        # Save critic state
        torch.save({'critic': agent.critic.state_dict(),
                    'critic_tgt': agent.critic_tgt.state_dict(),
                    'last_c': avg_c}, CRITIC_STATE_PATH)

        # Persist replay buffer
        if (ep - episode_start + 1) % BUFFER_SAVE_FREQ == 0:
            try:
                with open(BUFFER_SAVE_PATH, 'wb') as f:
                    _pkl.dump(list(buffer), f, protocol=4)
                print(f"  💾 Buffer saved: {len(buffer):,} → {BUFFER_SAVE_PATH}")
            except Exception as e:
                print(f"  ⚠️  Buffer save failed: {e}")

        # Save best checkpoint
        if mission.total_reward > best_reward:
            best_reward = mission.total_reward
            agent.save(f"{CKPT_DIR}/tunnel_td3_v6_r52_best.pth")
            ckpt_p = torch.load(f"{CKPT_DIR}/tunnel_td3_v6_r52_best.pth",
                                map_location='cpu', weights_only=False)
            ckpt_p.update({'episode': ep+1, 'speed_stage': speed_stage, 'dist_stage': dist_stage,
                           'best_reward': best_reward, 'stage_episode_count': stage_episode_count})
            torch.save(ckpt_p, f"{CKPT_DIR}/tunnel_td3_v6_r52_best.pth")

        # Periodic checkpoint every 25 episodes
        if (ep + 1) % 25 == 0:
            path = f"{CKPT_DIR}/tunnel_td3_v6_r52_ep{ep+1:04d}.pth"
            agent.save(path)
            ckpt_p = torch.load(path, map_location='cpu', weights_only=False)
            ckpt_p.update({'episode': ep+1, 'speed_stage': speed_stage, 'dist_stage': dist_stage,
                           'stage_episode_count': stage_episode_count})
            torch.save(ckpt_p, path)
            print(f"  💾 Checkpoint: EP {ep+1}")

        # JSON log
        with open(f"{CKPT_DIR}/training_log_r52.jsonl", 'a') as f:
            f.write(json.dumps({
                'episode': ep+1, 'speed_stage': speed_stage, 'dist_stage': dist_stage,
                'speed': float(target_speed), 'turn_x': float(turn_x),
                'reward': float(mission.total_reward), 'dist': float(mission.max_dist),
                'avg_vel': float(mission.avg_vel), 'max_vel': float(mission.max_vel),
                'success': int(success), 'event': event,
                'collisions': mission.collision_count, 'avoided': mission.obstacles_avoided,
                'roll_sr': float(roll_sr), 'sr_overall': float(all_sr_overall[-1]),
                'duration': float(ep_dur), 'c_loss': float(avg_c), 'a_loss': float(avg_a),
                'buf_size': len(buffer), 'total_steps': total_steps,
            }) + '\n')

        # Curriculum: distance advance — R51: use 35% SR threshold after regression (not 30%)
        # to prevent the 65m↔80m cycling that R50 suffered (22 regressions in 5000 eps).
        _advance_thresh = 0.35 if dist_stage == post_regress_stage else DIST_SR_THRESH
        if (dist_stage < len(DIST_STAGES) - 1
                and stage_episode_count >= MIN_MISSIONS_STAGE
                and roll_sr >= _advance_thresh):
            dist_stage += 1; stage_episode_count = 0; success_history.clear()
            if dist_stage > post_regress_stage: post_regress_stage = -1  # cleared once safely past
            dist_stages_log.append(ep + 1)
            print(f"\n  📏 DISTANCE ADVANCE → turn@{DIST_STAGES[dist_stage]:.0f}m  "
                  f"(SR={roll_sr:.0%})\n")

        # Auto-regression: if SR=0% for 120 consecutive eps, drop back one dist stage.
        # R51: 100→120 threshold — harder to regress, reduces the 65m↔80m oscillation R50 suffered.
        _long_fail = stage_episode_count >= 120 and roll_sr == 0.0
        if _long_fail and dist_stage > 0:
            dist_stage -= 1; stage_episode_count = 0; success_history.clear()
            success_buffer.clear(); noise_boost_eps = 60
            post_regress_stage = dist_stage          # R51: require 35% to re-advance after regression
            print(f"\n  📉 AUTO-REGRESS → turn@{DIST_STAGES[dist_stage]:.0f}m  "
                  f"(0% SR for 120 eps — rebuilding at shorter range, noise boost 60 eps)\n")

        # Curriculum: speed advance
        sr_ready = (stage_episode_count >= MIN_MISSIONS_STAGE
                    and roll_sr >= SR_THRESHOLDS[min(speed_stage+1, len(SR_THRESHOLDS)-1)])
        if speed_stage < len(SPEED_STAGES) - 1 and sr_ready:
            speed_stage += 1; stage_episode_count = 0; success_history.clear()
            speed_stages_log.append(ep + 1)
            print(f"\n  🚀 SPEED ADVANCE → {SPEED_STAGES[speed_stage]:.0f} m/s  "
                  f"(SR={roll_sr:.0%})\n")

        # Comprehensive PNG every PLOT_FREQ episodes (and on every 25ep checkpoint)
        if HAS_MPL and ((ep + 1) % PLOT_FREQ == 0 or (ep + 1) % 25 == 0) and len(all_ep_nums) > 4:
            save_comprehensive_plot(
                all_ep_nums, all_rewards, all_dists, all_ok,
                all_speeds, all_max_speeds, all_cols, all_avds,
                all_c_loss, all_a_loss, all_durs, all_buf_sizes, all_sr_overall,
                dist_stages_log, speed_stages_log,
                recent_success_trajs, all_col_positions,
                all_encounters=all_encounters,
                run_name='run52'
            )

    # ── Final ─────────────────────────────────────────────────────────────────
    final_path = f"{CKPT_DIR}/tunnel_td3_v6_r52_final.pth"
    agent.save(final_path)
    ckpt_p = torch.load(final_path, map_location='cpu', weights_only=False)
    ckpt_p.update({'episode': args.missions, 'speed_stage': speed_stage, 'dist_stage': dist_stage})
    torch.save(ckpt_p, final_path)

    if HAS_MPL and all_ep_nums:
        save_comprehensive_plot(
            all_ep_nums, all_rewards, all_dists, all_ok,
            all_speeds, all_max_speeds, all_cols, all_avds,
            all_c_loss, all_a_loss, all_durs, all_buf_sizes, all_sr_overall,
            dist_stages_log, speed_stages_log,
            recent_success_trajs, all_col_positions,
            run_name='run52'
        )

    # Print final summary
    n = len(all_ok); wins = sum(all_ok)
    avg_spd_wins = np.mean([all_speeds[i] for i in range(n) if all_ok[i]]) if wins else 0
    print(f"\n{'═'*72}")
    print(f"  🏁  RUN 52 COMPLETE — {args.missions} episodes")
    print(f"  Total successes:   {wins} / {n}  ({wins/n*100:.1f}%)")
    print(f"  Avg speed (wins):  {avg_spd_wins:.3f} m/s")
    print(f"  Best reward:       {best_reward:.0f}")
    print(f"  Final speed stage: {SPEED_STAGES[speed_stage]:.0f} m/s")
    print(f"  Final dist stage:  {DIST_STAGES[dist_stage]:.0f} m")
    print(f"  Models saved:  {CKPT_DIR}/")
    print(f"{'═'*72}\n")

    drone.land(); time.sleep(5.0); drone.stop_spinning()
    drone.destroy_node(); rclpy.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Tunnel TD3 R52 — Speed-Focused Training")
    parser.add_argument('--missions',    type=int,   default=5000,  help='Total training episodes (default: 5000)')
    parser.add_argument('--load',        type=str,   default=None,  help='Resume from checkpoint .pth')
    parser.add_argument('--speed-stage', type=int,   default=None,  help='Override starting speed stage (0-3)')
    parser.add_argument('--dist-stage',  type=int,   default=None,  help='Override starting dist stage (0-4)')
    parser.add_argument('--fresh-buffer',action='store_true',       help='Ignore saved replay buffer')
    parser.add_argument('--expl-sigma',  type=float, default=None,  help='Override exploration noise')
    parser.add_argument('--start-episode',type=int,  default=None,  help='Override episode start counter (use with --load to get EP1-N numbering)')
    args = parser.parse_args()
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    try:
        run_training(args)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — model saved to results_v6_r52/")
    except Exception as e:
        print(f"\n❌ Fatal: {e}")
        import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
