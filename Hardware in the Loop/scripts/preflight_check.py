#!/usr/bin/env python3
"""
Pre-Flight Check — HITL R54 Test
==================================
Runs a sequence of safety checks before authorising HITL test episodes.
All checks must pass before running run_hitl_r54.py.

Usage:
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 "Hardware in the Loop/scripts/preflight_check.py"
"""

import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu
import numpy as np

PROJECT    = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT / "src/tunnel_drl/results_r54/r54_best.pth"

SAFE_ALT      = 1.5
CORRIDOR_HW   = 2.5
NUM_H_SECTORS = 9
DEPTH_FAR     = 10.0
DEPTH_PERCENTILE = 5


class PreflightNode(Node):
    def __init__(self):
        super().__init__("hitl_preflight")
        self.state: State    = None
        self.odom:  Odometry = None
        self.imu:   Imu      = None
        self.depth_img: Image = None
        self.depth_min  = DEPTH_FAR

        self.create_subscription(State,    "/mavros/state",               self._s_cb, 10)
        self.create_subscription(Odometry, "/mavros/local_position/odom", self._o_cb, 10)
        self.create_subscription(Imu,      "/mavros/imu/data",            self._i_cb, 10)
        self.create_subscription(Image,    "/camera/camera/depth/image_raw",
                                 self._d_cb, 10)

    def _s_cb(self, m): self.state = m
    def _o_cb(self, m): self.odom  = m
    def _i_cb(self, m): self.imu   = m
    def _d_cb(self, m):
        self.depth_img = m
        if m.encoding == "32FC1":
            raw = np.frombuffer(m.data, dtype=np.float32).reshape(m.height, m.width)
        elif m.encoding == "16UC1":
            raw = np.frombuffer(m.data, dtype=np.uint16).reshape(
                m.height, m.width).astype(np.float32) / 1000.0
        else:
            return
        raw[(raw <= 0.05) | ~np.isfinite(raw)] = DEPTH_FAR
        self.depth_min = float(np.min(raw))


def wait_for(fn, label, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn():
            return True
        time.sleep(0.2)
    return False


CHECK = "✓"; CROSS = "✗"; WARN = "⚠"


def main():
    rclpy.init()
    node = PreflightNode()
    ex   = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    t = threading.Thread(target=ex.spin, daemon=True)
    t.start()

    results = []

    print("\n" + "="*60)
    print("  PRE-FLIGHT CHECK — R54 HITL")
    print("="*60)

    def chk(ok, label, detail=""):
        sym = CHECK if ok else CROSS
        line = f"  [{sym}] {label}"
        if detail:
            line += f"  ({detail})"
        print(line)
        results.append(ok)
        return ok

    # 1. Checkpoint file exists
    chk(CHECKPOINT.exists(), "R54 checkpoint (r54_best.pth)", str(CHECKPOINT))

    # 2. MAVROS connected
    ok = wait_for(lambda: node.state is not None and node.state.connected,
                  "MAVROS heartbeat", timeout=12.0)
    chk(ok, "MAVROS ↔ FC heartbeat",
        f"mode={node.state.mode} armed={node.state.armed}" if node.state else "TIMEOUT")

    # 3. FC not armed at start (safe state)
    if node.state:
        chk(not node.state.armed, "FC disarmed (safe start state)",
            "armed — DISARM before test" if node.state.armed else "DISARMED")
    else:
        chk(False, "FC disarmed (safe start state)", "no FC state")

    # 4. Local position estimate
    ok = wait_for(lambda: node.odom is not None, "local position estimate", timeout=10.0)
    if ok:
        p = node.odom.pose.pose.position
        chk(True, "EKF position estimate",
            f"x={p.x:.2f} y={p.y:.2f} z={p.z:.2f} m")
    else:
        chk(False, "EKF position estimate", "TIMEOUT — check HITL_ENABLE=1 on FC")

    # 5. IMU data
    ok = wait_for(lambda: node.imu is not None, "IMU", timeout=8.0)
    if ok:
        a = node.imu.linear_acceleration
        chk(True, "IMU data", f"az={a.z:.2f} m/s² (gravity check)")
    else:
        chk(False, "IMU data", "TIMEOUT")

    # 6. Depth camera
    ok = wait_for(lambda: node.depth_img is not None, "depth camera", timeout=10.0)
    if ok:
        img = node.depth_img
        chk(True, "Depth camera", f"{img.width}x{img.height}  enc={img.encoding}  min_d={node.depth_min:.2f}m")
        # Warn if drone is too close to something at spawn
        if node.depth_min < 0.5:
            print(f"  [{WARN}] Min depth={node.depth_min:.2f}m — drone may be clipped into obstacle at spawn")
    else:
        chk(False, "Depth camera", "TIMEOUT — Gazebo running?")

    # 7. Spawn position within corridor
    if node.odom:
        p = node.odom.pose.pose.position
        in_corridor = abs(p.y) <= CORRIDOR_HW
        chk(in_corridor, "Spawn within corridor bounds",
            f"|y|={abs(p.y):.2f}m ≤ {CORRIDOR_HW}m")
    else:
        chk(False, "Spawn within corridor bounds", "no odom")

    # 8. PX4 HITL mode parameter (inferred from mode string)
    if node.state:
        mode = node.state.mode
        hitl_likely = mode in ("MANUAL", "STABILIZED", "POSCTL", "OFFBOARD", "HITL", "")
        chk(hitl_likely, "FC mode compatible with HITL", f"mode={mode}")
    else:
        chk(False, "FC mode", "no state")

    # ── Summary ─────────────────────────────────────────────────────────────────
    n_pass = sum(results); n_total = len(results)
    print("-"*60)
    print(f"  Result: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print(f"  [{CHECK}] ALL CHECKS PASSED — safe to run run_hitl_r54.py")
    else:
        print(f"  [{CROSS}] {n_total - n_pass} check(s) failed — resolve before running episodes")
    print("="*60 + "\n")

    ex.shutdown(); rclpy.shutdown()
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
