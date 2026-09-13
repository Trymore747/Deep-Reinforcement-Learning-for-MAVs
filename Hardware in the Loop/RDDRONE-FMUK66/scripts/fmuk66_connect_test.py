#!/usr/bin/env python3
"""
RDDRONE-FMUK66 Connection Test
================================
Verifies the FMUK66 is connected, responsive via MAVROS, and ready for HITL.

Usage:
  source /opt/ros/humble/setup.bash
  python3 scripts/fmuk66_connect_test.py [--port /dev/ttyACM0] [--baud 57600]
"""

import sys
import time
import argparse
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

OK   = "\033[32m[OK]  \033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
INFO = "\033[34m[--]  \033[0m"

parser = argparse.ArgumentParser()
parser.add_argument("--port", default="/dev/ttyACM0")
parser.add_argument("--baud", type=int, default=57600)
args = parser.parse_args()


class ConnectTestNode(Node):
    def __init__(self):
        super().__init__("fmuk66_connect_test")
        self.state = None
        self.odom  = None
        self.imu   = None
        self.create_subscription(State,    "/mavros/state",               self._s, 10)
        self.create_subscription(Odometry, "/mavros/local_position/odom", self._o, 10)
        self.create_subscription(Imu,      "/mavros/imu/data",            self._i, 10)

    def _s(self, m): self.state = m
    def _o(self, m): self.odom  = m
    def _i(self, m): self.imu   = m


def wait(fn, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn():
            return True
        time.sleep(0.2)
    return False


def main():
    print()
    print("══════════════════════════════════════════════════")
    print("  RDDRONE-FMUK66 Connection Test")
    print(f"  Port: {args.port}  Baud: {args.baud}")
    print("══════════════════════════════════════════════════")

    rclpy.init()
    node = ConnectTestNode()
    ex   = SingleThreadedExecutor()
    ex.add_node(node)
    t = threading.Thread(target=ex.spin, daemon=True)
    t.start()

    results = []

    def chk(ok, label, detail=""):
        sym = OK if ok else FAIL
        print(f"  {sym} {label}" + (f"  ({detail})" if detail else ""))
        results.append(ok)

    # 1. MAVROS heartbeat
    ok = wait(lambda: node.state is not None and node.state.connected, timeout=15.0)
    s  = node.state
    chk(ok, "MAVROS ↔ FMUK66 heartbeat",
        f"connected={s.connected} mode={s.mode}" if s else "TIMEOUT — check USB cable and MAVROS launch")

    # 2. HITL mode active
    if s:
        hitl_mode = s.mode in ("MANUAL", "STABILIZED", "POSCTL", "OFFBOARD", "HITL", "")
        chk(hitl_mode, "FC mode compatible with HITL", f"mode={s.mode}")
    else:
        chk(False, "FC mode", "no state received")

    # 3. FC disarmed
    if s:
        chk(not s.armed, "FC disarmed at start",
            "DISARMED ✓" if not s.armed else "ARMED — disarm before test!")
    else:
        chk(False, "FC disarmed", "no state")

    # 4. EKF position estimate
    ok = wait(lambda: node.odom is not None, timeout=12.0)
    if ok:
        p = node.odom.pose.pose.position
        chk(True, "EKF local position",
            f"x={p.x:.2f} y={p.y:.2f} z={p.z:.2f} m")
    else:
        chk(False, "EKF local position",
            "TIMEOUT — is HITL_ENABLE=1 set on the FC? Is Gazebo running?")

    # 5. IMU data
    ok = wait(lambda: node.imu is not None, timeout=8.0)
    if ok:
        az = node.imu.linear_acceleration.z
        chk(True, "IMU data", f"az={az:.2f} m/s² (gravity ≈ ±9.8)")
    else:
        chk(False, "IMU data", "TIMEOUT")

    # ── Summary ──────────────────────────────────────────────────────────────
    n_pass = sum(results)
    n_total = len(results)
    print("──────────────────────────────────────────────────")
    print(f"  Result: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print(f"  {OK}All checks passed — FMUK66 ready for HITL")
    else:
        print(f"  {FAIL}{n_total - n_pass} check(s) failed — resolve before running episodes")
        print()
        print("  Common fixes:")
        print("  • No heartbeat  → check cable, run hitl.launch.py first, reboot FC")
        print("  • No EKF       → set HITL_ENABLE=1 in QGC, reboot FC")
        print("  • Armed        → disarm via QGC or: ros2 run mavros mavcmd long 400 0 0 0 0 0 0 0")
    print("══════════════════════════════════════════════════")

    ex.shutdown(); rclpy.shutdown()
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
