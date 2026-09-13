#!/usr/bin/env python3
"""
Emergency Stop — HITL Kill Switch
===================================
Immediately disarms the flight controller and sets STABILIZED mode.
Run from a dedicated terminal at any time during HITL testing.

Usage:
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 "Hardware in the Loop/scripts/emergency_stop.py"
"""

import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from mavros_msgs.srv import CommandBool, SetMode


def main():
    rclpy.init()
    node = Node("hitl_emergency_stop")

    arm_cli  = node.create_client(CommandBool, "/mavros/cmd/arming")
    mode_cli = node.create_client(SetMode,     "/mavros/set_mode")

    ex = SingleThreadedExecutor()
    ex.add_node(node)
    t = threading.Thread(target=ex.spin, daemon=True)
    t.start()

    print("\n[EMERGENCY STOP] Sending DISARM + STABILIZED mode...")

    # Set STABILIZED first (exits OFFBOARD safely)
    if mode_cli.wait_for_service(timeout_sec=3.0):
        req = SetMode.Request(); req.custom_mode = "STABILIZED"
        fut = mode_cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 3.0:
            time.sleep(0.05)
        if fut.done() and fut.result().mode_sent:
            print("[EMERGENCY STOP] Mode → STABILIZED")
        else:
            print("[EMERGENCY STOP] Mode change failed — forcing DISARM anyway")
    else:
        print("[EMERGENCY STOP] SetMode service unavailable — forcing DISARM")

    time.sleep(0.3)

    # Disarm
    if arm_cli.wait_for_service(timeout_sec=3.0):
        req = CommandBool.Request(); req.value = False
        fut = arm_cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 3.0:
            time.sleep(0.05)
        if fut.done() and fut.result().success:
            print("[EMERGENCY STOP] DISARMED")
        else:
            print("[EMERGENCY STOP] Disarm call returned failure — check FC manually")
    else:
        print("[EMERGENCY STOP] Arming service unavailable — check MAVROS connection")

    print("[EMERGENCY STOP] Done.\n")
    ex.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
