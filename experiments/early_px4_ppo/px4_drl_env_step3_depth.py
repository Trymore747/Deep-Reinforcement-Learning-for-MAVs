#!/usr/bin/env python3

import asyncio
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from mavsdk import System


# =========================
# ROS 2 DEPTH NODE
# =========================
class DepthNode(Node):
    def __init__(self):
        super().__init__("depth_node")
        self.bridge = CvBridge()
        self.depth = None

        self.sub = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depth_callback,
            10
        )

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            self.depth = np.array(depth, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"Depth error: {e}")


# =========================
# PX4 + DEPTH ENV
# =========================
class PX4DepthEnv:
    def __init__(self):
        # PX4
        self.drone = System()

        # ROS
        rclpy.init()
        self.depth_node = DepthNode()

    # ---------- DEPTH UTILS ----------
    def safe_min(self, region, max_range=10.0):
        region = np.array(region, dtype=np.float32)

        # Remove invalid values
        region = region[np.isfinite(region)]
        region = region[region > 0.05]   # ignore zeros / noise

        if region.size == 0:
            return max_range

        return float(np.clip(region.min(), 0.05, max_range))

    def get_depth_obs(self):
        rclpy.spin_once(self.depth_node, timeout_sec=0.0)
        depth = self.depth_node.depth

        if depth is None:
            return np.ones(3) * 10.0

        h, w = depth.shape

        front = depth[:, w//2 - 10 : w//2 + 10]
        left  = depth[:, : w//3]
        right = depth[:, 2*w//3 :]

        return np.array([
            self.safe_min(front),
            self.safe_min(left),
            self.safe_min(right)
        ])

    # ---------- PX4 ----------
    async def connect(self):
        print("Connecting to PX4...")
        await self.drone.connect(system_address="udp://:14540")

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                break
        print("Connected to PX4")

    async def close(self):
        rclpy.shutdown()


# =========================
# TEST LOOP
# =========================
async def main():
    env = PX4DepthEnv()
    await env.connect()

    for _ in range(20):
        obs = env.get_depth_obs()
        print(f"OBS depth [front, left, right]: {obs}")
        await asyncio.sleep(0.2)

    await env.close()


if __name__ == "__main__":
    asyncio.run(main())

