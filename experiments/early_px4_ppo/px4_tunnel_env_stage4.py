import asyncio
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DepthSubscriber(Node):
    """ROS2 node to subscribe to depth camera"""
    def __init__(self):
        super().__init__('depth_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
        self.latest_depth = np.array([10.0, 10.0, 10.0], dtype=np.float32)  # front, left, right

    def listener_callback(self, msg):
        # Convert ROS Image to numpy array
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # Simple 3-region depth extraction
        h, w = depth_image.shape
        front = np.min(depth_image[:, w//3:w//3*2])
        left = np.min(depth_image[:, :w//3])
        right = np.min(depth_image[:, w//3*2:])
        self.latest_depth = np.array([front, left, right], dtype=np.float32)

class PX4TunnelEnvStage4(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Observation: depth + forward distance + yaw
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -180.0]),
            high=np.array([10.0, 10.0, 10.0, 50.0, 180.0]),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -0.5, -30.0]),
            high=np.array([3.0, 3.0, 0.5, 30.0]),
            dtype=np.float32
        )

        # Async MAVSDK
        self.loop = asyncio.get_event_loop()
        self.drone = System()
        self.connected = False

        # ROS2 depth subscriber
        rclpy.init()
        self.depth_node = DepthSubscriber()

        self.forward_distance = 0.0
        self.yaw = 0.0

    async def _connect_px4(self):
        print("Connecting to PX4...")
        await self.drone.connect(system_address="udp://:14540")

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Connected to PX4")
                break

        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok:
                print("Global position OK")
                break

        print("Arming...")
        await self.drone.action.arm()

        await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0,0,0,0))
        try:
            await self.drone.offboard.start()
            print("Offboard started successfully")
        except OffboardError as e:
            print(f"Offboard start failed: {e._result.result_str}")
            await self.drone.action.disarm()

        self.connected = True

    def reset(self, seed=None, options=None):
        if not self.connected:
            self.loop.run_until_complete(self._connect_px4())

        self.forward_distance = 0.0
        obs = np.concatenate([self.depth_node.latest_depth, [self.forward_distance, self.yaw]])
        return obs, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        fwd, lat, vert, yaw_rate = action

        asyncio.run(self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(fwd, lat, vert, yaw_rate)
        ))

        # Use actual depth from ROS2
        depth = self.depth_node.latest_depth

        self.forward_distance += fwd * 0.1
        collision_penalty = -10.0 if np.any(depth < 0.5) else 0.0
        reward = fwd * 0.5 + collision_penalty
        done = collision_penalty != 0.0 or self.forward_distance >= 50.0

        obs = np.concatenate([depth, [self.forward_distance, self.yaw]])
        info = {}
        return obs, reward, done, False, info

    def render(self, mode="human"):
        print(f"Forward distance: {self.forward_distance:.2f}, Depth: {self.depth_node.latest_depth}")

    def close(self):
        if self.connected:
            asyncio.run(self.drone.offboard.stop())
            asyncio.run(self.drone.action.disarm())
            self.connected = False
        rclpy.shutdown()

