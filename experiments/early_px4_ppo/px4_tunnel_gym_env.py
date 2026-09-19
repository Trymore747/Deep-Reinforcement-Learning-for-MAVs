import gym
import numpy as np
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
import time

class PX4TunnelEnv(gym.Env):
    def __init__(self, forward_distance=40.0, cruise_alt=1.0):
        super(PX4TunnelEnv, self).__init__()

        self.forward_distance = forward_distance  # meters
        self.cruise_alt = cruise_alt             # meters
        self.loop = asyncio.get_event_loop()
        self.drone = System()
        self.connected = False

        # Observation: front, left, right depth + position x + velocity
        self.observation_space = gym.spaces.Box(
            low=0, high=10, shape=(5,), dtype=np.float32
        )
        # Action: velocity in x, y, z, yaw rate
        self.action_space = gym.spaces.Box(
            low=np.array([-2, -2, -1, -30]),
            high=np.array([2, 2, 1, 30]),
            dtype=np.float32
        )

    async def _connect_px4(self):
        if not self.connected:
            print("Connecting to PX4...")
            await self.drone.connect(system_address="udp://:14540")
            print("Connected to PX4")

            # Wait for health
            await self._wait_for_arm_ready()

            # Arm
            await self.drone.action.arm()
            print("Drone armed!")

            # Start OFFBOARD
            try:
                await self.drone.offboard.start()
                print("OFFBOARD started")
            except OffboardError as e:
                print(f"Offboard start failed: {e._result}")

            self.connected = True

    async def _wait_for_arm_ready(self):
        print("Waiting for drone health to arm...")
        async for health in self.drone.telemetry.health():
            if health.is_gyrometer_calibration_ok and \
               health.is_accelerometer_calibration_ok and \
               health.is_local_position_ok:
                print("Drone health OK, ready to arm!")
                break
            await asyncio.sleep(0.1)

    def reset(self):
        self.loop.run_until_complete(self._connect_px4())
        # Takeoff to cruise altitude
        self.loop.run_until_complete(self._takeoff())
        # Initialize obs
        obs = self._get_obs()
        return obs

    async def _takeoff(self):
        await self.drone.action.takeoff()
        await asyncio.sleep(3)
        print(f"Takeoff to {self.cruise_alt} m reached")

    def step(self, action):
        # Convert action to velocity
        vx, vy, vz, yaw_rate = action
        # Send velocity command
        self.loop.run_until_complete(
            self.drone.offboard.set_velocity_ned(PositionNedYaw(vx, vy, vz, yaw_rate))
        )
        time.sleep(0.1)

        # Get observation
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._check_done(obs)
        info = {}

        return obs, reward, done, info

    def _get_obs(self):
        # Placeholder depth sensor reading
        front = 10.0
        left = 10.0
        right = 10.0
        # Position and velocity
        x = 0.0
        vel = 0.0
        return np.array([front, left, right, x, vel], dtype=np.float32)

    def _compute_reward(self, obs):
        # Reward proportional to forward progress
        reward = 2.0
        return reward

    def _check_done(self, obs):
        # Done if reached target distance
        if obs[3] >= self.forward_distance:
            return True
        return False

    def close(self):
        try:
            self.loop.run_until_complete(self.drone.offboard.stop())
            self.loop.run_until_complete(self.drone.action.land())
            print("Drone landed and OFFBOARD stopped")
        except:
            pass

