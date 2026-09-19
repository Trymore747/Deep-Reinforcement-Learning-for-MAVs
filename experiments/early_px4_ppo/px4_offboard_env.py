import asyncio
import numpy as np
from mavsdk import System
from mavsdk.offboard import Offboard, VelocityBodyYawspeed

class PX4OffboardEnv:
    def __init__(self, duration_per_step=0.1):
        self.drone = System()
        self.duration_per_step = duration_per_step  # seconds per action
        self.connected = False

    async def connect(self):
        await self.drone.connect(system_address="udp://:14540")
        print("Waiting for drone to connect...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Drone connected!")
                break

        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_local_position_ok:
                print("Drone ready to arm!")
                break

        await self.drone.action.arm()
        print("Drone armed!")

        # Start OFFBOARD mode
        await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await self.drone.offboard.start()
        print("OFFBOARD started!")
        self.connected = True

    async def step(self, action):
        # Map discrete action to velocity
        vx, vy, vz = 0.0, 0.0, 0.0
        speed = 1.0  # m/s
        if action == 0:
            vx = speed
        elif action == 1:
            vx = -speed
        elif action == 2:
            vy = -speed
        elif action == 3:
            vy = speed
        elif action == 4:
            vz = -speed  # Up in NED frame
        elif action == 5:
            vz = speed   # Down in NED frame
        elif action == 6:
            vx = vy = vz = 0.0  # Hover

        # Send velocity command
        await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, vz, 0.0))
        await asyncio.sleep(self.duration_per_step)

        # Dummy observation, reward, done for now
        obs = np.zeros(22)  # replace with real state
        reward = 0.3        # replace with real reward function
        done = False        # episode termination condition
        return obs, reward, done

    async def reset(self):
        # Hover to reset
        await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await asyncio.sleep(1.0)
        obs = np.zeros(22)
        return obs

    async def close(self):
        # Stop the drone safely
        await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        await self.drone.offboard.stop()
        await self.drone.action.disarm()
        print("Drone disarmed and environment closed!")
