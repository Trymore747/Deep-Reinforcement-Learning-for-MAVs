import asyncio
import numpy as np
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed


class PX4Step2Observation:
    def __init__(self):
        self.drone = System()
        self.dt = 0.1
        self.speed = 0.5
        self.distance = 5.0
        self.altitude = 1.0

        self.front = np.inf
        self.left = np.inf
        self.right = np.inf

    async def read_lidar(self):
        async for dist in self.drone.telemetry.distance_sensor():
            if dist.orientation.name == "FORWARD":
                self.front = dist.current_distance_m
            elif dist.orientation.name == "LEFT":
                self.left = dist.current_distance_m
            elif dist.orientation.name == "RIGHT":
                self.right = dist.current_distance_m

    async def run(self):
        await self.drone.connect(system_address="udp://:14540")

        async for s in self.drone.core.connection_state():
            if s.is_connected:
                break

        async for h in self.drone.telemetry.health():
            if h.is_local_position_ok:
                break

        await self.drone.action.arm()
        await self.drone.action.takeoff()

        async for p in self.drone.telemetry.position():
            if p.relative_altitude_m >= self.altitude - 0.05:
                break

        for _ in range(10):
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0, 0, 0, 0)
            )
            await asyncio.sleep(0.1)

        await self.drone.offboard.start()

        asyncio.create_task(self.read_lidar())

        steps = int(self.distance / self.speed / self.dt)
        traveled = 0.0

        for _ in range(steps):
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(-self.speed, 0, 0, 0)
            )

            traveled += self.speed * self.dt

            obs = [
                round(self.front, 2),
                round(self.left, 2),
                round(self.right, 2),
                round(traveled, 2),
                round(self.speed, 2)
            ]

            print("OBS:", obs)

            await asyncio.sleep(self.dt)

        await self.drone.offboard.stop()
        await self.drone.action.land()

        async for in_air in self.drone.telemetry.in_air():
            if not in_air:
                break

        await self.drone.action.disarm()
        print("Step 2 complete")


if __name__ == "__main__":
    asyncio.run(PX4Step2Observation().run())

