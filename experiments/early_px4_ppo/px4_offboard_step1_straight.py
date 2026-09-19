import asyncio
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed


class PX4Step1Straight:
    def __init__(self):
        self.drone = System()
        self.step_dt = 0.1
        self.speed = 0.5      # m/s
        self.distance = 10.0  # meters
        self.altitude = 1.0   # meters

    async def run(self):
        await self.drone.connect(system_address="udp://:14540")

        print("Waiting for drone...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                break
        print("Connected")

        async for health in self.drone.telemetry.health():
            if health.is_local_position_ok:
                break
        print("Position OK")

        await self.drone.action.arm()
        print("Armed")

        await self.drone.action.takeoff()
        print(f"Taking off to {self.altitude} m")

        async for pos in self.drone.telemetry.position():
            if pos.relative_altitude_m >= self.altitude - 0.05:
                break
            await asyncio.sleep(0.1)

        print("Altitude reached")

        # Send initial setpoints BEFORE starting offboard
        for _ in range(10):
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0, 0, 0, 0)
            )
            await asyncio.sleep(0.1)

        await self.drone.offboard.start()
        print("OFFBOARD started")

        # Fly straight forward
        steps = int(self.distance / self.speed / self.step_dt)
        traveled = 0.0

        print("Flying forward...")
        for _ in range(steps):
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(-self.speed, 0.0, 0.0, 0.0)
            )
            traveled += self.speed * self.step_dt
            await asyncio.sleep(self.step_dt)

        print(f"Finished flight: {traveled:.2f} m")

        # Stop
        for _ in range(20):
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0, 0, 0, 0)
            )
            await asyncio.sleep(0.1)

        await self.drone.offboard.stop()
        print("OFFBOARD stopped")

        await self.drone.action.land()
        print("Landing")

        async for in_air in self.drone.telemetry.in_air():
            if not in_air:
                break
            await asyncio.sleep(0.5)

        await self.drone.action.disarm()
        print("Disarmed — Step 1 complete")


if __name__ == "__main__":
    asyncio.run(PX4Step1Straight().run())

