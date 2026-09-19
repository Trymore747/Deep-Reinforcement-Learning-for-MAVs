import asyncio
import numpy as np
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed


class PX4OffboardEnvContinuous:
    def __init__(self, step_duration=0.1, max_speed=3.0, takeoff_alt=1.0):
        self.drone = System()
        self.step_duration = step_duration
        self.max_speed = max_speed
        self.takeoff_alt = takeoff_alt  # cruise altitude now 1.0 m

        # LiDAR readings
        self.front_distance = 10.0
        self.left_distance = 10.0
        self.right_distance = 10.0
        self.lidar_task = None

        # Safety parameters
        self.safety_distance = 2.0
        self.min_distance = 0.8

        # Current speed for smooth acceleration
        self.current_speed = 0.0

    async def connect(self):
        await self.drone.connect(system_address="udp://:14540")
        print("Waiting for drone to connect...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Drone connected!")
                break

        async for health in self.drone.telemetry.health():
            if health.is_local_position_ok:
                print("Drone ready to arm!")
                break

        await self.drone.action.arm()
        print("Drone armed!")

        print(f"Taking off to {self.takeoff_alt} meters...")
        await self.drone.action.takeoff()
        async for pos in self.drone.telemetry.position():
            if pos.relative_altitude_m >= self.takeoff_alt - 0.05:
                break
            await asyncio.sleep(0.1)

        print("Takeoff altitude reached.")

        # Start LiDAR listener
        self.lidar_task = asyncio.create_task(self._lidar_listener())

        # Send initial OFFBOARD setpoints
        for _ in range(10):
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
            await asyncio.sleep(0.1)

        await self.drone.offboard.start()
        print("OFFBOARD started successfully.")

    async def _lidar_listener(self):
        async for dist in self.drone.telemetry.distance_sensor():
            if dist.current_distance_m > 0:
                if dist.sensor_orientation_deg == 0:
                    self.front_distance = dist.current_distance_m
                elif dist.sensor_orientation_deg == -90:
                    self.left_distance = dist.current_distance_m
                elif dist.sensor_orientation_deg == 90:
                    self.right_distance = dist.current_distance_m

    def _predictive_avoidance(self, desired_speed):
        """
        Compute smooth speed and yaw based on LiDAR lookahead
        """
        speed = desired_speed
        yaw_rate = 0.0

        # Emergency stop if obstacle too close
        if self.front_distance < self.min_distance:
            speed = 0.0
            yaw_rate = 30.0 if self.left_distance > self.right_distance else -30.0
        # Moderate slowing if obstacle near
        elif self.front_distance < self.safety_distance:
            speed = desired_speed * 0.4
            yaw_rate = (self.right_distance - self.left_distance) * 15.0 / self.safety_distance
        else:
            # Tunnel centering
            delta = self.right_distance - self.left_distance
            yaw_rate = np.clip(delta * 5.0 / 2.0, -10.0, 10.0)

        return speed, yaw_rate

    async def fly_distance_predictive(self, distance_m, speed_mps):
        speed_mps = np.clip(speed_mps, -self.max_speed, self.max_speed)
        travelled = 0.0
        steps = int(abs(distance_m / speed_mps) / self.step_duration)
        direction = "FORWARD" if speed_mps < 0 else "BACKWARD"
        print(f"Flying {direction} for {distance_m:.2f} m with predictive avoidance")

        for _ in range(steps):
            speed, yaw_rate = self._predictive_avoidance(speed_mps)
            self.current_speed += np.clip(speed - self.current_speed, -0.3, 0.3)

            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(self.current_speed, 0.0, 0.0, np.deg2rad(yaw_rate))
            )

            travelled += abs(self.current_speed) * self.step_duration
            await asyncio.sleep(self.step_duration)

        print(f"Finished {direction} flight, travelled ≈ {travelled:.2f} m")

    async def hover(self, duration=2.0):
        steps = int(duration / self.step_duration)
        for _ in range(steps):
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
            await asyncio.sleep(self.step_duration)

    async def land_and_disarm(self):
        await self.drone.offboard.stop()
        print("OFFBOARD stopped.")

        print("Landing...")
        await self.drone.action.land()
        async for in_air in self.drone.telemetry.in_air():
            if not in_air:
                break
            await asyncio.sleep(0.5)

        await self.drone.action.disarm()
        print("Drone landed and disarmed at spawn.")

