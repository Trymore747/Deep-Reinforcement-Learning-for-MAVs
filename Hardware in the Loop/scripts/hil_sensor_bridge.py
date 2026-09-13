#!/usr/bin/env python3
"""
HIL State Bridge: Gazebo odom → FC via /uas1/mavlink_sink

Publishes HIL_STATE_QUATERNION at 50 Hz directly to MAVROS's mavlink_sink
ROS2 topic so PX4's EKF receives valid state and enables OFFBOARD arming.

Usage:
  python3 hil_sensor_bridge.py
"""
import math
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from mavros_msgs.msg import Mavlink
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

try:
    from pymavlink import mavutil
except ImportError:
    import sys
    sys.exit("[hil_bridge] pymavlink not installed: pip install pymavlink")

# ── Config ─────────────────────────────────────────────────────────────────────
RATE_HZ     = 50
SRC_SYSTEM  = 254   # MAV_TYPE_ONBOARD_CONTROLLER
SRC_COMP    = 0
REF_LAT_DEG = 47.397742
REF_LON_DEG =  8.545594
REF_ALT_M   =  488.0
DEG2RAD     = math.pi / 180.0
GRAVITY     = 9.80665
# ──────────────────────────────────────────────────────────────────────────────


def _pack_mavlink_v1(msgid: int, payload: bytes, seq: int) -> Mavlink:
    """Pack a MAVLink v1 payload into a mavros_msgs/Mavlink ROS message."""
    # MAVLink v1 CRC-extra bytes for HIL_STATE_QUATERNION (id=115) = 4
    # We let pymavlink handle CRC, then we just re-read the frame bytes.
    # Build frame manually to avoid pymavlink connection overhead.
    # CRC_EXTRA for common messages — only need #115 here.
    CRC_EXTRA = {115: 4}

    frame = bytearray()
    frame += bytes([0xFE, len(payload), seq & 0xFF, SRC_SYSTEM, SRC_COMP, msgid & 0xFF])
    frame += payload

    # Compute CRC-16/MCRF4XX
    crc = _crc16_mcrf4xx(bytes(frame[1:]), CRC_EXTRA.get(msgid, 0))

    ros_msg = Mavlink()
    ros_msg.header = Header()
    ros_msg.header.frame_id = ""
    ros_msg.framing_status = Mavlink.FRAMING_OK
    ros_msg.magic    = 0xFE        # MAVLINK_V10
    ros_msg.len      = len(payload)
    ros_msg.incompat_flags = 0
    ros_msg.compat_flags   = 0
    ros_msg.seq      = seq & 0xFF
    ros_msg.sysid    = SRC_SYSTEM
    ros_msg.compid   = SRC_COMP
    ros_msg.msgid    = msgid
    ros_msg.checksum = crc

    # Pack payload bytes into uint64 array (little-endian, padded to 8-byte boundary)
    padded = payload + b'\x00' * ((8 - len(payload) % 8) % 8)
    n = len(padded) // 8
    ros_msg.payload64 = list(struct.unpack(f'<{n}Q', padded))

    return ros_msg


def _crc16_mcrf4xx(data: bytes, crc_extra: int) -> int:
    crc = 0xFFFF
    for b in data:
        tmp = b ^ (crc & 0xFF)
        tmp ^= tmp << 4
        tmp &= 0xFF
        crc = (crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)
        crc &= 0xFFFF
    # Append CRC_EXTRA
    tmp = crc_extra ^ (crc & 0xFF)
    tmp ^= tmp << 4
    tmp &= 0xFF
    crc = (crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)
    return crc & 0xFFFF


class HILBridge(Node):
    def __init__(self):
        super().__init__("hil_state_bridge")
        self._lock = threading.Lock()
        self._odom = None
        self._seq  = 0

        self.create_subscription(
            Odometry, "/CERLAB/quadcopter/odom_raw", self._odom_cb, 10
        )
        self._sink_pub = self.create_publisher(Mavlink, "/uas1/mavlink_sink", 10)
        self.create_timer(1.0 / RATE_HZ, self._send_hil_state)
        self.get_logger().info("HIL bridge ready — publishing to /uas1/mavlink_sink")

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self._odom = msg

    def _send_hil_state(self):
        with self._lock:
            odom = self._odom
        if odom is None:
            return

        now_us = int(time.time() * 1e6)
        p = odom.pose.pose.position
        o = odom.pose.pose.orientation
        v = odom.twist.twist.linear
        w = odom.twist.twist.angular

        # Attitude quaternion: MAVLink expects [w, x, y, z]
        att_q = [o.w, o.x, o.y, o.z]

        rollspeed  = w.x
        pitchspeed = w.y
        yawspeed   = w.z

        # ENU position → GPS
        d_lat = p.y / 111320.0
        d_lon = p.x / (111320.0 * math.cos(REF_LAT_DEG * DEG2RAD))
        lat_int = int((REF_LAT_DEG + d_lat) * 1e7)
        lon_int = int((REF_LON_DEG + d_lon) * 1e7)
        alt_mm  = int((p.z + REF_ALT_M) * 1000)

        # ENU velocity → NED (cm/s)
        vx_ned = int( v.y * 100)
        vy_ned = int( v.x * 100)
        vz_ned = int(-v.z * 100)

        # Acceleration (hover approximation — gravity downward in body frame)
        xacc_mg = 0
        yacc_mg = 0
        zacc_mg = int(GRAVITY * 1000)

        # Encode HIL_STATE_QUATERNION payload (MAVLink msg ID 115)
        # Field order from MAVLink common.xml:
        # time_usec(u64), attitude_quaternion(float[4]), rollspeed(f), pitchspeed(f),
        # yawspeed(f), lat(i32), lon(i32), alt(i32), vx(i16), vy(i16), vz(i16),
        # ind_airspeed(u16), true_airspeed(u16), xacc(i16), yacc(i16), zacc(i16)
        payload = struct.pack(
            '<Q4f3f3i3hHH3h',
            now_us,
            att_q[0], att_q[1], att_q[2], att_q[3],
            rollspeed, pitchspeed, yawspeed,
            lat_int, lon_int, alt_mm,
            vx_ned, vy_ned, vz_ned,
            0, 0,           # ind_airspeed, true_airspeed
            xacc_mg, yacc_mg, zacc_mg,
        )

        ros_msg = _pack_mavlink_v1(115, payload, self._seq)
        now = self.get_clock().now().to_msg()
        ros_msg.header.stamp = now
        self._seq = (self._seq + 1) & 0xFF
        self._sink_pub.publish(ros_msg)


def main():
    print("\n" + "=" * 58)
    print("  HIL Bridge: Gazebo odom → FC via /uas1/mavlink_sink")
    print(f"  HIL_STATE_QUATERNION at {RATE_HZ} Hz")
    print("  Keep running during ALL HITL episodes.")
    print("=" * 58 + "\n")

    rclpy.init()
    node = HILBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[bridge] Stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
