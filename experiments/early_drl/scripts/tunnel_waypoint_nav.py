#!/usr/bin/env python3
"""
Simple waypoint navigation through tunnel
Flies straight down the tunnel with basic obstacle avoidance using LiDAR
"""

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import numpy as np

class TunnelNavigator:
    def __init__(self):
        rospy.init_node('tunnel_waypoint_nav')
        
        # Parameters
        self.target_x = rospy.get_param('~target_x', 150.0)
        self.cruise_speed = rospy.get_param('~cruise_speed', 2.0)  # m/s
        self.safety_distance = rospy.get_param('~safety_distance', 1.0)  # m
        self.waypoint_spacing = rospy.get_param('~waypoint_spacing', 10.0)  # m
        
        # State
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.current_vel = np.array([0.0, 0.0, 0.0])
        self.min_front_distance = 10.0
        self.navigation_started = False
        
        # Publishers
        self.vel_pub = rospy.Publisher('/CERLAB/quadcopter/cmd_vel', TwistStamped, queue_size=1)
        self.pose_pub = rospy.Publisher('/CERLAB/quadcopter/setpoint_pose', PoseStamped, queue_size=1)
        self.posctrl_pub = rospy.Publisher('/CERLAB/quadcopter/posctrl', Bool, queue_size=1)
        self.velctrl_pub = rospy.Publisher('/CERLAB/quadcopter/vel_mode', Bool, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/CERLAB/quadcopter/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        rospy.loginfo(f"[TunnelNav]: Target: {self.target_x}m, Speed: {self.cruise_speed}m/s")
        
    def odom_callback(self, msg):
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.current_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        
    def scan_callback(self, msg):
        """Get minimum distance from front-facing lidar rays"""
        if len(msg.ranges) == 0:
            return
        
        # Front 60 degree cone (assuming 360 degree scan)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        front_mask = (angles > -np.pi/6) & (angles < np.pi/6)
        
        front_ranges = np.array(msg.ranges)[front_mask]
        valid_ranges = front_ranges[np.isfinite(front_ranges) & (front_ranges > 0.1)]
        
        if len(valid_ranges) > 0:
            self.min_front_distance = np.min(valid_ranges)
        else:
            self.min_front_distance = msg.range_max
            
    def takeoff(self):
        """Takeoff to 1m height"""
        rospy.loginfo("[TunnelNav]: Taking off...")
        
        # Enable position control
        self.posctrl_pub.publish(Bool(data=True))
        rospy.sleep(0.5)
        
        # Send takeoff position
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 1.0
        pose.pose.orientation.w = 1.0
        
        for _ in range(50):
            self.pose_pub.publish(pose)
            rospy.sleep(0.1)
            
        rospy.loginfo("[TunnelNav]: Takeoff complete")
        
    def navigate(self):
        """Navigate through tunnel using velocity commands"""
        self.takeoff()
        
        # Switch to velocity control
        rospy.sleep(1.0)
        self.velctrl_pub.publish(Bool(data=True))
        rospy.sleep(0.5)
        
        rospy.loginfo("[TunnelNav]: Starting navigation")
        self.navigation_started = True
        
        rate = rospy.Rate(20)  # 20 Hz
        
        while not rospy.is_shutdown():
            # Check if reached goal
            distance_to_goal = self.target_x - self.current_pos[0]
            
            if distance_to_goal < 2.0:
                rospy.loginfo("[TunnelNav]: Goal reached!")
                self.stop()
                break
                
            # Calculate desired velocity
            vel_cmd = TwistStamped()
            vel_cmd.header.stamp = rospy.Time.now()
            vel_cmd.header.frame_id = 'world'
            
            # Forward speed based on obstacle distance
            if self.min_front_distance < self.safety_distance:
                # Too close - stop
                forward_speed = 0.0
                rospy.logwarn(f"[TunnelNav]: Obstacle at {self.min_front_distance:.2f}m - stopping")
            elif self.min_front_distance < self.safety_distance * 2:
                # Slow down proportionally
                forward_speed = self.cruise_speed * (self.min_front_distance - self.safety_distance) / self.safety_distance
            else:
                # Full speed
                forward_speed = self.cruise_speed
            
            # Lateral centering (keep y close to 0)
            lateral_speed = -0.5 * self.current_pos[1]  # P controller
            lateral_speed = np.clip(lateral_speed, -0.5, 0.5)
            
            # Height control (maintain 1m)
            vertical_speed = 0.3 * (1.0 - self.current_pos[2])
            vertical_speed = np.clip(vertical_speed, -0.3, 0.3)
            
            vel_cmd.twist.linear.x = forward_speed
            vel_cmd.twist.linear.y = lateral_speed
            vel_cmd.twist.linear.z = vertical_speed
            
            self.vel_pub.publish(vel_cmd)
            
            if self.current_pos[0] % 10 < 0.5:  # Log every 10m
                rospy.loginfo(f"[TunnelNav]: Position: {self.current_pos[0]:.1f}m, Distance: {self.min_front_distance:.2f}m, Speed: {forward_speed:.2f}m/s")
            
            rate.sleep()
            
    def stop(self):
        """Stop the drone"""
        vel_cmd = TwistStamped()
        vel_cmd.header.stamp = rospy.Time.now()
        vel_cmd.header.frame_id = 'world'
        
        for _ in range(10):
            self.vel_pub.publish(vel_cmd)
            rospy.sleep(0.1)
            
        rospy.loginfo("[TunnelNav]: Navigation complete")

if __name__ == '__main__':
    try:
        nav = TunnelNavigator()
        nav.navigate()
    except rospy.ROSInterruptException:
        pass
