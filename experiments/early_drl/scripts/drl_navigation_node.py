#!/usr/bin/env python3
"""
ROS Node for DRL-based High-Speed Tunnel Navigation
Integrates deep reinforcement learning agent with ROS navigation system
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud2, Range
from std_msgs.msg import Bool, Float64, String
from visualization_msgs.msg import Marker
from collections import deque
import sys
import os
import math

# Import DRL agent (lightweight version)
from drl_agent_lite import DRLNavigationAgentLite


class DRLNavigationNode:
    """
    ROS node for DRL-based navigation
    Subscribes to sensor data and odometry
    Publishes velocity commands for high-speed navigation
    """
    def __init__(self):
        rospy.init_node('drl_navigation_node', anonymous=False)
        
        # Parameters
        self.load_parameters()
        
        # Initialize DRL agent (lightweight version)
        rospy.loginfo("[DRL Nav] Initializing lightweight DRL agent...")
        self.agent = DRLNavigationAgentLite(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            model_path=self.model_path
        )
        rospy.loginfo("[DRL Nav] DRL agent initialized successfully")
        
        # State variables
        self.current_odom = None
        self.current_goal = None
        self.lidar_ranges = np.ones(8) * 10.0  # Initialize with large values
        self.last_update_time = rospy.Time.now()
        self.navigation_active = False
        self.artifact_detected = False
        self.tunnel_end_reached = False
        
        # History for state estimation
        self.velocity_history = deque(maxlen=5)
        self.acceleration_history = deque(maxlen=5)
        
        # Publishers
        self.setup_publishers()
        
        # Subscribers
        self.setup_subscribers()
        
        # Timer for control loop
        self.control_timer = rospy.Timer(rospy.Duration(1.0/self.control_freq), self.control_loop)
        
        rospy.loginfo("[DRL Nav] Node initialized and ready")
        rospy.loginfo("[DRL Nav] ========================================")
        rospy.loginfo("[DRL Nav] Waiting for:")
        rospy.loginfo("[DRL Nav]   1. Goal position: rostopic pub /drl_navigation/goal ...")
        rospy.loginfo("[DRL Nav]   2. Start command: rostopic pub /drl_navigation/start std_msgs/Bool 'data: true'")
        rospy.loginfo("[DRL Nav] ========================================")
    
    def load_parameters(self):
        """Load ROS parameters"""
        # DRL parameters
        self.state_dim = rospy.get_param('~state_dim', 21)
        self.action_dim = rospy.get_param('~action_dim', 4)
        self.model_path = rospy.get_param('~model_path', '')
        
        # Control parameters
        self.control_freq = rospy.get_param('~control_frequency', 50.0)  # Hz
        self.min_speed = rospy.get_param('~min_speed', 5.0)  # m/s
        self.max_speed = rospy.get_param('~max_speed', 10.0)  # m/s
        self.use_deterministic_policy = rospy.get_param('~deterministic', True)
        
        # Safety parameters
        self.min_obstacle_distance = rospy.get_param('~min_obstacle_distance', 1.5)  # meters
        self.emergency_stop_distance = rospy.get_param('~emergency_stop_distance', 0.8)  # meters
        
        # Goal/Artifact detection
        self.goal_reach_threshold = rospy.get_param('~goal_reach_threshold', 1.0)  # meters
        self.use_artifact_detection = rospy.get_param('~use_artifact_detection', True)
        self.artifact_types = rospy.get_param('~artifact_types', ['cube', 'cylinder', 'rope'])
        
        # Tunnel parameters
        self.tunnel_length = rospy.get_param('~tunnel_length', 100.0)  # meters
        self.tunnel_width = rospy.get_param('~tunnel_width', 4.0)  # meters
        self.tunnel_height = rospy.get_param('~tunnel_height', 3.0)  # meters
        
        rospy.loginfo(f"[DRL Nav] Loaded parameters: speed range [{self.min_speed}, {self.max_speed}] m/s")
        rospy.loginfo(f"[DRL Nav] Control frequency: {self.control_freq} Hz")
    
    def setup_publishers(self):
        """Setup ROS publishers"""
        # Velocity command publisher - publish to simulator topic directly
        self.cmd_vel_pub = rospy.Publisher('/CERLAB/quadcopter/cmd_vel', 
                                          TwistStamped, queue_size=10)
        
        # Velocity mode enable
        self.vel_mode_pub = rospy.Publisher('/CERLAB/quadcopter/vel_mode',
                                           Bool, queue_size=10)
        
        # Alternative: position setpoint for position control mode
        self.cmd_pos_pub = rospy.Publisher('/mavros/setpoint_position/local',
                                          PoseStamped, queue_size=10)
        
        # Status publishers
        self.speed_pub = rospy.Publisher('/drl_navigation/current_speed', 
                                        Float64, queue_size=10)
        self.navigation_status_pub = rospy.Publisher('/drl_navigation/status',
                                                    String, queue_size=10)
        self.artifact_detection_pub = rospy.Publisher('/drl_navigation/artifact_detected',
                                                     Bool, queue_size=10)
        
        # Visualization
        self.trajectory_viz_pub = rospy.Publisher('/drl_navigation/predicted_trajectory',
                                                 Path, queue_size=10)
    
    def setup_subscribers(self):
        """Setup ROS subscribers"""
        # Odometry
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom',
                                        Odometry, self.odom_callback, queue_size=10)
        
        # Goal position
        self.goal_sub = rospy.Subscriber('/drl_navigation/goal',
                                        PoseStamped, self.goal_callback, queue_size=10)
        
        # LiDAR/Depth sensor (choose based on your setup)
        self.scan_sub = rospy.Subscriber('/scan',
                                        LaserScan, self.scan_callback, queue_size=10)
        
        # Alternative: depth camera ranges
        self.depth_sub = rospy.Subscriber('/depth_ranges',
                                         Range, self.depth_callback, queue_size=10)
        
        # Artifact detection
        if self.use_artifact_detection:
            self.artifact_sub = rospy.Subscriber('/artifact_detector/detection',
                                                String, self.artifact_callback, queue_size=10)
        
        # Start/stop commands
        self.start_sub = rospy.Subscriber('/drl_navigation/start',
                                         Bool, self.start_callback, queue_size=10)
    
    def odom_callback(self, msg):
        """Process odometry data"""
        self.current_odom = msg
        
        # Extract velocity
        vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        self.velocity_history.append(vel)
        
        # Compute acceleration (numerical derivative)
        if len(self.velocity_history) >= 2:
            dt = 1.0 / self.control_freq
            accel = (self.velocity_history[-1] - self.velocity_history[-2]) / dt
            self.acceleration_history.append(accel)
    
    def goal_callback(self, msg):
        """Process goal position"""
        self.current_goal = msg
        rospy.loginfo(f"[DRL Nav] New goal received: [{msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f}]")
    
    def scan_callback(self, msg):
        """Process LiDAR scan data"""
        # Convert 360-degree scan to 8 directional ranges
        # Directions: front, front-left, left, back-left, back, back-right, right, front-right
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = msg.range_max
        
        num_ranges = len(ranges)
        sector_size = num_ranges // 8
        
        for i in range(8):
            start_idx = i * sector_size
            end_idx = (i + 1) * sector_size if i < 7 else num_ranges
            self.lidar_ranges[i] = np.min(ranges[start_idx:end_idx])
    
    def depth_callback(self, msg):
        """Process depth sensor data"""
        # Simple range sensor callback
        # You may need to modify based on your sensor setup
        pass
    
    def artifact_callback(self, msg):
        """Process artifact detection messages"""
        if msg.data in self.artifact_types:
            self.artifact_detected = True
            rospy.logwarn(f"[DRL Nav] Artifact detected: {msg.data}")
            self.artifact_detection_pub.publish(Bool(data=True))
    
    def start_callback(self, msg):
        """Start/stop navigation"""
        self.navigation_active = msg.data
        if msg.data:
            # Enable velocity mode
            self.vel_mode_pub.publish(Bool(data=True))
            rospy.sleep(0.1)
            rospy.loginfo("[DRL Nav] Velocity mode enabled")
            rospy.loginfo("[DRL Nav] Navigation started")
            self.navigation_status_pub.publish(String(data="ACTIVE"))
        else:
            rospy.loginfo("[DRL Nav] Navigation stopped")
            self.navigation_status_pub.publish(String(data="STOPPED"))
            # Send stop command
            self.send_stop_command()
    
    def construct_state(self):
        """
        Construct state vector from sensor data
        
        State: [position(3), velocity(3), orientation(4), lidar_ranges(8), goal_direction(3)]
        Total: 21 dimensions
        """
        if self.current_odom is None or self.current_goal is None:
            return None
        
        # Position
        pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y,
            self.current_odom.pose.pose.position.z
        ])
        
        # Velocity
        vel = np.array([
            self.current_odom.twist.twist.linear.x,
            self.current_odom.twist.twist.linear.y,
            self.current_odom.twist.twist.linear.z
        ])
        
        # Orientation (quaternion)
        quat = np.array([
            self.current_odom.pose.pose.orientation.x,
            self.current_odom.pose.pose.orientation.y,
            self.current_odom.pose.pose.orientation.z,
            self.current_odom.pose.pose.orientation.w
        ])
        
        # LiDAR ranges (8 directions)
        lidar = self.lidar_ranges.copy()
        
        # Goal direction (normalized)
        goal_pos = np.array([
            self.current_goal.pose.position.x,
            self.current_goal.pose.position.y,
            self.current_goal.pose.position.z
        ])
        goal_dir = goal_pos - pos
        goal_dist = np.linalg.norm(goal_dir)
        if goal_dist > 0:
            goal_dir = goal_dir / goal_dist
        
        # Concatenate state
        state = np.concatenate([pos, vel, quat, lidar, goal_dir])
        
        return state
    
    def check_termination_conditions(self):
        """Check if navigation should terminate"""
        if self.current_odom is None or self.current_goal is None:
            return False
        
        # Check if goal reached
        pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y,
            self.current_odom.pose.pose.position.z
        ])
        goal_pos = np.array([
            self.current_goal.pose.position.x,
            self.current_goal.pose.position.y,
            self.current_goal.pose.position.z
        ])
        dist_to_goal = np.linalg.norm(goal_pos - pos)
        
        if dist_to_goal < self.goal_reach_threshold:
            self.tunnel_end_reached = True
            rospy.loginfo("[DRL Nav] Goal reached!")
            return True
        
        # Check if artifact detected
        if self.use_artifact_detection and self.artifact_detected:
            rospy.loginfo("[DRL Nav] Navigation stopped: Artifact detected")
            return True
        
        # Check tunnel bounds
        if abs(pos[1]) > self.tunnel_width / 2 or pos[2] < 0.5 or pos[2] > self.tunnel_height:
            rospy.logwarn("[DRL Nav] Warning: Approaching tunnel boundaries!")
        
        # Emergency stop for obstacle
        if np.min(self.lidar_ranges) < self.emergency_stop_distance:
            rospy.logerr("[DRL Nav] Emergency stop: Obstacle too close!")
            return True
        
        return False
    
    def control_loop(self, event):
        """Main control loop"""
        if not self.navigation_active:
            return
        
        # Check termination conditions
        if self.check_termination_conditions():
            self.navigation_active = False
            self.send_stop_command()
            return
        
        # Construct state
        state = self.construct_state()
        if state is None:
            rospy.logwarn_throttle(1.0, "[DRL Nav] Waiting for odometry and goal...")
            return
        
        # Get action from DRL agent
        action = self.agent.get_action(state, deterministic=self.use_deterministic_policy)
        
        # Parse action: [forward_vel, lateral_vel, vertical_vel, yaw_rate]
        forward_vel = np.clip(action[0], self.min_speed, self.max_speed)
        lateral_vel = action[1]
        vertical_vel = action[2]
        yaw_rate = action[3]
        
        # Safety check
        if np.min(self.lidar_ranges) < self.min_obstacle_distance:
            # Reduce speed when close to obstacles
            speed_factor = np.min(self.lidar_ranges) / self.min_obstacle_distance
            forward_vel *= speed_factor
            rospy.logwarn_throttle(0.5, f"[DRL Nav] Reducing speed due to obstacle: {speed_factor:.2f}x")
        
        # Publish velocity command
        self.publish_velocity_command(forward_vel, lateral_vel, vertical_vel, yaw_rate)
        
        # Publish current speed
        current_speed = np.linalg.norm([forward_vel, lateral_vel, vertical_vel])
        self.speed_pub.publish(Float64(data=current_speed))
        
        # Log status
        rospy.loginfo_throttle(1.0, 
            f"[DRL Nav] Speed: {current_speed:.2f} m/s | "
            f"Min obstacle dist: {np.min(self.lidar_ranges):.2f} m")
    
    def publish_velocity_command(self, forward_vel, lateral_vel, vertical_vel, yaw_rate):
        """Publish velocity command"""
        # Get current orientation for body-to-world transform
        if self.current_odom is None:
            return
        
        quat = (
            self.current_odom.pose.pose.orientation.x,
            self.current_odom.pose.pose.orientation.y,
            self.current_odom.pose.pose.orientation.z,
            self.current_odom.pose.pose.orientation.w
        )
        # Calculate yaw from quaternion (lightweight)
        yaw = math.atan2(2.0*(quat[3]*quat[2] + quat[0]*quat[1]), 
                         1.0 - 2.0*(quat[1]*quat[1] + quat[2]*quat[2]))
        
        # Convert body frame velocities to world frame
        vx_world = forward_vel * np.cos(yaw) - lateral_vel * np.sin(yaw)
        vy_world = forward_vel * np.sin(yaw) + lateral_vel * np.cos(yaw)
        vz_world = vertical_vel
        
        # Create and publish command
        cmd_vel = TwistStamped()
        cmd_vel.header.stamp = rospy.Time.now()
        cmd_vel.header.frame_id = "world"
        cmd_vel.twist.linear.x = vx_world
        cmd_vel.twist.linear.y = vy_world
        cmd_vel.twist.linear.z = vz_world
        cmd_vel.twist.angular.z = yaw_rate
        
        self.cmd_vel_pub.publish(cmd_vel)
    
    def send_stop_command(self):
        """Send stop command"""
        cmd_vel = TwistStamped()
        cmd_vel.header.stamp = rospy.Time.now()
        cmd_vel.header.frame_id = "world"
        self.cmd_vel_pub.publish(cmd_vel)
        rospy.loginfo("[DRL Nav] Stop command sent")


if __name__ == '__main__':
    try:
        node = DRLNavigationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
