#!/usr/bin/env python3
"""
Data Recorder for High-Speed Tunnel Navigation
Records comprehensive drone parameters during DRL navigation
"""

import rospy
import csv
import json
import os
from datetime import datetime
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu, BatteryState
from std_msgs.msg import Float64, String, Bool


class NavigationDataRecorder:
    """
    Records drone telemetry and navigation data
    """
    def __init__(self):
        rospy.init_node('navigation_data_recorder', anonymous=False)
        
        # Parameters
        self.load_parameters()
        
        # Data storage
        self.data_buffer = []
        self.recording_active = False
        self.episode_count = 0
        self.start_time = None
        
        # Latest sensor data
        self.latest_odom = None
        self.latest_imu = None
        self.latest_battery = None
        self.latest_speed = 0.0
        self.latest_lidar_min = 999.0
        self.artifact_detected = False
        self.goal_reached = False
        
        # Create output directory
        self.setup_output_directory()
        
        # Setup subscribers
        self.setup_subscribers()
        
        # Recording timer
        self.record_timer = rospy.Timer(rospy.Duration(1.0/self.recording_freq), 
                                       self.record_data_callback)
        
        rospy.loginfo("[Data Recorder] Initialized and ready")
    
    def load_parameters(self):
        """Load ROS parameters"""
        self.recording_freq = rospy.get_param('~recording_frequency', 50.0)  # Hz
        self.output_dir = rospy.get_param('~output_directory', 
                                         os.path.expanduser('~/tunnel_navigation_data'))
        self.save_format = rospy.get_param('~save_format', 'csv')  # csv or json
        self.auto_start = rospy.get_param('~auto_start', True)
        
        rospy.loginfo(f"[Data Recorder] Recording frequency: {self.recording_freq} Hz")
        rospy.loginfo(f"[Data Recorder] Output directory: {self.output_dir}")
    
    def setup_output_directory(self):
        """Create output directory structure"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            rospy.loginfo(f"[Data Recorder] Created output directory: {self.output_dir}")
        
        # Create subdirectories
        self.episodes_dir = os.path.join(self.output_dir, 'episodes')
        self.summary_dir = os.path.join(self.output_dir, 'summaries')
        
        for directory in [self.episodes_dir, self.summary_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def setup_subscribers(self):
        """Setup ROS subscribers"""
        # Odometry
        rospy.Subscriber('/mavros/local_position/odom', Odometry, 
                        self.odom_callback, queue_size=10)
        
        # IMU
        rospy.Subscriber('/mavros/imu/data', Imu, 
                        self.imu_callback, queue_size=10)
        
        # Battery
        rospy.Subscriber('/mavros/battery', BatteryState, 
                        self.battery_callback, queue_size=10)
        
        # LiDAR
        rospy.Subscriber('/scan', LaserScan, 
                        self.scan_callback, queue_size=10)
        
        # DRL Navigation status
        rospy.Subscriber('/drl_navigation/current_speed', Float64,
                        self.speed_callback, queue_size=10)
        
        rospy.Subscriber('/drl_navigation/artifact_detected', Bool,
                        self.artifact_callback, queue_size=10)
        
        # Recording control
        rospy.Subscriber('/data_recorder/start', Bool,
                        self.start_callback, queue_size=10)
        
        rospy.Subscriber('/data_recorder/save', Bool,
                        self.save_callback, queue_size=10)
        
        if self.auto_start:
            self.start_recording()
            rospy.loginfo("[Data Recorder] Auto-start enabled - recording")
    
    def odom_callback(self, msg):
        """Process odometry"""
        self.latest_odom = msg
    
    def imu_callback(self, msg):
        """Process IMU data"""
        self.latest_imu = msg
    
    def battery_callback(self, msg):
        """Process battery data"""
        self.latest_battery = msg
    
    def scan_callback(self, msg):
        """Process LiDAR scan"""
        ranges = np.array(msg.ranges)
        ranges = ranges[~np.isnan(ranges) & ~np.isinf(ranges)]
        if len(ranges) > 0:
            self.latest_lidar_min = np.min(ranges)
    
    def speed_callback(self, msg):
        """Process speed data"""
        self.latest_speed = msg.data
    
    def artifact_callback(self, msg):
        """Process artifact detection"""
        if msg.data:
            self.artifact_detected = True
    
    def start_callback(self, msg):
        """Start/stop recording"""
        if msg.data and not self.recording_active:
            self.start_recording()
        elif not msg.data and self.recording_active:
            self.stop_recording()
    
    def save_callback(self, msg):
        """Save current data"""
        if msg.data:
            self.save_episode_data()
    
    def start_recording(self):
        """Start a new recording episode"""
        self.recording_active = True
        self.start_time = rospy.Time.now()
        self.data_buffer = []
        self.episode_count += 1
        self.artifact_detected = False
        self.goal_reached = False
        rospy.loginfo(f"[Data Recorder] Started recording episode {self.episode_count}")
    
    def quaternion_to_euler(self, quat):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        x, y, z, w = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def stop_recording(self):
        """Stop recording and save data"""
        if self.recording_active:
            self.recording_active = False
            self.save_episode_data()
            rospy.loginfo("[Data Recorder] Stopped recording")
    
    def record_data_callback(self, event):
        """Record current sensor data"""
        if not self.recording_active or self.latest_odom is None:
            return
        
        # Calculate elapsed time
        elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
        
        # Extract position
        pos = self.latest_odom.pose.pose.position
        position = [pos.x, pos.y, pos.z]
        
        # Extract orientation (quaternion and Euler angles)
        quat_msg = self.latest_odom.pose.pose.orientation
        quaternion = [quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w]
        # Convert quaternion to Euler angles (lightweight)
        roll, pitch, yaw = self.quaternion_to_euler(quaternion)
        
        # Extract linear velocity
        vel = self.latest_odom.twist.twist.linear
        velocity = [vel.x, vel.y, vel.z]
        speed = np.linalg.norm(velocity)
        
        # Extract angular velocity
        ang_vel = self.latest_odom.twist.twist.angular
        angular_velocity = [ang_vel.x, ang_vel.y, ang_vel.z]
        
        # Extract acceleration from IMU
        acceleration = [0.0, 0.0, 0.0]
        if self.latest_imu is not None:
            acc = self.latest_imu.linear_acceleration
            acceleration = [acc.x, acc.y, acc.z]
        
        # Battery status
        battery_voltage = 0.0
        battery_percentage = 0.0
        if self.latest_battery is not None:
            battery_voltage = self.latest_battery.voltage
            battery_percentage = self.latest_battery.percentage * 100
        
        # Create data entry
        data_entry = {
            'timestamp': rospy.Time.now().to_sec(),
            'elapsed_time': elapsed_time,
            'episode': self.episode_count,
            
            # Position
            'pos_x': position[0],
            'pos_y': position[1],
            'pos_z': position[2],
            
            # Orientation (Quaternion)
            'quat_x': quaternion[0],
            'quat_y': quaternion[1],
            'quat_z': quaternion[2],
            'quat_w': quaternion[3],
            
            # Orientation (Euler)
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            
            # Linear velocity
            'vel_x': velocity[0],
            'vel_y': velocity[1],
            'vel_z': velocity[2],
            'speed': speed,
            
            # Angular velocity
            'ang_vel_x': angular_velocity[0],
            'ang_vel_y': angular_velocity[1],
            'ang_vel_z': angular_velocity[2],
            
            # Acceleration
            'acc_x': acceleration[0],
            'acc_y': acceleration[1],
            'acc_z': acceleration[2],
            
            # Sensors
            'lidar_min_distance': self.latest_lidar_min,
            
            # Power
            'battery_voltage': battery_voltage,
            'battery_percentage': battery_percentage,
            
            # Status
            'artifact_detected': int(self.artifact_detected),
            'goal_reached': int(self.goal_reached)
        }
        
        self.data_buffer.append(data_entry)
    
    def save_episode_data(self):
        """Save recorded data to file"""
        if len(self.data_buffer) == 0:
            rospy.logwarn("[Data Recorder] No data to save")
            return
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"episode_{self.episode_count:04d}_{timestamp}"
        
        if self.save_format == 'csv':
            filepath = os.path.join(self.episodes_dir, f"{filename_base}.csv")
            self.save_csv(filepath)
        elif self.save_format == 'json':
            filepath = os.path.join(self.episodes_dir, f"{filename_base}.json")
            self.save_json(filepath)
        else:
            # Save both formats
            self.save_csv(os.path.join(self.episodes_dir, f"{filename_base}.csv"))
            self.save_json(os.path.join(self.episodes_dir, f"{filename_base}.json"))
        
        # Save episode summary
        self.save_episode_summary(timestamp)
        
        rospy.loginfo(f"[Data Recorder] Saved {len(self.data_buffer)} data points for episode {self.episode_count}")
    
    def save_csv(self, filepath):
        """Save data in CSV format"""
        if len(self.data_buffer) == 0:
            return
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = list(self.data_buffer[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for data_entry in self.data_buffer:
                writer.writerow(data_entry)
        
        rospy.loginfo(f"[Data Recorder] Saved CSV: {filepath}")
    
    def save_json(self, filepath):
        """Save data in JSON format"""
        with open(filepath, 'w') as jsonfile:
            json.dump(self.data_buffer, jsonfile, indent=2)
        
        rospy.loginfo(f"[Data Recorder] Saved JSON: {filepath}")
    
    def save_episode_summary(self, timestamp):
        """Save episode summary statistics"""
        if len(self.data_buffer) == 0:
            return
        
        # Calculate statistics
        positions = np.array([[d['pos_x'], d['pos_y'], d['pos_z']] for d in self.data_buffer])
        velocities = np.array([[d['vel_x'], d['vel_y'], d['vel_z']] for d in self.data_buffer])
        speeds = np.array([d['speed'] for d in self.data_buffer])
        
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        total_time = self.data_buffer[-1]['elapsed_time']
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        min_speed = np.min(speeds)
        
        summary = {
            'episode': self.episode_count,
            'timestamp': timestamp,
            'total_time': total_time,
            'total_distance': total_distance,
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'min_speed': min_speed,
            'start_position': positions[0].tolist(),
            'end_position': positions[-1].tolist(),
            'artifact_detected': self.artifact_detected,
            'goal_reached': self.goal_reached,
            'num_data_points': len(self.data_buffer)
        }
        
        # Save summary
        summary_file = os.path.join(self.summary_dir, f"episode_{self.episode_count:04d}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        rospy.loginfo(f"[Data Recorder] Episode summary: Distance={total_distance:.2f}m, "
                     f"Time={total_time:.2f}s, Avg Speed={avg_speed:.2f}m/s")


if __name__ == '__main__':
    try:
        recorder = NavigationDataRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
