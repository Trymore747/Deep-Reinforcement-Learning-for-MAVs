#!/usr/bin/env python3
"""
Comprehensive Performance Data Collector
Collects detailed metrics for RRT, A*, and PPO comparison
"""

import rospy
import numpy as np
import json
import csv
from datetime import datetime
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import time
import os


class PerformanceDataCollector:
    """Collect comprehensive performance metrics"""
    
    def __init__(self, algorithm_name, output_dir="/tmp/nav_comparison"):
        self.algorithm = algorithm_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Timing
        self.mission_start_time = None
        self.mission_end_time = None
        self.goal_received_time = None
        
        # Position and trajectory
        self.positions = []
        self.timestamps = []
        self.velocities = []
        self.accelerations = []
        
        # Goals
        self.goals = []
        self.goal_times = []
        self.start_pos = None
        self.final_goal = None
        
        # Performance metrics
        self.collision_detected = False
        self.collision_time = None
        self.max_speed_achieved = 0
        self.min_obstacle_distances = []
        self.planning_times = []
        
        # Safety metrics
        self.safety_violations = 0  # Count of close calls
        self.safety_threshold = 0.3  # meters
        
        # Initialize ROS
        rospy.init_node(f'{algorithm_name}_data_collector', anonymous=True)
        
        # Subscribers
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        self.rate = rospy.Rate(50)  # 50 Hz data collection
        
        rospy.loginfo(f"Data collector initialized for {algorithm_name}")
        rospy.loginfo(f"Output directory: {output_dir}")
    
    def odom_callback(self, msg):
        """Record position and velocity data"""
        current_time = rospy.Time.now().to_sec()
        
        if self.mission_start_time is None:
            self.mission_start_time = current_time
        
        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        
        # Record data
        self.positions.append(pos)
        self.velocities.append(vel)
        self.timestamps.append(current_time - self.mission_start_time)
        
        # Calculate metrics
        speed = np.linalg.norm(vel)
        if speed > self.max_speed_achieved:
            self.max_speed_achieved = speed
        
        # Calculate acceleration
        if len(self.velocities) > 1:
            dt = self.timestamps[-1] - self.timestamps[-2]
            if dt > 0:
                acc = (self.velocities[-1] - self.velocities[-2]) / dt
                self.accelerations.append(acc)
        
        # Check for collision (velocity suddenly becomes zero and position unchanged)
        if len(self.positions) > 10:
            recent_movement = np.linalg.norm(self.positions[-1] - self.positions[-10])
            if recent_movement < 0.1 and speed < 0.1 and self.mission_start_time is not None:
                elapsed = current_time - self.mission_start_time
                if elapsed > 5.0 and not self.collision_detected:  # After 5 seconds of movement
                    self.collision_detected = True
                    self.collision_time = elapsed
    
    def goal_callback(self, msg):
        """Record goal information"""
        goal = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        current_time = rospy.Time.now().to_sec()
        if self.mission_start_time is None:
            self.mission_start_time = current_time
        
        self.goals.append(goal)
        self.goal_times.append(current_time - self.mission_start_time)
        self.final_goal = goal
        
        rospy.loginfo(f"Goal recorded: {goal}")
    
    def compute_metrics(self):
        """Compute all performance metrics"""
        if len(self.positions) < 2:
            rospy.logwarn("Insufficient data for metrics")
            return None
        
        positions = np.array(self.positions)
        velocities = np.array(self.velocities)
        timestamps = np.array(self.timestamps)
        
        # 1. Mission time
        mission_duration = timestamps[-1] if len(timestamps) > 0 else 0
        
        # 2. Path metrics
        path_segments = np.diff(positions, axis=0)
        path_length = np.sum(np.linalg.norm(path_segments, axis=1))
        
        # 3. Speed metrics
        speeds = np.linalg.norm(velocities, axis=1)
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        
        # 4. Path smoothness (jerk - rate of change of acceleration)
        if len(self.accelerations) > 1:
            accelerations = np.array(self.accelerations)
            acc_magnitudes = np.linalg.norm(accelerations, axis=1)
            smoothness_std = np.std(acc_magnitudes)
            max_acceleration = np.max(acc_magnitudes)
            avg_acceleration = np.mean(acc_magnitudes)
        else:
            smoothness_std = 0
            max_acceleration = 0
            avg_acceleration = 0
        
        # 5. Efficiency
        if self.final_goal is not None and len(positions) > 0:
            start = positions[0]
            end = positions[-1]
            straight_line_dist = np.linalg.norm(self.final_goal - start)
            actual_dist_to_goal = np.linalg.norm(end - self.final_goal)
            path_efficiency = straight_line_dist / path_length if path_length > 0 else 0
            goal_reached = actual_dist_to_goal < 2.0
        else:
            straight_line_dist = 0
            actual_dist_to_goal = float('inf')
            path_efficiency = 0
            goal_reached = False
        
        # 6. Energy consumption (integral of acceleration magnitude)
        if len(self.accelerations) > 0:
            energy = np.sum(np.linalg.norm(self.accelerations, axis=1))
        else:
            energy = 0
        
        # 7. Control smoothness (velocity variance)
        velocity_changes = np.diff(speeds)
        control_smoothness = np.std(velocity_changes) if len(velocity_changes) > 0 else 0
        
        metrics = {
            'algorithm': self.algorithm,
            'timestamp': datetime.now().isoformat(),
            
            # Time metrics
            'mission_duration': mission_duration,
            'collision_detected': self.collision_detected,
            'collision_time': self.collision_time,
            
            # Distance metrics
            'path_length': path_length,
            'straight_line_distance': straight_line_dist,
            'distance_to_goal': actual_dist_to_goal,
            'path_efficiency': path_efficiency,
            
            # Speed metrics
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'max_speed_achieved': self.max_speed_achieved,
            
            # Smoothness metrics
            'acceleration_std': smoothness_std,
            'max_acceleration': max_acceleration,
            'avg_acceleration': avg_acceleration,
            'control_smoothness': control_smoothness,
            
            # Success metrics
            'goal_reached': goal_reached,
            'success': goal_reached and not self.collision_detected,
            
            # Energy and efficiency
            'energy_consumption': energy,
            'safety_violations': self.safety_violations,
            
            # Data quality
            'num_samples': len(positions),
            'sampling_rate': len(positions) / mission_duration if mission_duration > 0 else 0
        }
        
        return metrics
    
    def save_raw_data(self):
        """Save raw trajectory data"""
        filename = f"{self.output_dir}/{self.algorithm}_trajectory.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
            
            for i in range(len(self.positions)):
                row = [self.timestamps[i]]
                row.extend(self.positions[i])
                row.extend(self.velocities[i])
                writer.writerow(row)
        
        rospy.loginfo(f"Raw data saved to {filename}")
    
    def save_metrics(self, metrics):
        """Save computed metrics"""
        filename = f"{self.output_dir}/{self.algorithm}_metrics.json"
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        rospy.loginfo(f"Metrics saved to {filename}")
    
    def generate_report(self):
        """Generate text report"""
        metrics = self.compute_metrics()
        if metrics is None:
            return
        
        report = f"""
{'='*60}
PERFORMANCE REPORT: {self.algorithm}
{'='*60}

Mission Overview:
  Duration: {metrics['mission_duration']:.2f} seconds
  Success: {'YES' if metrics['success'] else 'NO'}
  Goal Reached: {'YES' if metrics['goal_reached'] else 'NO'}
  Collision: {'YES' if metrics['collision_detected'] else 'NO'}

Path Metrics:
  Total Length: {metrics['path_length']:.2f} m
  Straight-Line Distance: {metrics['straight_line_distance']:.2f} m
  Path Efficiency: {metrics['path_efficiency']:.2%}
  Final Distance to Goal: {metrics['distance_to_goal']:.2f} m

Speed Performance:
  Average Speed: {metrics['avg_speed']:.2f} m/s
  Maximum Speed: {metrics['max_speed']:.2f} m/s
  Peak Speed Achieved: {metrics['max_speed_achieved']:.2f} m/s

Smoothness & Control:
  Acceleration Std Dev: {metrics['acceleration_std']:.3f} m/s²
  Max Acceleration: {metrics['max_acceleration']:.2f} m/s²
  Avg Acceleration: {metrics['avg_acceleration']:.2f} m/s²
  Control Smoothness: {metrics['control_smoothness']:.3f}

Energy & Safety:
  Energy Consumption: {metrics['energy_consumption']:.2f}
  Safety Violations: {metrics['safety_violations']}

Data Quality:
  Samples Collected: {metrics['num_samples']}
  Sampling Rate: {metrics['sampling_rate']:.1f} Hz

{'='*60}
"""
        
        # Save report
        report_file = f"{self.output_dir}/{self.algorithm}_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        rospy.loginfo(f"Report saved to {report_file}")
        
        return metrics
    
    def run(self, duration=120):
        """Run data collection for specified duration"""
        rospy.loginfo(f"Starting data collection for {duration} seconds...")
        start_time = time.time()
        
        try:
            while not rospy.is_shutdown() and (time.time() - start_time) < duration:
                self.rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("Data collection interrupted by user")
        
        rospy.loginfo("Data collection complete. Generating report...")
        
        # Save all data
        self.save_raw_data()
        metrics = self.generate_report()
        if metrics:
            self.save_metrics(metrics)
        
        return metrics


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 performance_data_collector.py <ALGORITHM_NAME> [duration]")
        print("Example: python3 performance_data_collector.py RRT 60")
        sys.exit(1)
    
    algorithm = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    
    try:
        collector = PerformanceDataCollector(algorithm)
        collector.run(duration)
    except rospy.ROSInterruptException:
        pass
