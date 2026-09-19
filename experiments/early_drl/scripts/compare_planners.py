#!/usr/bin/env python3
"""
Performance Comparison Script for RRT, A*, and PPO Navigation
Evaluates: completion time, path length, smoothness, safety margins, success rate
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import time
import sys
from datetime import datetime


class NavigationBenchmark:
    """Benchmark navigation algorithms"""
    
    def __init__(self, algorithm_name):
        self.algorithm = algorithm_name
        
        # Metrics
        self.start_time = None
        self.end_time = None
        self.trajectory = []
        self.velocities = []
        self.timestamps = []
        self.start_pos = None
        self.goal_pos = None
        self.goal_reached = False
        self.collision = False
        self.min_obstacle_distance = float('inf')
        
        # Subscribers
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)
        
        rospy.loginfo(f"Benchmark initialized for {algorithm_name}")
    
    def odom_callback(self, msg):
        """Record trajectory data"""
        if self.start_time is None:
            return
        
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
        
        self.trajectory.append(pos)
        self.velocities.append(vel)
        self.timestamps.append(time.time() - self.start_time)
        
        # Check goal
        if self.goal_pos is not None:
            goal_dist = np.linalg.norm(pos - self.goal_pos)
            if goal_dist < 0.5:
                self.goal_reached = True
                self.end_time = time.time()
    
    def start_mission(self, start_pos, goal_pos):
        """Start benchmark mission"""
        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        self.start_time = time.time()
        self.trajectory = []
        self.velocities = []
        self.timestamps = []
        self.goal_reached = False
        
        rospy.loginfo(f"Mission started: {start_pos} -> {goal_pos}")
    
    def compute_metrics(self):
        """Compute performance metrics"""
        if len(self.trajectory) < 2:
            return None
        
        trajectory = np.array(self.trajectory)
        velocities = np.array(self.velocities)
        
        # 1. Completion time
        completion_time = self.end_time - self.start_time if self.end_time else None
        
        # 2. Path length
        path_segments = np.diff(trajectory, axis=0)
        path_length = np.sum(np.linalg.norm(path_segments, axis=1))
        
        # 3. Average speed
        speeds = np.linalg.norm(velocities, axis=1)
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        
        # 4. Path smoothness (acceleration variance)
        accelerations = np.diff(velocities, axis=0)
        acc_magnitudes = np.linalg.norm(accelerations, axis=1)
        smoothness = np.std(acc_magnitudes)
        
        # 5. Path efficiency (actual length / straight line distance)
        straight_line = np.linalg.norm(self.goal_pos - self.start_pos)
        efficiency = straight_line / path_length if path_length > 0 else 0
        
        # 6. Energy consumption (integral of acceleration)
        energy = np.sum(acc_magnitudes)
        
        metrics = {
            'algorithm': self.algorithm,
            'success': self.goal_reached,
            'completion_time': completion_time,
            'path_length': path_length,
            'straight_line_distance': straight_line,
            'efficiency': efficiency,
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'smoothness': smoothness,
            'energy': energy,
            'n_waypoints': len(trajectory)
        }
        
        return metrics
    
    def print_report(self):
        """Print benchmark report"""
        metrics = self.compute_metrics()
        
        if metrics is None:
            rospy.logwarn("Insufficient data for report")
            return
        
        print("\n" + "="*60)
        print(f"BENCHMARK REPORT - {metrics['algorithm']}")
        print("="*60)
        print(f"Success: {'YES' if metrics['success'] else 'NO'}")
        
        if metrics['completion_time']:
            print(f"Completion Time: {metrics['completion_time']:.2f} seconds")
        
        print(f"Path Length: {metrics['path_length']:.2f} meters")
        print(f"Straight-Line Distance: {metrics['straight_line_distance']:.2f} meters")
        print(f"Path Efficiency: {metrics['efficiency']:.2%}")
        print(f"Average Speed: {metrics['avg_speed']:.2f} m/s")
        print(f"Max Speed: {metrics['max_speed']:.2f} m/s")
        print(f"Path Smoothness (std acc): {metrics['smoothness']:.3f}")
        print(f"Energy (sum acc): {metrics['energy']:.2f}")
        print(f"Waypoints Recorded: {metrics['n_waypoints']}")
        print("="*60 + "\n")
        
        return metrics


def run_comparison_mission(algorithms=['RRT', 'A*', 'PPO'], goal_distance=60):
    """Run comparison mission for all algorithms"""
    
    rospy.init_node('navigation_comparison', anonymous=True)
    
    # Test waypoints
    start_pos = [0, 0, 1.0]
    waypoints = [
        [20, 0, 1.0],
        [40, 0, 1.0],
        [goal_distance, 0, 1.0]
    ]
    
    all_results = []
    
    for algorithm in algorithms:
        rospy.loginfo(f"\n{'='*60}")
        rospy.loginfo(f"Testing {algorithm} Navigation")
        rospy.loginfo(f"{'='*60}")
        
        # Initialize benchmark
        benchmark = NavigationBenchmark(algorithm)
        
        # Publish goals
        goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        rospy.sleep(2)  # Wait for publisher
        
        benchmark.start_mission(start_pos, waypoints[-1])
        
        # Send waypoints
        for waypoint in waypoints:
            goal_msg = PoseStamped()
            goal_msg.header.stamp = rospy.Time.now()
            goal_msg.header.frame_id = "map"
            goal_msg.pose.position.x = waypoint[0]
            goal_msg.pose.position.y = waypoint[1]
            goal_msg.pose.position.z = waypoint[2]
            goal_msg.pose.orientation.w = 1.0
            
            goal_pub.publish(goal_msg)
            rospy.loginfo(f"Goal sent: {waypoint}")
            
            # Wait for progress
            rospy.sleep(15)
        
        # Wait for completion or timeout
        timeout = 60
        start = time.time()
        rate = rospy.Rate(10)
        
        while not benchmark.goal_reached and (time.time() - start) < timeout:
            rate.sleep()
        
        # Get results
        metrics = benchmark.print_report()
        all_results.append(metrics)
        
        rospy.loginfo(f"Waiting before next algorithm test...")
        rospy.sleep(5)
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<10} {'Success':<10} {'Time(s)':<10} {'Length(m)':<12} {'Efficiency':<12} {'Avg Speed':<10}")
    print("-"*80)
    
    for result in all_results:
        success = "YES" if result['success'] else "NO"
        time_str = f"{result['completion_time']:.2f}" if result['completion_time'] else "N/A"
        print(f"{result['algorithm']:<10} {success:<10} {time_str:<10} "
              f"{result['path_length']:<12.2f} {result['efficiency']:<12.2%} "
              f"{result['avg_speed']:<10.2f}")
    
    print("="*80 + "\n")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/tmp/navigation_comparison_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("Navigation Algorithm Comparison\n")
        f.write(f"Date: {timestamp}\n\n")
        
        for result in all_results:
            f.write(f"\n{result['algorithm']}:\n")
            for key, value in result.items():
                f.write(f"  {key}: {value}\n")
    
    rospy.loginfo(f"Results saved to {filename}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        algorithm = sys.argv[1].upper()
        rospy.init_node('navigation_benchmark', anonymous=True)
        benchmark = NavigationBenchmark(algorithm)
        
        # Wait for mission
        rospy.loginfo(f"Benchmark ready for {algorithm}. Send goals to /move_base_simple/goal")
        rospy.loginfo("Press Ctrl+C when done to see results")
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            benchmark.print_report()
    else:
        print("Usage:")
        print("  Single algorithm: python3 compare_planners.py RRT")
        print("  Full comparison: Run separately for each algorithm and compare results")
