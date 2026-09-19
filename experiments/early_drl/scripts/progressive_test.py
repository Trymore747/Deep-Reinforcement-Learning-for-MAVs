#!/usr/bin/env python3
"""
Simple A* vs RRT comparison - Same mission
Tests navigation to increasing distances to see where each fails
"""

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import time
import sys

class SimpleNavigationTest:
    def __init__(self):
        rospy.init_node('simple_nav_test', anonymous=True)
        self.current_pos = None
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)
        rospy.sleep(2)
    
    def odom_callback(self, msg):
        self.current_pos = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]
    
    def send_goal(self, x, y=0, z=1.0):
        """Send navigation goal"""
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        print(f"Goal sent: {x}m")
    
    def wait_and_monitor(self, goal_x, timeout=30):
        """Wait for goal and monitor progress"""
        start_time = time.time()
        max_x = 0
        
        rate = rospy.Rate(5)
        while not rospy.is_shutdown() and (time.time() - start_time) < timeout:
            if self.current_pos:
                x = self.current_pos[0]
                if x > max_x:
                    max_x = x
                    print(f"  Progress: {x:.1f}m")
                
                # Check if goal reached
                dist = abs(x - goal_x)
                if dist < 2.0:
                    print(f"✓ Goal reached: {x:.1f}m")
                    return True
            
            rate.sleep()
        
        print(f"✗ Timeout. Max distance: {max_x:.1f}m")
        return False
    
    def run_test(self):
        """Run progressive distance test"""
        print("\n" + "="*50)
        print("Progressive Distance Navigation Test")
        print("="*50)
        
        # Test increasing distances
        test_distances = [20, 40, 60, 80]
        
        for dist in test_distances:
            print(f"\n[Test] Target: {dist}m")
            self.send_goal(dist)
            success = self.wait_and_monitor(dist, timeout=30)
            
            if not success:
                print(f"\n⚠ Failed at {dist}m - stopping test")
                break
            
            time.sleep(2)
        
        print("\n" + "="*50)
        print("Test complete!")
        print("="*50)


if __name__ == '__main__':
    try:
        tester = SimpleNavigationTest()
        tester.run_test()
    except rospy.ROSInterruptException:
        pass
