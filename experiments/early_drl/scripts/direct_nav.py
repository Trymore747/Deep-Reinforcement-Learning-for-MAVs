#!/usr/bin/env python3
"""Direct navigation using goal-based approach with closer waypoints"""
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import subprocess
import time

class DirectNavigator:
    def __init__(self):
        rospy.init_node('direct_navigator')
        self.current_pos = None
        
        # Subscribe to odometry
        rospy.Subscriber('/CERLAB/quadcopter/odom', Odometry, self.odom_callback)
        
        # Publish goals
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        
    def odom_callback(self, msg):
        self.current_pos = msg.pose.pose.position
        
    def send_goal(self, x, z=1.5):
        """Send a goal position"""
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = x
        goal.pose.position.y = 0.0
        goal.pose.position.z = z
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        print(f"  Goal sent: {x}m")
        
    def wait_for_position(self, target_x, timeout=20):
        """Wait until close to target or timeout"""
        start = time.time()
        while time.time() - start < timeout:
            if self.current_pos and abs(self.current_pos.x - target_x) < 5.0:
                return True
            rospy.sleep(0.5)
        return False
        
    def navigate(self):
        rospy.sleep(3)
        
        print("===== INCREMENTAL NAVIGATION TO 60m @ 3 m/s =====")
        
        # Go in 20m increments
        waypoints = [20, 40, 60]
        
        for wp in waypoints:
            print(f"\n[Waypoint] Target: {wp}m")
            self.send_goal(wp)
            
            # Wait and monitor
            for i in range(15):
                rospy.sleep(1)
                if self.current_pos:
                    print(f"    Position: {self.current_pos.x:.1f}m")
                    if abs(self.current_pos.x - wp) < 5.0:
                        print(f"    Reached {wp}m!")
                        break
            
            rospy.sleep(2)
        
        print("\n[Return] Going home...")
        self.send_goal(0.0, z=1.0)
        
        # Monitor return
        for i in range(30):
            rospy.sleep(2)
            if self.current_pos:
                print(f"    Position: {self.current_pos.x:.1f}m")
                if self.current_pos.x < 5.0:
                    print("    Home reached!")
                    break
        
        rospy.sleep(3)
        
        print("\n[Landing]")
        subprocess.call("rostopic pub -1 /CERLAB/quadcopter/posctrl std_msgs/Bool 'data: true'", shell=True)
        rospy.sleep(2)
        
        land_proc = subprocess.Popen(
            "rostopic pub /CERLAB/quadcopter/setpoint_pose geometry_msgs/PoseStamped "
            "'header: {stamp: now, frame_id: map} pose: {position: {x: 0, y: 0, z: 0}, orientation: {w: 1}}' -r 10",
            shell=True
        )
        rospy.sleep(10)
        land_proc.terminate()
        
        print("\n===== COMPLETE: 60m mission @ 3 m/s =====")

if __name__ == '__main__':
    try:
        nav = DirectNavigator()
        nav.navigate()
    except rospy.ROSInterruptException:
        pass
