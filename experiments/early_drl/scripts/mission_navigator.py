#!/usr/bin/env python3
"""
Mission: Fly to 50m, turn around, return to start, and land
"""

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import numpy as np
import math

class MissionNavigator:
    def __init__(self):
        rospy.init_node('mission_navigator')
        
        # Mission parameters
        self.forward_target = rospy.get_param('~forward_target', 50.0)  # Go to 50m
        self.cruise_speed = rospy.get_param('~cruise_speed', 2.0)  # m/s
        self.safety_distance = rospy.get_param('~safety_distance', 1.0)  # m
        
        # State
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.current_yaw = 0.0
        self.min_front_distance = 10.0
        
        # Mission state
        self.mission_state = 'TAKEOFF'  # TAKEOFF -> FORWARD -> TURN -> RETURN -> LAND
        
        # Publishers
        self.vel_pub = rospy.Publisher('/CERLAB/quadcopter/cmd_vel', TwistStamped, queue_size=1)
        self.pose_pub = rospy.Publisher('/CERLAB/quadcopter/setpoint_pose', PoseStamped, queue_size=1)
        self.posctrl_pub = rospy.Publisher('/CERLAB/quadcopter/posctrl', Bool, queue_size=1)
        self.velctrl_pub = rospy.Publisher('/CERLAB/quadcopter/vel_mode', Bool, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/CERLAB/quadcopter/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        rospy.loginfo(f"[Mission]: Forward to {self.forward_target}m, then return and land")
        
    def odom_callback(self, msg):
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        # Extract yaw from quaternion
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.current_yaw = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
        
    def scan_callback(self, msg):
        if len(msg.ranges) == 0:
            self.min_front_distance = 10.0
            return
        
        # Check multiple directions for better obstacle awareness
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges) & (ranges > 0.1), ranges, 100.0)
        
        # Get minimum distance in front
        self.min_front_distance = np.min(ranges) if len(ranges) > 0 else 10.0
            
    def takeoff(self):
        """Takeoff to 1m height"""
        rospy.loginfo("[Mission]: TAKEOFF - Rising to 1m")
        
        # Enable position control
        self.posctrl_pub.publish(Bool(data=True))
        rospy.sleep(0.5)
        
        # Send takeoff position
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 1.0
        pose.pose.orientation.w = 1.0
        
        for _ in range(50):
            pose.header.stamp = rospy.Time.now()
            self.pose_pub.publish(pose)
            rospy.sleep(0.1)
            
        rospy.loginfo("[Mission]: Takeoff complete")
        
    def navigate_forward(self):
        """Navigate forward to target distance"""
        rospy.loginfo(f"[Mission]: FORWARD - Flying to {self.forward_target}m")
        
        # Switch to velocity control
        self.velctrl_pub.publish(Bool(data=True))
        rospy.sleep(0.5)
        
        rate = rospy.Rate(20)
        
        while not rospy.is_shutdown():
            distance_to_target = self.forward_target - self.current_pos[0]
            
            if distance_to_target < 1.0:
                rospy.loginfo(f"[Mission]: Reached {self.current_pos[0]:.1f}m - Starting turn")
                self.stop_briefly()
                return True
                
            # Calculate velocity
            vel_cmd = TwistStamped()
            vel_cmd.header.stamp = rospy.Time.now()
            vel_cmd.header.frame_id = 'world'
            
            # Forward speed with aggressive obstacle avoidance
            if self.min_front_distance < 0.8:
                forward_speed = 0.0
                rospy.logwarn(f"[Mission]: STOP! Obstacle at {self.min_front_distance:.2f}m")
            elif self.min_front_distance < 1.5:
                # Slow down significantly near obstacles
                forward_speed = 0.3 * (self.min_front_distance - 0.8) / 0.7
                rospy.logwarn(f"[Mission]: Slowing down - obstacle at {self.min_front_distance:.2f}m")
            elif self.min_front_distance < 3.0:
                # Moderate speed
                forward_speed = min(1.0, distance_to_target * 0.5)
            else:
                # Full speed when clear
                forward_speed = min(self.cruise_speed, distance_to_target * 0.5)
            
            # Lateral centering
            lateral_speed = -0.5 * self.current_pos[1]
            lateral_speed = np.clip(lateral_speed, -0.5, 0.5)
            
            # Height control
            vertical_speed = 0.3 * (1.0 - self.current_pos[2])
            vertical_speed = np.clip(vertical_speed, -0.3, 0.3)
            
            vel_cmd.twist.linear.x = forward_speed
            vel_cmd.twist.linear.y = lateral_speed
            vel_cmd.twist.linear.z = vertical_speed
            
            self.vel_pub.publish(vel_cmd)
            
            if abs(self.current_pos[0] - int(self.current_pos[0])) < 0.1 and int(self.current_pos[0]) % 10 == 0:
                rospy.loginfo(f"[Mission]: Position: {self.current_pos[0]:.1f}m / {self.forward_target}m")
            
            rate.sleep()
            
        return False
        
    def turn_around(self):
        """Turn 180 degrees"""
        rospy.loginfo("[Mission]: TURN - Rotating 180 degrees")
        
        target_yaw = self.current_yaw + math.pi
        if target_yaw > math.pi:
            target_yaw -= 2 * math.pi
        elif target_yaw < -math.pi:
            target_yaw += 2 * math.pi
            
        rate = rospy.Rate(20)
        
        while not rospy.is_shutdown():
            yaw_error = target_yaw - self.current_yaw
            
            # Normalize to [-pi, pi]
            while yaw_error > math.pi:
                yaw_error -= 2 * math.pi
            while yaw_error < -math.pi:
                yaw_error += 2 * math.pi
                
            if abs(yaw_error) < 0.1:
                rospy.loginfo("[Mission]: Turn complete")
                self.stop_briefly()
                return True
                
            vel_cmd = TwistStamped()
            vel_cmd.header.stamp = rospy.Time.now()
            vel_cmd.header.frame_id = 'world'
            
            # Yaw control
            yaw_rate = np.clip(yaw_error * 0.5, -0.5, 0.5)
            vel_cmd.twist.angular.z = yaw_rate
            
            # Height control
            vel_cmd.twist.linear.z = 0.3 * (1.0 - self.current_pos[2])
            
            self.vel_pub.publish(vel_cmd)
            rate.sleep()
            
        return False
        
    def navigate_return(self):
        """Return to starting point"""
        rospy.loginfo("[Mission]: RETURN - Flying back to start")
        
        rate = rospy.Rate(20)
        
        while not rospy.is_shutdown():
            distance_to_home = abs(self.current_pos[0])
            
            if distance_to_home < 2.0:
                rospy.loginfo("[Mission]: Reached home position")
                self.stop_briefly()
                return True
                
            vel_cmd = TwistStamped()
            vel_cmd.header.stamp = rospy.Time.now()
            vel_cmd.header.frame_id = 'world'
            
            # Forward speed (in body frame, now facing backward)
            if self.min_front_distance < 0.8:
                forward_speed = 0.0
                rospy.logwarn(f"[Mission]: STOP! Obstacle at {self.min_front_distance:.2f}m")
            elif self.min_front_distance < 1.5:
                forward_speed = 0.3 * (self.min_front_distance - 0.8) / 0.7
                rospy.logwarn(f"[Mission]: Slowing down - obstacle at {self.min_front_distance:.2f}m")
            elif self.min_front_distance < 3.0:
                forward_speed = min(1.0, distance_to_home * 0.5)
            else:
                forward_speed = min(self.cruise_speed, distance_to_home * 0.5)
            
            # Lateral centering
            lateral_speed = -0.5 * self.current_pos[1]
            lateral_speed = np.clip(lateral_speed, -0.5, 0.5)
            
            # Height control
            vertical_speed = 0.3 * (1.0 - self.current_pos[2])
            vertical_speed = np.clip(vertical_speed, -0.3, 0.3)
            
            vel_cmd.twist.linear.x = forward_speed
            vel_cmd.twist.linear.y = lateral_speed
            vel_cmd.twist.linear.z = vertical_speed
            
            self.vel_pub.publish(vel_cmd)
            
            if abs(self.current_pos[0] - int(self.current_pos[0])) < 0.1 and int(self.current_pos[0]) % 10 == 0:
                rospy.loginfo(f"[Mission]: Position: {self.current_pos[0]:.1f}m from home")
            
            rate.sleep()
            
        return False
        
    def land(self):
        """Land at current position"""
        rospy.loginfo("[Mission]: LAND - Descending")
        
        # Switch to position control for landing
        self.posctrl_pub.publish(Bool(data=True))
        rospy.sleep(0.5)
        
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = self.current_pos[0]
        pose.pose.position.y = self.current_pos[1]
        pose.pose.position.z = 0.0  # Land
        pose.pose.orientation.w = 1.0
        
        for _ in range(100):
            pose.header.stamp = rospy.Time.now()
            self.pose_pub.publish(pose)
            rospy.sleep(0.1)
            
        rospy.loginfo("[Mission]: MISSION COMPLETE!")
        
    def stop_briefly(self):
        """Stop and hover briefly"""
        vel_cmd = TwistStamped()
        vel_cmd.header.frame_id = 'world'
        
        for _ in range(10):
            vel_cmd.header.stamp = rospy.Time.now()
            self.vel_pub.publish(vel_cmd)
            rospy.sleep(0.1)
            
    def run_mission(self):
        """Execute full mission"""
        rospy.loginfo("="*50)
        rospy.loginfo("[Mission]: Starting mission sequence")
        rospy.loginfo(f"[Mission]: 1. Takeoff to 1m")
        rospy.loginfo(f"[Mission]: 2. Fly forward to {self.forward_target}m")
        rospy.loginfo(f"[Mission]: 3. Turn 180 degrees")
        rospy.loginfo(f"[Mission]: 4. Return to start")
        rospy.loginfo(f"[Mission]: 5. Land")
        rospy.loginfo("="*50)
        
        self.takeoff()
        rospy.sleep(1.0)
        
        if self.navigate_forward():
            rospy.sleep(1.0)
            
            if self.turn_around():
                rospy.sleep(1.0)
                
                if self.navigate_return():
                    rospy.sleep(1.0)
                    self.land()

if __name__ == '__main__':
    try:
        navigator = MissionNavigator()
        navigator.run_mission()
    except rospy.ROSInterruptException:
        pass
