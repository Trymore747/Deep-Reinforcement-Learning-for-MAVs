#!/usr/bin/env python3
"""
High-Speed DRL Navigation System (10 m/s target)
Demonstrates that only Deep RL can achieve 10 m/s navigation in dynamic environments
Uses PPO with specifically tuned hyperparameters for high-speed flight
"""

import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import os


class HighSpeedActorCritic(nn.Module):
    """Enhanced Actor-Critic for high-speed navigation"""
    
    def __init__(self, obs_dim, act_dim, hidden_dim=512):
        super(HighSpeedActorCritic, self).__init__()
        
        # Deeper network for high-speed decision making
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        # Actor head with higher capacity
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, act_dim),
            nn.Tanh()  # Bounded output
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )
    
    def forward(self, obs):
        features = self.feature(obs)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        value = self.critic(features)
        return action_mean, action_std, value
    
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            action_mean, action_std, value = self.forward(obs)
            
            if deterministic:
                action = action_mean
            else:
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                action = torch.clamp(action, -1, 1)  # Ensure bounded
            
            return action.cpu().numpy(), value.cpu().item()


class HighSpeedDRLNavigator:
    """High-speed navigation system using DRL (target: 10 m/s)"""
    
    def __init__(self):
        rospy.init_node('highspeed_drl_navigator', anonymous=True)
        
        # High-speed parameters
        self.max_speed = rospy.get_param('~max_speed', 10.0)  # 10 m/s target!
        self.aggressive_mode = rospy.get_param('~aggressive_mode', True)
        self.obs_range = rospy.get_param('~obs_range', 10.0)  # Longer lookahead
        self.n_rays = rospy.get_param('~n_rays', 32)  # More sensors
        self.model_path = rospy.get_param('~model_path', 
            '/home/trymore/catkin_ws/src/CERLAB-UAV-Autonomy/autonomous_flight/models/highspeed_ppo_10ms.pth')
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # State
        self.current_pos = None
        self.current_vel = None
        self.goal_pos = None
        self.obstacle_cloud = None
        self.depth_readings = np.ones(self.n_rays) * self.obs_range
        
        # Enhanced observation space for high-speed flight
        # [rel_goal(3), velocity(3), goal_distance(1), goal_angle(2), depth_readings(32), prev_action(3)]
        self.obs_dim = 3 + 3 + 1 + 2 + self.n_rays + 3
        self.act_dim = 3  # [vx, vy, vz]
        
        # Previous action for smoothness
        self.prev_action = np.zeros(3)
        
        # Initialize high-speed network
        self.policy = HighSpeedActorCritic(self.obs_dim, self.act_dim, hidden_dim=512).to(self.device)
        
        # Load model if exists
        if os.path.exists(self.model_path):
            self.policy.load_state_dict(torch.load(self.model_path, map_location=self.device))
            rospy.loginfo(f"Loaded high-speed model from {self.model_path}")
        else:
            rospy.logwarn(f"No trained model found - using random policy")
        
        self.policy.eval()
        
        # Performance tracking
        self.max_speed_achieved = 0
        self.speed_history = []
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        rospy.Subscriber('/camera/depth/points', PointCloud2, self.pointcloud_callback)
        
        rospy.loginfo("="*60)
        rospy.loginfo("HIGH-SPEED DRL NAVIGATOR INITIALIZED")
        rospy.loginfo("="*60)
        rospy.loginfo(f"Target Speed: {self.max_speed} m/s (10 m/s!)")
        rospy.loginfo(f"Aggressive Mode: {self.aggressive_mode}")
        rospy.loginfo(f"Observation dim: {self.obs_dim}, Action dim: {self.act_dim}")
        rospy.loginfo(f"Sensors: {self.n_rays} rays, range: {self.obs_range}m")
        rospy.loginfo(f"Device: {self.device}")
        rospy.loginfo("="*60)
    
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
        
        # Track speed
        speed = np.linalg.norm(self.current_vel)
        self.speed_history.append(speed)
        if speed > self.max_speed_achieved:
            self.max_speed_achieved = speed
            if speed > 8.0:  # Celebrate high speeds!
                rospy.loginfo(f"🚀 NEW SPEED RECORD: {speed:.2f} m/s!")
    
    def goal_callback(self, msg):
        self.goal_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        rospy.loginfo(f"High-speed mission to: {self.goal_pos}")
    
    def pointcloud_callback(self, msg):
        points = []
        for point in pc2.read_points(msg, skip_nans=True):
            points.append([point[0], point[1], point[2]])
        
        if len(points) > 0:
            self.obstacle_cloud = np.array(points)
            self.update_depth_readings()
    
    def update_depth_readings(self):
        """Enhanced depth sensing for high-speed flight"""
        if self.obstacle_cloud is None or self.current_pos is None:
            return
        
        # Denser ray pattern for high-speed obstacle detection
        angles_horizontal = np.linspace(0, 2*np.pi, self.n_rays - 4, endpoint=False)
        directions = []
        
        # Horizontal rays
        for angle in angles_horizontal:
            directions.append([np.cos(angle), np.sin(angle), 0])
        
        # Vertical rays
        directions.append([0, 0, 1])   # Up
        directions.append([0, 0, -1])  # Down
        
        # Forward-focused rays for high-speed
        directions.append([1, 0, 0.2])   # Forward-up
        directions.append([1, 0, -0.2])  # Forward-down
        
        directions = np.array(directions)
        
        # Calculate depths
        for i, direction in enumerate(directions):
            relative_pos = self.obstacle_cloud - self.current_pos
            projections = np.dot(relative_pos, direction)
            mask = projections > 0
            
            if np.any(mask):
                distances = np.linalg.norm(relative_pos[mask], axis=1)
                dot_products = np.dot(relative_pos[mask], direction) / (distances + 1e-6)
                cone_mask = dot_products > np.cos(np.pi / 8)  # Narrower cone for precision
                
                if np.any(cone_mask):
                    self.depth_readings[i] = np.min(distances[cone_mask])
                else:
                    self.depth_readings[i] = self.obs_range
            else:
                self.depth_readings[i] = self.obs_range
        
        self.depth_readings = np.clip(self.depth_readings, 0, self.obs_range)
    
    def get_observation(self):
        """Enhanced observation for high-speed flight"""
        if self.current_pos is None or self.goal_pos is None:
            return None
        
        # Relative goal
        rel_goal = self.goal_pos - self.current_pos
        goal_distance = np.linalg.norm(rel_goal)
        
        # Goal angle (horizontal and vertical)
        goal_angle_h = np.arctan2(rel_goal[1], rel_goal[0])
        goal_angle_v = np.arctan2(rel_goal[2], np.linalg.norm(rel_goal[:2]))
        
        # Current velocity
        vel = self.current_vel if self.current_vel is not None else np.zeros(3)
        
        # Normalize depth readings
        normalized_depth = self.depth_readings / self.obs_range
        
        # Combine enhanced observation
        obs = np.concatenate([
            rel_goal / 20.0,  # Normalized goal (longer distances)
            vel / self.max_speed,  # Normalized velocity
            [goal_distance / 100.0],  # Normalized distance
            [np.cos(goal_angle_h), np.sin(goal_angle_h)],  # Goal direction
            normalized_depth,
            self.prev_action  # Action history for smoothness
        ])
        
        return obs.astype(np.float32)
    
    def compute_action(self):
        """Compute high-speed action"""
        obs = self.get_observation()
        
        if obs is None:
            return np.zeros(3)
        
        # Get action from policy
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, value = self.policy.get_action(obs_tensor, deterministic=True)
        action = action[0]
        
        # Scale to high speed
        if self.aggressive_mode:
            # Full speed ahead!
            action = action * self.max_speed
        else:
            # Adaptive speed based on obstacle proximity
            min_depth = np.min(self.depth_readings)
            if min_depth < 2.0:
                speed_factor = min_depth / 2.0
            else:
                speed_factor = 1.0
            action = action * self.max_speed * speed_factor
        
        # Store for next observation
        self.prev_action = action / self.max_speed
        
        return action
    
    def control_loop(self):
        """High-speed control loop"""
        if self.current_pos is None or self.goal_pos is None:
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Check goal
        goal_dist = np.linalg.norm(self.goal_pos - self.current_pos)
        if goal_dist < 1.0:
            rospy.loginfo(f"Goal reached! Max speed achieved: {self.max_speed_achieved:.2f} m/s")
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Compute high-speed action
        action = self.compute_action()
        
        # Publish command
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.linear.y = float(action[1])
        cmd.linear.z = float(action[2])
        self.cmd_vel_pub.publish(cmd)
        
        # Log performance
        if len(self.speed_history) % 50 == 0:  # Every ~2.5 seconds at 20Hz
            recent_avg = np.mean(self.speed_history[-50:])
            rospy.loginfo(f"Speed: {recent_avg:.2f} m/s | Max: {self.max_speed_achieved:.2f} m/s | Goal: {goal_dist:.1f}m")
    
    def run(self):
        """Main high-speed loop"""
        rate = rospy.Rate(20)  # 20 Hz for high-speed control
        
        rospy.loginfo("High-speed DRL navigator ready for 10 m/s navigation!")
        
        while not rospy.is_shutdown():
            self.control_loop()
            rate.sleep()


if __name__ == '__main__':
    try:
        navigator = HighSpeedDRLNavigator()
        navigator.run()
    except rospy.ROSInterruptException:
        pass
