#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Navigation for Dynamic Environment
Deep reinforcement learning based high-speed navigation
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
from collections import deque
import os


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs):
        """Forward pass"""
        features = self.feature(obs)
        
        # Actor
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        
        # Critic
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy"""
        with torch.no_grad():
            action_mean, action_std, value = self.forward(obs)
            
            if deterministic:
                action = action_mean
            else:
                dist = Normal(action_mean, action_std)
                action = dist.sample()
            
            return action.cpu().numpy(), value.cpu().item()
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions for training"""
        action_mean, action_std, value = self.forward(obs)
        
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_probs, entropy, value


class PPOPlanner:
    """PPO-based planner for high-speed navigation"""
    
    def __init__(self):
        rospy.init_node('ppo_planner', anonymous=True)
        
        # Parameters
        self.max_speed = rospy.get_param('~max_speed', 3.0)
        self.obs_range = rospy.get_param('~obs_range', 5.0)
        self.n_rays = rospy.get_param('~n_rays', 16)  # Lidar-like rays
        self.model_path = rospy.get_param('~model_path', 
            '/home/trymore/catkin_ws/src/CERLAB-UAV-Autonomy/autonomous_flight/models/ppo_tunnel_nav.pth')
        
        # Device
        self.device = torch.device('cpu')  # Use CPU for real-time inference
        
        # State
        self.current_pos = None
        self.current_vel = None
        self.goal_pos = None
        self.obstacle_cloud = None
        self.depth_readings = np.ones(self.n_rays) * self.obs_range
        
        # Observation and action dimensions
        # Obs: [rel_goal_x, rel_goal_y, rel_goal_z, vel_x, vel_y, vel_z, n_rays depth readings]
        self.obs_dim = 6 + self.n_rays
        self.act_dim = 3  # [vx, vy, vz]
        
        # Initialize network
        self.policy = ActorCritic(self.obs_dim, self.act_dim, hidden_dim=256).to(self.device)
        
        # Load trained model if exists
        if os.path.exists(self.model_path):
            self.policy.load_state_dict(torch.load(self.model_path, map_location=self.device))
            rospy.loginfo(f"Loaded PPO model from {self.model_path}")
        else:
            rospy.logwarn(f"No trained model found at {self.model_path} - using random policy")
        
        self.policy.eval()
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        rospy.Subscriber('/camera/depth/points', PointCloud2, self.pointcloud_callback)
        
        rospy.loginfo("PPO Planner initialized - Deep RL navigation mode")
        rospy.loginfo(f"Observation dim: {self.obs_dim}, Action dim: {self.act_dim}")
        rospy.loginfo(f"Max speed: {self.max_speed} m/s, Ray sensors: {self.n_rays}")
    
    def odom_callback(self, msg):
        """Update current position and velocity"""
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
    
    def goal_callback(self, msg):
        """Receive new goal"""
        self.goal_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        rospy.loginfo(f"New goal received: {self.goal_pos}")
    
    def pointcloud_callback(self, msg):
        """Update obstacle perception from point cloud"""
        points = []
        for point in pc2.read_points(msg, skip_nans=True):
            points.append([point[0], point[1], point[2]])
        
        if len(points) > 0:
            self.obstacle_cloud = np.array(points)
            self.update_depth_readings()
    
    def update_depth_readings(self):
        """Compute depth readings in radial directions (lidar-like)"""
        if self.obstacle_cloud is None or self.current_pos is None:
            return
        
        # Create ray directions in horizontal plane (more rays) + vertical
        angles = np.linspace(0, 2*np.pi, self.n_rays - 2, endpoint=False)
        directions = []
        
        # Horizontal rays
        for angle in angles:
            directions.append([np.cos(angle), np.sin(angle), 0])
        
        # Vertical rays
        directions.append([0, 0, 1])   # Up
        directions.append([0, 0, -1])  # Down
        
        directions = np.array(directions)
        
        # Calculate depth for each ray
        for i, direction in enumerate(directions):
            # Find closest obstacle in this direction
            relative_pos = self.obstacle_cloud - self.current_pos
            
            # Project onto ray direction
            projections = np.dot(relative_pos, direction)
            
            # Only consider points in front of ray
            mask = projections > 0
            
            if np.any(mask):
                # Get distances
                distances = np.linalg.norm(relative_pos[mask], axis=1)
                
                # Filter by direction cone (30 degree cone)
                dot_products = np.dot(relative_pos[mask], direction) / (distances + 1e-6)
                cone_mask = dot_products > np.cos(np.pi / 6)
                
                if np.any(cone_mask):
                    self.depth_readings[i] = np.min(distances[cone_mask])
                else:
                    self.depth_readings[i] = self.obs_range
            else:
                self.depth_readings[i] = self.obs_range
        
        # Clip to observation range
        self.depth_readings = np.clip(self.depth_readings, 0, self.obs_range)
    
    def get_observation(self):
        """Construct observation vector"""
        if self.current_pos is None or self.goal_pos is None:
            return None
        
        # Relative goal position
        rel_goal = self.goal_pos - self.current_pos
        
        # Current velocity
        vel = self.current_vel if self.current_vel is not None else np.zeros(3)
        
        # Normalize depth readings
        normalized_depth = self.depth_readings / self.obs_range
        
        # Combine observation
        obs = np.concatenate([
            rel_goal / 10.0,  # Normalize goal distance
            vel / self.max_speed,  # Normalize velocity
            normalized_depth
        ])
        
        return obs.astype(np.float32)
    
    def compute_action(self):
        """Compute action using PPO policy"""
        obs = self.get_observation()
        
        if obs is None:
            return np.zeros(3)
        
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        # Get action from policy (deterministic for deployment)
        action, value = self.policy.get_action(obs_tensor, deterministic=True)
        
        # Scale action to max speed
        action = action[0] * self.max_speed
        
        # Apply safety: slow down near obstacles
        min_depth = np.min(self.depth_readings)
        if min_depth < 1.0:
            safety_factor = min_depth / 1.0
            action = action * safety_factor
        
        return action
    
    def control_loop(self):
        """Main control loop"""
        if self.current_pos is None or self.goal_pos is None:
            # Hover in place
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Check if goal reached
        goal_dist = np.linalg.norm(self.goal_pos - self.current_pos)
        if goal_dist < 0.5:
            rospy.loginfo("Goal reached!")
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return
        
        # Compute action
        action = self.compute_action()
        
        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.linear.y = float(action[1])
        cmd.linear.z = float(action[2])
        self.cmd_vel_pub.publish(cmd)
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(20)  # 20 Hz control
        
        while not rospy.is_shutdown():
            self.control_loop()
            rate.sleep()


class PPOTrainer:
    """PPO training implementation for tunnel navigation"""
    
    def __init__(self, env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        # Hyperparameters
        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        self.n_epochs = 10
        self.batch_size = 64
        self.buffer_size = 2048
        
        # Network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Buffers
        self.obs_buffer = []
        self.act_buffer = []
        self.rew_buffer = []
        self.val_buffer = []
        self.logp_buffer = []
        self.done_buffer = []
        
        rospy.loginfo(f"PPO Trainer initialized on device: {self.device}")
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return np.array(advantages), np.array(returns)
    
    def update(self):
        """Update policy using PPO"""
        # Convert to tensors
        obs = torch.FloatTensor(np.array(self.obs_buffer)).to(self.device)
        acts = torch.FloatTensor(np.array(self.act_buffer)).to(self.device)
        old_logps = torch.FloatTensor(np.array(self.logp_buffer)).to(self.device)
        
        # Compute advantages
        advantages, returns = self.compute_gae(
            self.rew_buffer, self.val_buffer, self.done_buffer
        )
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for epoch in range(self.n_epochs):
            # Mini-batch updates
            indices = np.arange(len(self.obs_buffer))
            np.random.shuffle(indices)
            
            for start in range(0, len(self.obs_buffer), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Evaluate actions
                logps, entropy, values = self.policy.evaluate_actions(
                    obs[batch_indices], acts[batch_indices]
                )
                
                # Policy loss
                ratio = torch.exp(logps - old_logps[batch_indices])
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages[batch_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (returns[batch_indices] - values).pow(2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Clear buffers
        self.obs_buffer = []
        self.act_buffer = []
        self.rew_buffer = []
        self.val_buffer = []
        self.logp_buffer = []
        self.done_buffer = []
    
    def train(self, n_episodes=1000):
        """Train PPO agent"""
        for episode in range(n_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_mean, action_std, value = self.policy(obs_tensor)
                    dist = Normal(action_mean, action_std)
                    action = dist.sample()
                    logp = dist.log_prob(action).sum(dim=-1)
                
                action_np = action.cpu().numpy()[0]
                value_np = value.cpu().item()
                logp_np = logp.cpu().item()
                
                # Step environment
                next_obs, reward, done, info = self.env.step(action_np)
                
                # Store transition
                self.obs_buffer.append(obs)
                self.act_buffer.append(action_np)
                self.rew_buffer.append(reward)
                self.val_buffer.append(value_np)
                self.logp_buffer.append(logp_np)
                self.done_buffer.append(done)
                
                obs = next_obs
                episode_reward += reward
                
                # Update policy
                if len(self.obs_buffer) >= self.buffer_size:
                    self.update()
            
            rospy.loginfo(f"Episode {episode}: Reward = {episode_reward:.2f}")
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                save_path = f'/home/trymore/catkin_ws/src/CERLAB-UAV-Autonomy/autonomous_flight/models/ppo_tunnel_nav_ep{episode+1}.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.policy.state_dict(), save_path)
                rospy.loginfo(f"Model saved to {save_path}")


if __name__ == '__main__':
    try:
        planner = PPOPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
