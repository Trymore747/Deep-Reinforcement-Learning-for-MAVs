#!/usr/bin/env python3
"""
Gym Environment for Training DRL Agent in Tunnel Navigation
Compatible with OpenAI Gym and Stable-Baselines3
"""

import gym
from gym import spaces
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
import tf


class TunnelNavigationEnv(gym.Env):
    """
    Custom Gym environment for high-speed tunnel navigation training
    
    State Space:
        - Position (3): x, y, z
        - Velocity (3): vx, vy, vz
        - Orientation (4): quaternion
        - LiDAR ranges (8): directional obstacle distances
        - Goal direction (3): normalized vector to goal
        Total: 21 dimensions
    
    Action Space:
        - Forward velocity (1): 5-10 m/s
        - Lateral velocity (1): -3 to 3 m/s
        - Vertical velocity (1): -2 to 2 m/s
        - Yaw rate (1): -pi/2 to pi/2 rad/s
        Total: 4 dimensions (continuous)
    
    Reward Function:
        - Progress toward goal: +1 per meter
        - Speed bonus: +0.1 * (speed - 5) for speed > 5 m/s
        - Collision penalty: -100
        - Boundary penalty: -10
        - Goal reached: +500
        - Time penalty: -0.1 per step
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, max_episode_steps=1000, use_ros=True):
        super(TunnelNavigationEnv, self).__init__()
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.use_ros = use_ros
        
        # State and action space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )
        
        # Normalized action space [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Action bounds (actual physical limits)
        self.action_low = np.array([5.0, -3.0, -2.0, -np.pi/2])
        self.action_high = np.array([10.0, 3.0, 2.0, np.pi/2])
        
        # Tunnel parameters
        self.tunnel_length = 100.0  # meters
        self.tunnel_width = 4.0  # meters
        self.tunnel_height = 3.0  # meters
        
        # Goal parameters
        self.goal_reach_threshold = 1.0  # meters
        self.collision_threshold = 0.5  # meters
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.last_distance_to_goal = None
        
        # ROS integration
        if self.use_ros:
            self.setup_ros()
        
        # State variables
        self.current_state = None
        self.current_goal = None
        self.drone_position = np.zeros(3)
        self.drone_velocity = np.zeros(3)
        self.drone_orientation = np.array([0, 0, 0, 1])
        self.lidar_ranges = np.ones(8) * 10.0
        
        # Reset environment
        self.reset()
    
    def setup_ros(self):
        """Setup ROS publishers and subscribers for sim interaction"""
        rospy.loginfo("[Gym Env] Setting up ROS interface...")
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel',
                                          TwistStamped, queue_size=1)
        self.reset_pub = rospy.Publisher('/gazebo/reset_simulation',
                                        Empty, queue_size=1)
        
        # Subscribers
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom',
                                        Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan',
                                        LaserScan, self.scan_callback)
        
        rospy.loginfo("[Gym Env] ROS interface ready")
    
    def odom_callback(self, msg):
        """Process odometry from simulator"""
        self.drone_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.drone_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        self.drone_orientation = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
    
    def scan_callback(self, msg):
        """Process LiDAR scan"""
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = msg.range_max
        
        num_ranges = len(ranges)
        sector_size = num_ranges // 8
        
        for i in range(8):
            start_idx = i * sector_size
            end_idx = (i + 1) * sector_size if i < 7 else num_ranges
            self.lidar_ranges[i] = np.min(ranges[start_idx:end_idx])
    
    def denormalize_action(self, action):
        """Convert normalized action to physical command"""
        action = np.clip(action, -1.0, 1.0)
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
    
    def get_state(self):
        """Construct state observation"""
        # Goal direction
        goal_dir = self.current_goal - self.drone_position
        goal_dist = np.linalg.norm(goal_dir)
        if goal_dist > 0:
            goal_dir_norm = goal_dir / goal_dist
        else:
            goal_dir_norm = np.zeros(3)
        
        # Concatenate state
        state = np.concatenate([
            self.drone_position,
            self.drone_velocity,
            self.drone_orientation,
            self.lidar_ranges,
            goal_dir_norm
        ]).astype(np.float32)
        
        return state
    
    def calculate_reward(self, action, done_info):
        """Calculate reward for current step"""
        reward = 0.0
        
        # Distance to goal
        goal_dist = np.linalg.norm(self.current_goal - self.drone_position)
        
        # Progress reward
        if self.last_distance_to_goal is not None:
            progress = self.last_distance_to_goal - goal_dist
            reward += progress * 1.0  # +1 per meter of progress
        
        self.last_distance_to_goal = goal_dist
        
        # Speed bonus (encourage high speed)
        speed = np.linalg.norm(self.drone_velocity)
        if speed > 5.0:
            reward += 0.1 * (speed - 5.0)
        
        # Time penalty (encourage efficiency)
        reward -= 0.1
        
        # Terminal rewards/penalties
        if done_info['collision']:
            reward -= 100.0
        elif done_info['boundary']:
            reward -= 10.0
        elif done_info['goal_reached']:
            reward += 500.0
        
        # Penalty for getting too close to obstacles
        min_obstacle_dist = np.min(self.lidar_ranges)
        if min_obstacle_dist < 1.5:
            reward -= (1.5 - min_obstacle_dist) * 2.0
        
        return reward
    
    def check_done(self):
        """Check if episode is done"""
        done_info = {
            'goal_reached': False,
            'collision': False,
            'boundary': False,
            'timeout': False
        }
        
        # Goal reached
        goal_dist = np.linalg.norm(self.current_goal - self.drone_position)
        if goal_dist < self.goal_reach_threshold:
            done_info['goal_reached'] = True
            return True, done_info
        
        # Collision
        min_obstacle_dist = np.min(self.lidar_ranges)
        if min_obstacle_dist < self.collision_threshold:
            done_info['collision'] = True
            return True, done_info
        
        # Boundary violation
        if (abs(self.drone_position[1]) > self.tunnel_width / 2 or
            self.drone_position[2] < 0.3 or
            self.drone_position[2] > self.tunnel_height):
            done_info['boundary'] = True
            return True, done_info
        
        # Timeout
        if self.current_step >= self.max_episode_steps:
            done_info['timeout'] = True
            return True, done_info
        
        return False, done_info
    
    def step(self, action):
        """
        Execute one time step
        
        Args:
            action: normalized action in [-1, 1]
        
        Returns:
            observation: state after action
            reward: reward for this step
            done: whether episode is done
            info: additional information
        """
        self.current_step += 1
        
        # Denormalize action
        physical_action = self.denormalize_action(action)
        
        # Execute action (send to simulator if using ROS)
        if self.use_ros:
            self.execute_action_ros(physical_action)
            rospy.sleep(0.02)  # 50 Hz control rate
        else:
            # Simulate dynamics (simple model)
            self.simulate_dynamics(physical_action, dt=0.02)
        
        # Get new state
        state = self.get_state()
        
        # Check if done
        done, done_info = self.check_done()
        
        # Calculate reward
        reward = self.calculate_reward(physical_action, done_info)
        self.episode_reward += reward
        
        # Info dictionary
        info = {
            'episode_step': self.current_step,
            'episode_reward': self.episode_reward,
            'goal_distance': np.linalg.norm(self.current_goal - self.drone_position),
            'speed': np.linalg.norm(self.drone_velocity),
            **done_info
        }
        
        return state, reward, done, info
    
    def execute_action_ros(self, action):
        """Execute action by publishing to ROS"""
        # Get current yaw
        quat = self.drone_orientation
        euler = tf.transformations.euler_from_quaternion(quat)
        yaw = euler[2]
        
        # Convert body frame to world frame
        forward_vel, lateral_vel, vertical_vel, yaw_rate = action
        vx_world = forward_vel * np.cos(yaw) - lateral_vel * np.sin(yaw)
        vy_world = forward_vel * np.sin(yaw) + lateral_vel * np.cos(yaw)
        
        # Publish command
        cmd = TwistStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.twist.linear.x = vx_world
        cmd.twist.linear.y = vy_world
        cmd.twist.linear.z = vertical_vel
        cmd.twist.angular.z = yaw_rate
        
        self.cmd_vel_pub.publish(cmd)
    
    def simulate_dynamics(self, action, dt):
        """Simple dynamics simulation (for non-ROS mode)"""
        # Extract action components
        forward_vel, lateral_vel, vertical_vel, yaw_rate = action
        
        # Get current yaw
        quat = self.drone_orientation
        euler = tf.transformations.euler_from_quaternion(quat)
        roll, pitch, yaw = euler
        
        # Update yaw
        yaw += yaw_rate * dt
        
        # Update position (body frame to world frame)
        vx_world = forward_vel * np.cos(yaw) - lateral_vel * np.sin(yaw)
        vy_world = forward_vel * np.sin(yaw) + lateral_vel * np.cos(yaw)
        
        self.drone_position[0] += vx_world * dt
        self.drone_position[1] += vy_world * dt
        self.drone_position[2] += vertical_vel * dt
        
        # Update velocity
        self.drone_velocity = np.array([vx_world, vy_world, vertical_vel])
        
        # Update orientation
        new_quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        self.drone_orientation = np.array(new_quat)
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.episode_reward = 0.0
        self.last_distance_to_goal = None
        
        # Reset drone to start position
        self.drone_position = np.array([0.0, 0.0, 1.5])
        self.drone_velocity = np.zeros(3)
        self.drone_orientation = np.array([0, 0, 0, 1])
        self.lidar_ranges = np.ones(8) * 10.0
        
        # Set goal at end of tunnel
        self.current_goal = np.array([self.tunnel_length, 0.0, 1.5])
        
        # Reset simulator if using ROS
        if self.use_ros:
            self.reset_pub.publish(Empty())
            rospy.sleep(1.0)  # Wait for reset
        
        return self.get_state()
    
    def render(self, mode='human'):
        """Render environment (placeholder)"""
        if mode == 'human':
            print(f"Step: {self.current_step}, "
                  f"Position: [{self.drone_position[0]:.2f}, {self.drone_position[1]:.2f}, {self.drone_position[2]:.2f}], "
                  f"Speed: {np.linalg.norm(self.drone_velocity):.2f} m/s, "
                  f"Reward: {self.episode_reward:.2f}")
    
    def close(self):
        """Cleanup"""
        pass


# Training script using Stable-Baselines3
if __name__ == '__main__':
    """
    Example training script
    Requires: pip install stable-baselines3
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.callbacks import CheckpointCallback
        
        print("Creating environment...")
        env = TunnelNavigationEnv(use_ros=False)  # Use simulation mode for testing
        
        print("Checking environment...")
        check_env(env)
        print("Environment check passed!")
        
        print("Creating PPO agent...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./ppo_tunnel_navigation_tensorboard/"
        )
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path='./models/',
            name_prefix='ppo_tunnel_nav'
        )
        
        print("Starting training...")
        model.learn(
            total_timesteps=1000000,
            callback=checkpoint_callback
        )
        
        print("Saving final model...")
        model.save("ppo_tunnel_navigation_final")
        
        print("Training complete!")
        
    except ImportError:
        print("This training script requires stable-baselines3")
        print("Install with: pip install stable-baselines3")
        print("\nEnvironment class can still be used independently")
