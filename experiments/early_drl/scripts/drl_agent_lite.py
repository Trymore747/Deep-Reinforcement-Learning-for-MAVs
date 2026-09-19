#!/usr/bin/env python3
"""
Lightweight DRL Agent for High-Speed Tunnel Navigation
Uses simple rule-based policy instead of PyTorch (no heavy dependencies)
Can be replaced with trained model later
"""

import numpy as np
import os
import json


class DRLNavigationAgentLite:
    """
    Lightweight DRL Agent without PyTorch dependency
    Uses rule-based policy for navigation until a trained model is available
    """
    def __init__(self, state_dim=21, action_dim=4, model_path=None):
        """
        Args:
            state_dim: Dimension of state space (21)
            action_dim: Dimension of action space (4)
            model_path: Path to model weights (currently ignored in lite version)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Action bounds
        self.action_low = np.array([5.0, -3.0, -2.0, -np.pi/2])
        self.action_high = np.array([10.0, 3.0, 2.0, np.pi/2])
        
        # State normalization (default values)
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        
        # Load model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"[DRL Agent Lite] Loaded model from {model_path}")
            self.use_model = True
        else:
            print(f"[DRL Agent Lite] Using rule-based policy (no model loaded)")
            self.use_model = False
    
    def normalize_state(self, state):
        """Normalize state"""
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def denormalize_action(self, action):
        """Convert normalized action to actual command range"""
        action = np.clip(action, -1.0, 1.0)
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
    
    def get_action(self, state, deterministic=True):
        """
        Get action from current state
        
        Args:
            state: numpy array [pos(3), vel(3), quat(4), lidar(8), goal_dir(3)]
            deterministic: unused in lite version
            
        Returns:
            action: [forward_vel, lateral_vel, vertical_vel, yaw_rate]
        """
        # Parse state
        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]
        lidar = state[10:18]
        goal_dir = state[18:21]
        
        # Rule-based policy
        action = self._rule_based_policy(pos, vel, quat, lidar, goal_dir)
        
        return action
    
    def _rule_based_policy(self, pos, vel, quat, lidar, goal_dir):
        """
        Conservative obstacle avoidance policy - prioritizes safety
        """
        # Parse LiDAR: front, front-left, left, back-left, back, back-right, right, front-right
        front = lidar[0]
        front_left = lidar[1]
        left = lidar[2]
        back_left = lidar[3]
        back = lidar[4]
        back_right = lidar[5]
        right = lidar[6]
        front_right = lidar[7]
        
        min_dist = np.min(lidar)
        
        # CRITICAL: Emergency stop if too close
        if min_dist < 0.8:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Calculate yaw from quaternion
        yaw = np.arctan2(2.0*(quat[3]*quat[2] + quat[0]*quat[1]), 
                        1.0 - 2.0*(quat[1]*quat[1] + quat[2]*quat[2]))
        
        # Goal direction
        goal_yaw = np.arctan2(goal_dir[1], goal_dir[0])
        yaw_error = self._normalize_angle(goal_yaw - yaw)
        
        # Speed based on minimum obstacle distance
        if min_dist < 1.5:
            forward_vel = 0.0  # Stop
        elif min_dist < 2.5:
            forward_vel = 3.0  # Very slow
        elif min_dist < 4.0:
            forward_vel = 5.0  # Slow
        else:
            forward_vel = 7.0  # Normal
        
        # Additional front obstacle check
        if front < 3.0:
            forward_vel = min(forward_vel, 2.0 * (front - 1.0))  # Proportional to distance
        
        forward_vel = max(0.0, min(forward_vel, 10.0))
        
        # Lateral: Find clearest direction
        left_clearance = (left + front_left) / 2.0
        right_clearance = (right + front_right) / 2.0
        
        lateral_vel = 0.0
        if min_dist < 3.0:
            # Strong avoidance
            if right_clearance > left_clearance + 0.5:
                lateral_vel = -1.5  # Move right (negative y)
            elif left_clearance > right_clearance + 0.5:
                lateral_vel = 1.5   # Move left (positive y)
        
        lateral_vel = np.clip(lateral_vel, -3.0, 3.0)
        
        # Vertical: Maintain height, climb if obstacle ahead
        target_height = 1.5
        vertical_vel = 1.0 * (target_height - pos[2])
        
        if front < 3.0 and pos[2] < 2.5:
            vertical_vel = 1.0  # Climb
        
        vertical_vel = np.clip(vertical_vel, -2.0, 2.0)
        
        # Yaw: Turn toward clear space or goal
        if front < 3.0:
            # Obstacle ahead - turn toward clearer side
            if right_clearance > left_clearance + 0.5:
                yaw_rate = -0.5  # Turn right
            elif left_clearance > right_clearance + 0.5:
                yaw_rate = 0.5   # Turn left
            else:
                yaw_rate = 0.0  # Don't turn
        else:
            # Clear ahead - turn toward goal
            yaw_rate = 0.3 * yaw_error
        
        yaw_rate = np.clip(yaw_rate, -0.5, 0.5)  # Slower turning
        
        return np.array([forward_vel, lateral_vel, vertical_vel, yaw_rate])
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return angle
    
    def load_model(self, model_path):
        """Load model parameters (lite version loads config only)"""
        try:
            # Try to load as JSON config
            config_path = model_path.replace('.pth', '.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'state_mean' in config:
                        self.state_mean = np.array(config['state_mean'])
                    if 'state_std' in config:
                        self.state_std = np.array(config['state_std'])
        except Exception as e:
            print(f"[DRL Agent Lite] Could not load config: {e}")
    
    def save_model(self, model_path):
        """Save configuration"""
        config = {
            'state_mean': self.state_mean.tolist(),
            'state_std': self.state_std.tolist(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'type': 'rule_based'
        }
        
        config_path = model_path.replace('.pth', '.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[DRL Agent Lite] Config saved to {config_path}")


if __name__ == '__main__':
    # Test the agent
    print("Testing Lightweight DRL Navigation Agent...")
    
    agent = DRLNavigationAgentLite(state_dim=21, action_dim=4)
    
    # Dummy state
    dummy_state = np.random.randn(21)
    dummy_state[10:18] = np.abs(dummy_state[10:18]) * 5  # LiDAR ranges
    
    # Get action
    action = agent.get_action(dummy_state)
    print(f"State shape: {dummy_state.shape}")
    print(f"Action: {action}")
    print(f"  Forward velocity: {action[0]:.2f} m/s")
    print(f"  Lateral velocity: {action[1]:.2f} m/s")
    print(f"  Vertical velocity: {action[2]:.2f} m/s")
    print(f"  Yaw rate: {action[3]:.2f} rad/s")
    
    print("\nLightweight agent initialized successfully!")
    print("Note: This uses rule-based policy. Train a model for better performance.")
