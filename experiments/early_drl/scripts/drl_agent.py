#!/usr/bin/env python3
"""
Deep Reinforcement Learning Agent for High-Speed Tunnel Navigation
Uses PPO (Proximal Policy Optimization) for stable training and inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import os


class DroneActorCritic(nn.Module):
    """
    Actor-Critic network for drone navigation
    Actor: outputs velocity commands and yaw rate
    Critic: estimates state value
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DroneActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (policy network)
        self.actor_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value network)
        self.critic_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_value = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        # Shared layers
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        # Actor output
        actor_out = F.relu(self.actor_fc1(x))
        action_mean = self.actor_mean(actor_out)
        action_std = torch.exp(self.actor_logstd)
        
        # Critic output
        critic_out = F.relu(self.critic_fc1(x))
        state_value = self.critic_value(critic_out)
        
        return action_mean, action_std, state_value
    
    def get_action(self, state, deterministic=False):
        """Sample action from the policy"""
        action_mean, action_std, _ = self.forward(state)
        
        if deterministic:
            return action_mean
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            return action
    
    def evaluate_actions(self, state, action):
        """Evaluate log probability and entropy of actions"""
        action_mean, action_std, state_value = self.forward(state)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return action_log_probs, state_value, dist_entropy


class DRLNavigationAgent:
    """
    DRL Agent for high-speed tunnel navigation
    Handles inference and model loading
    """
    def __init__(self, state_dim=21, action_dim=4, device='cuda', model_path=None):
        """
        Args:
            state_dim: Dimension of state space
                - Position (3): x, y, z
                - Velocity (3): vx, vy, vz
                - Orientation (4): qx, qy, qz, qw
                - LiDAR/Depth ranges (8): front, front-left, left, back-left, back, back-right, right, front-right
                - Goal direction (3): normalized vector to goal
            action_dim: Dimension of action space
                - Forward velocity command (1): 5-10 m/s
                - Lateral velocity command (1): -3 to 3 m/s
                - Vertical velocity command (1): -2 to 2 m/s
                - Yaw rate command (1): -pi/2 to pi/2 rad/s
            device: 'cuda' or 'cpu'
            model_path: Path to pretrained model weights
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize network
        self.policy = DroneActorCritic(state_dim, action_dim).to(self.device)
        
        # Load pretrained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"[DRL Agent] Loaded model from {model_path}")
        else:
            print(f"[DRL Agent] Initialized with random weights")
        
        self.policy.eval()  # Set to evaluation mode
        
        # Action bounds for safety
        self.action_low = np.array([5.0, -3.0, -2.0, -np.pi/2])  # [forward_vel, lateral_vel, vertical_vel, yaw_rate]
        self.action_high = np.array([10.0, 3.0, 2.0, np.pi/2])
        
        # State normalization parameters (to be updated during training)
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
    
    def normalize_state(self, state):
        """Normalize state for better network performance"""
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def denormalize_action(self, action):
        """Convert normalized action [-1, 1] to actual command range"""
        action = np.clip(action, -1.0, 1.0)
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
    
    def get_action(self, state, deterministic=True):
        """
        Get action from current state
        
        Args:
            state: numpy array of current state
            deterministic: if True, return mean action; if False, sample from distribution
            
        Returns:
            action: numpy array [forward_vel, lateral_vel, vertical_vel, yaw_rate]
        """
        # Normalize state
        state_norm = self.normalize_state(state)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_tensor = self.policy.get_action(state_tensor, deterministic)
        
        # Convert to numpy and denormalize
        action_norm = action_tensor.cpu().numpy().flatten()
        action = self.denormalize_action(action_norm)
        
        return action
    
    def load_model(self, model_path):
        """Load pretrained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        # Load normalization parameters if available
        if 'state_mean' in checkpoint:
            self.state_mean = checkpoint['state_mean']
        if 'state_std' in checkpoint:
            self.state_std = checkpoint['state_std']
    
    def save_model(self, model_path):
        """Save model weights"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        torch.save(checkpoint, model_path)
        print(f"[DRL Agent] Model saved to {model_path}")


class ReplayBuffer:
    """Experience replay buffer for training"""
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class PPOTrainer:
    """
    PPO trainer for the DRL agent
    For offline training in simulation
    """
    def __init__(self, agent, learning_rate=3e-4, gamma=0.99, epsilon=0.2, 
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.agent = agent
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = torch.optim.Adam(agent.policy.parameters(), lr=learning_rate)
    
    def update(self, rollouts):
        """
        Update policy using PPO algorithm
        
        Args:
            rollouts: dictionary containing states, actions, rewards, values, log_probs
        """
        states = torch.FloatTensor(rollouts['states']).to(self.agent.device)
        actions = torch.FloatTensor(rollouts['actions']).to(self.agent.device)
        old_log_probs = torch.FloatTensor(rollouts['log_probs']).to(self.agent.device)
        returns = torch.FloatTensor(rollouts['returns']).to(self.agent.device)
        advantages = torch.FloatTensor(rollouts['advantages']).to(self.agent.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        log_probs, values, entropy = self.agent.policy.evaluate_actions(states, actions)
        
        # Policy loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': loss.item()
        }


if __name__ == '__main__':
    # Test the agent
    print("Testing DRL Navigation Agent...")
    
    agent = DRLNavigationAgent(state_dim=21, action_dim=4, device='cpu')
    
    # Dummy state
    dummy_state = np.random.randn(21)
    
    # Get action
    action = agent.get_action(dummy_state)
    print(f"State shape: {dummy_state.shape}")
    print(f"Action: {action}")
    print(f"  Forward velocity: {action[0]:.2f} m/s")
    print(f"  Lateral velocity: {action[1]:.2f} m/s")
    print(f"  Vertical velocity: {action[2]:.2f} m/s")
    print(f"  Yaw rate: {action[3]:.2f} rad/s")
    
    print("\nAgent initialized successfully!")
