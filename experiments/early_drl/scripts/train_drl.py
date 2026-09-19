#!/usr/bin/env python3
"""
Training script for DRL tunnel navigation agent
Trains the agent in simulation using the Gym environment
"""

import rospy
import argparse
import os
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import torch

from tunnel_gym_env import TunnelNavigationEnv


def make_env(rank, seed=0, use_ros=False):
    """Create a single environment"""
    def _init():
        env = TunnelNavigationEnv(use_ros=use_ros)
        env.seed(seed + rank)
        return Monitor(env)
    return _init


def train_ppo(args):
    """Train using PPO algorithm"""
    print("="*60)
    print("Training DRL Agent for High-Speed Tunnel Navigation")
    print("Algorithm: Proximal Policy Optimization (PPO)")
    print("="*60)
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create vectorized environment
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env(i, use_ros=args.use_ros) for i in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env(0, use_ros=args.use_ros)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(999, use_ros=args.use_ros)])
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.num_envs,
        save_path=args.model_dir,
        name_prefix='ppo_tunnel_nav',
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq // args.num_envs,
        n_eval_episodes=5,
        deterministic=True
    )
    
    # Load existing model or create new one
    if args.continue_training and os.path.exists(args.load_model):
        print(f"Loading existing model from {args.load_model}")
        model = PPO.load(args.load_model, env=env, tensorboard_log=args.log_dir)
    else:
        print("Creating new PPO model")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device
        )
    
    print(f"\nTraining for {args.total_timesteps} timesteps...")
    print(f"Using device: {args.device}")
    print(f"Number of parallel environments: {args.num_envs}")
    
    # Train
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="PPO"
    )
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, "ppo_tunnel_nav_final")
    model.save(final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    
    # Convert to PyTorch format for ROS deployment
    convert_to_pytorch(model, args.model_dir)
    
    return model


def train_sac(args):
    """Train using SAC algorithm (alternative)"""
    print("="*60)
    print("Training DRL Agent for High-Speed Tunnel Navigation")
    print("Algorithm: Soft Actor-Critic (SAC)")
    print("="*60)
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([make_env(0, use_ros=args.use_ros)])
    eval_env = DummyVecEnv([make_env(999, use_ros=args.use_ros)])
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.model_dir,
        name_prefix='sac_tunnel_nav'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=5
    )
    
    # Create or load model
    if args.continue_training and os.path.exists(args.load_model):
        print(f"Loading existing model from {args.load_model}")
        model = SAC.load(args.load_model, env=env, tensorboard_log=args.log_dir)
    else:
        print("Creating new SAC model")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=args.gamma,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device
        )
    
    print(f"\nTraining for {args.total_timesteps} timesteps...")
    
    # Train
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="SAC"
    )
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, "sac_tunnel_nav_final")
    model.save(final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    
    return model


def convert_to_pytorch(model, output_dir):
    """Convert trained model to PyTorch format for ROS deployment"""
    from drl_agent import DRLNavigationAgent
    
    # Create agent
    agent = DRLNavigationAgent(state_dim=21, action_dim=4, device='cpu')
    
    # Extract policy weights from Stable-Baselines3 model
    # This is a simplified conversion - you may need to adjust based on network architecture
    sb3_policy = model.policy
    
    # Copy weights (this is approximate and may need adjustment)
    try:
        # Map SB3 policy to our custom policy
        with torch.no_grad():
            # Feature extractor
            agent.policy.shared_fc1.weight.data = sb3_policy.mlp_extractor.policy_net[0].weight.data
            agent.policy.shared_fc1.bias.data = sb3_policy.mlp_extractor.policy_net[0].bias.data
            agent.policy.shared_fc2.weight.data = sb3_policy.mlp_extractor.policy_net[2].weight.data
            agent.policy.shared_fc2.bias.data = sb3_policy.mlp_extractor.policy_net[2].bias.data
            
            # Actor head
            agent.policy.actor_mean.weight.data = sb3_policy.action_net.weight.data
            agent.policy.actor_mean.bias.data = sb3_policy.action_net.bias.data
            
            # Critic head
            agent.policy.critic_value.weight.data = sb3_policy.value_net.weight.data
            agent.policy.critic_value.bias.data = sb3_policy.value_net.bias.data
        
        # Save in our custom format
        pytorch_model_path = os.path.join(output_dir, "drl_tunnel_nav.pth")
        agent.save_model(pytorch_model_path)
        print(f"Converted model saved to {pytorch_model_path}")
        
    except Exception as e:
        print(f"Warning: Could not convert model: {e}")
        print("You may need to manually convert the model or retrain using drl_agent.py")


def main():
    parser = argparse.ArgumentParser(description='Train DRL agent for tunnel navigation')
    
    # Algorithm
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'sac'],
                       help='RL algorithm to use')
    
    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                       help='Total number of training timesteps')
    parser.add_argument('--num_envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--n_steps', type=int, default=2048,
                       help='Number of steps per update (PPO)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='Number of epochs per update (PPO)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--clip_range', type=float, default=0.2,
                       help='PPO clip range')
    
    # Directories
    parser.add_argument('--model_dir', type=str, default='./models',
                       help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for tensorboard logs')
    
    # Checkpointing
    parser.add_argument('--save_freq', type=int, default=50000,
                       help='Save model every N steps')
    parser.add_argument('--eval_freq', type=int, default=25000,
                       help='Evaluate model every N steps')
    parser.add_argument('--continue_training', action='store_true',
                       help='Continue training from existing model')
    parser.add_argument('--load_model', type=str, default='./models/best_model',
                       help='Path to model to continue training')
    
    # Simulation
    parser.add_argument('--use_ros', action='store_true',
                       help='Use ROS simulation (Gazebo)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: cpu, cuda, or auto')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train
    if args.algorithm == 'ppo':
        model = train_ppo(args)
    elif args.algorithm == 'sac':
        model = train_sac(args)
    
    print("\nTo view training progress, run:")
    print(f"tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    main()
