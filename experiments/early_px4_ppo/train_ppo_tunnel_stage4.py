import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from px4_tunnel_env_stage4 import PX4TunnelEnv

# Create environment
env = PX4TunnelEnv()

# Checkpoints
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./logs_stage4/",
                                         name_prefix="ppo_tunnel")

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=50000, callback=checkpoint_callback)

# Save final model
model.save("ppo_tunnel_stage4")

