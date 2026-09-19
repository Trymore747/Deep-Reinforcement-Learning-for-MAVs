from px4_tunnel_env_stage4 import PX4TunnelEnvStage4
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

env = DummyVecEnv([lambda: PX4TunnelEnvStage4()])
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='ppo_tunnel')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000, callback=checkpoint_callback)

