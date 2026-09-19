from px4_tunnel_gym_env import PX4TunnelEnv
import numpy as np

env = PX4TunnelEnv()
obs, _ = env.reset()

for _ in range(50):
    action = np.array([2.0, 0.0, 0.0])  # slow forward
    obs, reward, done, _, _ = env.step(action)
    print("obs:", obs, "reward:", reward)
    if done:
        break

env.close()

