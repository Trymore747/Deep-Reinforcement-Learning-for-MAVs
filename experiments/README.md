# Experiments — Early Research Iterations

This folder contains early-stage experiments and prototype code that preceded the final R54 TD3 policy. Kept for research lineage and reproducibility.

## early_px4_ppo/
Prototype PPO-based tunnel navigation environments developed during the initial PX4 + ROS 2 integration phase. These were written before switching to TD3 and the CERLAB simulator.

Key files:
- `px4_offboard_env.py` — first PX4 offboard Gym environment
- `px4_tunnel_env_stage4.py` — 4-stage curriculum version
- `train_ppo_tunnel.py` — PPO training entry point
- `px4_tunnel_gym_env.py` — final Gym wrapper

## early_drl/
First-generation DRL navigation scripts using the CERLAB simulator, written before the v5/v6 TD3 architecture. Includes world files and comparative study notes.

Key files:
- `drl_agent.py` — early TD3/PPO agent
- `drl_navigation_node.py` — ROS 2 navigation node
- `compare_planners.py` — A* vs DRL benchmark
- `worlds/` — 10 early tunnel world files

## early_results_plots/
Comparative results plots from early training runs (R1–R53 progression).
