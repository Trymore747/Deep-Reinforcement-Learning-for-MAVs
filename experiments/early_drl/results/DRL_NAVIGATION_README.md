# DRL High-Speed Tunnel Navigation System

This package implements **Deep Reinforcement Learning (DRL)** based high-speed autonomous navigation for UAVs in dynamic tunnel environments. The system enables drones to navigate at speeds between **5-10 m/s** while avoiding obstacles and detecting artifacts.

## Features

- 🚀 **High-Speed Navigation**: 5-10 m/s autonomous flight in confined spaces
- 🧠 **Deep Reinforcement Learning**: PPO-based policy for real-time decision making
- 📊 **Comprehensive Data Recording**: Records position, velocity, acceleration, orientation, and sensor data
- 🎯 **Artifact Detection**: Stops navigation when target objects are detected
- 🔄 **Dynamic Environment Handling**: Adapts to moving obstacles and changing conditions
- 📈 **Training Framework**: OpenAI Gym environment for offline training

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DRL Navigation System                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Sensors    │───▶│  DRL Agent   │───▶│  Controller  │  │
│  │ LiDAR/Camera │    │ (PPO Policy) │    │   Commands   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │         │
│         │                    ▼                    │         │
│         │            ┌──────────────┐             │         │
│         └───────────▶│ Data Recorder│◀────────────┘         │
│                      └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
autonomous_flight/
├── scripts/
│   ├── drl_agent.py              # DRL agent implementation (PPO)
│   ├── drl_navigation_node.py    # ROS node for DRL inference
│   ├── data_recorder.py          # Data recording module
│   ├── tunnel_gym_env.py         # Gym environment for training
│   └── train_drl.py              # Training script
├── cfg/dynamic_navigation/
│   ├── drl_param.yaml            # DRL configuration
│   ├── drl_flight_base.yaml      # Flight parameters
│   └── data_recorder_param.yaml  # Recording configuration
├── launch/
│   ├── drl_dynamic_navigation.launch  # DRL navigation launch file
│   └── dynamic_navigation.launch      # Traditional navigation (fallback)
├── models/
│   └── drl_tunnel_nav.pth        # Trained model weights (to be added)
└── requirements.txt               # Python dependencies
```

## Installation

### 1. Install Python Dependencies

```bash
cd ~/catkin_ws/src/CERLAB-UAV-Autonomy/autonomous_flight
pip install -r requirements.txt
```

### 2. Make Scripts Executable

```bash
chmod +x scripts/drl_agent.py
chmod +x scripts/drl_navigation_node.py
chmod +x scripts/data_recorder.py
chmod +x scripts/tunnel_gym_env.py
chmod +x scripts/train_drl.py
```

### 3. Build the Workspace

```bash
cd ~/catkin_ws
catkin build autonomous_flight
source devel/setup.bash
```

## Usage

### Training the DRL Agent

#### Option 1: Train with Simulated Dynamics (No ROS required)

```bash
cd ~/catkin_ws/src/CERLAB-UAV-Autonomy/autonomous_flight/scripts
python3 train_drl.py --algorithm ppo --total_timesteps 1000000 --num_envs 4
```

#### Option 2: Train with Gazebo Simulation

```bash
# Terminal 1: Launch Gazebo simulation
roslaunch autonomous_flight tunnel_simulation.launch

# Terminal 2: Train agent
python3 train_drl.py --algorithm ppo --use_ros --total_timesteps 1000000
```

#### Monitor Training Progress

```bash
tensorboard --logdir ./logs
# Open browser to http://localhost:6006
```

### Running High-Speed Navigation

#### 1. Launch the DRL Navigation System

```bash
roslaunch autonomous_flight drl_dynamic_navigation.launch use_drl:=true use_data_recorder:=true
```

#### 2. Set the Goal Position

```bash
rostopic pub /drl_navigation/goal geometry_msgs/PoseStamped "header:
  frame_id: 'world'
pose:
  position: {x: 100.0, y: 0.0, z: 1.5}
  orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}"
```

#### 3. Start Navigation

```bash
rostopic pub /drl_navigation/start std_msgs/Bool "data: true"
```

#### 4. Monitor Status

```bash
# View current speed
rostopic echo /drl_navigation/current_speed

# View navigation status
rostopic echo /drl_navigation/status

# View recorded data location
roscd autonomous_flight
ls ~/tunnel_navigation_data/episodes/
```

## Configuration

### Speed Parameters

Edit `cfg/dynamic_navigation/drl_param.yaml`:

```yaml
min_speed: 5.0   # m/s
max_speed: 10.0  # m/s
max_lateral_speed: 3.0  # m/s
max_vertical_speed: 2.0  # m/s
```

### Safety Parameters

```yaml
min_obstacle_distance: 1.5  # meters - slow down threshold
emergency_stop_distance: 0.8  # meters - emergency brake
```

### Artifact Detection

```yaml
use_artifact_detection: true
artifact_types: ['cube', 'cylinder', 'rope', 'backpack', 'phone', 'helmet']
```

## Data Recording

Data is automatically recorded during navigation and saved to:
- **Episodes**: `~/tunnel_navigation_data/episodes/`
- **Summaries**: `~/tunnel_navigation_data/summaries/`

### Recorded Parameters

- Position (x, y, z)
- Velocity (vx, vy, vz)
- Acceleration (ax, ay, az)
- Orientation (quaternion and Euler angles)
- Angular velocity
- LiDAR minimum distance
- Battery status
- Speed
- Artifact detection events

### Analyzing Data

```bash
cd ~/tunnel_navigation_data/episodes
python3 -c "
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('episode_0001_<timestamp>.csv')

# Plot trajectory
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(df['pos_x'], df['pos_y'])
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Trajectory (Top View)')

plt.subplot(132)
plt.plot(df['elapsed_time'], df['speed'])
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Speed Profile')

plt.subplot(133)
plt.plot(df['elapsed_time'], df['lidar_min_distance'])
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Minimum Obstacle Distance')

plt.tight_layout()
plt.savefig('navigation_analysis.png')
plt.show()
"
```

## State and Action Space

### State Space (21 dimensions)
- Position: [x, y, z] (3)
- Velocity: [vx, vy, vz] (3)
- Orientation: [qx, qy, qz, qw] (4)
- LiDAR ranges: 8 directional measurements (8)
- Goal direction: normalized [dx, dy, dz] (3)

### Action Space (4 dimensions)
- Forward velocity: 5-10 m/s
- Lateral velocity: -3 to 3 m/s
- Vertical velocity: -2 to 2 m/s
- Yaw rate: -π/2 to π/2 rad/s

## Termination Conditions

Navigation stops when:
1. ✅ Goal position reached (within 1.0m threshold)
2. 🎯 Target artifact detected
3. ⚠️ Emergency: obstacle too close (< 0.8m)
4. ⚠️ Tunnel boundary violation

## Troubleshooting

### Model Not Found
```bash
# Download or train a model first
python3 scripts/train_drl.py --total_timesteps 100000
```

### CUDA Out of Memory
```yaml
# Edit drl_param.yaml
device: "cpu"  # Use CPU instead of CUDA
```

### Data Not Recording
```bash
# Check recorder status
rostopic echo /data_recorder/status

# Manually start recording
rostopic pub /data_recorder/start std_msgs/Bool "data: true"
```

## Performance Tips

1. **Use GPU for training**: Set `device: "cuda"` for 10x faster training
2. **Parallel environments**: Use `--num_envs 8` for faster convergence
3. **Pretrained models**: Fine-tune existing models instead of training from scratch
4. **Adjust reward function**: Modify `tunnel_gym_env.py` to optimize for specific behaviors

## Safety Notes

⚠️ **Important**: Always test in simulation before deploying on real hardware!

- Start with lower speeds (5 m/s) and gradually increase
- Ensure adequate sensor coverage (LiDAR/depth cameras)
- Set conservative safety margins for real-world deployment
- Monitor battery levels during long missions
- Have emergency stop procedures in place

## Citation

If you use this work, please cite:

```bibtex
@misc{drl_tunnel_navigation,
  title={Deep Reinforcement Learning for High-Speed UAV Tunnel Navigation},
  author={CERLAB UAV Autonomy Team},
  year={2025}
}
```

## License

See main repository for license information.

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review training logs in TensorBoard
