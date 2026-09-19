# Navigation Algorithm Comparison: RRT vs A* vs PPO

This directory contains implementations of three different navigation algorithms for high-speed autonomous flight in dynamic environments.

## Algorithms Implemented

### 1. RRT (Rapidly-exploring Random Tree) - **Current System**
- **File**: Uses existing autonomous_flight node
- **Launch**: `roslaunch autonomous_flight dynamic_navigation.launch`
- **Description**: Sampling-based path planning with B-spline trajectory optimization
- **Pros**: Fast replanning, good for dynamic obstacles
- **Cons**: Suboptimal paths, can be jerky

### 2. A* (A-star Path Planning)
- **File**: `scripts/astar_planner.py`
- **Launch**: `roslaunch autonomous_flight astar_navigation.launch`
- **Description**: Grid-based optimal path search with 3D occupancy mapping
- **Pros**: Optimal paths, predictable behavior
- **Cons**: Computational cost, fixed grid resolution

### 3. PPO (Proximal Policy Optimization)
- **File**: `scripts/ppo_planner.py`
- **Launch**: `roslaunch autonomous_flight ppo_navigation.launch`
- **Description**: Deep reinforcement learning agent with continuous action space
- **Pros**: Smooth control, learns from experience
- **Cons**: Requires training, needs trained model

## Quick Start

### Test Individual Algorithms

```bash
# Test RRT (current system)
cd /home/trymore/catkin_ws/src/CERLAB-UAV-Autonomy/autonomous_flight
bash scripts/test_rrt.sh

# Test A*
bash scripts/test_astar.sh

# Test PPO
bash scripts/test_ppo.sh
```

### Run Full Comparison

```bash
cd /home/trymore/catkin_ws/src/CERLAB-UAV-Autonomy/autonomous_flight
bash scripts/run_comparison.sh
```

This will:
1. Test all three algorithms sequentially
2. Collect performance metrics
3. Generate a comparison report
4. Save results to `/tmp/navigation_comparison_TIMESTAMP/`

## Manual Testing

### 1. RRT Navigation (Current)
```bash
# Terminal 1: Simulation
roslaunch uav_simulator tunnel_straight_dynamic_5.launch

# Terminal 2: Navigation
roslaunch autonomous_flight dynamic_navigation.launch

# Terminal 3: Send goals
python3 scripts/direct_nav.py
```

### 2. A* Navigation
```bash
# Terminal 1: Launch A* system
roslaunch autonomous_flight astar_navigation.launch

# Terminal 2: Send goals
python3 scripts/direct_nav.py

# Or send single goal:
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped '{header: {frame_id: "map"}, pose: {position: {x: 60.0, y: 0.0, z: 1.0}, orientation: {w: 1.0}}}'
```

### 3. PPO Navigation
```bash
# Terminal 1: Launch PPO system
roslaunch autonomous_flight ppo_navigation.launch

# Terminal 2: Send goals
python3 scripts/direct_nav.py
```

**Note**: PPO requires a trained model. See Training section below.

## Performance Benchmarking

### Collect Metrics for One Algorithm
```bash
# Start navigation system first, then:
python3 scripts/compare_planners.py RRT    # or ASTAR or PPO

# Send goals and let it complete
# Press Ctrl+C to see benchmark report
```

### Metrics Collected
- **Completion Time**: Time to reach goal
- **Path Length**: Total distance traveled
- **Path Efficiency**: Straight-line distance / actual path length
- **Average Speed**: Mean velocity during mission
- **Smoothness**: Standard deviation of accelerations
- **Energy**: Sum of acceleration magnitudes
- **Success Rate**: Whether goal was reached

## Training PPO (Optional)

The PPO planner can be trained on the tunnel environment:

```bash
# Create training environment wrapper (use existing tunnel_gym_env.py)
# Train PPO agent
python3 scripts/train_ppo.py

# Model will be saved to: models/ppo_tunnel_nav.pth
```

## Configuration

### A* Parameters (in `astar_navigation.launch`)
```xml
<param name="grid_resolution" value="0.5"/>      <!-- Grid cell size (m) -->
<param name="obstacle_clearance" value="0.8"/>   <!-- Static obstacle margin -->
<param name="dynamic_clearance" value="1.0"/>    <!-- Dynamic obstacle margin -->
<param name="max_speed" value="3.0"/>            <!-- Maximum velocity (m/s) -->
<param name="replan_rate" value="2.0"/>          <!-- Replanning frequency (Hz) -->
```

### PPO Parameters (in `ppo_navigation.launch`)
```xml
<param name="max_speed" value="3.0"/>            <!-- Maximum velocity (m/s) -->
<param name="obs_range" value="5.0"/>            <!-- Sensor range (m) -->
<param name="n_rays" value="16"/>                <!-- Number of depth sensors -->
<param name="model_path" value="...pth"/>        <!-- Path to trained model -->
```

### RRT Parameters (in `cfg/dynamic_navigation/`)
- `flight_base.yaml`: velocity, acceleration limits
- `planner_param.yaml`: RRT sampling, B-spline optimization
- `mapping_param.yaml`: occupancy grid, robot size

## Comparison Results

After running tests, you'll get a report like:

```
Algorithm  Success  Time(s)  Length(m)  Efficiency  Avg Speed
--------------------------------------------------------------
RRT        YES      22.45    65.32      0.92        2.91
ASTAR      YES      20.15    62.18      0.96        3.08
PPO        YES      21.87    63.45      0.94        2.90
```

## Expected Performance

### Test Environment
- **Tunnel**: 138m straight tunnel
- **Dynamic Obstacles**: 
  - 3 humans crossing at 15m/35m/45m (0.5-0.7 m/s)
  - 3 rocks moving at 20-25m/30m/40-43m (0.3-0.4 m/s)
- **Target**: Navigate to 60m
- **Speed**: 3.0 m/s maximum

### Algorithm Characteristics

**RRT** (Current):
- ✓ Fast replanning (adapts quickly to dynamic obstacles)
- ✓ Works without prior training
- ✗ Paths can be suboptimal and jerky
- Best for: Real-time dynamic environments

**A***:
- ✓ Optimal paths (shortest in discrete grid)
- ✓ Predictable, reproducible behavior
- ✗ Computational cost increases with grid resolution
- Best for: Known environments where optimality matters

**PPO**:
- ✓ Smooth, continuous control
- ✓ Can learn complex avoidance strategies
- ✗ Requires significant training time
- ✗ Performance depends on training quality
- Best for: Repeated similar tasks after proper training

## Troubleshooting

### A* Issues
- **"No path found"**: Increase `grid_resolution` or decrease `obstacle_clearance`
- **Slow planning**: Decrease map size or increase grid resolution
- **Crashes into obstacles**: Increase `obstacle_clearance` or `dynamic_clearance`

### PPO Issues
- **Random behavior**: Model not trained - use random policy or train first
- **Model not found**: Train using `train_ppo.py` or use existing model
- **Poor performance**: Requires more training episodes

### General Issues
- **Collision with rocks**: Increase `dynamic_clearance` in config (currently 0.8m)
- **Too slow**: Increase `max_speed` parameter
- **System crash**: Reduce speed or use more conservative parameters

## Files Created

```
autonomous_flight/
├── scripts/
│   ├── astar_planner.py          # A* implementation
│   ├── ppo_planner.py             # PPO implementation
│   ├── compare_planners.py        # Benchmarking tool
│   ├── test_rrt.sh                # Test RRT
│   ├── test_astar.sh              # Test A*
│   ├── test_ppo.sh                # Test PPO
│   └── run_comparison.sh          # Full comparison
├── launch/
│   ├── astar_navigation.launch    # A* launch file
│   └── ppo_navigation.launch      # PPO launch file
└── COMPARISON_README.md           # This file
```

## Next Steps

1. **Test RRT** (baseline): `bash scripts/test_rrt.sh`
2. **Test A***: `bash scripts/test_astar.sh`
3. **Compare results**: Check `/tmp/navigation_comparison_*/`
4. **Optional**: Train and test PPO for learning-based approach

## Contact & Credits

- RRT implementation: Original autonomous_flight package
- A* implementation: Custom grid-based planner
- PPO implementation: PyTorch-based deep RL agent
