# 01 — Environment Setup

This guide sets up the full software stack: ROS 2, Gazebo 11, MAVROS, PX4 SITL, and Python dependencies.

## Tested Configuration

| Component | Version |
|---|---|
| OS | Ubuntu 20.04 LTS |
| ROS 2 | Foxy |
| Gazebo | 11 (Classic) |
| Python | 3.8 / 3.10 |
| MAVROS | 2.x |
| PX4 | v1.12 |

---

## 1 — ROS 2 Foxy

```bash
sudo apt update && sudo apt install curl gnupg lsb-release -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-foxy-desktop python3-colcon-common-extensions -y
echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 2 — Gazebo 11

```bash
sudo apt install gazebo11 libgazebo11-dev -y
```

## 3 — MAVROS + MAVLink

```bash
sudo apt install ros-foxy-mavros ros-foxy-mavros-extras -y
wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
sudo bash install_geographiclib_datasets.sh
```

## 4 — PX4 Autopilot (for SITL / HITL)

```bash
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
bash ./Tools/setup/ubuntu.sh
make px4_sitl gazebo     # builds SITL — verify Gazebo launch works
```

For HITL with the RDDRONE-FMUK66, see [04_hitl_testing.md](04_hitl_testing.md).

## 5 — Build the UAV Simulator ROS 2 Package

```bash
cd ~/ros2_ws/src   # or wherever your workspace is
ln -s /path/to/Deep-Reinforcement-Learning-for-MAVs/uav_simulator .
cd ~/ros2_ws
colcon build --packages-select uav_simulator
source install/setup.bash
```

## 6 — Python Dependencies

```bash
cd /path/to/Deep-Reinforcement-Learning-for-MAVs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 7 — Environment Variables

Add to `~/.bashrc`:

```bash
source /opt/ros/foxy/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/path/to/Deep-Reinforcement-Learning-for-MAVs/uav_simulator/models
```

## 8 — Verify

```bash
ros2 launch uav_simulator start.launch  # should open Gazebo with the drone
```

If Gazebo opens and you see the quadrotor in an empty world, the setup is complete.

---

## Troubleshooting

**`libgazebo_ros_api_plugin.so` not found** — make sure `libgazebo11-dev` is installed and you sourced the ROS workspace.

**`rmw_cyclonedds_cpp` not found** — `sudo apt install ros-foxy-rmw-cyclonedds-cpp`

**MAVROS `fcu_url` connection refused** — check the USB device (`ls /dev/ttyACM*`) and set `fcu_url:=udp://:14540@127.0.0.1:14557` for SITL or `fcu_url:=serial:///dev/ttyACM0:57600` for HITL.
