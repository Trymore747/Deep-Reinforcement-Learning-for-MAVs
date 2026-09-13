#!/usr/bin/env python3
"""
HITL Launch File — R54 Hardware-in-the-Loop Testing
Starts Gazebo (real-world tunnel + quadcopter) + MAVROS → FMUK66.
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    SetEnvironmentVariable, TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix, get_package_share_directory

PROJECT  = Path(__file__).resolve().parents[2]
CFG_DIR  = Path(__file__).resolve().parent.parent / "config"

def generate_launch_description():
    # ── Paths ──────────────────────────────────────────────────────────────────
    pkg_prefix   = get_package_prefix("uav_simulator")
    share_dir    = os.path.join(pkg_prefix, "share", "uav_simulator")
    plugin_dir   = os.path.join(pkg_prefix, "lib")
    models_dir   = os.path.join(share_dir, "models")
    worlds_dir   = os.path.join(share_dir, "worlds", "tunnel")
    urdf_path    = os.path.join(share_dir, "urdf", "quadcopter.urdf")
    spawn_script = os.path.join(
        get_package_prefix("gazebo_ros"), "lib", "gazebo_ros", "spawn_entity.py")

    world_file   = os.path.join(worlds_dir, "tunnel_100m_realworld.world")

    plugin_path   = ":".join(filter(None, [plugin_dir, os.environ.get("GAZEBO_PLUGIN_PATH", "")]))
    model_path    = ":".join(filter(None, [models_dir, os.path.expanduser("~/.gazebo/models"), os.environ.get("GAZEBO_MODEL_PATH", "")]))
    resource_path = ":".join(filter(None, [
        str(PROJECT / "uav_simulator" / "worlds"),
        share_dir,
        "/usr/share/gazebo-11",
        "/usr/share/gazebo",
        os.environ.get("GAZEBO_RESOURCE_PATH", ""),
    ]))

    # ── Args ───────────────────────────────────────────────────────────────────
    declare_port  = DeclareLaunchArgument("fcu_port",  default_value="/dev/ttyACM0")
    declare_baud  = DeclareLaunchArgument("fcu_baud",  default_value="57600")
    declare_gui   = DeclareLaunchArgument("gui",       default_value="true")

    # ── Gazebo (reuse gazebo_ros launch for proper env setup) ──────────────────
    launch_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world":   world_file,
            "verbose": "true",
            "pause":   "false",
            "gui":     LaunchConfiguration("gui"),
        }.items(),
    )

    # ── Spawn quadcopter at tunnel entrance ────────────────────────────────────
    spawn_drone = TimerAction(
        period=7.0,
        actions=[ExecuteProcess(
            cmd=["/usr/bin/python3", spawn_script,
                 "-entity", "quadcopter",
                 "-file",   urdf_path,
                 "-x", "0", "-y", "0", "-z", "0.3"],
            output="screen",
        )],
    )

    # ── MAVROS → FMUK66 ────────────────────────────────────────────────────────
    mavros = TimerAction(
        period=8.0,
        actions=[Node(
            package="mavros",
            executable="mavros_node",
            name="mavros",
            output="screen",
            parameters=[
                str(CFG_DIR / "mavros_hitl.yaml"),
                {"fcu_url": ["serial://", LaunchConfiguration("fcu_port"),
                             ":", LaunchConfiguration("fcu_baud")]},
            ],
        )],
    )

    # ── Static TF ─────────────────────────────────────────────────────────────
    tf_nodes = TimerAction(
        period=9.0,
        actions=[
            Node(package="tf2_ros", executable="static_transform_publisher",
                 name="tf_map_odom",
                 arguments=["0","0","0","0","0","0","map","odom"]),
            Node(package="tf2_ros", executable="static_transform_publisher",
                 name="tf_odom_base",
                 arguments=["0","0","0","0","0","0","odom","base_link"]),
        ],
    )

    return LaunchDescription([
        declare_port, declare_baud, declare_gui,
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
        SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
        SetEnvironmentVariable("GAZEBO_PLUGIN_PATH",   plugin_path),
        SetEnvironmentVariable("GAZEBO_MODEL_PATH",    model_path),
        SetEnvironmentVariable("GAZEBO_RESOURCE_PATH", resource_path),
        launch_gazebo,
        spawn_drone,
        mavros,
        tf_nodes,
    ])
