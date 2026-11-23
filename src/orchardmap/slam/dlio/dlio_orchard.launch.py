"""DLIO on the orchard rig: IMU-fix relay + DLIO odom and map nodes.

    ros2 launch dlio/dlio_orchard.launch.py [params_override:=<yaml>] [imu_fix:=true]

params_override is an optional second ROS params file layered on top of
dlio_orchard.yaml (tools/dlio_test_run.sh writes one per run).
Outputs: /dlio/odom_node/odom, /dlio/odom_node/path, /dlio/map_node/map,
TF odom -> base_link (base_link == velodyne).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.realpath(__file__))
IMU_FIX = os.path.join(HERE, '..', 'liosam', 'imu_fix_quat_node.py')


def nodes(context):
    params = [os.path.join(HERE, 'dlio_orchard.yaml')]
    override = LaunchConfiguration('params_override').perform(context)
    if override:
        params.append(override)
    remap = [
        ('pointcloud', LaunchConfiguration('pointcloud_topic')),
        ('imu', LaunchConfiguration('imu_topic')),
        ('odom', 'dlio/odom_node/odom'),
        ('pose', 'dlio/odom_node/pose'),
        ('path', 'dlio/odom_node/path'),
        ('kf_pose', 'dlio/odom_node/keyframes'),
        ('kf_cloud', 'dlio/odom_node/pointcloud/keyframe'),
        ('deskewed', 'dlio/odom_node/pointcloud/deskewed'),
    ]
    return [
        Node(package='direct_lidar_inertial_odometry', executable='dlio_odom_node',
             name='dlio_odom_node', output='screen', parameters=params, remappings=remap),
        Node(package='direct_lidar_inertial_odometry', executable='dlio_map_node',
             name='dlio_map_node', output='screen', parameters=params,
             remappings=[('keyframes', 'dlio/odom_node/pointcloud/keyframe')]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('pointcloud_topic', default_value='/velodyne_points'),
        DeclareLaunchArgument('imu_topic', default_value='/camera/camera/imu_fixed'),
        DeclareLaunchArgument('params_override', default_value=''),
        DeclareLaunchArgument('imu_fix', default_value='true',
                              description='Run liosam/imu_fix_quat_node.py (raw IMU -> velodyne axes)'),
        DeclareLaunchArgument('imu_fix_args', default_value='',
                              description='Extra "-p name:=value" args for the IMU fix node'),
        ExecuteProcess(
            cmd=['python3', IMU_FIX, '--ros-args', '-p', 'use_sim_time:=true',
                 LaunchConfiguration('imu_fix_args')],
            shell=True, output='screen', condition=IfCondition(LaunchConfiguration('imu_fix'))),
        OpaqueFunction(function=nodes),
    ])
