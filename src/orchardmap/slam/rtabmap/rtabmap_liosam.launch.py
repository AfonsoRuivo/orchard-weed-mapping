"""RTAB-Map graph SLAM on top of LIO-SAM odometry (LIO-SAM launched separately).

Used by tools/liosam_param_sweep.py (variant 'rtabmap_on_liosam'): the sweep starts
LIO-SAM, the IMU fix node and the static TFs as usual and then launches this file
with out_dir:=<run dir>. RTAB-Map consumes /lio_sam/mapping/odometry (the
scan-to-map optimised pose, not the IMU-rate TF) plus the raw Velodyne cloud.
Loop-closure settings match rtabmap_kissicp_loop.launch.py. RTAB-Map does not
publish TF, because LIO-SAM already owns map->odom->base_link.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
import os

HERE = os.path.dirname(os.path.realpath(__file__))


def generate_launch_description():
    out_dir = LaunchConfiguration('out_dir')
    sim_time = {'use_sim_time': True}
    rtabmap_params = {
        **sim_time,
        'frame_id':                    'base_link',
        'odom_frame_id':               '',          # use the odometry topic, not TF
        'map_frame_id':                'rtabmap_map',
        'publish_tf':                  False,
        'subscribe_depth':             False,
        'subscribe_rgb':               False,
        'subscribe_scan_cloud':        True,
        'approx_sync':                 True,
        'wait_for_transform':          1.0,
        'RGBD/LinearUpdate':           '0.1',
        'RGBD/AngularUpdate':          '0.09',
        'RGBD/ProximityBySpace':       'true',
        'RGBD/ProximityMaxGraphDepth': '0',
        'RGBD/ProximityPathMaxNeighbors': '10',
        'RGBD/ProximityAngle':         '180',
        'RGBD/LocalRadius':            '10',
        'Optimizer/Strategy':          '2',
        'Optimizer/Robust':            'true',
        'Reg/Strategy':                '1',
        'Icp/VoxelSize':               '0.3',
        'Icp/PointToPlane':            'true',
        'Icp/MaxCorrespondenceDistance': '1.0',
        'Icp/Iterations':              '30',
        'Icp/CorrespondenceRatio':     '0.2',
        'Grid/FromDepth':              'false',
        'Grid/3D':                     'true',
        'Grid/CellSize':               '0.2',
        'Grid/RangeMax':               '30.0',
        'Mem/STMSize':                 '30',
        'Rtabmap/MemoryThr':           '300',
        'cloud_output_voxel_size':     '0.2',
        'cloud_max_depth':             '30.0',
        'database_path':               PathJoinSubstitution([out_dir, 'rtabmap_liosam.db']),
    }
    return LaunchDescription([
        DeclareLaunchArgument('out_dir', default_value='/tmp'),
        Node(
            package='rtabmap_slam', executable='rtabmap', name='rtabmap', output='screen',
            parameters=[rtabmap_params],
            remappings=[('scan_cloud', '/velodyne_points'),
                        ('odom', '/lio_sam/mapping/odometry')],
            arguments=['--delete_db_on_start'],
        ),
        ExecuteProcess(
            cmd=['python3', os.path.join(HERE, '..', 'trajectories', 'recorders', 'trajectory_recorder.py'),
                 PathJoinSubstitution([out_dir, 'rtabmap_mappath.csv'])],
            output='screen',
        ),
    ])
