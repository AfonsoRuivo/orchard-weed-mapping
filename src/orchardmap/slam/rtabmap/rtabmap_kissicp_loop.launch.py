from launch import LaunchDescription
import os
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    sim_time = {'use_sim_time': True}
    kiss_voxel     = LaunchConfiguration('kiss_voxel_size')
    kiss_min_range = LaunchConfiguration('kiss_min_range')
    kiss_max_range = LaunchConfiguration('kiss_max_range')
    kiss_deskew    = LaunchConfiguration('kiss_deskew')
    ground_z_min   = LaunchConfiguration('ground_z_min')

    rtabmap_params = {
        **sim_time,
        'frame_id':                    'velodyne',
        'odom_frame_id':               'odom',
        'map_frame_id':                'map',
        'subscribe_depth':             False,
        'subscribe_rgb':               False,
        'subscribe_scan_cloud':        True,
        'subscribe_odom_info':         False,
        'approx_sync':                 True,
        'tf_buffer_size':              60.0,
        'wait_for_transform':          1.0,
        'RGBD/LinearUpdate':           '0.1',
        'RGBD/AngularUpdate':          '0.09',
        'Rtabmap/LoopThr':             '0.11',
        'RGBD/LocalImmunizationRatio': '0.5',
        
        # --- LiDAR loop closure (proximity by space) ---
        # Was disabled to avoid orchard-symmetry false matches; re-enabled with
        # the robust optimizer (Vertigo) to reject bad links instead.
        'RGBD/ProximityBySpace':       'true',
        'RGBD/ProximityMaxGraphDepth': '0',     # allow links to any older node
        'RGBD/ProximityPathMaxNeighbors': '10', # one-to-many: match against local scan path
        'RGBD/ProximityAngle':         '180',   # 360 deg LiDAR: revisit in any heading
        'RGBD/LocalRadius':            '10',
        'Icp/MaxCorrespondenceDistance': '1.0',
        'Icp/Iterations':              '30',
        'Icp/CorrespondenceRatio':     '0.2',   # reject weak loop registrations
        
        'Rtabmap/LoopGPS':             'true',
        'qos_gps':                      1,
        'Optimizer/Strategy':          '2',
        'Optimizer/Robust':            'true',
        'Optimizer/GravitySigma':      '0.3',

        # --- REQUIRED FOR LIDAR LOOP CLOSURES ---
        'Reg/Strategy':                '1',    
        'Icp/VoxelSize':               '0.3',  
        'Icp/PointToPlane':            'true', 

        'Grid/FromDepth':              'false',
        'Grid/3D':                     'true',
        'Grid/CellSize':               '0.2',   # 0.05 m 3D grid of the whole orchard swapped this 8 GB machine
        'Grid/RangeMax':               '30.0',
        'Mem/STMSize':                 '30',
        'Rtabmap/MemoryThr':           '300',
        'cloud_output_voxel_size':     '0.2',
        'cloud_max_depth':             '30.0',
        'database_path':               '/tmp/rtabmap_kissicp_loop.db',
    }

    return LaunchDescription([
        # Off by default: the bridge adds load/latency during bag playback
        DeclareLaunchArgument('use_foxglove', default_value='false'),
        DeclareLaunchArgument('kiss_voxel_size', default_value='0.5'),
        DeclareLaunchArgument('kiss_min_range',  default_value='1.0'),
        DeclareLaunchArgument('kiss_max_range',  default_value='60.0'),
        # Tuned on session_20260706_140610(best) (2026-10-04): constant-velocity
        # deskew OFF (heavy vibration makes the velocity estimate noise) and ground
        # points removed before KISS-ICP -> 1.80 m ATE vs GPS on the full bag.
        DeclareLaunchArgument('kiss_deskew',     default_value='false'),
        DeclareLaunchArgument('ground_z_min',    default_value='-0.45'),

        # --- ground removal for KISS-ICP input (RTAB-Map still gets the full cloud) ---
        ExecuteProcess(
            cmd=['python3', os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'kiss_icp', 'cloud_filter_node.py'),
                 '--ros-args', '-p', ['z_min:=', ground_z_min], '-p', 'use_sim_time:=true'],
            output='screen',
        ),

        # --- static TFs (Extrinsics kept exactly as requested) ---
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_velodyne',
            parameters=[sim_time],
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'velodyne']
        ),

        # --- KISS-ICP odometry ---
        Node(
            package='kiss_icp',
            executable='kiss_icp_node',
            name='kiss_icp',
            output='screen',
            parameters=[{
                **sim_time,
                'base_frame':                           'velodyne',
                'lidar_odom_frame':                     'odom',
                'publish_odom_tf':                      True,
                'publish_debug_clouds':                 False,
                
                # --- PREVENT TURN BLINDNESS ---
                'data.max_range':                       kiss_max_range,
                'data.min_range':                       kiss_min_range,
                
                # --- PREVENT LIDAR WARPING DURING ROTATION ---
                'data.deskew':                          kiss_deskew,
                
                'mapping.voxel_size':                   kiss_voxel,
                'mapping.max_points_per_voxel':         20,
                'adaptive_threshold.initial_threshold': 0.5,
                'adaptive_threshold.min_motion_th':     0.1,
                'registration.max_num_iterations':      500,
                'registration.max_num_threads':         4,
            }],
            remappings=[('pointcloud_topic', '/velodyne_points_filtered')],
        ),

        # --- RTAB-Map SLAM ---
        # Uses KISS-ICP's odom directly. No EKF in the middle.
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap',
            output='screen',
            parameters=[rtabmap_params],
            remappings=[
                ('scan_cloud', '/velodyne_points'),
                ('odom',       '/kiss/odometry'),
                ('gps/fix',    '/gnss/fix'),
            ],
            arguments=['--delete_db_on_start']
        ),

        # --- Foxglove bridge (optional) ---
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            parameters=[{'port': 8765, 'address': '0.0.0.0'}],
            condition=IfCondition(LaunchConfiguration('use_foxglove'))
        ),

        # --- Map assembler ---
        Node(
            package='rtabmap_util',
            executable='map_assembler',
            name='map_assembler',
            output='screen',
            parameters=[{**sim_time, 'assembling': True, 'voxel_size': 0.2}],
            remappings=[('mapData', 'rtabmap/mapData')],
        ),
    ])
