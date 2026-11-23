from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    sim_time = {'use_sim_time': True}

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
        
        # --- PREVENT ORCHARD SYMMETRY TURN DRIFT ---
        'RGBD/ProximityBySpace':       'false', 
        
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
        'Grid/CellSize':               '0.05',
        'Grid/RangeMax':               '30.0',
        'Mem/STMSize':                 '30',
        'Rtabmap/MemoryThr':           '300',
        'cloud_output_voxel_size':     '0.05',
        'cloud_max_depth':             '30.0',
        'database_path':               '/tmp/rtabmap_kissicp_gps.db',
    }

    return LaunchDescription([
        # Off by default: the bridge adds load/latency during bag playback
        DeclareLaunchArgument('use_foxglove', default_value='false'),

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
                'data.max_range':                       60.0, 
                'data.min_range':                       1.0,
                
                # --- PREVENT LIDAR WARPING DURING ROTATION ---
                'data.deskew':                          True,
                
                'mapping.voxel_size':                   0.5,
                'mapping.max_points_per_voxel':         20,
                'adaptive_threshold.initial_threshold': 0.5,
                'adaptive_threshold.min_motion_th':     0.1,
                'registration.max_num_iterations':      500,
                'registration.max_num_threads':         2,
            }],
            remappings=[('pointcloud_topic', '/velodyne_points')],
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
            parameters=[{**sim_time, 'assembling': True, 'voxel_size': 0.05}],
            remappings=[('mapData', 'rtabmap/mapData')],
        ),
    ])
