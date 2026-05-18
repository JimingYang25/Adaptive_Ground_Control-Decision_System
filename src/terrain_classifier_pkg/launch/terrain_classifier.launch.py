#launch for classifier_node --Maintainer: Jiming Yang

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    pkg_share = get_package_share_directory('terrain_classifier_pkg')
    
    
    declare_config_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_share, 'config', 'terrain_params.yaml'),
        description='Path to the parameters YAML file'
    )
    
    declare_window_size = DeclareLaunchArgument(
        'window_size',
        default_value='30',
        description='Sliding window size (K frames)'
    )
    
    declare_publish_freq = DeclareLaunchArgument(
        'publish_freq_hz',
        default_value='10.0',
        description='Publishing frequency (Hz)'
    )
    
    declare_imu_topic = DeclareLaunchArgument(
        'imu_topic',
        default_value='/imu/data_raw',
        description='IMU data topic name'
    )
    
    
    terrain_node = Node(
        package='terrain_classifier_pkg',
        executable='terrain_node.py',
        name='terrain_classifier',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),   
            {
                'window_size': LaunchConfiguration('window_size'),
                'publish_freq_hz': LaunchConfiguration('publish_freq_hz'),
                'imu_topic': LaunchConfiguration('imu_topic'),
            }
        ],
        emulate_tty=True,   
    )
    
    return LaunchDescription([
        declare_config_file,
        declare_window_size,
        declare_publish_freq,
        declare_imu_topic,
        terrain_node,
    ])

#launch for classifier_node --Maintainer: Jiming Yang
