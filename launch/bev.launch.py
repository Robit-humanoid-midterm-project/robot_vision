from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = str(Path(get_package_share_directory('robot_vision')) / 'config' / 'obstacle_distance.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=config),
        Node(package='robot_vision', executable='bev_node',
             name='camera_bev_preview', output='screen',
             arguments=['--config', LaunchConfiguration('config')]),
    ])
