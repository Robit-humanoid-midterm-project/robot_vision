from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=PathJoinSubstitution([
            FindPackageShare('robot_vision'), 'config', 'obstacle_distance.yaml'])),
        DeclareLaunchArgument('image_topic', default_value='/camera1/camera/compressed_image'),
        DeclareLaunchArgument('viewer', default_value='true'),
        DeclareLaunchArgument('device', default_value='auto',
                              description='Camera device path; auto uses Insta360 discovery'),
        DeclareLaunchArgument('start_camera', default_value='true',
                              description='Launch Insta360 Link together with distance detection'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('insta360_usb_cam'), 'launch', 'usb_cam.launch.py'])),
            condition=IfCondition(LaunchConfiguration('start_camera')),
            launch_arguments={'device': LaunchConfiguration('device')}.items()),
        Node(package='robot_vision', executable='obstacle_distance_node',
             name='obstacle_distance', output='screen', parameters=[
                 LaunchConfiguration('config_file'),
                 {'image_topic': LaunchConfiguration('image_topic'),
                  'viewer': ParameterValue(LaunchConfiguration('viewer'), value_type=bool)}]),
    ])
