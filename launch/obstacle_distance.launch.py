import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def detection_node(context):
    overrides = {}
    topic = LaunchConfiguration('image_topic').perform(context)
    viewer = LaunchConfiguration('viewer').perform(context)
    if topic:
        overrides['image_topic'] = topic
    if viewer:
        if viewer.lower() not in ('true', 'false'):
            raise ValueError('viewer must be true or false')
        overrides['viewer'] = viewer.lower() == 'true'
    weights = LaunchConfiguration('weights').perform(context) or os.environ.get('ROBOT_VISION_WEIGHTS', '')
    if weights:
        weights = str(Path(os.path.expandvars(weights)).expanduser().resolve())
    else:
        data_home = Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share')))
        candidates = [Path(get_package_share_directory('robot_vision')) / 'models/weights.pt',
                      data_home / 'robot_vision/models/weights.pt',
                      Path.home() / 'Downloads/weights.pt']
        weights = str(next((p for p in candidates if p.is_file()), candidates[0]))
    if not Path(weights).is_file():
        raise RuntimeError('YOLO weights not found: ' + weights +
                           '. Copy weights.pt to this computer and pass weights:=/absolute/path/weights.pt')
    yolo_overrides = {'weights': weights,
                      'device': LaunchConfiguration('yolo_device').perform(context)}
    if topic:
        yolo_overrides['image_topic'] = topic
    return [Node(package='robot_vision', executable='run_yolo_lane.sh',
                 name='yolo_lane', output='screen',
                 parameters=[LaunchConfiguration('config_file'), yolo_overrides]),
            Node(package='robot_vision', executable='obstacle_distance_node',
                 name='obstacle_distance', output='screen',
                 parameters=[LaunchConfiguration('config_file'), overrides])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=PathJoinSubstitution([
            FindPackageShare('robot_vision'), 'config', 'obstacle_distance.yaml'])),
        DeclareLaunchArgument('image_topic', default_value='', description='Empty uses YAML'),
        DeclareLaunchArgument('viewer', default_value='', description='Empty uses YAML'),
        DeclareLaunchArgument('weights', default_value='',
                              description='Checkpoint path; empty searches this user home or ROBOT_VISION_WEIGHTS'),
        DeclareLaunchArgument('yolo_device', default_value='cpu', description='cpu or GPU index, e.g. 0'),
        DeclareLaunchArgument('device', default_value='auto',
                              description='Camera device path; auto uses Insta360 discovery'),
        DeclareLaunchArgument('start_camera', default_value='true',
                              description='Launch Insta360 Link together with distance detection'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('insta360_usb_cam'), 'launch', 'usb_cam.launch.py'])),
            condition=IfCondition(LaunchConfiguration('start_camera')),
            launch_arguments={'device': LaunchConfiguration('device')}.items()),
        OpaqueFunction(function=detection_node),
    ])
