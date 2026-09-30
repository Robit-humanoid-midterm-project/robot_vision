"""Calibrate the actual published image using a measured checkerboard."""
import math
import re

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def calibration_actions(context):
    size = LaunchConfiguration('board_size').perform(context)
    square = LaunchConfiguration('square_size').perform(context)
    if not re.fullmatch(r'[1-9][0-9]*x[1-9][0-9]*', size):
        raise ValueError('board_size must be internal corners, e.g. 8x6')
    if min(map(int, size.split('x'))) < 3:
        raise ValueError('At least 3 internal corners per direction are required')
    try:
        meters = float(square)
    except ValueError as exc:
        raise ValueError('square_size must be one measured square side in meters') from exc
    if not math.isfinite(meters) or meters <= 0:
        raise ValueError('square_size must be a finite positive length in meters')
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('insta360_usb_cam'), 'launch', 'usb_cam.launch.py'
            ])),
            condition=IfCondition(LaunchConfiguration('start_camera')),
        ),
        Node(
            package='camera_calibration', executable='cameracalibrator',
            name='cameracalibrator', output='screen',
            arguments=['--size', size, '--square', square,
                       '--camera_name', 'insta360', '--no-service-check'],
            remappings=[('image', LaunchConfiguration('image_topic'))],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('board_size', description='Measured internal corners: columns x rows, e.g. 8x6'),
        DeclareLaunchArgument('square_size', description='Measured square side in meters, e.g. 0.025 for 25 mm'),
        DeclareLaunchArgument('image_topic', default_value='/camera1/camera/compressed_image',
                              description='sensor_msgs/Image topic; current driver uses this name'),
        DeclareLaunchArgument('start_camera', default_value='true',
                              description='Set false if the camera node is already running'),
        OpaqueFunction(function=calibration_actions),
    ])
