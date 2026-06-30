"""Multi-camera RealSense + ArUco fiducial pose demo.

Reads a config file listing cameras and the ArUco settings, then for each camera starts:
  * a RealSense node (via realsense2_camera/rs_launch.py), namespaced by the camera name, and
  * an aruco_pose_node subscribed to that camera's color stream.

Each camera's markers are published as tf frames '<camera>_marker_<id>' and a PoseArray on
'/<camera>/camera/aruco_poses', so multiple cameras don't collide.

Args:
  config_file      path to the cameras/aruco yaml (default: this package's config/cameras.yaml)
  launch_cameras   'true' to start the RealSense nodes; 'false' to only start the detectors
                   (use when the cameras are already running elsewhere)
"""

import os

import yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    config_file = LaunchConfiguration('config_file').perform(context)
    launch_cameras = LaunchConfiguration('launch_cameras').perform(context).lower() == 'true'

    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    cameras = cfg.get('cameras', []) or []
    aruco = cfg.get('aruco', {}) or {}
    dictionary = aruco.get('dictionary', 'DICT_4X4_50')
    marker_size = float(aruco.get('marker_size_m', 0.05))

    rs_launch = os.path.join(
        get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')

    actions = []
    for cam in cameras:
        name = cam['name']
        serial = str(cam.get('serial_no', '') or '')

        if launch_cameras:
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(rs_launch),
                launch_arguments={
                    'camera_namespace': name,
                    'camera_name': 'camera',
                    'serial_no': serial,
                    'enable_depth': 'false',
                    'enable_color': 'true',
                    'pointcloud.enable': 'false',
                }.items()))

        # Detector runs in the camera's namespace, so 'color/image_raw' etc. resolve to
        # /<name>/camera/color/...; tf frames are global, hence the per-camera prefix.
        actions.append(Node(
            package='ur_vision_demo',
            executable='aruco_pose_node',
            namespace=f'{name}/camera',
            name='aruco_pose_node',
            output='screen',
            parameters=[{
                'image_topic': 'color/image_raw',
                'camera_info_topic': 'color/camera_info',
                'aruco_dictionary': dictionary,
                'marker_size_m': marker_size,
                'marker_frame_prefix': f'{name}_marker_',
                'publish_debug_image': True,
            }]))

    return actions


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare('ur_vision_demo'), 'config', 'cameras.yaml'])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='YAML listing cameras and ArUco settings.'),
        DeclareLaunchArgument(
            'launch_cameras', default_value='true',
            description="Start RealSense nodes ('false' = detectors only)."),
        OpaqueFunction(function=launch_setup),
    ])
