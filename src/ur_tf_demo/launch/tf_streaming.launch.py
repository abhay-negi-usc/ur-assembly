"""Publish the hand-eye static transforms and stream tool + marker poses in the base frame.

Reads hand_eye.yaml and, for each camera, starts a static_transform_publisher
(base_frame -> camera optical frame). Then starts the pose_streamer node, which looks up
base -> tool and base -> markers via tf2 and republishes them as PoseStamped.

Args:
  config_file        path to the hand-eye/frames yaml (default: this package's config)
  publish_static_tf  'true' to publish the hand-eye static transforms here; set 'false' if
                     you publish them elsewhere (e.g. a calibration package)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

import yaml


def launch_setup(context, *args, **kwargs):
    config_file = LaunchConfiguration('config_file').perform(context)
    publish_static = LaunchConfiguration('publish_static_tf').perform(context).lower() == 'true'

    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    base_frame = cfg.get('base_frame', 'base_link')
    tool_frame = cfg.get('tool_frame', 'tool0')
    rate = float(cfg.get('publish_rate_hz', 10.0))
    prefixes = cfg.get('marker_frame_prefixes', ['camera1_marker_'])
    cameras = cfg.get('cameras', []) or []

    actions = []

    if publish_static:
        for cam in cameras:
            cam_frame = cam['camera_frame']
            # Frame the camera is mounted to: tool frame for eye-in-hand (camera moves with the
            # arm), or base_frame for a fixed/world camera. Defaults to tool_frame.
            parent = cam.get('parent_frame', tool_frame)
            xyz = cam.get('xyz', [0.0, 0.0, 0.0])
            rpy = cam.get('rpy', [0.0, 0.0, 0.0])
            actions.append(Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name=f'handeye_{cam_frame}',
                arguments=[
                    '--x', str(xyz[0]), '--y', str(xyz[1]), '--z', str(xyz[2]),
                    '--roll', str(rpy[0]), '--pitch', str(rpy[1]), '--yaw', str(rpy[2]),
                    '--frame-id', parent, '--child-frame-id', cam_frame,
                ]))

    actions.append(Node(
        package='ur_tf_demo',
        executable='pose_streamer',
        name='pose_streamer',
        output='screen',
        parameters=[{
            'base_frame': base_frame,
            'tool_frame': tool_frame,
            'marker_frame_prefixes': prefixes,
            'publish_rate_hz': rate,
        }]))

    return actions


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare('ur_tf_demo'), 'config', 'hand_eye.yaml'])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Hand-eye / frames yaml.'),
        DeclareLaunchArgument(
            'publish_static_tf', default_value='true',
            description="Publish the hand-eye static transforms here ('false' to skip)."),
        OpaqueFunction(function=launch_setup),
    ])
