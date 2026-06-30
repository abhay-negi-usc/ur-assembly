"""Launch the fiducial-guided pick-and-place node.

This launches ONLY the pick_place node. It assumes these are already running:
  * the integrated bringup (arm + gripper under one controller_manager) -- ur_gripper_bringup,
  * move_group (for /compute_ik) -- e.g. ur_moveit_config,
  * the vision + hand-eye tf stack -- ur_vision_demo + ur_tf_demo,
so that the object marker resolves in the base frame and the gripper action is available.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare('ur_pick_place_demo'), 'config', 'pick_place.yaml'])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Pick-and-place config yaml.'),
        Node(
            package='ur_pick_place_demo',
            executable='pick_place',
            name='pick_place',
            output='screen',
            emulate_tty=True,   # so confirm_each_step input() prompts render
            parameters=[{'config_file': LaunchConfiguration('config_file')}],
        ),
    ])
