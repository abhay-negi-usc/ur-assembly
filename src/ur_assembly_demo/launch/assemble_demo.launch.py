"""Launch the pick-and-assemble node.

Launches ONLY the assemble node. It assumes these are already running:
  * the integrated bringup (arm + gripper under one controller_manager) -- ur_gripper_bringup,
  * move_group (for /compute_ik) -- e.g. ur_moveit_config,
  * the vision + hand-eye tf stack -- ur_vision_demo + ur_tf_demo,
  * the ros2_control admittance_controller LOADED (inactive) on the controller_manager
    (see ur_admittance_demo for install + spawn), for the compliant mate.

NOTE: with confirm_each_step the node reads stdin -- use `ros2 run` in an interactive terminal so
the prompts render (input() does not work under `ros2 launch`). Keep the e-stop in hand: this demo
moves the arm autonomously and applies force during the mate.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare('ur_assembly_demo'), 'config', 'assemble.yaml'])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Pick-and-assemble config yaml.'),
        Node(
            package='ur_assembly_demo',
            executable='assemble',
            name='assemble',
            output='screen',
            emulate_tty=True,
            parameters=[{'config_file': LaunchConfiguration('config_file')}],
        ),
    ])
