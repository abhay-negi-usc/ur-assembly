"""Launch the eye-in-hand visual servoing node.

This launches ONLY the servo node. It assumes these are already running:
  * the arm driver / integrated bringup (scaled_joint_trajectory_controller active),
  * move_group (for /compute_ik),
  * the vision + hand-eye tf stack (ur_vision_demo + ur_tf_demo) so the marker resolves in base
    and tool0 -> camera is published.

NOTE: with confirm_start / confirm_each_move the node reads stdin, which only works under
`ros2 run` in an interactive terminal -- not `ros2 launch`. For a hands-off run, set both false
(here or in the yaml). Keep the e-stop in hand: this demo moves the arm autonomously.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare('ur_visual_servo_demo'), 'config', 'visual_servo.yaml'])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Visual servo config yaml.'),
        Node(
            package='ur_visual_servo_demo',
            executable='visual_servo',
            name='visual_servo',
            output='screen',
            emulate_tty=True,   # so confirm prompts render
            parameters=[{'config_file': LaunchConfiguration('config_file')}],
        ),
    ])
