"""Launch the kinematic assembly node.

Launches ONLY the assembly node. It assumes these are already running:
  * the arm driver / bringup with scaled_joint_trajectory_controller ACTIVE,
  * move_group (for /compute_ik),
  * (admittance mode only) the ros2_control admittance_controller LOADED (inactive) on the
    controller_manager -- see ur_admittance_demo.

NOTE: with confirm_each_step the node reads stdin -- use `ros2 run` so the prompts render (input()
does not work under `ros2 launch`). Keep the e-stop in hand: this demo moves the arm autonomously
and (in admittance mode) applies force.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare('ur_kinematic_assembly_demo'), 'config', 'kinematic_assembly.yaml'])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Kinematic assembly config yaml.'),
        Node(
            package='ur_kinematic_assembly_demo',
            executable='kinematic_assembly',
            name='kinematic_assembly',
            output='screen',
            emulate_tty=True,
            parameters=[{'config_file': LaunchConfiguration('config_file')}],
        ),
    ])
