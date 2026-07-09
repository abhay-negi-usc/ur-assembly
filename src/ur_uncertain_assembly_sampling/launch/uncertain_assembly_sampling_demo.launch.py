"""Launch the uncertain assembly sampling node.

Launches ONLY the sampling node. It assumes these are already running:
  * the arm driver / bringup with scaled_joint_trajectory_controller ACTIVE + a live wrist F/T,
  * move_group (for /compute_ik),
  * (control_mode: admittance -- the default here) the ros2_control admittance_controller LOADED
    (inactive) on the controller_manager (see ur_admittance_demo).

The arm-only bringup in ur_kinematic_assembly_demo works well:
  ros2 launch ur_kinematic_assembly_demo arm_bringup.launch.py load_admittance:=true

NOTE: with confirm_each_step the node reads stdin -- use `ros2 run` so the trial prompts render.
Keep the e-stop in hand: this demo repeatedly drives a perturbed part into contact.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare('ur_uncertain_assembly_sampling'),
         'config', 'uncertain_assembly_sampling.yaml'])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Uncertain assembly sampling config yaml.'),
        Node(
            package='ur_uncertain_assembly_sampling',
            executable='uncertain_assembly_sampling',
            name='uncertain_assembly_sampling',
            output='screen',
            emulate_tty=True,
            parameters=[{'config_file': LaunchConfiguration('config_file')}],
        ),
    ])
