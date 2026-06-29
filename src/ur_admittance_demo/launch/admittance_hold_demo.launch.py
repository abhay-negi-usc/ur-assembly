"""Launch the UR10e admittance compliant-hold demo node.

This launches only the demo node. The admittance_controller itself must already be loaded
and active on the UR controller_manager -- see the README for the spawn/switch commands.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'controller_name', default_value='admittance_controller',
            description='Name of the running admittance controller.'),
        DeclareLaunchArgument(
            'publish_rate_hz', default_value='20.0',
            description='Rate at which the hold reference is republished.'),
        DeclareLaunchArgument(
            'hold_duration_s', default_value='0.0',
            description='Seconds to hold; 0 = until Ctrl-C.'),
    ]

    param_names = ['controller_name', 'publish_rate_hz', 'hold_duration_s']

    demo_node = Node(
        package='ur_admittance_demo',
        executable='admittance_hold_demo',
        name='admittance_hold_demo',
        output='screen',
        emulate_tty=True,
        parameters=[{name: LaunchConfiguration(name) for name in param_names}],
    )

    return LaunchDescription(args + [demo_node])
