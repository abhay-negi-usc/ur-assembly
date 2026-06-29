"""Launch the UR10e cartesian pose demo with the common parameters exposed as args."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'controller_name', default_value='pose_based_cartesian_traj_controller',
            description='Cartesian trajectory controller to command.'),
        DeclareLaunchArgument(
            'base_frame', default_value='base',
            description='Reference frame the target poses are expressed in.'),
        DeclareLaunchArgument(
            'tip_frame', default_value='tool0',
            description='Controlled TCP frame.'),
        DeclareLaunchArgument(
            'linear_step_m', default_value='0.010',
            description='Linear step for X/Y/Z moves (meters).'),
        DeclareLaunchArgument(
            'angular_step_deg', default_value='10.0',
            description='Angular step for roll/pitch/yaw moves (degrees).'),
        DeclareLaunchArgument(
            'move_duration_s', default_value='4.0',
            description='Time per move (seconds). Lower for sim, keep conservative on HW.'),
        DeclareLaunchArgument(
            'confirm_each_move', default_value='true',
            description='Prompt on the console before each move (safety on real HW).'),
        DeclareLaunchArgument(
            'rotate_in_tool_frame', default_value='true',
            description='Apply roll/pitch/yaw about the tool axes (true) or base axes (false).'),
        DeclareLaunchArgument(
            'auto_switch_controllers', default_value='false',
            description='Deactivate scaled JTC and activate the cartesian controller on start.'),
    ]

    param_names = [
        'controller_name', 'base_frame', 'tip_frame', 'linear_step_m',
        'angular_step_deg', 'move_duration_s', 'confirm_each_move',
        'rotate_in_tool_frame', 'auto_switch_controllers',
    ]

    demo_node = Node(
        package='ur_cartesian_demo',
        executable='cartesian_pose_demo',
        name='cartesian_pose_demo',
        output='screen',
        emulate_tty=True,  # needed so input() prompts render when confirm_each_move:=true
        parameters=[{name: LaunchConfiguration(name) for name in param_names}],
    )

    return LaunchDescription(args + [demo_node])
