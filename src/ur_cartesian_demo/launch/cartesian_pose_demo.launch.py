"""Launch the UR10e cartesian pose demo with the common parameters exposed as args."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'planning_group', default_value='ur_manipulator',
            description='MoveIt planning group used for IK.'),
        DeclareLaunchArgument(
            'reference_frame', default_value='base_link',
            description='Frame the target poses are expressed in.'),
        DeclareLaunchArgument(
            'tip_frame', default_value='tool0',
            description='Controlled TCP / IK tip link.'),
        DeclareLaunchArgument(
            'controller_action',
            default_value='/scaled_joint_trajectory_controller/follow_joint_trajectory',
            description='FollowJointTrajectory action of the active joint controller.'),
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
            'avoid_collisions', default_value='true',
            description='Ask MoveIt IK to reject self-colliding solutions.'),
        DeclareLaunchArgument(
            'rotate_in_tool_frame', default_value='true',
            description='Apply roll/pitch/yaw about the tool axes (true) or base axes (false).'),
        DeclareLaunchArgument(
            'confirm_each_move', default_value='true',
            description='Prompt on the console before each move (safety on real HW).'),
    ]

    param_names = [
        'planning_group', 'reference_frame', 'tip_frame', 'controller_action',
        'linear_step_m', 'angular_step_deg', 'move_duration_s', 'avoid_collisions',
        'rotate_in_tool_frame', 'confirm_each_move',
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
