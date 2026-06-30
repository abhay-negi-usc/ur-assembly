"""Launch the Robotiq gripper demo node (gripper only, no arm)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'action_name', default_value='/robotiq_gripper_controller/gripper_cmd',
            description='GripperCommand action server of the running gripper controller.'),
        DeclareLaunchArgument(
            'open_position', default_value='0.0',
            description='GripperCommand position for fully open.'),
        DeclareLaunchArgument(
            'closed_position', default_value='0.8',
            description='GripperCommand position for fully closed.'),
        DeclareLaunchArgument(
            'max_effort', default_value='50.0',
            description='Grip force/effort sent with each goal.'),
        DeclareLaunchArgument(
            'dwell_s', default_value='1.5',
            description='Pause between positions (seconds).'),
        DeclareLaunchArgument(
            'cycles', default_value='1',
            description='Number of open->closed->open cycles.'),
        DeclareLaunchArgument(
            'activate_first', default_value='false',
            description='Call the reactivate service before cycling.'),
    ]

    param_names = [
        'action_name', 'open_position', 'closed_position', 'max_effort',
        'dwell_s', 'cycles', 'activate_first',
    ]

    demo_node = Node(
        package='ur_gripper_demo',
        executable='gripper_demo',
        name='gripper_demo',
        output='screen',
        emulate_tty=True,
        parameters=[{name: LaunchConfiguration(name) for name in param_names}],
    )

    return LaunchDescription(args + [demo_node])
