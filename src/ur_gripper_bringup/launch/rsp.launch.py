"""robot_state_publisher for the combined UR10e + Robotiq 2F-85 description.

This is handed to ur_control.launch.py via its `description_launchfile` argument, so the UR
driver's controller_manager reads this (combined) robot_description and loads BOTH the arm and
the gripper hardware.

ur_control.launch.py forwards only a fixed set of args here (ur_type, robot_ip,
kinematics_parameters_file, use_mock_hardware, mock_sensor_commands, headless_mode). The
gripper-specific values (com_port, coupler, gripper_enabled) are NOT forwarded, so we read them
directly from config/coupler.yaml.
"""

import os

import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('ur_gripper_bringup')

    # Gripper/coupler config (not forwarded by ur_control.launch.py).
    with open(os.path.join(pkg, 'config', 'coupler.yaml'), 'r') as f:
        cfg = yaml.safe_load(f) or {}
    com_port = str(cfg.get('com_port', '/dev/ttyUSB1'))
    gripper_enabled = str(cfg.get('gripper_enabled', True)).lower()
    coupler = cfg.get('coupler', {}) or {}
    xyz = coupler.get('xyz', [0.0, 0.0, 0.0])
    rpy = coupler.get('rpy', [0.0, 0.0, 0.0])

    xacro_file = os.path.join(pkg, 'urdf', 'ur10e_with_2f85.urdf.xacro')

    robot_description = Command([
        FindExecutable(name='xacro'), ' ', xacro_file,
        ' ur_type:=', LaunchConfiguration('ur_type'),
        ' robot_ip:=', LaunchConfiguration('robot_ip'),
        ' kinematics_parameters_file:=', LaunchConfiguration('kinematics_parameters_file'),
        ' use_mock_hardware:=', LaunchConfiguration('use_mock_hardware'),
        ' mock_sensor_commands:=', LaunchConfiguration('mock_sensor_commands'),
        ' headless_mode:=', LaunchConfiguration('headless_mode'),
        ' gripper_enabled:=', gripper_enabled,
        ' com_port:=', com_port,
        ' coupler_x:=', str(xyz[0]), ' coupler_y:=', str(xyz[1]), ' coupler_z:=', str(xyz[2]),
        ' coupler_roll:=', str(rpy[0]), ' coupler_pitch:=', str(rpy[1]),
        ' coupler_yaw:=', str(rpy[2]),
    ])

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[{'robot_description': ParameterValue(robot_description, value_type=str)}],
    )
    return [rsp_node]


def generate_launch_description():
    # These are the args ur_control.launch.py forwards to the description_launchfile.
    args = [
        DeclareLaunchArgument('ur_type', default_value='ur10e'),
        DeclareLaunchArgument('robot_ip', default_value='0.0.0.0'),
        DeclareLaunchArgument(
            'kinematics_parameters_file',
            default_value=os.path.join(
                get_package_share_directory('ur_description'),
                'config', 'ur10e', 'default_kinematics.yaml')),
        DeclareLaunchArgument('use_mock_hardware', default_value='false'),
        DeclareLaunchArgument('mock_sensor_commands', default_value='false'),
        DeclareLaunchArgument('headless_mode', default_value='false'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
