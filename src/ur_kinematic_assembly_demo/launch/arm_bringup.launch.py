"""One-shot ARM-ONLY bring-up for the kinematic assembly demo (no gripper, no camera).

Starts, in one command:
  * ur_robot_driver ur_control.launch.py -- the bare UR10e arm (scaled_joint_trajectory_controller
    + force_torque_sensor_broadcaster), with this robot's kinematic calibration,
  * (launch_moveit, default true) ur_moveit_config ur_moveit.launch.py -- move_group for /compute_ik,
  * (load_admittance, default false) the ros2_control admittance_controller, LOADED INACTIVE
    (needed only for control_mode: admittance).

You still start the External Control program on the pendant after this comes up. Then run the demo:
  ros2 run ur_kinematic_assembly_demo kinematic_assembly

Args:
  robot_ip (192.168.125.2), ur_type (ur10e),
  kinematics_params_file (defaults to ur_gripper_bringup's ur10e_calibration.yaml; falls back to the
    generic ur_description kinematics with a WARNING if that file is missing),
  launch_moveit (true), load_admittance (false), launch_rviz (false -> MoveIt RViz when true).
"""

import os

import launch.logging
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    logger = launch.logging.get_logger('ur_kinematic_assembly_bringup')
    ur_type = LaunchConfiguration('ur_type').perform(context)
    robot_ip = LaunchConfiguration('robot_ip')
    launch_rviz = LaunchConfiguration('launch_rviz')
    launch_moveit = LaunchConfiguration('launch_moveit').perform(context).lower() == 'true'
    load_admittance = LaunchConfiguration('load_admittance').perform(context).lower() == 'true'

    # Always use this robot's calibration; fall back to the generic kinematics with a loud warning.
    cal = LaunchConfiguration('kinematics_params_file').perform(context)
    if not cal or not os.path.isfile(cal):
        cal = os.path.join(get_package_share_directory('ur_description'),
                           'config', ur_type, 'default_kinematics.yaml')
        logger.warning(
            f'Calibration file not found; using GENERIC {ur_type} kinematics (TCP off by mm-cm). '
            'Extract this robot\'s calibration with ur_calibration -- see ur_gripper_bringup/README.md.')
    else:
        logger.info(f'Using robot calibration: {cal}')

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ur_robot_driver'), 'launch', 'ur_control.launch.py')),
        launch_arguments={
            'ur_type': ur_type,
            'robot_ip': robot_ip,
            'kinematics_params_file': cal,
            'launch_rviz': 'false',   # the MoveIt RViz (below) is the useful one
        }.items())
    actions = [driver]

    if launch_moveit:
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('ur_moveit_config'), 'launch', 'ur_moveit.launch.py')),
            launch_arguments={
                'ur_type': ur_type,
                'kinematics_params_file': cal,
                'launch_rviz': launch_rviz,
            }.items()))

    if load_admittance:
        try:
            adm_params = os.path.join(
                get_package_share_directory('ur_admittance_demo'),
                'config', 'ur_admittance_controller.yaml')
        except Exception:
            logger.warning('ur_admittance_demo not found; skipping the admittance_controller load.')
        else:
            # Delay so the controller_manager is up before we load the controller.
            actions.append(TimerAction(period=8.0, actions=[Node(
                package='controller_manager',
                executable='spawner',
                output='screen',
                arguments=[
                    'admittance_controller',
                    '--param-file', adm_params,
                    '--controller-manager', '/controller_manager',
                    '--controller-manager-timeout', '60',
                    '--inactive',
                ])]))

    return actions


def generate_launch_description():
    try:
        default_cal = os.path.join(
            get_package_share_directory('ur_gripper_bringup'), 'config', 'ur10e_calibration.yaml')
    except Exception:
        default_cal = ''
    args = [
        DeclareLaunchArgument('ur_type', default_value='ur10e'),
        DeclareLaunchArgument('robot_ip', default_value='192.168.125.2'),
        DeclareLaunchArgument(
            'kinematics_params_file', default_value=default_cal,
            description="This robot's calibration; falls back to generic kinematics if missing."),
        DeclareLaunchArgument('launch_moveit', default_value='true',
                              description='Also start move_group (for /compute_ik).'),
        DeclareLaunchArgument('load_admittance', default_value='false',
                              description='Also load admittance_controller (inactive), for admittance mode.'),
        DeclareLaunchArgument('launch_rviz', default_value='false',
                              description='Start the MoveIt RViz.'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
