"""Integrated UR10e + Robotiq 2F-85 bringup under ONE controller_manager.

Starts the UR driver with the combined description (arm + gripper, via rsp.launch.py), then
spawns the gripper controllers into the driver's controller_manager. Because both hardware
interfaces live in one controller_manager, the arm and gripper controllers coexist with no
conflict (unlike running ur_control.launch.py and robotiq_control.launch.py separately).

Also runs two preflight checks (warnings only, never fatal):
  * gripper serial probe on com_port (Modbus id 9), and
  * coupler-is-identity warning.
"""

import os

import yaml

import launch.logging
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _read_config():
    pkg = get_package_share_directory('ur_gripper_bringup')
    with open(os.path.join(pkg, 'config', 'coupler.yaml'), 'r') as f:
        return yaml.safe_load(f) or {}


def preflight(context, *args, **kwargs):
    """Warn (never fail) about an identity coupler and a missing gripper."""
    logger = launch.logging.get_logger('ur_gripper_bringup')
    cfg = _read_config()
    com_port = str(cfg.get('com_port', '/dev/ttyUSB1'))
    coupler = cfg.get('coupler', {}) or {}
    xyz = [float(v) for v in coupler.get('xyz', [0.0, 0.0, 0.0])]
    rpy = [float(v) for v in coupler.get('rpy', [0.0, 0.0, 0.0])]

    if xyz == [0.0, 0.0, 0.0] and rpy == [0.0, 0.0, 0.0]:
        logger.warning(
            'Coupler transform is IDENTITY (placeholder). '
            'TODO: set the measured tool0 -> gripper-mount xyz/rpy in config/coupler.yaml.')

    if not bool(cfg.get('gripper_enabled', True)):
        logger.warning('gripper_enabled is false -- bringing up the ARM ONLY.')
        return []

    # Quick Modbus probe so a missing/unpowered gripper is flagged before the driver claims
    # the port. Non-fatal: we always continue.
    try:
        from pymodbus.client import ModbusSerialClient
    except ImportError:
        logger.warning(
            f'pymodbus not installed; skipped gripper preflight on {com_port}.')
        return []

    try:
        client = ModbusSerialClient(port=com_port, baudrate=115200, bytesize=8,
                                    parity='N', stopbits=1, timeout=0.5)
        responding = False
        if client.connect():
            for kw in ('device_id', 'slave', 'unit'):   # pymodbus API varies by version
                try:
                    r = client.read_input_registers(address=0x07D0, count=3, **{kw: 9})
                    responding = not (r is None or r.isError())
                    break
                except TypeError:
                    continue
                except Exception:
                    break
        client.close()
        if responding:
            logger.info(f'Robotiq gripper detected on {com_port}.')
        else:
            logger.warning(
                f'Robotiq gripper NOT detected on {com_port} (no Modbus id 9 response). '
                'Continuing, but the gripper hardware may fail to initialize -- check power/'
                'wiring, or set gripper_enabled: false in config/coupler.yaml for arm-only.')
    except Exception as exc:
        logger.warning(f'Gripper preflight on {com_port} failed: {exc}. Continuing.')
    return []


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('ur_gripper_bringup')
    cfg = _read_config()
    gripper_enabled = bool(cfg.get('gripper_enabled', True))

    # Always use this robot's extracted calibration (ur_calibration) for accurate TCP poses.
    # Falls back to the generic ur10e kinematics with a loud warning if the file is missing.
    logger = launch.logging.get_logger('ur_gripper_bringup')
    kinematics_file = LaunchConfiguration('kinematics_params_file').perform(context)
    if not os.path.isfile(kinematics_file):
        default_kin = os.path.join(
            get_package_share_directory('ur_description'),
            'config', 'ur10e', 'default_kinematics.yaml')
        logger.warning(
            f'Calibration file not found at {kinematics_file}. Using GENERIC ur10e kinematics '
            '-- TCP poses will be off by mm-cm. Extract your robot calibration with:\n'
            '  ros2 launch ur_calibration calibration_correction.launch.py '
            f'robot_ip:=<ip> target_filename:={kinematics_file}\n'
            'then relaunch.')
        kinematics_file = default_kin
    else:
        logger.info(f'Using robot calibration: {kinematics_file}')

    ur_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ur_robot_driver'), 'launch', 'ur_control.launch.py')),
        launch_arguments={
            'ur_type': 'ur10e',
            'robot_ip': LaunchConfiguration('robot_ip'),
            'use_mock_hardware': LaunchConfiguration('use_mock_hardware'),
            'launch_rviz': LaunchConfiguration('launch_rviz'),
            'kinematics_params_file': kinematics_file,
            'description_launchfile': os.path.join(pkg, 'launch', 'rsp.launch.py'),
        }.items())

    actions = [ur_control]

    if gripper_enabled:
        gripper_controllers = os.path.join(pkg, 'config', 'gripper_controllers.yaml')
        # Delay so the controller_manager is up and the gripper hardware is active before we
        # load/activate the gripper controllers into it.
        actions.append(TimerAction(period=12.0, actions=[Node(
            package='controller_manager',
            executable='spawner',
            output='screen',
            arguments=[
                'robotiq_gripper_controller',
                'robotiq_activation_controller',
                '--param-file', gripper_controllers,
                '--controller-manager', '/controller_manager',
                '--controller-manager-timeout', '60',
            ])]))

    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument('robot_ip', default_value='192.168.125.2',
                              description='UR10e IP address.'),
        DeclareLaunchArgument('use_mock_hardware', default_value='false',
                              description='Mock both arm and gripper hardware (no robot/serial).'),
        DeclareLaunchArgument('launch_rviz', default_value='false',
                              description='Start RViz with the driver.'),
        DeclareLaunchArgument(
            'kinematics_params_file',
            default_value=os.path.join(
                get_package_share_directory('ur_gripper_bringup'),
                'config', 'ur10e_calibration.yaml'),
            description='This robot\'s ur_calibration file. Defaults to '
                        'config/ur10e_calibration.yaml; falls back to generic if missing.'),
    ]
    return LaunchDescription(args + [
        OpaqueFunction(function=preflight),
        OpaqueFunction(function=launch_setup),
    ])
