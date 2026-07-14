"""One-shot launch for the WHOLE cable pick-and-place stack (everything except the demo itself).

Replaces the 6 terminals from the README with one command. It starts, in order:

  1. ur_gripper_bringup ....... arm + gripper under one controller_manager
                                (then press PLAY on the pendant's External Control program)
  2. move_group ............... /compute_ik, with this robot's kinematic calibration
  3. RealSense ................ camera_name:=camera1, publish_tf:=false (the hand-eye owns that frame)
  4. ur_tf_demo ............... hand-eye tf: tool0 -> camera1_color_optical_frame
  5. SAM3 detector (Node 1) ... cable_neck_ros_node -- runs in the SAM3 VENV (torch/GPU)
  6. SAM3 fusion  (Node 2) ... connector_pose_node -- plain system python; publishes base_link -> connector

The DEMO is deliberately NOT started here: it prompts on stdin (confirm_each_step), and stdin does not
work under `ros2 launch`. Run it in its own terminal once this stack is up:

    ros2 run ur_cable_pick_place_demo cable_pick_place

Usage:
    ros2 launch ur_cable_pick_place_demo cable_stack.launch.py
    # skip pieces you already have running:
    ros2 launch ur_cable_pick_place_demo cable_stack.launch.py bringup:=false camera:=false
    # point at a different camera / SAM3 install:
    ros2 launch ur_cable_pick_place_demo cable_stack.launch.py camera_serial:=_123456789012

NOTE the topic names: realsense2_camera nests topics under camera_namespace AND camera_name, so with
camera_name:=camera1 the image lands on /camera/camera1/color/image_raw (NOT /camera1/...). The image
FRAME is still camera1_color_optical_frame, which is what the hand-eye tf publishes -- that's the one
that has to match.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ur_type = LaunchConfiguration('ur_type')
    sam3_python = LaunchConfiguration('sam3_python')
    sam3_scripts = LaunchConfiguration('sam3_scripts')
    image_topic = LaunchConfiguration('image_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')

    # This robot's extracted kinematic calibration -- move_group must use the SAME one as the driver,
    # or IK targets are systematically off.
    calibration = PathJoinSubstitution(
        [FindPackageShare('ur_gripper_bringup'), 'config', 'ur10e_calibration.yaml'])

    args = [
        # --- component toggles (set false for anything you already have running) ---
        DeclareLaunchArgument('bringup', default_value='true',
                              description='Start ur_gripper_bringup (arm + gripper).'),
        DeclareLaunchArgument('moveit', default_value='true',
                              description='Start move_group (needed for /compute_ik).'),
        DeclareLaunchArgument('camera', default_value='true',
                              description='Start the RealSense camera.'),
        DeclareLaunchArgument('handeye', default_value='true',
                              description='Start the hand-eye tf (ur_tf_demo).'),
        DeclareLaunchArgument('sam3', default_value='true',
                              description='Start the two SAM3 nodes (detector + fusion).'),

        # --- robot / camera ---
        DeclareLaunchArgument('ur_type', default_value='ur10e'),
        DeclareLaunchArgument('camera_name', default_value='camera1',
                              description='Sets the image FRAME to <camera_name>_color_optical_frame, '
                                          'which must match the hand-eye tf.'),
        DeclareLaunchArgument('camera_serial', default_value='_218622272137',
                              description='RealSense serial (leading underscore = string).'),

        # --- SAM3 ---
        DeclareLaunchArgument('sam3_python', default_value='/opt/sam3_venv/bin/python',
                              description='Interpreter for the SAM3 DETECTOR (needs torch; the venv).'),
        DeclareLaunchArgument('sam3_scripts', default_value='/abhay_ws/sam3-abhay/scripts',
                              description='Directory holding cable_neck_ros_node.py / connector_pose_node.py.'),
        DeclareLaunchArgument('image_topic', default_value='/camera/camera1/color/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera1/color/camera_info'),
        DeclareLaunchArgument('neck_diameter', default_value='0.0034',
                              description='Physical connector diameter at the neck (m).'),
        DeclareLaunchArgument('tf_cache_s', default_value='60.0',
                              description='TF history for Node 2. Necks carry the IMAGE stamp, so they '
                                          'lag by one SAM3 inference -- keep this well above that.'),

        # ADAPTIVE THRESHOLD. With a fixed threshold SAM3 flips between labelling the connector "cable"
        # and vice versa; whichever class comes up empty kills the neck outright, because a neck IS the
        # cable/connector contact. Adaptive mode runs SAM3 ONCE at confidence_floor and then searches
        # the threshold PAIR in software, keeping the most confident masks that still yield a valid
        # neck -- at no extra inference cost.
        DeclareLaunchArgument('adaptive', default_value='true'),
        DeclareLaunchArgument('confidence_floor', default_value='0.2',
                              description='Never admit a mask below this, however the search relaxes.'),
    ]

    # 1. Arm + gripper (t=0). Press PLAY on the pendant once this is up.
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('ur_gripper_bringup'), 'launch', 'ur_gripper_control.launch.py'])),
        condition=IfCondition(LaunchConfiguration('bringup')))

    # 3. RealSense (t=0). publish_tf:=false so it does NOT also publish camera1_color_optical_frame --
    #    the hand-eye tf is the sole parent of that frame (two parents would break the tf tree).
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
        launch_arguments={
            'camera_name': LaunchConfiguration('camera_name'),
            'serial_no': LaunchConfiguration('camera_serial'),
            'enable_color': 'true',
            'enable_depth': 'false',
            'publish_tf': 'false',
        }.items(),
        condition=IfCondition(LaunchConfiguration('camera')))

    # 4. Hand-eye tf (t=4): after the bringup, so base_link -> tool0 already exists and the
    #    pose_streamer doesn't spend its first seconds warning about a missing transform.
    handeye = TimerAction(period=4.0, actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('ur_tf_demo'), 'launch', 'tf_streaming.launch.py'])),
            condition=IfCondition(LaunchConfiguration('handeye'))),
    ])

    # 2. move_group (t=6): needs the driver's robot_description up first.
    moveit = TimerAction(period=6.0, actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('ur_moveit_config'), 'launch', 'ur_moveit.launch.py'])),
            launch_arguments={
                'ur_type': ur_type,
                'kinematics_params_file': calibration,
            }.items(),
            condition=IfCondition(LaunchConfiguration('moveit'))),
    ])

    # 5. SAM3 detector (t=8). ExecuteProcess (not Node) because it must run under the SAM3 VENV's
    #    interpreter -- a normal Node action would use the system python, which has no torch.
    sam3_detector = TimerAction(period=8.0, actions=[
        ExecuteProcess(
            cmd=[sam3_python,
                 PathJoinSubstitution([sam3_scripts, 'cable_neck_ros_node.py']),
                 '--ros-args',
                 '-p', ['image_topic:=', image_topic],
                 '-p', ['adaptive:=', LaunchConfiguration('adaptive')],
                 '-p', ['confidence_floor:=', LaunchConfiguration('confidence_floor')],
                 '-p', 'publish_debug:=true'],
            additional_env={'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'},
            name='cable_neck_detector', output='screen',
            condition=IfCondition(LaunchConfiguration('sam3'))),
    ])

    # 6. SAM3 fusion (t=10): plain system python (no torch). Publishes base_link -> connector.
    sam3_fusion = TimerAction(period=10.0, actions=[
        ExecuteProcess(
            cmd=['python3',
                 PathJoinSubstitution([sam3_scripts, 'connector_pose_node.py']),
                 '--ros-args',
                 '-p', 'world_frame:=base_link',
                 '-p', 'connector_frame:=connector',
                 '-p', 'necks_topic:=/cable_neck_detector/necks',
                 '-p', ['camera_info_topic:=', camera_info_topic],
                 '-p', ['neck_diameter:=', LaunchConfiguration('neck_diameter')],
                 '-p', ['tf_cache_s:=', LaunchConfiguration('tf_cache_s')]],
            name='connector_pose_estimator', output='screen',
            condition=IfCondition(LaunchConfiguration('sam3'))),
    ])

    return LaunchDescription(
        args + [bringup, camera, handeye, moveit, sam3_detector, sam3_fusion])
