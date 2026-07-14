"""One-shot launch for the WHOLE cable TOUCH-pick stack (everything except the demo itself).

Same as ur_cable_pick_place_demo's cable_stack.launch.py, with ONE difference in the perception: it
runs the SAM3 **TIP** detector instead of the neck detector.

  1. ur_gripper_bringup ....... arm + gripper (then press PLAY on the pendant)
  2. move_group ............... /compute_ik, with this robot's kinematic calibration
  3. RealSense ................ camera_name:=camera1, publish_tf:=false (the hand-eye owns that frame)
  4. ur_tf_demo ............... hand-eye tf: tool0 -> camera1_color_optical_frame
  5. SAM3 TIP detector ........ cable_tip_ros_node -- runs in the SAM3 VENV (torch/GPU)
  6. SAM3 fusion .............. connector_pose_node, pointed at the TIP topic -> base_link -> connector_tip

Why the tip detector: SAM3 routinely labels the whole assembly "cable" and returns NO connector mask,
which starves the neck pipeline entirely (compute_necks iterates over CONNECTOR masks). The tip node
unions both prompts and works on the shape, so it survives that. Its ~/tips messages use the SAME
convention as ~/necks, so connector_pose_node fuses them unchanged -- including its RANSAC outlier
rejection, which is what discards background cables/connectors.

The DEMO is NOT started here: it prompts on stdin (confirm_each_step), which does not work under
`ros2 launch`. Run it in its own terminal:

    ros2 run ur_cable_touch_pick_place_demo cable_touch_pick_place
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
    tip_frame = LaunchConfiguration('tip_frame')

    calibration = PathJoinSubstitution(
        [FindPackageShare('ur_gripper_bringup'), 'config', 'ur10e_calibration.yaml'])

    args = [
        DeclareLaunchArgument('bringup', default_value='true'),
        DeclareLaunchArgument('moveit', default_value='true'),
        DeclareLaunchArgument('camera', default_value='true'),
        DeclareLaunchArgument('handeye', default_value='true'),
        DeclareLaunchArgument('sam3', default_value='true'),

        DeclareLaunchArgument('ur_type', default_value='ur10e'),
        DeclareLaunchArgument('camera_name', default_value='camera1',
                              description='Sets the image FRAME (<name>_color_optical_frame); must '
                                          'match the hand-eye tf.'),
        DeclareLaunchArgument('camera_serial', default_value='_218622272137'),

        DeclareLaunchArgument('sam3_python', default_value='/opt/sam3_venv/bin/python',
                              description='Interpreter for the SAM3 detector (needs torch; the venv).'),
        DeclareLaunchArgument('sam3_scripts', default_value='/abhay_ws/sam3-abhay/scripts'),
        DeclareLaunchArgument('image_topic', default_value='/camera/camera1/color/image_raw',
                              description='realsense2_camera nests topics under namespace AND name.'),
        DeclareLaunchArgument('camera_info_topic',
                              default_value='/camera/camera1/color/camera_info'),
        DeclareLaunchArgument('tip_frame', default_value='connector_tip',
                              description='TF the fusion publishes; must match the demo yaml.'),

        # Prompts / thresholds. The connector is the harder class -- SAM3 tends to call the whole
        # assembly "cable". The TIP node survives that (it unions the masks), but a connector mask,
        # when it exists, is still the best evidence for WHICH end is the connector.
        DeclareLaunchArgument('cable_prompt', default_value='cable'),
        DeclareLaunchArgument('connector_prompt', default_value='electrical connector'),
        DeclareLaunchArgument('threshold', default_value='0.5'),
        DeclareLaunchArgument('connector_threshold', default_value='0.20'),
        DeclareLaunchArgument('curve_px', default_value='40',
                              description='Predefined curve length (px) walked back from the tip to '
                                          'measure the connector axis.'),

        DeclareLaunchArgument('min_inlier_views', default_value='3',
                              description='RANSAC consensus: the true tip must be seen by at least '
                                          'this many views. Raises rejection of background cables.'),
        DeclareLaunchArgument('max_range_m', default_value='0.80',
                              description='Reject tips farther than this from the camera (background).'),
        DeclareLaunchArgument('tf_cache_s', default_value='60.0'),
    ]

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('ur_gripper_bringup'), 'launch', 'ur_gripper_control.launch.py'])),
        condition=IfCondition(LaunchConfiguration('bringup')))

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
        launch_arguments={
            'camera_name': LaunchConfiguration('camera_name'),
            'serial_no': LaunchConfiguration('camera_serial'),
            'enable_color': 'true',
            'enable_depth': 'false',
            'publish_tf': 'false',      # the hand-eye tf is the SOLE parent of the camera frame
        }.items(),
        condition=IfCondition(LaunchConfiguration('camera')))

    handeye = TimerAction(period=4.0, actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('ur_tf_demo'), 'launch', 'tf_streaming.launch.py'])),
            condition=IfCondition(LaunchConfiguration('handeye'))),
    ])

    moveit = TimerAction(period=6.0, actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare('ur_moveit_config'), 'launch', 'ur_moveit.launch.py'])),
            launch_arguments={'ur_type': ur_type,
                              'kinematics_params_file': calibration}.items(),
            condition=IfCondition(LaunchConfiguration('moveit'))),
    ])

    # SAM3 TIP detector -- ExecuteProcess (not Node) because it must run under the SAM3 VENV's
    # interpreter; a Node action would use system python, which has no torch.
    sam3_tip = TimerAction(period=8.0, actions=[
        ExecuteProcess(
            cmd=[sam3_python,
                 PathJoinSubstitution([sam3_scripts, 'cable_tip_ros_node.py']),
                 '--ros-args',
                 '-p', ['image_topic:=', image_topic],
                 '-p', ['cable_prompt:=', LaunchConfiguration('cable_prompt')],
                 '-p', ['connector_prompt:=', LaunchConfiguration('connector_prompt')],
                 '-p', ['threshold:=', LaunchConfiguration('threshold')],
                 '-p', ['connector_threshold:=', LaunchConfiguration('connector_threshold')],
                 '-p', ['curve_px:=', LaunchConfiguration('curve_px')],
                 '-p', 'publish_debug:=true'],
            additional_env={'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'},
            name='cable_tip_detector', output='screen',
            condition=IfCondition(LaunchConfiguration('sam3'))),
    ])

    # Fusion: the SAME estimator as the neck pipeline, just pointed at ~/tips. Its RANSAC consensus is
    # what rejects background cables/connectors (they form their own, smaller, view-clusters).
    sam3_fusion = TimerAction(period=10.0, actions=[
        ExecuteProcess(
            cmd=['python3',
                 PathJoinSubstitution([sam3_scripts, 'connector_pose_node.py']),
                 '--ros-args',
                 '-p', 'world_frame:=base_link',
                 '-p', ['connector_frame:=', tip_frame],
                 '-p', 'necks_topic:=/cable_tip_detector/tips',
                 '-p', ['camera_info_topic:=', camera_info_topic],
                 '-p', ['tf_cache_s:=', LaunchConfiguration('tf_cache_s')],
                 '-p', ['min_inlier_views:=', LaunchConfiguration('min_inlier_views')],
                 '-p', ['max_range_m:=', LaunchConfiguration('max_range_m')]],
            name='connector_tip_estimator', output='screen',
            condition=IfCondition(LaunchConfiguration('sam3'))),
    ])

    return LaunchDescription(
        args + [bringup, camera, handeye, moveit, sam3_tip, sam3_fusion])
