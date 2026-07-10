"""Launch the cable pick-and-place node.

Launches ONLY the pick/place node. It assumes these are already running:
  * the integrated bringup (arm + gripper under one controller_manager) -- ur_gripper_bringup,
  * move_group (for /compute_ik) -- e.g. ur_moveit_config,
  * the hand-eye tf (ur_tf_demo) so base_link -> camera is published,
  * a RealSense camera publishing color image + camera_info,
  * the external SAM3 nodes (see the sam3-abhay repo), started separately:
      python scripts/cable_neck_ros_node.py --ros-args -p image_topic:=/camera1/color/image_raw
      python scripts/connector_pose_node.py --ros-args -p world_frame:=base_link \
          -p connector_frame:=connector -p camera_info_topic:=/camera1/color/camera_info
    so that base_link -> connector is broadcast for this demo to read.

See the package README for the full per-terminal run guide.

NOTE: with confirm_each_step the node reads stdin -- use `ros2 run` so the prompts render.
Keep the e-stop in hand.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare('ur_cable_pick_place_demo'), 'config', 'cable_pick_place.yaml'])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Cable pick-and-place config yaml.'),
        Node(
            package='ur_cable_pick_place_demo',
            executable='cable_pick_place',
            name='cable_pick_place',
            output='screen',
            emulate_tty=True,
            parameters=[{'config_file': LaunchConfiguration('config_file')}],
        ),
    ])
