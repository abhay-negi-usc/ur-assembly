#!/usr/bin/env python3
"""Stream tool and fiducial-marker poses w.r.t. the robot base, using tf2.

Standard practice: rather than re-deriving transforms by hand, this node runs a
``tf2_ros.TransformListener`` over the shared tf tree and periodically looks up and
republishes, as ``geometry_msgs/PoseStamped`` in the robot base frame:

  * the tool pose:        base_frame -> tool_frame          on  ~/tool_pose
  * each visible marker:  base_frame -> <marker frame>      on  ~/markers/<marker_frame>

and all markers together as a ``geometry_msgs/PoseArray`` on ``~/marker_poses`` (for RViz).

The lookups compose chains published by other nodes:
  base -> ... -> tool0        from the UR driver (robot_state_publisher),
  base -> camera optical      from the hand-eye static transform (see tf_streaming.launch.py),
  camera -> marker            from the ArUco detector (ur_vision_demo).
So a hand-eye calibration must connect each camera to the base for the marker lookups to
resolve; until then base->tool0 still streams.
"""

import yaml

import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration

import tf2_ros
from geometry_msgs.msg import PoseStamped, PoseArray, Pose


def transform_to_pose(transform):
    """geometry_msgs/Transform -> geometry_msgs/Pose."""
    pose = Pose()
    pose.position.x = transform.translation.x
    pose.position.y = transform.translation.y
    pose.position.z = transform.translation.z
    pose.orientation = transform.rotation
    return pose


class PoseStreamer(Node):
    """Looks up base->tool and base->markers via tf2 and republishes them as poses."""

    def __init__(self):
        super().__init__('pose_streamer')

        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        self.tool_frame = self.declare_parameter('tool_frame', 'tool0').value
        # Marker tf frames are matched by these prefixes (one per camera).
        self.marker_prefixes = self.declare_parameter(
            'marker_frame_prefixes', ['camera1_marker_']).value
        self.publish_rate = self.declare_parameter('publish_rate_hz', 10.0).value
        # Skip markers whose latest transform is older than this (stale / out of view).
        self.max_age_s = self.declare_parameter('max_marker_age_s', 1.0).value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.tool_pub = self.create_publisher(PoseStamped, '~/tool_pose', 10)
        self.markers_pub = self.create_publisher(PoseArray, '~/marker_poses', 10)
        self._marker_pubs = {}   # marker frame -> PoseStamped publisher (created on demand)

        self.create_timer(1.0 / max(self.publish_rate, 1.0), self._tick)
        self.get_logger().info(
            f"Streaming tool '{self.tool_frame}' and markers {self.marker_prefixes} "
            f"in base frame '{self.base_frame}'")

    def _lookup(self, source_frame):
        """Latest base_frame -> source_frame transform, or None if unavailable."""
        try:
            return self.tf_buffer.lookup_transform(self.base_frame, source_frame, Time())
        except tf2_ros.TransformException:
            return None

    def _discover_marker_frames(self):
        """All tf frames whose name matches one of the marker prefixes."""
        try:
            frames = yaml.safe_load(self.tf_buffer.all_frames_as_yaml()) or {}
        except yaml.YAMLError:
            return []
        return [f for f in frames
                if any(f.startswith(p) for p in self.marker_prefixes)]

    def _tick(self):
        now = self.get_clock().now()
        stamp = now.to_msg()

        # --- tool pose ---------------------------------------------------
        tf = self._lookup(self.tool_frame)
        if tf is not None:
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = self.base_frame
            ps.pose = transform_to_pose(tf.transform)
            self.tool_pub.publish(ps)
        else:
            self.get_logger().warn(
                f'No transform {self.base_frame} -> {self.tool_frame} '
                '(is the UR driver running?)', throttle_duration_sec=5.0)

        # --- marker poses ------------------------------------------------
        max_age = Duration(seconds=self.max_age_s)
        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = self.base_frame

        for frame in self._discover_marker_frames():
            tf = self._lookup(frame)
            if tf is None:
                continue
            if (now - Time.from_msg(tf.header.stamp)) > max_age:
                continue   # marker not seen recently
            pose = transform_to_pose(tf.transform)
            pose_array.poses.append(pose)

            pub = self._marker_pubs.get(frame)
            if pub is None:
                pub = self.create_publisher(PoseStamped, f'~/markers/{frame}', 10)
                self._marker_pubs[frame] = pub
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = self.base_frame
            ps.pose = pose
            pub.publish(ps)

            p = pose.position
            self.get_logger().info(
                f'{frame}: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} (in {self.base_frame})',
                throttle_duration_sec=2.0)

        self.markers_pub.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)
    node = PoseStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
