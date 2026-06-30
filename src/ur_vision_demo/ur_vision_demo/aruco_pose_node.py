#!/usr/bin/env python3
"""ArUco fiducial pose estimator (ROS2 Jazzy).

Subscribes to one camera's color image + camera_info, detects ArUco markers, estimates each
marker's 6-DOF pose in the camera optical frame, and streams the results as:

  * tf2 transforms:  <camera_optical_frame> -> <marker_frame_prefix><id>
  * geometry_msgs/PoseArray on  ~/aruco_poses  (header = camera optical frame)
  * (optional) an annotated image on  ~/debug_image

Run one instance per camera (namespaced) so multiple cameras can stream simultaneously; give
each a distinct ``marker_frame_prefix`` so their tf frames don't collide. The ArUco dictionary
and marker size come from parameters (set from the demo config).

Pose is estimated with ``cv2.solvePnP(..., SOLVEPNP_IPPE_SQUARE)`` from the marker's known
physical size, which works across OpenCV versions (no deprecated estimatePoseSingleMarkers).
"""

import numpy as np

import rclpy
from rclpy.node import Node

import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_matrix


def get_aruco_dictionary(name):
    """Return a cv2.aruco dictionary for a name like 'DICT_4X4_50', across OpenCV versions."""
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError(
            'cv2.aruco not available. Install opencv with contrib '
            '(e.g. `pip install opencv-contrib-python`).')
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary '{name}'.")
    dict_id = getattr(cv2.aruco, name)
    if hasattr(cv2.aruco, 'getPredefinedDictionary'):       # OpenCV >= 4.7
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)                # OpenCV < 4.7


class ArucoPoseNode(Node):
    """Detects ArUco markers in one camera stream and publishes their poses."""

    def __init__(self):
        super().__init__('aruco_pose_node')

        self.image_topic = self.declare_parameter('image_topic', 'color/image_raw').value
        self.info_topic = self.declare_parameter(
            'camera_info_topic', 'color/camera_info').value
        self.dictionary_name = self.declare_parameter(
            'aruco_dictionary', 'DICT_4X4_50').value
        self.marker_size = self.declare_parameter('marker_size_m', 0.05).value
        self.marker_frame_prefix = self.declare_parameter(
            'marker_frame_prefix', 'marker_').value
        # If empty, use the frame_id from camera_info (the camera optical frame).
        self.camera_frame_override = self.declare_parameter('camera_frame', '').value
        self.publish_debug = self.declare_parameter('publish_debug_image', True).value

        self.bridge = CvBridge()
        dictionary = get_aruco_dictionary(self.dictionary_name)
        if hasattr(cv2.aruco, 'ArucoDetector'):             # OpenCV >= 4.7
            detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
            self._detect = detector.detectMarkers
        else:                                               # OpenCV < 4.7
            params = cv2.aruco.DetectorParameters_create()
            self._detect = lambda gray: cv2.aruco.detectMarkers(
                gray, dictionary, parameters=params)

        # Marker corner model (matches ArUco order: top-left, top-right, bottom-right,
        # bottom-left), centered on the marker, Z out of the marker face.
        h = self.marker_size / 2.0
        self.obj_points = np.array(
            [[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float32)

        self.K = None
        self.D = None
        self.camera_frame = self.camera_frame_override

        self.create_subscription(CameraInfo, self.info_topic, self._info_cb, 10)
        self.create_subscription(Image, self.image_topic, self._image_cb, 10)
        self.pose_pub = self.create_publisher(PoseArray, 'aruco_poses', 10)
        self.debug_pub = (self.create_publisher(Image, 'debug_image', 1)
                          if self.publish_debug else None)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            f"ArUco detector up: dict={self.dictionary_name}, "
            f"marker={self.marker_size * 1000:.0f} mm, image='{self.image_topic}'")

    # ------------------------------------------------------------- callbacks
    def _info_cb(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        d = np.array(msg.d, dtype=np.float64)
        self.D = d if d.size else np.zeros(5, dtype=np.float64)
        if not self.camera_frame_override:
            self.camera_frame = msg.header.frame_id

    def _image_cb(self, msg):
        if self.K is None:
            self.get_logger().warn('Waiting for camera_info...', throttle_duration_sec=5.0)
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)

        pose_array = PoseArray()
        pose_array.header.stamp = msg.header.stamp
        pose_array.header.frame_id = self.camera_frame or msg.header.frame_id

        if ids is not None and len(ids) > 0:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                img_pts = marker_corners.reshape(4, 2).astype(np.float32)
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_points, img_pts, self.K, self.D,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue

                t = tvec.flatten()
                rot = np.eye(4)
                rot[:3, :3], _ = cv2.Rodrigues(rvec)
                q = quaternion_from_matrix(rot)             # [x, y, z, w]

                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = (
                    float(t[0]), float(t[1]), float(t[2]))
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
                    float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                pose_array.poses.append(pose)

                tf = TransformStamped()
                tf.header.stamp = msg.header.stamp
                tf.header.frame_id = pose_array.header.frame_id
                tf.child_frame_id = f'{self.marker_frame_prefix}{int(marker_id)}'
                tf.transform.translation.x, tf.transform.translation.y, \
                    tf.transform.translation.z = float(t[0]), float(t[1]), float(t[2])
                tf.transform.rotation = pose.orientation
                self.tf_broadcaster.sendTransform(tf)

                self.get_logger().info(
                    f'id {int(marker_id)}: x={t[0]:.3f} y={t[1]:.3f} z={t[2]:.3f} m',
                    throttle_duration_sec=1.0)

                if self.debug_pub is not None:
                    cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec,
                                      self.marker_size * 0.5)

            if self.debug_pub is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        self.pose_pub.publish(pose_array)

        if self.debug_pub is not None:
            dbg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            dbg.header = msg.header
            self.debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseNode()
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
