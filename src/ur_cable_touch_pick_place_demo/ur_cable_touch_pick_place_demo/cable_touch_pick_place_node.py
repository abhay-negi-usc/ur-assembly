#!/usr/bin/env python3
"""Cable TOUCH-then-pick demo for the UR10e + Robotiq 2F-85 (ROS2 Jazzy).

Subclasses ur_cable_pick_place_demo's CablePickPlace, so the scan, fingertip frame, grasp check, speed
caps and image saving are all reused. The difference is HOW the connector's pose is obtained.

THE IDEA
--------
Vision and touch are each used only for what they are actually good at:

  * The camera looks DOWN, so the connector's Z is the camera's DEPTH direction -- the weakest axis of
    a monocular multi-view fit (error grows as Z^2 / baseline; several mm here). But x, y and the yaw
    are LATERAL in the image and are well determined.
  * So: take x, y and YAW from vision, and MEASURE z by touching. Roll and pitch are assumed zero (the
    connector lies flat), which is what makes a single touch sufficient to pin down z.

The probe is the gripper itself with the fingers FULLY CLOSED -- no extra hardware.

SEQUENCE
--------
  close gripper (fingers = probe) -> scan (multi-view) -> estimate TIP (x, y, yaw only)
    -> servo-align over the touch point while hovering
    -> descend in small steps until a VERY LOW force triggers  -> contact z  =>  the connector's z
    -> retract -> open gripper -> align -> grasp -> [grasp check] -> lift
    -> pre-place -> place -> open -> retreat -> home

The touch point is defined RELATIVE TO THE TIP (touch_offset, default 5 cm along the tip's -x, i.e.
back along the cable from the tip). The grasp point defaults to the same place -- deliberately: z is
only KNOWN where we touched, so grasping anywhere else re-introduces the height uncertainty we just
spent a touch removing.

PERCEPTION
----------
Reads the TF published by the SAM3 tip pipeline (see the sam3-abhay repo):
    cable_tip_ros_node.py  -> ~/tips  (classification-free: cable+connector masks are UNIONED, so it
                                       survives SAM3 labelling the whole assembly "cable")
    connector_pose_node.py -> base_link -> connector_tip   (fuses the tips; RANSAC outlier rejection)
"""

import os

import numpy as np

import rclpy
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger

from ur_cable_pick_place_demo.cable_pick_place_node import CablePickPlace
from ur_pick_place_demo.pick_place_node import xyzrpy_to_matrix


class CableTouchPickPlace(CablePickPlace):
    """Vision for x/y/yaw, TOUCH for z, then pick."""

    def __init__(self):
        super().__init__(node_name='cable_touch_pick_place',
                         default_config=self._touch_config())
        c = self.cfg

        # The TF the SAM3 tip pipeline publishes. NOTE: the inherited scan machinery (including the
        # axis-aware refine orbit) keys off self.connector_frame, so set connector_frame to this in the
        # yaml and the scan refines the TIP's axis for free.
        self.tip_name = c.get('connector_frame', 'connector_tip')

        # Touch / grasp points, expressed in the TIP frame (its x = the connector axis, pointing out of
        # the cable toward the tip; so -x is BACK along the cable).
        self.T_tip_touch = xyzrpy_to_matrix(**self._xyzrpy(c.get('touch_offset', {})))
        g = c.get('grasp_offset')
        # Default the grasp to the TOUCHED point: z is only measured there. Grasping elsewhere along
        # the cable would re-introduce the height uncertainty the touch just removed.
        self.T_tip_grasp = (self.T_tip_touch.copy() if g is None
                            else xyzrpy_to_matrix(**self._xyzrpy(g)))

        t = c.get('touch', {}) or {}
        self.touch_force = float(t.get('force_n', 3.0))            # VERY low: this is a probe, not a push
        self.touch_step = float(t.get('step_m', 0.001))
        self.touch_max_descent = float(t.get('max_descent_m', 0.06))
        self.touch_settle_s = float(t.get('settle_s', 0.25))
        self.hover_height = float(t.get('hover_height_m', 0.05))
        self.contact_z_offset = float(t.get('contact_z_offset_m', 0.0))
        self.wrench_topic = t.get('wrench_topic', '/force_torque_sensor_broadcaster/wrench')
        self.ft_zero_service = t.get('ft_zero_service', '/io_and_status_controller/zero_ftsensor')
        self.tare_before = bool(t.get('tare_before', True))

        a = c.get('align', {}) or {}
        self.align_max_iters = int(a.get('max_iterations', 8))
        self.align_pos_deadband = float(a.get('pos_deadband_m', 0.002))
        self.align_ang_deadband = np.radians(float(a.get('ang_deadband_deg', 1.5)))

        # Interfaces
        self._force = None
        self.create_subscription(WrenchStamped, self.wrench_topic, self._wrench_cb, 10)
        self.ft_zero_client = self.create_client(Trigger, self.ft_zero_service)

        # Estimated state
        self._tip_xy = None       # from vision (trusted)
        self._tip_yaw = None      # from vision (trusted)
        self._z_vision = None     # from vision (NOT trusted -- only a starting height for the descent)
        self._z_touch = None      # from the touch (trusted)

    @staticmethod
    def _touch_config():
        src = os.path.normpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), '..', 'config',
            'cable_touch_pick_place.yaml'))
        if os.path.isfile(src):
            return src
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('ur_cable_touch_pick_place_demo'),
                            'config', 'cable_touch_pick_place.yaml')

    # ------------------------------------------------------------------ setup
    def setup(self):
        if not super().setup():
            return False
        if not self.ft_zero_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn(
                f"F/T zero service '{self.ft_zero_service}' not up -- the touch threshold will be "
                'biased by the tool weight unless the sensor is already tared.')
        self.get_logger().warn(
            'This demo PROBES BY TOUCH with the fingers closed. Keep the e-stop in hand: the descent '
            f'stops on a {self.touch_force:.1f} N contact, but it is POSITION-controlled, so an '
            'unexpectedly rigid obstacle will build force within one step.')
        return True

    # ---------------------------------------------------------------- F/T
    def _wrench_cb(self, msg):
        f = msg.wrench.force
        self._force = float((f.x ** 2 + f.y ** 2 + f.z ** 2) ** 0.5)

    def _tare_ft(self):
        if not self.tare_before:
            return True
        if not self.ft_zero_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('F/T zero service unavailable; skipping tare.')
            return False
        future = self.ft_zero_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        resp = future.result()
        if resp is None or not resp.success:
            self.get_logger().warn('F/T tare did not report success; continuing.')
            return False
        self.get_logger().info('F/T sensor tared.')
        return True

    def _contact_force(self):
        """Magnitude of the (tared) contact force in N. 0 until the first wrench arrives."""
        return 0.0 if self._force is None else self._force

    # -------------------------------------------------------------- geometry
    def _probe_pose(self):
        """The PROBE (the closed fingertip) in the base frame, from tf. This is the grasp reference
        inherited from CablePickPlace -- with the fingers shut it doubles as the touch probe."""
        T = self._tf_matrix(self.base_frame, self.tip_frame)
        return None if T is None else T @ self.T_tool0_grasp

    def _tip_flat(self, z):
        """The tip pose, FLATTENED to this demo's assumption: x/y/yaw from vision, z supplied,
        roll = pitch = 0. Flattening is what makes one touch enough -- with roll/pitch free, a single
        contact point could not pin down the height."""
        c, s = np.cos(self._tip_yaw), np.sin(self._tip_yaw)
        T = np.eye(4)
        T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T[:3, 3] = [self._tip_xy[0], self._tip_xy[1], z]
        return T

    def _touch_target(self, z):
        """Where the probe should touch: an offset from the tip, in the TIP frame (-x = back along
        the cable from the tip)."""
        return self._tip_flat(z) @ self.T_tip_touch

    def _grasp_target(self, z):
        return self._tip_flat(z) @ self.T_tip_grasp

    # ----------------------------------------------------------- estimate tip
    def _estimate_tip(self):
        """Read the fused tip pose and keep ONLY what vision is good for: x, y and yaw.

        The camera looks down, so the triangulated z IS the depth direction -- the weakest axis of the
        fit. x/y are lateral in the image and well determined; the yaw comes from the axis, which the
        scan's refine orbit is specifically designed to sharpen. z is deliberately discarded (kept only
        as a starting height for the descent) and measured by touching instead."""
        T = self._tf_matrix(self.base_frame, self.tip_name,
                            max_age_s=self.connector_max_age, timeout_s=self.connector_wait_s)
        if T is None:
            self.get_logger().error(
                f"No fresh '{self.base_frame} -> {self.tip_name}' tf. Is the SAM3 tip pipeline running "
                f'(cable_tip_ros_node + connector_pose_node with connector_frame:={self.tip_name})?')
            return False
        self._tip_xy = T[:3, 3][:2].copy()
        axis = T[:3, 0]
        self._tip_yaw = float(np.arctan2(axis[1], axis[0]))
        self._z_vision = float(T[2, 3])
        self.get_logger().info(
            f'Tip (vision): xy=({self._tip_xy[0]:.4f}, {self._tip_xy[1]:.4f}) m  '
            f'yaw={np.degrees(self._tip_yaw):+.1f} deg  |  z={self._z_vision:.4f} m from vision is '
            'NOT trusted (depth axis) -- it will be measured by touch.')
        return True

    # ------------------------------------------------------------ align / touch
    def _servo_align_hover(self):
        """Closed-loop align the probe over the touch point at hover height, re-reading the tip each
        iteration so a drifting/improving estimate is tracked rather than committed to once."""
        for i in range(self.align_max_iters):
            if not self._estimate_tip():          # re-read: the estimate keeps improving during the scan
                return False
            T_target = self._touch_target(self._z_vision + self.hover_height)
            T_cur = self._probe_pose()
            if T_cur is None:
                self.get_logger().error(f'No {self.base_frame} -> {self.tip_frame} tf.')
                return False
            lin, ang = self._pose_error(T_cur, T_target)
            self.get_logger().info(
                f'[align] iter {i + 1}: err lin={lin * 1000:.1f} mm ang={np.degrees(ang):.1f} deg')
            if lin <= self.align_pos_deadband and ang <= self.align_ang_deadband:
                self.get_logger().info('[align] converged over the touch point.')
                return True
            T_cmd = self._interpolate_pose(T_cur, T_target)
            if not self.move_grasp_tcp_to(T_cmd, f'align iter {i + 1}'):
                return False
        self.get_logger().warn(
            f'[align] hit max iterations ({self.align_max_iters}); proceeding with the current pose.')
        return True

    def _touch_descend(self):
        """Descend the closed-fingertip probe straight down until the tared contact force trips.

        Position-controlled stepping, NOT admittance: the force is checked BEFORE each step, so the
        descent stops the moment the threshold is crossed and the overshoot is bounded by one step
        (~1 mm). That is why step_m must stay small and force_n low -- a position-controlled move into
        a rigid object builds force fast. The UR's protective stop is the backstop.

        Sets self._z_touch (the connector's true height) and returns bool."""
        if not self._tare_ft():
            self.get_logger().warn('Proceeding untared -- the tool weight biases the force reading.')
        self._sleep(max(0.5, self.touch_settle_s))     # let a fresh, tared sample arrive

        T_probe = self._probe_pose()
        if T_probe is None:
            return False
        z_start = float(T_probe[2, 3])
        T_xyyaw = self._touch_target(0.0)              # x/y/yaw fixed; only z varies below

        steps = max(1, int(self.touch_max_descent / self.touch_step))
        self.get_logger().info(
            f'Descending from z={z_start:.4f} m in {self.touch_step * 1000:.1f} mm steps '
            f'(max {self.touch_max_descent * 1000:.0f} mm) until contact >= {self.touch_force:.1f} N...')

        for i in range(steps + 1):
            f = self._contact_force()
            if f >= self.touch_force:
                T_now = self._probe_pose()
                if T_now is None:
                    return False
                z_contact = float(T_now[2, 3])
                self._z_touch = z_contact + self.contact_z_offset
                self.get_logger().info(
                    f'CONTACT after {i} step(s): force {f:.2f} N >= {self.touch_force:.2f} N. '
                    f'probe z={z_contact:.4f} m (descended {(z_start - z_contact) * 1000:.1f} mm). '
                    f'connector z = {self._z_touch:.4f} m '
                    f'(contact_z_offset {self.contact_z_offset * 1000:+.1f} mm). '
                    f'Vision said {self._z_vision:.4f} m -- off by '
                    f'{(self._z_vision - self._z_touch) * 1000:+.1f} mm.')
                return True

            T = T_xyyaw.copy()
            T[2, 3] = z_start - (i + 1) * self.touch_step
            if not self.move_grasp_tcp_to(T, f'touch step {i + 1}/{steps}'):
                return False
            self._sleep(self.touch_settle_s)

        self.get_logger().error(
            f'No contact within {self.touch_max_descent * 1000:.0f} mm of descent. The vision xy may '
            'be wrong (probe missed the cable), or force_n is set above the noise floor. Aborting '
            'rather than driving deeper.')
        return False

    def _retract_hover(self):
        T = self._touch_target(self._z_touch + self.hover_height)
        return self.move_grasp_tcp_to(T, 'retract to hover')

    def _go_grasp(self):
        """Grasp using vision x/y/yaw and the TOUCHED z -- the whole point of the exercise."""
        self.T_base_grasp = self._grasp_target(self._z_touch)
        p = self.T_base_grasp[:3, 3]
        self.get_logger().info(
            f'Grasp target: xy from vision, z from TOUCH -> '
            f'({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m, yaw={np.degrees(self._tip_yaw):+.1f} deg.')
        return self.move_grasp_tcp_to(self.T_base_grasp, 'grasp')

    # --------------------------------------------------------------------- run
    def run(self):
        home_joints = self._current_joints()

        ok = (
            # Fingers FULLY CLOSED: the closed fingertip is the touch probe.
            self._do('close gripper (fingers = probe)',
                     lambda: self.gripper_to(self.grasp_close, 'close'))
            and self._do('scan cable (multi-view)', self._scan)
            and self._do('estimate tip (x, y, yaw)', self._estimate_tip)
            and self._do('servo-align over the touch point (hovering)', self._servo_align_hover)
            and self._do('TOUCH: descend until contact', self._touch_descend)
            and self._do('retract to hover', self._retract_hover)
            and self._do('open gripper', lambda: self.gripper_to(self.gripper_open, 'open'))
            and self._do('move to grasp-align (hover)', self._align_above_grasp)
        )
        if not ok:
            return False

        # Grasp, with the inherited "cable not seated in the fingertip groove" check + recovery.
        attempt = 0
        while True:
            if not (self._log_grasp_delta('pre-grasp')
                    and self._do('move to grasp', self._go_grasp)
                    and self._log_grasp_delta('at-grasp')
                    and self._do('close gripper (grasp)',
                                 lambda: self.gripper_to(self.grasp_close, 'close'))):
                return False
            if not self.grasp_check_enabled or self._grasp_succeeded():
                break
            if attempt >= self.grasp_max_retries:
                self.get_logger().error(
                    f'Grasp failed on all {self.grasp_max_retries + 1} attempts; aborting.')
                return False
            attempt += 1
            self.get_logger().warn(
                f'Pick-up failed (cable not in the fingertip groove). Recovering and retrying '
                f'(attempt {attempt + 1}/{self.grasp_max_retries + 1})...')
            # The z is still known from the touch, so recovery only has to reopen and re-approach.
            if not (self._do('recover: open gripper (drop)',
                             lambda: self.gripper_to(self.gripper_open, 'open'))
                    and self._do('recover: back to hover', self._retract_hover)):
                return False

        ok = (
            self._do('lift', lambda: self.move_grasp_tcp_to(self._lift_pose(), 'lift'))
            and self._do('move to pre-place',
                         lambda: self.move_grasp_tcp_to(self._pre_place_pose(), 'pre-place'))
            and self._do('move to place',
                         lambda: self.move_grasp_tcp_to(self._place_pose(), 'place'))
            and self._do('open gripper (release)',
                         lambda: self.gripper_to(self.gripper_open, 'open'))
            and self._do('retreat',
                         lambda: self.move_grasp_tcp_to(self._pre_place_pose(), 'retreat'))
            and self._do('return home', lambda: self.send_joints(home_joints)))
        if ok:
            self.get_logger().info('Cable touch-pick-and-place complete.')
        return ok

    def _align_above_grasp(self):
        """Hover over the GRASP point (which may differ from the touch point) before descending."""
        T = self._grasp_target(self._z_touch + self.hover_height)
        return self.move_grasp_tcp_to(T, 'grasp-align (hover)')


def main(args=None):
    rclpy.init(args=args)
    node = CableTouchPickPlace()
    try:
        if node.setup():
            node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
