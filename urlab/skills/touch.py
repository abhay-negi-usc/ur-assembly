"""Touch probing -- measure the connector's height by driving the closed fingertip into it.

Ported from CableTouchPickPlace (_touch_descend, _servo_align_hover, _tip_flat, _touch_target).

WHY TOUCH AT ALL. The camera looks down, so the connector's z IS the camera's depth axis -- the
weakest direction of a monocular multi-view fit (error ~ Z^2 / baseline, several mm here). Its x,
y and yaw are LATERAL in the image and are well determined. So this skill takes x/y/yaw from
vision and MEASURES z by touching: it closes the fingers into a probe (no extra hardware) and
descends until a very low force trips. Roll and pitch are assumed zero (the connector lies flat),
which is exactly what makes one contact point enough to pin the height.

WHY POSITION-CONTROLLED STEPS, not force mode. The force is checked BEFORE each step, so the
descent stops the instant the threshold is crossed and the overshoot is bounded by ONE step
(~1 mm). A position move into a rigid object builds force fast, so step_m stays small and force_n
low; the UR protective stop is the backstop. Force mode would be gentler but slower and would need
the height known to set the task frame -- which is the very thing we are trying to measure.
"""

import time

import numpy as np

from .. import log as urlog
from ..transforms import inverse, pose_error, step_toward, xyzrpy_to_matrix

log = urlog.get('touch')


class TouchConfig:
    def __init__(self, cfg):
        self.tip_frame_name = cfg.get('connector_frame', 'connector_tip')
        self.T_tip_touch = xyzrpy_to_matrix(**_xyzrpy(cfg.section('touch_offset')))
        g = cfg.get('grasp_offset')
        self.T_tip_grasp = (self.T_tip_touch.copy() if g is None
                            else xyzrpy_to_matrix(**_xyzrpy(g)))

        t = cfg.section('touch')
        self.force_n = float(t.get('force_n', 3.0))
        self.step_m = float(t.get('step_m', 0.001))
        self.max_descent_m = float(t.get('max_descent_m', 0.06))
        self.settle_s = float(t.get('settle_s', 0.25))
        self.hover_height_m = float(t.get('hover_height_m', 0.05))
        self.contact_z_offset_m = float(t.get('contact_z_offset_m', 0.0))
        self.tare_before = bool(t.get('tare_before', True))

        a = cfg.section('align')
        self.align_max_iters = int(a.get('max_iterations', 8))
        self.align_pos_deadband = float(a.get('pos_deadband_m', 0.002))
        self.align_ang_deadband = np.radians(float(a.get('ang_deadband_deg', 1.5)))


def _xyzrpy(d):
    d = d or {}
    return {'xyz': d.get('xyz', [0.0, 0.0, 0.0]), 'rpy': d.get('rpy', [0.0, 0.0, 0.0])}


class TouchProbe:
    """Estimates x/y/yaw from vision, then measures z by touch. Holds the running estimate."""

    def __init__(self, robot, tc):
        self.robot = robot
        self.tc = tc
        self.tip_xy = None
        self.tip_yaw = None
        self.z_vision = None       # NOT trusted -- only a starting height for the descent
        self.z_touch = None        # the measured truth

    # ------------------------------------------------------------------ geometry
    def tip_flat(self, z):
        """The tip pose flattened to the demo's assumption: x/y/yaw from vision, z supplied,
        roll = pitch = 0. Flattening is what makes ONE touch enough."""
        c, s = np.cos(self.tip_yaw), np.sin(self.tip_yaw)
        T = np.eye(4)
        T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        T[:3, 3] = [self.tip_xy[0], self.tip_xy[1], z]
        return T

    def touch_target(self, z):
        return self.tip_flat(z) @ self.tc.T_tip_touch

    def grasp_target(self, z):
        return self.tip_flat(z) @ self.tc.T_tip_grasp

    # ------------------------------------------------------------------ vision
    def update_from_pose(self, T_base_tip):
        """Take x/y/yaw from a fused tip pose; keep z only as a starting height."""
        self.tip_xy = T_base_tip[:3, 3][:2].copy()
        axis = T_base_tip[:3, 0]
        self.tip_yaw = float(np.arctan2(axis[1], axis[0]))
        self.z_vision = float(T_base_tip[2, 3])
        log.info('Tip (vision): xy=(%.4f, %.4f) yaw=%+.1f deg; z=%.4f from vision is NOT trusted '
                 '(depth axis) -- it will be measured by touch.',
                 self.tip_xy[0], self.tip_xy[1], np.degrees(self.tip_yaw), self.z_vision)

    # ------------------------------------------------------------------ align
    def servo_align_hover(self, reestimate_fn=None):
        """Align the probe over the touch point at hover height, re-reading the tip each iteration
        so a still-improving estimate is tracked rather than committed to once."""
        for i in range(self.tc.align_max_iters):
            if reestimate_fn is not None and not reestimate_fn():
                return False
            T_target = self.touch_target(self.z_vision + self.tc.hover_height_m)
            T_cur = self.robot.fingertip()
            lin, ang = pose_error(T_cur, T_target)
            log.info('[align] iter %d: err lin=%.1f mm ang=%.1f deg', i + 1, lin * 1000,
                     np.degrees(ang))
            if lin <= self.tc.align_pos_deadband and ang <= self.tc.align_ang_deadband:
                log.info('[align] converged over the touch point.')
                return True
            T_cmd = step_toward(T_cur, T_target, gain=1.0, max_lin=0.03, max_ang=np.radians(20))
            if not self.robot.move_fingertip(T_cmd, f'align iter {i + 1}'):
                return False
        log.warning('[align] hit max iterations; proceeding with the current pose.')
        return True

    # ------------------------------------------------------------------ touch
    def descend(self):
        """Descend the closed-fingertip probe straight down until the tared contact force trips.

        Sets self.z_touch and returns bool. The force is read from the arm each step and checked
        BEFORE moving, so the overshoot past the trigger is one step at most."""
        if self.tc.tare_before:
            self.robot.arm.zero_ft()
        time.sleep(max(0.5, self.tc.settle_s))

        T_probe = self.robot.fingertip()
        z_start = float(T_probe[2, 3])
        T_xyyaw = self.touch_target(0.0)                  # x/y/yaw fixed; only z varies
        steps = max(1, int(self.tc.max_descent_m / self.tc.step_m))
        log.info('Descending from z=%.4f in %.1f mm steps (max %.0f mm) until contact >= %.1f N...',
                 z_start, self.tc.step_m * 1000, self.tc.max_descent_m * 1000, self.tc.force_n)

        for i in range(steps + 1):
            f = self.robot.arm.force()
            if f >= self.tc.force_n:
                z_contact = float(self.robot.fingertip()[2, 3])
                self.z_touch = z_contact + self.tc.contact_z_offset_m
                log.info('CONTACT after %d step(s): %.2f N. probe z=%.4f (descended %.1f mm). '
                         'connector z=%.4f. Vision said %.4f -- off by %+.1f mm.',
                         i, f, z_contact, (z_start - z_contact) * 1000, self.z_touch,
                         self.z_vision, (self.z_vision - self.z_touch) * 1000)
                return True

            T = T_xyyaw.copy()
            T[2, 3] = z_start - (i + 1) * self.tc.step_m
            if not self.robot.move_fingertip(T, f'touch step {i + 1}/{steps}'):
                return False
            time.sleep(self.tc.settle_s)

        log.error('No contact within %.0f mm. The vision x/y may be wrong (probe missed the cable) '
                  'or force_n is below the noise floor. Aborting rather than driving deeper.',
                  self.tc.max_descent_m * 1000)
        return False

    def retract_hover(self):
        return self.robot.move_fingertip(self.touch_target(self.z_touch + self.tc.hover_height_m),
                                         'retract to hover')
