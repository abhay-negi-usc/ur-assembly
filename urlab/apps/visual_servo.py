"""Visual servo demo -- port of ur_visual_servo_demo.

Continuously drive the camera to a fixed standoff square to a marker: a position-based visual
servo loop. Holds when the marker is not visible or already within the deadband.
"""

import time

import numpy as np

from .. import log as urlog
from ..perception import ArucoDetector, MarkerTracker
from ..skills import servo
from ..transforms import pose_error
from ._runner import run_app

log = urlog.get('visual-servo')


def build_and_run(cfg, robot, camera, args):
    detector = ArucoDetector(cfg)
    marker_id = int(cfg.get_path('marker.id', 11))
    tracker = MarkerTracker(camera, detector, robot.frames, marker_id, cfg.get('base_frame'))
    cam_rpy = cfg.get('cam_rpy_in_marker', [np.pi, 0.0, 0.0])
    standoff = float(cfg.get('standoff_m', 0.10))
    gain = float(cfg.get('servo_gain', 0.4))
    max_lin = float(cfg.get('servo_max_linear_step_m', 0.02))
    max_ang = np.radians(float(cfg.get('servo_max_angular_step_deg', 15.0)))
    pos_db = float(cfg.get('pos_deadband_m', 0.005))
    ang_db = np.radians(float(cfg.get('ang_deadband_deg', 1.5)))
    rate = float(cfg.get('rate_hz', 2.0))
    max_age = float(cfg.get('marker_max_age_s', 0.5))

    if cfg.get('confirm_start', True):
        input('Visual servo will track marker %d. Enter to start (Ctrl-C to stop): ' % marker_id)

    log.info('Servoing to a %.0f cm standoff square to marker %d. Ctrl-C to stop.',
             standoff * 100, marker_id)
    period = 1.0 / max(rate, 0.1)
    while True:
        T_marker = tracker.observe()
        if T_marker is None:
            log.info('marker not in view -- holding.')
            time.sleep(period)
            continue
        from ..skills.servo import camera_on_marker
        from ..transforms import step_toward
        T_des = camera_on_marker(T_marker, standoff, cam_rpy)
        T_cur = robot.camera()
        lin, ang = pose_error(T_cur, T_des)
        if lin <= pos_db and ang <= ang_db:
            log.info('within deadband (%.1f mm, %.1f deg) -- holding.', lin * 1000, np.degrees(ang))
            time.sleep(period)
            continue
        T_cam_cmd = step_toward(T_cur, T_des, gain, max_lin, max_ang)
        robot.move_camera(T_cam_cmd, 'servo step')
        time.sleep(period)


def main():
    run_app('Visual servo (PBVS to a marker)', 'visual_servo', build_and_run,
            with_gripper=False, needs_camera=True)


if __name__ == '__main__':
    main()
