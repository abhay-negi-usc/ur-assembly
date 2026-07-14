"""Fiducial pick-and-place -- port of ur_pick_place_demo.

Detect an ArUco marker, visually approach it, estimate the grasp from the close view, then grasp
(blind or closed-loop) and place. The perception is a marker instead of a scanned cable, but the
grasp geometry and the place sequence are the shared ones.

    open -> detect + align/centre -> approach to standoff -> estimate grasp
      -> [blind: home then open-loop grasp | servo: closed-loop to grasp]
      -> close -> lift -> pre-place -> place -> open -> retreat -> home
"""

import numpy as np

from .. import log as urlog
from ..log import StepRunner
from ..perception import ArucoDetector, MarkerTracker
from ..skills import servo
from ..skills.pick import GraspGeometry
from ..transforms import from_cfg, inverse
from ._cable import make_confirm
from ._runner import run_app

log = urlog.get('pick-place')


def build_and_run(cfg, robot, camera, args):
    detector = ArucoDetector(cfg)
    marker_id = int(cfg.get_path('marker.id', 11))
    tracker = MarkerTracker(camera, detector, robot.frames, marker_id, cfg.get('base_frame'))
    geom = GraspGeometry(cfg)
    confirm = make_confirm(cfg)
    q_home = robot.arm.q()

    T_object_marker = from_cfg(cfg.section('object_marker'))
    T_object_grasp = from_cfg(cfg.section('object_grasp'))
    cam_rpy = cfg.get('servo_cam_rpy_in_marker', [np.pi, 0.0, 0.0])
    standoff = float(cfg.get('servo_standoff_m', 0.15))
    step = float(cfg.get('servo_step_m', 0.05))
    blind = bool(cfg.get('blind_pick', True))

    def estimate_grasp():
        T = tracker.acquire(max_age_s=cfg.get('marker_max_age_s', 1.0))
        if T is None:
            return False
        geom.T_base_grasp = T @ inverse(T_object_marker) @ T_object_grasp
        log.info('Grasp target %s', geom.T_base_grasp[:3, 3].round(3))
        return True

    def grasp_target():
        """Recompute the tool0 grasp target from a fresh marker (for the closed-loop servo)."""
        T = tracker.observe()
        if T is None:
            return None
        geom.T_base_grasp = T @ inverse(T_object_marker) @ T_object_grasp
        return geom.T_base_grasp @ inverse(robot.T_tool0_grasp)

    runner = StepRunner(log, confirm=confirm is not None)
    if not runner.run([
        ('open gripper', robot.gripper.open),
        ('detect + align/centre',
         lambda: servo.align_and_center(robot, tracker, cam_rpy,
                                        cfg.get('marker_max_age_s', 1.0))),
        ('approach to standoff',
         lambda: servo.approach_to_standoff(robot, tracker, standoff, step, cam_rpy,
                                            confirm=confirm)),
        ('estimate grasp @ standoff', estimate_grasp),
    ]):
        return False

    if blind:
        grasp_steps = [
            ('return home (blind)', lambda: robot.arm.move_j(q_home, label='home')),
            ('grasp-align (open-loop)', lambda: robot.move_grasp_tcp(geom.pre_grasp(), 'grasp-align')),
            ('grasp (open-loop)', lambda: robot.move_grasp_tcp(geom.T_base_grasp, 'grasp')),
        ]
    else:
        gain = float(cfg.get('servo_gain', 1.0))
        grasp_steps = [
            ('visual-servo to grasp',
             lambda: servo.visual_servo_to_pose(
                 robot, grasp_target, gain, cfg.get('servo_max_linear_step_m', 0.03),
                 np.radians(cfg.get('servo_max_angular_step_deg', 20.0)),
                 cfg.get('servo_pos_deadband_m', 0.01),
                 np.radians(cfg.get('servo_ang_deadband_deg', 5.0)),
                 label='grasp', finish_on_loss=True, confirm=confirm)),
        ]
    if not runner.run(grasp_steps + [('close gripper', robot.gripper.close)]):
        return False

    return runner.run([
        ('lift', lambda: robot.move_grasp_tcp(geom.lift(), 'lift')),
        ('pre-place', lambda: robot.move_grasp_tcp(geom.pre_place(), 'pre-place')),
        ('place', lambda: robot.move_grasp_tcp(geom.place(), 'place')),
        ('open gripper (release)', robot.gripper.open),
        ('retreat', lambda: robot.move_grasp_tcp(geom.pre_place(), 'retreat')),
        ('return home', lambda: robot.arm.move_j(q_home, label='home')),
    ])


def main():
    run_app('Fiducial pick-and-place (ArUco)', 'pick_place', build_and_run, needs_camera=True)


if __name__ == '__main__':
    main()
