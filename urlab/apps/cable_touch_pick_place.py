"""Cable touch-then-pick -- port of ur_cable_touch_pick_place_demo.

Vision gives x/y/yaw (well determined, lateral in the image); TOUCH gives z (the camera's weak
depth axis). The probe is the gripper's own fingers, fully closed. Roll/pitch are assumed zero, so
one contact pins the height.

    close (fingers = probe) -> scan -> estimate tip (x,y,yaw) -> align over the touch point
      -> descend to contact (z) -> retract -> open -> align -> grasp (vision xy + touched z)
      -> [check] -> lift -> place -> home
"""

from .. import log as urlog
from ..log import StepRunner
from ..skills.pick import GraspCheck, GraspGeometry, GraspImageRecorder
from ..skills.touch import TouchConfig, TouchProbe
from ._cable import build_scanner, make_confirm
from ._runner import run_app

log = urlog.get('cable-touch')


def build_and_run(cfg, robot, camera, args):
    scanner, detector, estimator = build_scanner(cfg, robot, camera)
    tc = TouchConfig(cfg)
    probe = TouchProbe(robot, tc)
    geom = GraspGeometry(cfg)
    check = GraspCheck(cfg)
    recorder = GraspImageRecorder(cfg)
    confirm = make_confirm(cfg)
    q_home = robot.arm.q()
    grasp_close = int(cfg.get_path('grasp_check.closed_counts', 228))

    def estimate_tip():
        """Re-fit the tip from the accumulated views and update the probe's x/y/yaw."""
        T = estimator.estimate()
        if T is None:
            log.error('No tip estimate available.')
            return False
        probe.update_from_pose(T)
        return True

    runner = StepRunner(log, confirm=confirm is not None)
    if not runner.run([
        ('close gripper (fingers = probe)', lambda: robot.gripper.go_to(grasp_close, 'close')),
        ('scan cable (multi-view)', lambda: scanner.scan(confirm=confirm) is not None),
        ('estimate tip (x, y, yaw)', estimate_tip),
        ('align over the touch point (hovering)',
         lambda: probe.servo_align_hover(reestimate_fn=estimate_tip)),
        ('TOUCH: descend until contact', probe.descend),
        ('retract to hover', probe.retract_hover),
        ('open gripper', robot.gripper.open),
        ('align over the grasp point (hover)',
         lambda: robot.move_fingertip(probe.grasp_target(probe.z_touch + tc.hover_height_m),
                                      'grasp-align (hover)')),
    ]):
        return False

    # Grasp with vision x/y/yaw + the TOUCHED z, with the counts-based check + recovery. Record
    # wrist images at grasp_check.capture_rate_hz (default 1 Hz) over the grasp/retry loop.
    with recorder.recording(scanner.camera):
        attempt = 0
        while True:
            geom.T_base_grasp = probe.grasp_target(probe.z_touch)
            if not runner.run([
                ('move to grasp', lambda: robot.move_fingertip(geom.T_base_grasp, 'grasp')),
                ('close gripper (grasp)', lambda: robot.gripper.go_to(grasp_close, 'close')),
            ]):
                return False
            if check.evaluate(robot.gripper) == 'ok':
                break
            if attempt >= check.max_retries:
                log.error('Grasp failed on all %d attempts; aborting.', check.max_retries + 1)
                return False
            attempt += 1
            log.warning('Grasp missed -- recovering (attempt %d/%d). z is still known from the touch.',
                        attempt + 1, check.max_retries + 1)
            if not (robot.gripper.open('drop') and probe.retract_hover()):
                return False

    return runner.run([
        ('lift', lambda: robot.move_fingertip(geom.lift(), 'lift')),
        ('move to pre-place', lambda: robot.move_fingertip(geom.pre_place(), 'pre-place')),
        ('move to place', lambda: robot.move_fingertip(geom.place(), 'place')),
        ('open gripper (release)', robot.gripper.open),
        ('retreat', lambda: robot.move_fingertip(geom.pre_place(), 'retreat')),
        ('return home', lambda: robot.arm.move_j(q_home, label='home')),
    ])


def main():
    run_app('Cable touch-then-pick (vision x/y/yaw + touch z)', 'cable_touch_pick_place',
            build_and_run, needs_camera=True)


if __name__ == '__main__':
    main()
