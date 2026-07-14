"""Gripper cycle demo -- port of ur_gripper_demo.

Steps the 2F-85 through a list of fractions (0 = open, 1 = closed), dwelling at each. A stall
(fingers meeting an object before the commanded position) is a SUCCESS, not a failure -- the
gripper reports it and we keep going.
"""

import time

from .. import log as urlog
from ._runner import run_app

log = urlog.get('gripper-demo')


def build_and_run(cfg, robot, camera, args):
    g = cfg.section('gripper_demo')
    fractions = g.get('fractions', [0.0, 0.25, 0.5, 0.75, 1.0, 0.0])
    cycles = max(int(g.get('cycles', 1)), 1)
    dwell = float(g.get('dwell_s', 1.5))

    for cycle in range(cycles):
        log.info('--- cycle %d/%d ---', cycle + 1, cycles)
        for frac in fractions:
            if not robot.gripper.go_to_fraction(frac):
                return False
            time.sleep(dwell)
    log.info('Gripper demo complete.')
    return True


def main():
    run_app('Gripper cycle demo (Robotiq 2F-85)', 'gripper', build_and_run)


if __name__ == '__main__':
    main()
