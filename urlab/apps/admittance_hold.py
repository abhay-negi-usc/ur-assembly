"""Admittance hold demo -- port of ur_admittance_demo.

Make the arm compliant IN PLACE: it holds its current pose but yields to a push, springing back
when released. In the ROS stack this needed the ros2_control admittance_controller (a separate
C++ plugin the Python node just fed a constant reference to). Here it is UR forceMode with all six
axes compliant and zero target wrench -- one call, and the whole controller apparatus is gone.

    tare -> forceMode(all axes soft, seek 0 force) -> hold until Ctrl-C -> endForceMode
"""

import time

from .. import log as urlog
from ._runner import run_app

log = urlog.get('admittance-hold')


def build_and_run(cfg, robot, camera, args):
    a = cfg.section('admittance')
    selected = [int(bool(v)) for v in a.get('selected_axes', [1, 1, 1, 1, 1, 1])]
    limits = [float(v) for v in a.get('force_limits', [0.1] * 3 + [0.35] * 3)]
    damping = float(a.get('damping', 0.005))
    gain = float(a.get('gain_scaling', 0.8))
    hold_s = float(a.get('hold_duration_s', 0.0))       # 0 = until Ctrl-C

    robot.arm.zero_ft()
    T_task = robot.tool0()                               # comply in the tool0 frame, where it is now
    robot.arm.force_mode(T_task, selected, [0.0] * 6, limits, damping=damping, gain_scaling=gain)
    log.info('Compliance ON (forceMode). Push the arm -- it yields and springs back. '
             '%s. Ctrl-C to stop.', f'Holding {hold_s:.0f}s' if hold_s > 0 else 'Holding until Ctrl-C')

    try:
        t0 = time.monotonic()
        while hold_s <= 0 or time.monotonic() - t0 < hold_s:
            # forceMode must be refreshed periodically or the controller times it out. Re-issue at
            # ~50 Hz against the SAME task frame so the compliance is steady.
            robot.arm.force_mode(T_task, selected, [0.0] * 6, limits, damping=damping,
                                 gain_scaling=gain)
            time.sleep(0.02)
    finally:
        robot.arm.end_force_mode()
        log.info('Compliance OFF.')
    return True


def main():
    run_app('Admittance hold (compliant in place)', 'admittance_hold', build_and_run,
            with_gripper=False)


if __name__ == '__main__':
    main()
