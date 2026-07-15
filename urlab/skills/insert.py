"""Assembly insertion -- stand-off, compliant chunked insertion, and a multi-step retract.

COMPLIANCE is a SOFTWARE ADMITTANCE law (see robot/admittance.py): a virtual spring-mass-damper
with a real, finite restoring stiffness (S = 2000 N/m by default), streamed over servoL. It
reimplements the dev branch's ros2_control admittance_controller in Python -- so there is a
restoring spring again (forceMode had none; it yields freely to zero force and cannot push a part
toward its mate), but none of the controller install/load/switch machinery. The insertion ramps
the reference stand-off -> target; the arm yields to contact and springs back toward the reference,
exactly as the dev insertion did.

A position-control fallback (insert_chunked) is kept for compliance disabled.
"""

import math
import time

import numpy as np

from .. import log as urlog
from ..transforms import inverse, slerp_matrix, translation_matrix

log = urlog.get('insert')


class InsertConfig:
    def __init__(self, cfg):
        a = cfg.section('assembly')
        self.method = str(a.get('method', 'kinematic')).lower()

        t = a.get('target', {}) or {}
        self.target_frame = str(t.get('frame', 'fingertip')).lower()
        from ..transforms import xyzrpy_to_matrix
        self.T_base_target = xyzrpy_to_matrix(t.get('xyz', [0, 0, 0]), t.get('rpy', [0, 0, 0]))

        s = a.get('standoff', {}) or {}
        self.standoff_axis = np.asarray(s.get('axis', [0.0, 0.0, 1.0]), dtype=float)
        self.standoff_dist = float(s.get('distance_m', 0.05))
        self.lift_after_pick = bool(a.get('lift_after_pick', True))

        # The compliance section is parsed by AdmittanceController (mass/stiffness/damping_ratio/
        # selected_axes/rate); the insert only needs whether it is on and whether to tare first.
        c = a.get('compliance', {}) or {}
        self.compliance = c
        self.compliance_enabled = bool(c.get('enabled', True))
        self.tare_before = bool(c.get('tare_before', True))

        fg = a.get('force_guard', {}) or {}
        self.max_force = float(fg.get('max_force_n', 30.0))
        self.max_torque = float(fg.get('max_torque_nm', 5.0))

        i = a.get('insertion', {}) or {}
        self.chunk_fraction = float(i.get('chunk_fraction', 0.25))
        self.chunk_time_s = float(i.get('chunk_time_s', 2.0))
        self.chunk_settle_s = float(i.get('settle_s', 0.5))

        r = a.get('retract', {}) or {}
        self.retract_frame = str(r.get('frame', 'target')).lower()
        steps = r.get('steps')
        if steps:
            self.retract_steps = [np.asarray(st['xyz'], dtype=float) for st in steps]
        else:
            axis = np.asarray(r.get('axis', [0.0, 0.0, 1.0]), dtype=float)
            dist = float(r.get('distance_m', 0.08))
            self.retract_steps = [axis * dist] if abs(dist) > 1e-9 else []


def fingertip_target(ic, T_connector_grasp, T_tool0_fingertip):
    """Reduce the configured target to a FINGERTIP pose (what the arm actually commands).

    'fingertip' -> the pose IS the fingertip target.
    'tool0'     -> the pose is the flange target; fingertip = target @ T_tool0_fingertip.
    'connector' -> the pose is where the held connector must land. The connector is rigidly held,
                   and at grasp the fingertip was commanded to connector @ connector_grasp, so
                   T_connector_fingertip == connector_grasp: fingertip = target @ connector_grasp.
    """
    T = ic.T_base_target
    if ic.target_frame == 'fingertip':
        return T
    if ic.target_frame == 'tool0':
        return T @ T_tool0_fingertip
    if ic.target_frame == 'connector':
        return T @ T_connector_grasp
    raise ValueError(f"assembly.target.frame {ic.target_frame!r} must be "
                     "'fingertip' | 'tool0' | 'connector'")


def standoff_of(ic, T_target):
    """Back off from the target along standoff.axis, IN THE TARGET FRAME."""
    return T_target @ translation_matrix(ic.standoff_axis * ic.standoff_dist)


def insert_compliant(robot, adm, guard, ic, T_standoff_ftip, T_target_ftip):
    """Insert stand-off -> target under SOFTWARE ADMITTANCE (robot/admittance.py). The reference
    ramps to the target while the spring-damper yields to contact and restores toward the
    reference; a force-guard trip means the part SEATED (success).

    Runs as ONE continuous servoL loop -- chunk_fraction only sets the guard/progress granularity
    within it. servoL cannot be paused for a per-chunk prompt without dropping servo control, so
    the veto is the single confirm the caller puts before this whole step (not per chunk)."""
    T_t0_ft = robot.T_tool0_fingertip

    def ref(T_ft):
        return T_ft @ inverse(T_t0_ft)                  # fingertip pose -> tool0 reference for servoL

    # Tare MID-WARMUP (servo engaged, static) so the guard baseline matches the servo-active
    # reading; taring while idle leaves the tool-weight offset when the payload is not configured.
    tare = (lambda: robot.arm.zero_ft(settle=False)) if ic.tare_before else None
    adm.reset()
    adm.warmup(ref(T_standoff_ftip), tare_fn=tare)  # settle the servo (+ tare) before the guard
    guard.reset()
    n = max(1, int(math.ceil(1.0 / max(1e-6, ic.chunk_fraction))))
    log.info('Inserting under ADMITTANCE (S=%.0f N/m trans, %.0f Nm/rad rot) in %d chunk(s) at '
             '%d Hz, guarded at %.0f N / %.1f Nm.',
             adm.S[0], adm.S[3], n, int(adm.rate), ic.max_force, ic.max_torque)
    try:
        for k in range(1, n + 1):
            if guard.check():
                log.info('Contact limit reached before chunk %d/%d -- part seated.', k, n)
                return True
            a0, a1 = (k - 1) * ic.chunk_fraction, min(1.0, k * ic.chunk_fraction)
            result = adm.ramp(ref(slerp_matrix(T_standoff_ftip, T_target_ftip, a0)),
                              ref(slerp_matrix(T_standoff_ftip, T_target_ftip, a1)),
                              ic.chunk_time_s, guard)
            log.info('  chunk %d/%d (%.0f%%): %s', k, n, a1 * 100, result)
            if result == 'seated':
                log.info('Contact limit reached -- part SEATED. Stopping the insertion.')
                return True
        adm.hold(ref(T_target_ftip), ic.chunk_settle_s, guard)   # settle at the target
        log.info('Insertion complete (contact limit not reached).')
        return True
    finally:
        robot.arm.servo_stop()


def insert_chunked(robot, guard, ic, T_start, T_target, confirm=None):
    """POSITION-control fallback (compliance disabled): stand-off -> target in force-guarded
    chunks, no yielding. A guard trip mid-chunk = the part SEATED (the one place a trip is
    success); anywhere else it is a collision."""
    n = max(1, int(math.ceil(1.0 / max(1e-6, ic.chunk_fraction))))
    log.info('Inserting in %d chunk(s) of %.0f%% (POSITION control -- STIFF, no yielding), '
             'guarded at %.0f N / %.1f Nm.', n, ic.chunk_fraction * 100, ic.max_force, ic.max_torque)

    for k in range(1, n + 1):
        if guard.check():
            log.info('Contact limit already reached before chunk %d/%d -- part seated.', k, n)
            return True
        alpha = min(1.0, k * ic.chunk_fraction)
        T = slerp_matrix(T_start, T_target, alpha)
        label = f'insert chunk {k}/{n} ({alpha * 100:.0f}%)'
        if confirm and not confirm(label):
            return False

        guard.reset()
        robot.arm.add_guard(guard)
        ok = robot.move_fingertip(T, label)
        robot.arm.clear_guards()
        if not ok:
            if guard.tripped_by:
                log.info('[%s] guard tripped (%s) -- part SEATED. Stopping here (success).',
                         label, guard.tripped_by)
                return True
            return False
        time.sleep(ic.chunk_settle_s)

    log.info('Insertion complete (contact limit not reached).')
    return True


def retract(robot, ic):
    """Retract as a SEQUENCE of displacement steps, each expressed in retract.frame.

    'base'/'target' are FIXED frames -- each step is the same world direction however the arm ends
    up (what an escape path almost always wants). 'tool0'/'fingertip' MOVE with the arm, so a
    later step's axes depend on where the earlier steps left the tool."""
    if not ic.retract_steps:
        log.info('No retract steps configured; skipping.')
        return True

    n = len(ic.retract_steps)
    fixed = ic.retract_frame in ('base', 'target')
    log.info('Retracting in %d step(s) in the %s frame (%s).',
             n, ic.retract_frame.upper(), 'FIXED' if fixed else 'MOVES WITH THE ARM')

    for k, d in enumerate(ic.retract_steps, start=1):
        R = _frame_rotation(robot, ic, ic.retract_frame)
        if R is None:
            log.error("retract.frame %r is not 'base'|'target'|'tool0'|'fingertip'.",
                      ic.retract_frame)
            return False
        T_now = robot.fingertip()
        T_new = np.array(T_now, dtype=float)
        T_new[:3, 3] = T_now[:3, 3] + R @ d
        label = f'retract {k}/{n}: [{d[0] * 100:+.0f}, {d[1] * 100:+.0f}, {d[2] * 100:+.0f}] cm'
        if not robot.move_fingertip(T_new, label):
            log.error('%s failed -- retract steps are large free-space moves; check the path.',
                      label)
            return False
    return True


def _frame_rotation(robot, ic, name):
    if name == 'base':
        return np.eye(3)
    if name == 'target':
        return ic.T_base_target[:3, :3]
    if name == 'tool0':
        return robot.tool0()[:3, :3]
    if name == 'fingertip':
        return robot.fingertip()[:3, :3]
    return None
