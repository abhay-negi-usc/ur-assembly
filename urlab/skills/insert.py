"""Assembly insertion -- stand-off, compliant chunked insertion, and a multi-step retract.

Ported from CablePickAssemble (_insert_chunked, _stream_to, _enable_compliance, _retract) and the
KinematicAssembly node. The behaviour is the same; the compliance implementation is far simpler.

WHAT forceMode DELETES. The ROS insertion needed the ros2_control admittance_controller, which
had to be installed, loaded inactive, parameterised over a service, and ACTIVATED -- and
activating it DEACTIVATED the trajectory controller, so everything from that point had to stream
joint references instead of using normal moves. Activation also made the arm jump to its
reference, which caused a joint-0 velocity fault, patched with an elaborate _hold_reference dance
that pinned the reference to the arm's current pose on both sides of both switches.

forceMode is a mode of the SAME controller. It starts from where the arm is, so there is no jump
and nothing to pin. There is no controller to switch out, so normal moves keep working. The
selection vector says which axes yield; everything else stays position-controlled. The entire
_enable_compliance / _hold_reference / _switch_to_position apparatus reduces to:

    arm.force_mode(...) ; do the insertion ; arm.end_force_mode()

with end_force_mode() in a finally so a crash mid-insert cannot leave the arm compliant.
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

        c = a.get('compliance', {}) or {}
        self.compliance_enabled = bool(c.get('enabled', True))
        self.tare_before = bool(c.get('tare_before', True))
        # forceMode: which axes yield, and toward what wrench. selected_axes marks compliant axes;
        # a compliant axis regulates toward `target_wrench` (0 = "go soft, seek no force"), a rigid
        # axis holds position. The task frame is the TARGET frame, so 'z compliant' means "float
        # along the mating direction" if the target's z is the insertion axis.
        self.selected_axes = [int(bool(v)) for v in c.get('selected_axes', [1, 1, 1, 1, 1, 1])]
        self.target_wrench = [float(v) for v in c.get('target_wrench', [0.0] * 6)]
        # Speed/deviation limits per axis: compliant axes -> max speed (m/s, rad/s); rigid axes ->
        # max deviation (m, rad). Conservative by default; contact should be gentle.
        self.force_limits = [float(v) for v in c.get('force_limits', [0.05] * 3 + [0.17] * 3)]
        self.force_damping = float(c.get('damping', 0.005))
        self.force_gain_scaling = float(c.get('gain_scaling', 0.8))

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


def insert_chunked(robot, guard, ic, T_start, T_target, confirm=None):
    """Execute the stand-off -> target trajectory in fractional chunks, force-guarded.

    Chunking is what makes the insertion inspectable: contact is checked BETWEEN chunks, so a jam
    is caught after a fraction of the travel, and (with confirm) each fraction can be vetoed.
    Under force mode the arm ALSO yields within a chunk.

    The force guard's trip means OPPOSITE things by phase, and here is the one place it means
    SUCCESS: a trip during insertion is the part SEATING against its mate. Everywhere else it is a
    collision. So this function reads the guard directly rather than arming it as a move-canceller,
    and treats a trip as 'done, seated'."""
    n = max(1, int(math.ceil(1.0 / max(1e-6, ic.chunk_fraction))))
    mode = 'FORCE MODE' if robot.arm.in_force_mode else 'POSITION'
    log.info('Inserting in %d chunk(s) of %.0f%% (%s), guarded at %.0f N / %.1f Nm.',
             n, ic.chunk_fraction * 100, mode, ic.max_force, ic.max_torque)

    for k in range(1, n + 1):
        if guard.check():
            log.info('Contact limit already reached before chunk %d/%d -- part seated.', k, n)
            return True
        alpha = min(1.0, k * ic.chunk_fraction)
        T = slerp_matrix(T_start, T_target, alpha)
        label = f'insert chunk {k}/{n} ({alpha * 100:.0f}%)'
        if confirm and not confirm(label):
            return False

        # Arm the guard as a move-canceller for THIS chunk so a seat stops the arm mid-travel.
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


def enter_compliance(robot, ic, T_target):
    """Enter force mode with the TARGET frame as the task frame. No-op if compliance is disabled.

    Returns True if the arm is now compliant (or compliance was intentionally off)."""
    if not ic.compliance_enabled:
        log.info('Compliance disabled; inserting under POSITION control.')
        return True
    if ic.tare_before:
        robot.arm.zero_ft()
    robot.arm.force_mode(T_target, ic.selected_axes, ic.target_wrench, ic.force_limits,
                         damping=ic.force_damping, gain_scaling=ic.force_gain_scaling)
    log.info('Compliance ON (force mode), task frame = assembly target.')
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
