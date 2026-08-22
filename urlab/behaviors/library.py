"""The behavior library -- the robot actions every app composes its tree from.

Each class is a py_trees leaf over ONE Robot motion primitive (robot/robot.py) or one shared
skill: behaviors add sequencing, logging and the confirm gate; the robot class owns the
frames and the motion math.  Poses and joint targets are passed as CALLABLES so they are
evaluated when the leaf runs, not when the tree is built -- most targets depend on state
produced by earlier leaves.

`LIBRARY` maps short names to the classes so a tree can also be assembled from a plain
dictionary lookup: `make('move_joints', robot, q_fn, label='home')` -- see
urlab/behaviors/script.py for stringing a whole new app together this way.
"""

from .. import log as urlog
from ..apps._common import ask
from ..skills import reset as reset_skill
from .core import Action, Check

log = urlog.get('behavior')


def _call(v):
    """Late-bind a target: callables are evaluated at tick time, values pass through."""
    return v() if callable(v) else v


class MoveJoints(Action):
    """robot.move_joints: joint move, optionally with the force guard armed as a canceller."""

    def __init__(self, robot, q, label, guard=None):
        def fn():
            target = _call(q)
            return False if target is None                 else robot.move_joints(target, label=label, guard=guard)
        super().__init__(label, fn)


class MoveToPose(Action):
    """robot.move_cartesian('ptp') to a tool0 pose. `seed` is a mutable dict holding
    {'q': ...}; the solved joints are written back so consecutive moves stay on the same IK
    branch."""

    def __init__(self, robot, pose, label, seed, guard=None):
        super().__init__(label, lambda: robot.move_cartesian(
            _call(pose), interpolation='ptp', label=label, seed=seed, guard=guard))


class MoveLinear(Action):
    """robot.move_cartesian('lin'): straight-line tool0 move, optionally guarded."""

    def __init__(self, robot, pose, label, guard=None):
        super().__init__(label, lambda: robot.move_cartesian(
            _call(pose), interpolation='lin', label=label, guard=guard))


class MoveFrame(Action):
    """The fully general primitive: put any robot-attached frame ON a target pose expressed
    in any registered frame (robot.move_cartesian with frame/reference)."""

    def __init__(self, robot, target, label, frame=None, reference=None,
                 interpolation='ptp', seed=None, guard=None):
        super().__init__(label, lambda: robot.move_cartesian(
            _call(target), frame=frame, reference=reference, interpolation=interpolation,
            label=label, seed=seed, guard=guard))


class MoveRelative(Action):
    """robot.move_relative: jog by a delta expressed in any registered frame."""

    def __init__(self, robot, delta, label, expressed_in=None, interpolation='lin',
                 guard=None):
        super().__init__(label, lambda: robot.move_relative(
            _call(delta), expressed_in=expressed_in, interpolation=interpolation,
            label=label, guard=guard))


class OpenGripper(Action):
    def __init__(self, robot, label='open gripper'):
        super().__init__(label, lambda: robot.gripper.open(label))


class CloseGripper(Action):
    def __init__(self, robot, label='close gripper'):
        super().__init__(label, lambda: robot.gripper.close(label))


class ResetRobot(Action):
    """The whole-run bookend: open the gripper, guarded moveJ home (skills/reset.py)."""

    def __init__(self, robot, cfg, label='reset'):
        super().__init__(label, lambda: reset_skill.reset_robot(robot, cfg, label))


class OperatorGate(Action):
    """An UNCONDITIONAL operator prompt (ignores confirm_each_step) -- for the moments a human
    must gate regardless, e.g. right before the first contact motion. Skipped on dry runs."""

    def __init__(self, robot, prompt, label='operator gate', on_eof=True, skip=False):
        def fn():
            if robot.arm.dry_run or skip:
                return True
            if not ask(prompt, on_eof=on_eof):
                log.info('Aborted by the user at %r.', label)
                return False
            return True
        super().__init__(label, fn)


class VerifyHeld(Action):
    """Re-close + counts check that the cable is still in the fingers (skills/pick.py)."""

    def __init__(self, robot, check, where):
        def fn():
            from ..skills.pick import verify_cable_held
            return verify_cable_held(robot, check, where)
        super().__init__(f'verify held ({where})', fn)


class Warmup(Action):
    """Engage the servo stream on a reference and let the F/T transient decay (optionally
    taring mid-warmup) before any guard is trusted."""

    def __init__(self, adm, ref, tare=None, label='servo warm-up'):
        def fn():
            adm.reset()
            adm.warmup(_call(ref), tare_fn=tare)
        super().__init__(label, fn)


class AdmittanceRamp(Action):
    """Ramp the compliant reference through a list of waypoints.

    `refs` yields the waypoint list at tick time; `seg_s(A, B)` paces each leg.  A guard trip
    ('seated') stops the advance and is recorded in `result` (a dict receiving 'last_ref',
    'tripped', 'reached_end').  With `stop_on_trip=False` (the default) a trip is still a
    SUCCESS, because for contact phases it is a normal outcome, not an error."""

    def __init__(self, adm, refs, seg_s, label, guard=None, result=None, on_step=None,
                 stop_on_trip=False):
        def fn():
            rows = list(_call(refs))
            res = result if result is not None else {}
            if guard is not None:
                guard.reset()
            tripped = False
            last = rows[0]
            for A, B in zip(rows, rows[1:]):
                status = adm.ramp(A, B, seg_s(A, B), guard, on_step=on_step)
                last = B
                if status == 'seated':
                    tripped = True
                    break
            res.update(last_ref=last, tripped=tripped, reached_end=not tripped)
            return not (tripped and stop_on_trip)
        super().__init__(label, fn)


class Hold(Action):
    """Hold a compliant reference for a fixed time (settle / noise-floor dwell)."""

    def __init__(self, adm, ref, seconds, label='hold', guard=None, on_step=None):
        def fn():
            adm.hold(_call(ref), _call(seconds), guard, on_step=on_step)
        super().__init__(label, fn)


class ServoStop(Action):
    """Leave the servo loop (always safe to run; pairs with any compliant phase)."""

    def __init__(self, adm, label='servo stop'):
        super().__init__(label, lambda: adm.stop())


class Say(Action):
    """Log a message as a step -- keeps the tree readable where nothing physical happens."""

    def __init__(self, message, fn=None):
        super().__init__(message, fn if fn is not None else (lambda: None))


LIBRARY = {
    'move_joints': MoveJoints,
    'move_to_pose': MoveToPose,
    'move_linear': MoveLinear,
    'move_frame': MoveFrame,
    'move_relative': MoveRelative,
    'open_gripper': OpenGripper,
    'close_gripper': CloseGripper,
    'reset': ResetRobot,
    'operator_gate': OperatorGate,
    'verify_held': VerifyHeld,
    'warmup': Warmup,
    'ramp': AdmittanceRamp,
    'hold': Hold,
    'servo_stop': ServoStop,
    'say': Say,
    'action': Action,
    'check': Check,
}


def make(name, *args, **kwargs):
    """Build a library behavior by name: make('open_gripper', robot)."""
    try:
        cls = LIBRARY[name]
    except KeyError:
        raise KeyError(f'unknown behavior {name!r}; known: {sorted(LIBRARY)}') from None
    return cls(*args, **kwargs)
