"""KEYBOARD TELEOP -- jog the end effector from the terminal, rigid or compliant.

    python -m urlab.apps.teleop [--set control_frame=tool0] [--dry-run]

One servoL loop at the admittance rate, with two ways of feeding it:

    POSITION    servoL straight to the reference. Rigid: the arm goes where it is told.
    COMPLIANCE  the software admittance law (robot/admittance.py) around the same reference --
                a spring of adjustable stiffness pulls the tool back onto it, so a push yields
                and springs back while you keep jogging.

Toggling between them never moves the arm. Leaving compliance re-references onto the pose the
spring was holding (the deflected one), so position control picks up where the tool IS rather
than snapping back onto a reference it had been pushed off. Entering it holds still for the
servo warm-up and tares there, so the law starts from zero force with the servo already engaged.

A KEY MOVES THE GOAL, NOT THE ARM. Each press shifts a goal pose by one step; the reference
chases the goal at the capped speed. The goal is never allowed to run more than `max_lead_steps`
ahead of the reference, so holding a key (the terminal's auto-repeat) gives smooth motion at the
speed cap, and letting go stops within that lead rather than working off a queue of presses.

THE FRAME the jogs are expressed in toggles between BASE (base_link axes) and TOOL (the control
frame's own current axes). Either way rotations turn about the CONTROL FRAME'S ORIGIN, which is
`control_frame` from configs/frames.yaml -- coupler_mate by default, so a rotation pivots on the
coupler tip rather than swinging it around the flange. The compliance law itself still runs in
tool0 (see AdmittanceController).

EVERY STEP IS IK-CHECKED before it is accepted (`check_ik`), so a jog that would leave the
workspace is refused where you pressed it instead of faulting the controller mid-servo.

THE FORCE GUARD CANCELS, IT DOES NOT LOCK. A trip freezes the goal onto the reference, which
throws away any motion still queued; the next key press is honoured, so you can always jog back
out of whatever you ran into.
"""

import os
import select
import sys
import termios
import tty

import numpy as np
from scipy.spatial.transform import Rotation

from .. import log as urlog
from .. import tool_frames
from ..robot.admittance import AdmittanceController
from ..robot.guard import ForceGuard
from ..transforms import fmt_pose, inverse, matrix_to_rtde, pose_error, step_toward
from ._runner import run_app

log = urlog.get('teleop')

# key -> (axis, sign). Axes 0-2 translate along x/y/z, 3-5 rotate about them.
JOG_KEYS = {
    'w': (0, +1), 's': (0, -1),
    'a': (1, +1), 'd': (1, -1),
    'q': (2, +1), 'e': (2, -1),
    'i': (3, +1), 'k': (3, -1),
    'j': (4, +1), 'l': (4, -1),
    'u': (5, +1), 'o': (5, -1),
}
AXIS_NAMES = ('x', 'y', 'z', 'rx', 'ry', 'rz')

HELP = """
  TRANSLATE   w/s  +x/-x     a/d  +y/-y     q/e  +z/-z
  ROTATE      i/k  +rx/-rx   j/l  +ry/-ry   u/o  +rz/-rz
  space       stop (drop any queued motion)
  f           toggle jog frame: BASE <-> TOOL
  c           toggle control:   POSITION <-> COMPLIANCE
  [ / ]       translational stiffness down / up     (compliance)
  { / }       rotational stiffness down / up        (compliance)
  - / =       jog step smaller / larger
  , / .       speed slower / faster
  t           tare the F/T sensor (do it touching nothing)
  r           rebase: accept the current deflection as the new reference (compliance)
  p           print status        h  this help        x  quit   (Ctrl-C also quits)
"""


def jog(T_ctrl, axis, amount, frame):
    """The control-frame pose moved by `amount` along/about one axis of `frame`.

    frame='base': the axis is a base_link axis. frame='tool': it is the control frame's own axis,
    read from T_ctrl. A ROTATION PIVOTS ON THE CONTROL FRAME'S ORIGIN either way -- the position
    is left alone and only the attitude turns, pre-multiplied for base axes and post-multiplied
    for tool axes. Pure, so the convention is testable without a robot."""
    T = np.array(T_ctrl, dtype=float)
    R = T[:3, :3]
    if axis < 3:
        d = np.zeros(3)
        d[axis] = amount
        T[:3, 3] = T[:3, 3] + (d if frame == 'base' else R @ d)
    else:
        rv = np.zeros(3)
        rv[axis - 3] = amount
        R_step = Rotation.from_rotvec(rv).as_matrix()
        T[:3, :3] = R_step @ R if frame == 'base' else R @ R_step
    return T


def split_keys(text):
    """The single-character keys in a burst of terminal input, with escape sequences dropped.

    An arrow key arrives as ESC [ A, and taken one character at a time its '[' would lower the
    stiffness. Anything starting with ESC is swallowed up to its final byte instead."""
    keys, i = [], 0
    while i < len(text):
        if text[i] != '\x1b':
            keys.append(text[i])
            i += 1
            continue
        i += 1
        if i < len(text) and text[i] in '[O':
            i += 1
            while i < len(text) and not (text[i].isalpha() or text[i] == '~'):
                i += 1
            i += 1
    return keys


class RawKeyboard:
    """Non-blocking single-key reads from the terminal. cbreak rather than raw, so Ctrl-C still
    raises KeyboardInterrupt and the log's newlines still return the carriage."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self._saved = None

    def __enter__(self):
        self._saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)

    def poll(self):
        if not select.select([self.fd], [], [], 0.0)[0]:
            return []
        return split_keys(os.read(self.fd, 64).decode(errors='ignore'))


class Teleop:
    """The loop state: reference, goal, mode, frame, step, speed and stiffness."""

    def __init__(self, cfg, robot):
        self.robot = robot
        self.arm = robot.arm
        t = cfg.section('teleop')
        name = t.get('control_frame', 'coupler_mate')
        frames = tool_frames.load_frames(cfg)
        if name not in frames:
            raise ValueError(f'control_frame {name!r} is not in {tool_frames.frames_path(cfg)}. '
                             f'Known: {", ".join(sorted(frames))}')
        self.frame_name = name
        self.T_tool0_ctrl = frames[name]
        self.T_ctrl_tool0 = inverse(self.T_tool0_ctrl)

        self.step_lin = float(t.get('step_m', 0.005))
        self.step_ang = float(t.get('step_rad', np.radians(2.0)))
        self.lead_steps = max(1.0, float(t.get('max_lead_steps', 2.0)))
        # The robot's own Cartesian caps are the ceiling; teleop runs at a fraction of them.
        self.cap_lin, self.cap_ang = self.arm.max_cart_vel, self.arm.max_cart_rot
        self.speed_frac = min(1.0, max(0.05, float(t.get('speed_fraction', 1.0))))
        self.check_ik = bool(t.get('check_ik', True))
        self.tare_on_enter = bool(t.get('tare_on_compliance', True))
        self.frame = 'tool' if t.get('start_frame', 'base') == 'tool' else 'base'

        comp = cfg.section('compliance')
        self.adm = AdmittanceController(self.arm, comp)
        self.zeta = np.asarray(comp.get('damping_ratio', [1.0] * 6), dtype=float)
        self.S_min = np.asarray(comp.get('min_stiffness', [100.0] * 3 + [1.0] * 3), dtype=float)
        self.S_max = np.asarray(comp.get('max_stiffness', [4000.0] * 3 + [50.0] * 3),
                                dtype=float)
        self.S_factor = max(1.05, float(comp.get('stiffness_factor', 1.5)))
        self.dt = 1.0 / self.adm.rate

        self.guard = ForceGuard(self.arm, cfg.section('force_guard'))
        self._guard_latched = False
        self._ik_blocked = False
        self.compliant = False
        self.T_ref = self.robot.tool0() @ self.T_tool0_ctrl       # control frame, base_link
        self.T_goal = self.T_ref.copy()

    # ---- helpers -------------------------------------------------------------------------------
    def tool0(self, T_ctrl):
        return T_ctrl @ self.T_ctrl_tool0

    def _reachable(self, T_ctrl):
        """Quiet IK check -- arm.ik() logs every refusal, and a held key would repeat it at the
        keyboard's auto-repeat rate."""
        if not self.check_ik or self.arm.dry_run:
            return True
        pose = matrix_to_rtde(self.tool0(T_ctrl))
        try:
            c = self.arm.rtde_c
            if not c.isPoseWithinSafetyLimits(pose):
                return False
            q = c.getInverseKinematics(pose, self.arm.q())
            return bool(q) and bool(c.isJointsWithinSafetyLimits(list(q)))
        except Exception:                          # noqa: BLE001 -- an IK error is a refusal
            return False

    def _set_stiffness(self, S):
        self.adm.S = np.clip(S, self.S_min, self.S_max)
        self.adm.D = self.zeta * 2.0 * np.sqrt(self.adm.M * self.adm.S)
        log.info('Stiffness: translation %s N/m, rotation %s Nm/rad.',
                 np.round(self.adm.S[:3], 0).tolist(), np.round(self.adm.S[3:], 2).tolist())

    def _commanded_ctrl(self):
        """Where the control frame is actually being commanded: the reference, displaced by the
        compliant deflection when the spring is running."""
        if not self.compliant:
            return self.T_ref.copy()
        d = self.adm.delta
        Delta = np.eye(4)
        Delta[:3, :3] = Rotation.from_rotvec(d[3:]).as_matrix()
        Delta[:3, 3] = d[:3]
        return self.tool0(self.T_ref) @ Delta @ self.T_tool0_ctrl

    def status(self):
        lin, ang = self.speed_frac * self.cap_lin, self.speed_frac * self.cap_ang
        log.info('%s control, jog frame %s, control frame %r | step %.2f mm / %.2f deg | '
                 'speed %.1f mm/s / %.1f deg/s',
                 'COMPLIANCE' if self.compliant else 'POSITION', self.frame.upper(),
                 self.frame_name, self.step_lin * 1e3, np.degrees(self.step_ang),
                 lin * 1e3, np.degrees(ang))
        log.info('  %s  %s', self.frame_name, fmt_pose(self._commanded_ctrl()))
        w = self.arm.wrench()
        log.info('  wrench  F %s N  |F| %.1f N', np.round(w[:3], 1).tolist(),
                 float(np.linalg.norm(w[:3])))
        if self.compliant:
            d = self.adm.delta
            log.info('  deflection %s mm, %s deg | stiffness %s N/m, %s Nm/rad',
                     np.round(d[:3] * 1e3, 1).tolist(), np.round(np.degrees(d[3:]), 1).tolist(),
                     np.round(self.adm.S[:3], 0).tolist(), np.round(self.adm.S[3:], 2).tolist())

    # ---- mode switches -------------------------------------------------------------------------
    def enter_compliance(self):
        """Hold still, tare with the servo engaged, and start the spring from zero deflection."""
        self.T_goal = self.T_ref.copy()
        tare = (lambda: self.arm.zero_ft(settle=False)) if self.tare_on_enter else None
        self.adm.warmup(self.tool0(self.T_ref), tare_fn=tare)      # also resets the integrator
        self.guard.reset()
        self.compliant = True
        log.info('COMPLIANCE on%s. Push the tool: it yields and springs back.',
                 ' (F/T tared)' if tare else '')
        self._set_stiffness(self.adm.S)

    def leave_compliance(self):
        """Re-reference onto the deflected pose, so position control does not snap back."""
        self.T_ref = self._commanded_ctrl()
        self.T_goal = self.T_ref.copy()
        self.adm.reset()
        self.compliant = False
        log.info('POSITION control. The reference is where the spring was holding the tool.')

    def rebase(self):
        if not self.compliant:
            log.info('Rebase only applies in compliance mode.')
            return
        self.T_ref = self._commanded_ctrl()
        self.T_goal = self.T_ref.copy()
        self.adm.reset()
        log.info('Rebased: the current deflection is now the reference.')

    # ---- input ---------------------------------------------------------------------------------
    def on_key(self, k):
        """Handle one key. Returns False to quit."""
        if k in JOG_KEYS:
            axis, sign = JOG_KEYS[k]
            amount = sign * (self.step_lin if axis < 3 else self.step_ang)
            goal = jog(self.T_goal, axis, amount, self.frame)
            lin, ang = pose_error(self.T_ref, goal)
            if (lin > self.lead_steps * self.step_lin * 1.001
                    or ang > self.lead_steps * self.step_ang * 1.001):
                return True                      # already far enough ahead: drop the repeat
            if not self._reachable(goal):
                if not self._ik_blocked:
                    log.warning('Jog %s%s refused: no IK solution there (out of reach, a joint '
                                'limit, or a singularity).', '+' if sign > 0 else '-',
                                AXIS_NAMES[axis])
                self._ik_blocked = True
                return True
            self._ik_blocked = False
            self.T_goal = goal
        elif k == ' ':
            self.T_goal = self.T_ref.copy()
        elif k == 'f':
            self.frame = 'tool' if self.frame == 'base' else 'base'
            log.info('Jog frame: %s', 'TOOL (%s axes)' % self.frame_name
                     if self.frame == 'tool' else 'BASE (base_link axes)')
        elif k == 'c':
            if self.compliant:
                self.leave_compliance()
            else:
                self.enter_compliance()
        elif k in '[]{}':
            if not self.compliant:
                log.info('(stiffness applies in compliance mode; the change is kept for it)')
            S = self.adm.S.copy()
            rows = slice(0, 3) if k in '[]' else slice(3, 6)
            S[rows] *= self.S_factor if k in ']}' else 1.0 / self.S_factor
            self._set_stiffness(S)
        elif k in '-=':
            f = 2.0 if k == '=' else 0.5
            self.step_lin = float(np.clip(self.step_lin * f, 0.00025, 0.05))
            self.step_ang = float(np.clip(self.step_ang * f, np.radians(0.1), np.radians(15.0)))
            log.info('Step: %.2f mm / %.2f deg', self.step_lin * 1e3, np.degrees(self.step_ang))
        elif k in ',.':
            self.speed_frac = float(np.clip(self.speed_frac * (1.5 if k == '.' else 1 / 1.5),
                                            0.05, 1.0))
            log.info('Speed: %.1f mm/s / %.1f deg/s (%.0f%% of the robot caps)',
                     self.speed_frac * self.cap_lin * 1e3,
                     np.degrees(self.speed_frac * self.cap_ang), self.speed_frac * 100)
        elif k == 't':
            self.arm.zero_ft(settle=False)
            log.info('F/T tared.')
        elif k == 'r':
            self.rebase()
        elif k == 'p':
            self.status()
        elif k in 'h?':
            print(HELP)
        elif k == 'x':
            return False
        return True

    # ---- the loop ------------------------------------------------------------------------------
    def cycle(self):
        """One servo period: advance the reference toward the goal, then command it."""
        self.T_ref = step_toward(self.T_ref, self.T_goal, 1.0,
                                 self.speed_frac * self.cap_lin * self.dt,
                                 self.speed_frac * self.cap_ang * self.dt)
        T_tool0 = self.tool0(self.T_ref)
        if self.compliant:
            self.adm.hold(T_tool0, self.dt)          # one admittance cycle
        else:
            self.arm.servo_l(T_tool0, self.dt, self.adm.lookahead, self.adm.gain)
        if self.guard():
            if not self._guard_latched:
                log.warning('FORCE GUARD: %s -- motion cancelled. Jog away to clear it.',
                            self.guard.tripped_by)
            self._guard_latched = True
            self.T_goal = self.T_ref.copy()
        elif self._guard_latched:
            self._guard_latched = False
            log.info('Force back under the guard limit.')

    def run(self):
        print(HELP)
        self.status()
        with RawKeyboard() as kb:
            while True:
                for k in kb.poll():
                    if not self.on_key(k):
                        return True
                self.cycle()


def build_and_run(cfg, robot, camera, args):
    if not sys.stdin.isatty():
        log.error('Teleop reads keys from the terminal; stdin is not one.')
        return False
    teleop = Teleop(cfg, robot)
    try:
        return teleop.run()
    finally:
        robot.arm.servo_stop()
        log.info('Teleop stopped; servo released.')


def main():
    run_app('Keyboard teleop: jog the end effector, position or compliance control',
            'teleop', build_and_run, with_gripper=False)


if __name__ == '__main__':
    main()
