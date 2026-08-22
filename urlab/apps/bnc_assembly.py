"""BNC ASSEMBLY -- cable_pick_estimate_assemble's pipeline with estimator_eval's estimator.

Same shape as cable_pick_estimate_assemble (scan, grasp, slip-checked lift, stand-off, then an
assemble/estimate loop that corrects the in-hand belief), but everything downstream of the grasp
is estimator_eval's:

  * ESTIMATOR: the same `estimation:` block, including `commit: argmin`. `_argmin_estimate` and
    `_landscape` are IMPORTED from estimator_eval rather than reimplemented, so the two apps
    cannot drift.
  * COLLECTION: `assembly.collection.mode` = attempts | offset_sweep | peck. offset_sweep
    commands one insertion per deliberate offset with the belief held FIXED across passes (so
    their evidence fuses exactly); peck keeps advancing past each force stop.
  * COMPLIANCE / GUARD: read from the TOP-LEVEL `compliance:` and `force_guard:` blocks, the same
    names estimator_eval uses, so tuned values copy across verbatim.
  * FINAL INSERTION: `assembly.final_insertion` -- stiffness, mass, damping, settle, dwell and the
    guard overrides in one place, each inheriting the shared value when null.

STATE VOCABULARY -- the four words this app reports progress in, in order:

    ENGAGED     the initial assembly mated the connector. Where the estimate/insert loop ends.
    SEATED      connector clocking succeeded: the bayonet cams pulled the connector home.
    LOCKED      collar clocking succeeded: the locking collar has been turned.
    ASSEMBLED   all of the above -- the connector/cable is done.

`CLOCK_STATES` is the progression and the code WALKS it rather than setting flags, so a state
cannot be reported without the step that earns it having succeeded.

CAREFUL -- 'seated' IS OVERLOADED. `AdmittanceController.ramp` returns 'seated' to mean "a guard
tripped and I stopped early"; that is the ROBOT layer's word and says nothing about the assembly
state. The clocking code reads that return into a local named `stopped`.

WHAT THIS APP ADDS BEYOND THE MATE -- two operations that run only once the connector is ENGAGED
and the operator has called the assembly successful, each with its own compliance, force guard and
speed scale (`assembly.connector_clocking`, `assembly.collar_clocking`; a failed screw and a finished
collar turn share one escape, `assembly.clocking_retract`):

  * CONNECTOR CLOCKING. An OSCILLATING screw about the connector's +X -- rock between the roll
    positions in `sweep_deg` while pushing along that same axis at a VIRTUAL target past where the
    connector can physically go, so compliance follows whatever path the bayonet cams allow. It
    rocks rather than turning further because a pin that missed its slot will not find it by
    turning harder -- it rides the rim and jams -- but it will find it by crossing back and forth
    under a steady axial load, which is held across every reversal. Progress is MEASURED (advance
    along +X) and ends the motion the moment it is reached; otherwise the legs run to `max_tries`.
    A leg stopped by the force guard is normal -- the next leg reverses from where it stopped. The
    gripper stays CLOSED throughout; there is no regrasp.
  * COLLAR CLOCKING. Only if the sweep succeeded. The gripper takes the ring AXIALLY -- fingers
    parallel to the cable, jaws closing across a diameter -- which puts tool0 ON the connector
    axis with its Z collinear, so turning the collar about that axis is a WRIST TWIST with the
    flange stationary. The socket is wall-mounted and that is what decides it: the obvious grasp
    (fingertip frame on the collar frame, approaching from the side) holds tool0 183 mm off the
    axis at +25 mm PAST the mating face and sweeps it through a 258 mm arc across the wall;
    axially it sits at -158 mm and travels 0 mm. Getting there is no longer an orbit, so the
    sequence seat-pushes where it stands, withdraws ALONG the cable, lifts the open fingers off,
    reorients in clear space, and advances back down the axis -- threading the cable through the
    open jaw. Every station is on the axis and gated against `wall_standoff_mm` before it moves.

TWO KINDS OF ANGLE, and keeping them apart is most of the arithmetic:

  * ABSOLUTE roll about the socket +X, wrt the TARGET frame -- `assembly.engage_clock_deg` and
    `connector_clocking.sweep_deg`. These are the angles obstacles and wrist limits live in.
  * ROTATIONS from the pose the connector was ENGAGED at -- what every stroke is built from, and
    what `collar_clocking.rotation_deg` / `prewind_deg` are measured in.

`engage_clock_deg` converts between them: it rolls the socket frame about its +X ONCE, and every
pose in the app is built from the rolled frame. As shipped, engage_clock_deg 0 with sweep_deg
[-75, +75] works the -75 .. +75 band: engage at 0, rock to -75, +75, -75, unwind to 0, turn the
collar to +90.

Two things are deliberately NOT rolled with it: `collar_clocking.axis_offset_mm`, a bench-measured
property of the FIXTURE that stays put while the plug turns (see `axis_offset_base`), and the
contact manifold, collected at one clock angle and describing different contact at another -- so
`insertion_mode: estimate` warns when the engage angle is non-zero.

THE BELIEF RESET at the start of connector clocking is the load-bearing idea. Once the mate is made the
connector's pose is known from a PHYSICAL CONSTRAINT -- it is at the target -- so that replaces the
estimate and the screw axis becomes the target's +X exactly instead of inheriting the accumulated
in-hand error.

WHAT IS NECESSARILY DIFFERENT from estimator_eval: the part is really picked and the true in-hand
pose is unknown, so there is no injected error, no err_before/after, and no truth to draw. Success
is the OPERATOR's call at the check (a dry run falls back to the kinematic tolerance).

Units: robot poses are metres/radians; the manifold space is mm/deg. The conversion happens only at
the observation/correction boundary.
"""

import csv as _csv
import os
from datetime import datetime

import time as _t

import numpy as np

from .. import config as urconfig
from .. import log as urlog
from .. import tool_frames
from ..log import StepRunner
from ..robot import AdmittanceController, ForceGuard
from ..skills import manifold_debug, marker_localize as mloc, reset
from ..skills import trajectory as traj
from ..skills import wiggle as wigmod
from ..skills.manifold import mats_from_vec6, vec6_from_mats
from ..skills.pick import (GraspCheck, GraspController, GraspGeometry, GraspImageRecorder,
                           GraspRecovery, belief_offset_m, connector_axis_height_m,
                           fingertip_in_connector, held_belief, offset_belief, retry_offset_x,
                           verify_cable_held)
from ..skills.solution_check import CheckedManifoldEstimator
from ..transforms import (from_cfg, inverse, matrix_to_xyzrpy, pose_error, rotate_about_axis,
                          slerp_matrix, translation_matrix, xyzrpy_to_matrix)
from ._cable import build_scanner, make_confirm
from ._common import prompts_off
from ._runner import run_app
from .cable_pick_assemble import _guarded, _pick
from .cable_pick_estimate_assemble import (_corr_to_m, _observe, _plot_run, _save_observations)
from .estimator_eval import _argmin_estimate, _landscape
from .uncertain_sampling import _retract_ref

log = urlog.get('bnc-assembly')

# The assembly state progression, walked in order (see the module docstring).
CLOCK_STATES = ('engaged', 'seated', 'locked')

# DISASSEMBLY walks the same ladder DOWNWARD and then one rung below the bottom. 'removed' is
# not a clocking state -- it means the connector is out of the socket and in the fingers, which
# is the only state from which placing it down is meaningful.
UNCLOCK_STATES = ('locked', 'seated', 'engaged', 'removed')


def _retreat_state(state, expected):
    """The state BELOW `expected`, asserting that is where we actually are -- the mirror of
    _advance_state, so a disassembly step cannot claim a rung it never undid."""
    if state != expected:
        raise AssertionError(
            f'disassembly step expected the connector to be {expected!r}, but it is {state!r}')
    return UNCLOCK_STATES[UNCLOCK_STATES.index(expected) + 1]

# The six pose axes, in the order every 6-vector in this app uses (mm, mm, mm, deg, deg, deg).
DIM_KEYS = ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')


def _num(block, key, default):
    """A number from a config block, treating an EXPLICIT null the same as absent -- null
    inherits, by this config's convention."""
    v = (block or {}).get(key)
    return float(default) if v is None else float(v)


def _path_time(lin_mm, ang_deg, v_mm_s, w_deg_s, min_s):
    """Seconds for a segment of `lin_mm` translation and `ang_deg` rotation under both caps.

    THE CAPS MUST ALREADY BE RESOLVED. A None here means a caller skipped its own defaulting,
    which once raised a TypeError partway into a clocking stroke with the gripper closed on the
    part. Module level so the null path is testable without a robot."""
    if v_mm_s is None or w_deg_s is None:
        raise ValueError(
            f'_path_time needs resolved speed caps, got v={v_mm_s!r} w={w_deg_s!r}. A null '
            f'override in the config must be defaulted by the caller before it gets here.')
    return max((lin_mm / v_mm_s) if v_mm_s > 0 else 0.0,
               (ang_deg / w_deg_s) if w_deg_s > 0 else 0.0, min_s)


def _wrap_near(angle, centre):
    """`angle` (rad) shifted by whole turns so it lands within pi of `centre`.

    Every achieved clock angle here is recovered from a measured pose -- an Euler roll or
    `dot(rotvec, axis)` -- and both return a value in (-pi, pi]. Near a half turn that sign is a
    coin flip, and the sign IS the direction the unwind orbits. The achieved angle lies inside the
    swept band, so wrapping near the band centre picks the right branch."""
    return float(angle) + 2.0 * np.pi * np.round((float(centre) - float(angle)) / (2.0 * np.pi))


def _clocking_plan(connector, collar):
    """Which post-mate maneuvers to run, from their `enabled` flags in configs/bnc_assembly.yaml.

    Both are optional and independently switchable; an absent block or key means OFF. With
    connector clocking off a run ends ENGAGED, with collar clocking off it ends SEATED.

    ONE DEPENDENCY: collar clocking requires connector clocking, and the combination is REJECTED
    rather than silently reinterpreted -- without the belief reset it has no connector pose to
    place the collar against, and locking a connector still proud of the socket is worse than not
    locking it.

    Returns (connector_on, collar_on); raises ValueError carrying the operator-facing reason.
    Pure, so the rule is testable without a robot."""
    connector_on = bool((connector or {}).get('enabled', False))
    collar_on = bool((collar or {}).get('enabled', False))
    if collar_on and not connector_on:
        raise ValueError(
            'assembly.collar_clocking.enabled is true but assembly.connector_clocking.enabled is '
            'false. Collar clocking only runs on a SEATED connector and takes the connector pose '
            'it places the collar against from connector clocking. Enable connector clocking, or disable '
            'collar clocking.')
    return connector_on, collar_on


def _advance_state(state, expected):
    """The next state after `expected`, asserting that is where we actually are -- so an edit
    that advances twice, or skips a step, fails here instead of logging 'locked' for a connector
    that was never seated."""
    if state != expected:
        raise AssertionError(f'cannot advance from {state!r}: expected {expected!r}')
    return CLOCK_STATES[CLOCK_STATES.index(state) + 1]


def _wrist3_window(arm, q_ref, span=2.0 * np.pi, tol=np.radians(0.25)):
    """The contiguous wrist_3 interval, in rad, the controller will accept around `q_ref[5]`.

    BISECTED against the controller's own joint-limit check rather than assuming the model's
    nominal +/-360: the usable wrist range is an INSTALLATION SAFETY SETTING, so an arm whose
    wrist has been restricted says so here instead of stalling part-way through a turn. Joints
    1..5 are held at `q_ref` -- exactly the configuration the turn runs in.

    Dry runs have no controller to ask, so they get the nominal +/-`span`."""
    q6 = float(q_ref[5])
    if arm.dry_run:
        return q6 - span, q6 + span

    def ok(v):
        q = list(q_ref)
        q[5] = float(v)
        return arm.joints_ok(q)

    def edge(direction):
        if not ok(q6 + direction * tol):
            return q6                                  # already at the limit
        if ok(q6 + direction * span):
            return q6 + direction * span               # nothing within a full turn
        good, bad = tol, span                          # good is inside, bad is outside
        while bad - good > tol:
            mid = 0.5 * (good + bad)
            if ok(q6 + direction * mid):
                good = mid
            else:
                bad = mid
        return q6 + direction * good

    return edge(-1.0), edge(+1.0)


def _fit_turn(theta, lo, hi):
    """`theta` shifted by whole turns into [lo, hi]. Returns (angle, clamped).

    A whole-turn shift is FREE: the collar is a body of revolution, so theta and theta +/- 360
    grip the same ring and differ only in which wrist_3 branch the turn then runs on. Clamping
    is the fallback and DOES move the grasp attitude, so it is reported separately.

    Pure, so the window arithmetic is testable without a robot."""
    theta = float(theta) + 2.0 * np.pi * np.round((0.5 * (lo + hi) - float(theta))
                                                  / (2.0 * np.pi))
    if lo <= theta <= hi:
        return float(theta), False
    return float(np.clip(theta, lo, hi)), True


class _ScrewAdvance:
    """Progress detector for a clocking screw, shaped like ForceGuard so that
    AdmittanceController.ramp can terminate on it (ramp returns 'seated' the cycle check() first
    returns True).

    That early termination is the point: once the cams have pulled the connector far enough there
    is nothing to gain by finishing the rotation.

    Advance is measured from the MEASURED arm pose, never the commanded reference. Under
    admittance the two differ by the compliant deflection, and here that deflection IS the signal
    -- the reference is deliberately a virtual target the connector cannot reach."""

    def __init__(self, robot, T_tool0_conn, T_base_conn_engaged, threshold_m):
        self.robot = robot
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)
        self._inv_engaged = inverse(np.asarray(T_base_conn_engaged, dtype=float))
        self.threshold_m = float(threshold_m)
        self.peak_m = 0.0                  # best advance seen across ALL tries (never reset)
        self.tripped_by = None

    def rebase(self, T_tool0_conn):
        """Adopt a new connector-in-gripper relationship, keeping the engaged reference frame.

        Needed after any REGRASP: the connector stays put in the socket while the gripper
        travels, so the two afterwards differ by exactly the progress made. Rebasing keeps advance
        measured from the ORIGINAL engaged pose instead of re-zeroing.

        NO CALLER IN THE CURRENT SWEEP -- the oscillating screw never lets go. Kept as the
        detector's contract for anything that does."""
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)

    def advance_m(self):
        """Connector translation along the ENGAGED connector frame's +X, in metres."""
        rel = self._inv_engaged @ (self.robot.tool0() @ self.T_tool0_conn)
        return float(rel[0, 3])

    def check(self):
        d = self.advance_m()
        self.peak_m = max(self.peak_m, d)
        if self.threshold_m > 0.0 and d >= self.threshold_m:
            self.tripped_by = (f'advance {d * 1000.0:.2f} mm >= '
                               f'{self.threshold_m * 1000.0:.2f} mm')
            return True
        return False

    def reset(self):
        self.tripped_by = None


class _AxialForce:
    """Trips on the contact force ALONG THE CONNECTOR'S OWN +X -- the insertion reaction.

    ForceGuard watches |f|, which a lateral graze raises without opposing the push at all -- so a
    magnitude limit tight enough to catch real resistance also stops on every glancing touch.
    Projecting onto the insertion axis separates the two, which lets the ENGAGE limit be set low
    without becoming a hair-trigger.

    Shaped like ForceGuard (check/reset/tripped_by) so _AnyGuard can OR it with the others.
    Persistence means the limit must hold CONTINUOUSLY for that long."""

    def __init__(self, robot, T_tool0_conn, max_force_n, persistence_s=0.0):
        self.robot = robot
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)
        self.max_force_n = float(max_force_n)
        self.persistence_s = float(persistence_s)
        self.peak_n = 0.0                  # best axial force seen; never reset, for the log
        self.tripped_by = None
        self._over_since = None

    def axial_n(self):
        """|force along the connector +X|, in newtons. Magnitude, not signed -- an inverted sign
        convention would turn this guard off rather than make it noisy."""
        T_base_tool0 = self.robot.tool0()
        T_base_conn = T_base_tool0 @ self.T_tool0_conn
        w = self.robot.arm.wrench_in(T_base_conn, T_base_tool0)
        return abs(float(w[0]))

    def check(self):
        f = self.axial_n()
        self.peak_n = max(self.peak_n, f)
        if self.max_force_n <= 0.0 or f < self.max_force_n:
            self._over_since = None
            return False
        now = _t.time()
        if self.persistence_s > 0.0:
            if self._over_since is None:
                self._over_since = now
                return False
            if now - self._over_since < self.persistence_s:
                return False
        self.tripped_by = (f'axial force {f:.1f} N >= {self.max_force_n:.1f} N'
                           + (f' for {now - self._over_since:.2f} s'
                              if self.persistence_s > 0.0 and self._over_since else ''))
        return True

    def reset(self):
        self.tripped_by = None
        self._over_since = None


class _AnyGuard:
    """ORs several ForceGuard-shaped watchdogs onto one ramp, remembering WHICH one tripped.

    The screw needs a success detector and a force limit watching the same motion, and the two
    mean opposite things -- satisfied vs jammed -- so which fired has to survive the call.
    `ramp` only reports 'seated'."""

    def __init__(self, *guards):
        self.guards = [g for g in guards if g is not None]
        self.tripped = None
        self.tripped_by = None

    def check(self):
        for g in self.guards:
            if g.check():
                self.tripped = g
                self.tripped_by = getattr(g, 'tripped_by', None)
                return True
        return False

    def reset(self):
        self.tripped = None
        self.tripped_by = None
        for g in self.guards:
            g.reset()



def _engage_report(status, s, depth_mm, det, guard, combo):
    """Print WHICH condition ended the engage, and where every OTHER one stood when it did.

    THREE THINGS CAN END THIS MOTION and they mean completely different things:

        PATH COMPLETE   the whole trajectory ran. Nothing resisted enough to stop it.
        AXIAL FORCE     the connector met the socket hard enough, along its own +X, for long
                        enough. A NORMAL end -- the bayonet screw drives the rest.
        FORCE GUARD     the general wrench limit, in any direction. A JAM.

    Reporting only the winner has cost real bench time: 'stopped on force' reads the same
    whether the axial limit was met at 2 mm or at 19.8 mm of a 20 mm path, and whether the
    general guard was idle or a hair under its own limit. So all three are printed every time,
    with the one that fired marked, and each shown against ITS OWN limit -- a bare number
    cannot be judged without the threshold it was tested against."""
    fired = {'complete': 'PATH COMPLETE', 'force': 'AXIAL FORCE LIMIT',
             'guard': 'GENERAL FORCE GUARD'}.get(status, status.upper())
    meaning = {
        'complete': 'the full path ran without meeting the axial limit',
        'force': 'a NORMAL end -- the clocking screw drives the rest',
        'guard': 'a JAM: the general wrench limit, not the axial one',
    }.get(status, '')
    say = log.warning if status == 'guard' else log.info

    def mark(name):
        return '>>' if name == status else '  '

    pct = 100.0 * s['elapsed_s'] / max(s['duration_s'], 1e-9)
    say('  --- ENGAGE ENDED: %s --- %s', fired, meaning)
    say('   %s path complete   %.2f of %.2f s (%.0f%%) -- drove %.1f of %.1f mm',
        mark('complete'), s['elapsed_s'], s['duration_s'], pct, s['driven_mm'], s['total_mm'])
    if s['axial_limit_n'] > 0:
        say('   %s axial force     %.1f N of %.1f N limit (peak %.1f N%s)',
            mark('force'), s['axial_n'], s['axial_limit_n'], s['axial_peak_n'],
            ', persistence %.2f s' % s['axial_persist_s'] if s['axial_persist_s'] > 0 else '')
    else:
        say('   %s axial force     NO LIMIT SET (peak %.1f N seen) -- this condition can never '
            'end the engage', mark('force'), s['axial_peak_n'])
    if guard is not None and getattr(guard, 'enabled', False):
        say('   %s force guard     |F| %.1f N of %.1f N (peak %.1f) | tau %.2f Nm of %.2f Nm '
            '(peak %.2f)', mark('guard'), s['force_n'], guard.max_force, guard.peak_force,
            s['torque_nm'], guard.max_torque, guard.peak_torque)
    else:
        say('   %s force guard     DISABLED -- nothing was watching for a jam', mark('guard'))
    say('      depth           %+.2f mm past the recorded mate', depth_mm)
    if status != 'complete' and combo is not None and combo.tripped_by:
        say('      tripped by      %s', combo.tripped_by)


def build_and_run(cfg, robot, camera, args):
    a = cfg.section('assembly')

    # The ESTIMATOR first: a missing/stale manifold CSV or a bad metric key must fail before
    # the robot moves, not after a part is in the fingers.
    estimator = CheckedManifoldEstimator(cfg.section('estimation'))
    commit = str(cfg.get_path('estimation.commit', 'aggregator')).strip().lower()
    if commit not in ('aggregator', 'argmin'):
        log.error("estimation.commit %r must be 'aggregator' or 'argmin'.", commit)
        return False
    log.info('Estimator: commit %s, dims %s, manifold %s.', commit, estimator.estimate_dims,
             cfg.get_path('estimation.manifold_csv'))

    scanner, _detector, _est = build_scanner(cfg, robot, camera)
    geom, check = GraspGeometry(cfg), GraspCheck(cfg)
    recovery, grasp = GraspRecovery(cfg), GraspController(cfg)
    recorder = GraspImageRecorder(cfg)
    confirm = make_confirm(cfg)

    # COMPLIANCE + GUARD from the top-level blocks -- same names estimator_eval reads, so a
    # tuned pair of blocks can be copied between the two configs without translation.
    adm = AdmittanceController(robot.arm, cfg.section('compliance'))
    guard_shared = ForceGuard(robot.arm, cfg.section('force_guard'))
    tare = (lambda: robot.arm.zero_ft(settle=False)) \
        if bool(cfg.get_path('compliance.tare_before', True)) else None
    settle_shared = float(cfg.get_path('compliance.settle_s', 0.5))
    hold_shared = float(cfg.get_path('compliance.hold_after_insertion_s', 0.0))

    # ---- FINAL INSERTION: one grouped block, each key inheriting the shared value when null.
    fi = a.get('final_insertion', {}) or {}
    fi_on = bool(fi.get('enabled', True))
    fi_settle = None if fi.get('settle_s') is None else float(fi['settle_s'])
    fi_hold = None if fi.get('hold_after_insertion_s') is None \
        else float(fi['hold_after_insertion_s'])
    # Pacing, dwell and noise for the COMMIT. Zero-noise by default: the jitter exists to gather
    # varied contact while PROBING and has no place in the attempt meant to seat.
    fi_v = None if fi.get('speed_translation_mm_s') is None \
        else float(fi['speed_translation_mm_s'])
    fi_wr = None if fi.get('speed_rotation_deg_s') is None \
        else float(fi['speed_rotation_deg_s'])
    fi_pause = float(fi.get('pause_s', 0.0) or 0.0)
    # PRELOAD -- the press, on the COMMIT only. Drives the commit's reference this far PAST the
    # mate along the connector's +X; the part stops at the mate, the reference keeps going, and
    # the spring turns the leftover travel into contact force. 0 disables.
    #
    # It must NOT live as a trajectory row: uncertain_sampling anchors that path, so the contact
    # map was collected with the press removed while this app drove it -- every observation 10 mm
    # deeper than anything the map held. Named here it survives anchoring and lands only on the
    # insertion meant to SEAT.
    #
    # It buys the break-in SPIKE, not a sustained press: it decays as the spring yields over
    # D/S ~ 6-9 s, and what it can HOLD is only stiffness x preload.
    fi_preload_mm = float(fi.get('preload_mm', 10.0) or 0.0)
    fin = fi.get('trajectory_noise', {}) or {}
    fi_noise_on = bool(fin.get('enabled', False))
    fi_noise_std = [float(v) for v in (fin.get('std') or [0.0] * 6)]
    fi_noise_w = max(1, int(fin.get('smooth_window', 10)))
    if fi_noise_on and len(fi_noise_std) != 6:
        log.error('final_insertion.trajectory_noise.std must have 6 entries.')
        return False
    for nm, v in (('speed_translation_mm_s', fi_v), ('speed_rotation_deg_s', fi_wr),
                  ('pause_s', fi_pause), ('preload_mm', fi_preload_mm)):
        if v is not None and v < 0:
            log.error('assembly.final_insertion.%s must be >= 0 (got %.2f).', nm, v)
            return False
    adm_final, guard_final = adm, None
    if fi_on:
        comp_final = dict(cfg.section('compliance'))
        for k in ('stiffness', 'mass', 'damping_ratio'):
            if fi.get(k) is not None:
                comp_final[k] = [float(v) for v in fi[k]]
        adm_final = AdmittanceController(robot.arm, comp_final)
        gsec = dict(cfg.section('force_guard'))
        over_g = {k: fi[k] for k in ('max_force_n', 'max_torque_nm', 'persistence_s')
                  if fi.get(k) is not None}
        guard_final = ForceGuard(robot.arm, {**gsec, **over_g}) if over_g else None
        log.info('Final insertion ON: stiffness %s, guard %.0f N.', comp_final.get('stiffness'),
                 float(over_g.get('max_force_n', gsec.get('max_force_n', 0.0))))
        if fi_preload_mm > 0:
            s_ins = max(float(v) for v in (comp_final.get('stiffness') or [0.0])[:3])
            log.info('   PRELOAD %.1f mm past the mate on the commit (probing passes do NOT '
                     'press). The spring holds %.1f N of it at %.0f N/m; the contact peak is a '
                     'transient several times larger.',
                     fi_preload_mm, s_ins * fi_preload_mm / 1000.0, s_ins)

    # ---- INSERTION MODE ----------------------------------------------------------------------
    # ESTIMATE  insert, observe, fit the contact manifold, correct the in-hand belief, repeat.
    #           Needs a map covering the contact states production actually visits.
    # ENGAGE    no estimation: drive the trajectory home with an optional oscillation superimposed,
    #           letting compliance find the mate. Needs no map, so it is immune to every failure
    #           mode of the estimator.
    ins_mode = str(a.get('insertion_mode') or 'estimate').strip().lower()
    if ins_mode not in ('estimate', 'engage'):
        log.error("assembly.insertion_mode %r must be 'estimate' or 'engage'. The old\n"
                  "standalone 'wiggle' is retired: engage does the same oscillation, but\n"
                  "superimposed on the trajectory rather than replacing it -- set\n"
                  "assembly.engage.amplitude instead.", ins_mode)
        return False
    # ---- ENGAGE: the trajectory-following insertion ------------------------------------------
    en = a.get('engage', {}) or {}
    en_amp = [float((en.get('amplitude') or {}).get(d, 0.0)) for d in DIM_KEYS]
    en_frq = [float((en.get('frequency_hz') or {}).get(d, 0.0)) for d in DIM_KEYS]
    en_rate = float(en.get('sample_rate_hz', 25.0))
    en_pre_mm = float(en.get('preload_mm', 0.0) or 0.0)
    en_speed = en.get('speed_mm_s')
    en_fmax = float(en.get('max_axial_force_n', 0.0) or 0.0)
    en_fpers = float(en.get('persistence_s', 0.0) or 0.0)
    # OSCILLATION SPEED CAP. speed_mm_s paces the PATH; the oscillation adds its own velocity
    # (amplitude x 2*pi*f) on top. The cap dilates the oscillation CLOCK only, leaving amplitude
    # and the frequency ratio untouched, so it costs wall-clock and nothing else. null = uncapped.
    #
    # ONE WIGGLE IMPLEMENTATION (urlab/skills/wiggle.py), with parameters from
    # configs/wiggle_sampling.yaml unless engage overrides them -- a second copy here is how this
    # app and the sampler would drive different excitations while both claiming to be "the wiggle",
    # and observations from one would stop being comparable with a map from the other.
    _eblk, _esrc = wigmod.from_shared(cfg, en.get('wiggle'),
                                      en.get('wiggle_from', 'wiggle_sampling.yaml'), 'engage')
    en_wig = None
    try:
        en_wig = wigmod.Wiggle.from_cfg(_eblk, 'engage')
        if en_wig is not None:
            en_wig.validate(rate_hz=en_rate,
                            cap_v=_eblk.get('max_speed_mm_s')
                            or en.get('max_oscillation_speed_mm_s'),
                            cap_w=_eblk.get('max_rotation_deg_s')
                            or en.get('max_oscillation_rotation_deg_s'))
            en_amp = list(en_wig.amp)
            en_frq = list(en_wig.frq)
            log.info('ENGAGE wiggle: %s (parameters from %s).', en_wig.describe(), _esrc)
    except wigmod.WiggleError as exc:
        log.error('%s', exc)
        return False
    en_scale, en_pv, en_pw, en_orbit = traj.wiggle_time_scale(
        en_amp, en_frq, en.get('max_oscillation_speed_mm_s'),
        en.get('max_oscillation_rotation_deg_s'))
    if ins_mode == 'engage':
        if en_rate <= 0:
            log.error('assembly.engage.sample_rate_hz must be > 0.')
            return False
        if en_pre_mm < 0:
            log.error('assembly.engage.preload_mm must be >= 0.')
            return False
        if en_speed is not None and float(en_speed) <= 0:
            log.error('assembly.engage.speed_mm_s must be > 0 when set (null = the assemble '
                      'phase scale).')
            return False
        for i, d in enumerate(DIM_KEYS):
            if abs(en_amp[i]) > 0.0 and en_frq[i] <= 0.0:
                log.error('assembly.engage.frequency_hz.%s must be > 0 when its amplitude is '
                          '%.3f (an amplitude with no frequency is a constant offset, which '
                          'belongs in the trajectory).', d, en_amp[i])
                return False
        # EFFECTIVE frequency: the speed cap dilates the oscillation clock, so the raw value
        # would reject a configuration that samples perfectly well.
        f_live = [en_frq[i] * en_scale for i in range(6) if abs(en_amp[i]) > 0.0]
        if f_live and en_rate < 4.0 * max(f_live):
            log.error('assembly.engage.sample_rate_hz %.1f Hz is too coarse for a %.2f Hz '
                      'component (need >= 4x) -- the sampled sine would alias.',
                      en_rate, max(f_live))
            return False
        if en_fmax <= 0:
            log.warning('assembly.engage.max_axial_force_n is 0 -- the engage will run the whole '
                        'trajectory no matter how hard it presses.')

    # ---- ENGAGE CLOCK ANGLE: the roll the connector is MATED at ------------------------------
    # Parsed here, ahead of the clocking blocks, because the connector sweep is stated in ABSOLUTE
    # roll positions about the socket +X and needs this to convert them into rotations from the
    # engaged pose. The frame it rolls (T_base_tconn) is built where the target is loaded.
    eng_clock = np.radians(_num(a, 'engage_clock_deg', 0.0))
    R_clock = xyzrpy_to_matrix([0.0, 0.0, 0.0], [eng_clock, 0.0, 0.0])

    # ---- CLOCKING (post-mate): CABLE clocking, then COLLAR clocking --------------------------
    # Both run only after a mate the operator called successful, each with its OWN compliance and
    # force guard. The guard override is not optional in practice: force_guard: is tuned for a
    # light probing insertion (5 N) and a press-and-twist exceeds that on the first cycle.
    cc = a.get('connector_clocking', {}) or {}
    cl = a.get('collar_clocking', {}) or {}
    if a.get('cable_clocking') is not None:
        # NOT a warning: `enabled` defaults to False, so a config still on the old name would read
        # as "the maneuver is off" and the run would end ENGAGED with nothing said about why.
        log.error('assembly.cable_clocking has been renamed to assembly.connector_clocking. '
                  'Rename the block -- leaving it under the old name would silently disable the '
                  'maneuver and end the run ENGAGED.')
        return False

    try:
        cc_on, cl_on = _clocking_plan(cc, cl)
    except ValueError as exc:
        log.error('%s', exc)
        return False
    log.info('Post-mate clocking: connector clocking %s, collar clocking %s (a run therefore ends %s '
             'at best).', 'ON' if cc_on else 'off', 'ON' if cl_on else 'off',
             'LOCKED/ASSEMBLED' if cl_on else ('SEATED' if cc_on else 'ENGAGED'))
    # ---- THE CONNECTOR SWEEP: absolute roll POSITIONS, visited in turn, one per try ---------------
    # `sweep_deg` is roll angles about the socket +X stated WRT THE TARGET FRAME, not wrt where
    # the connector was engaged -- the reachable band is a property of the FIXTURE, so stating it
    # absolutely means it does not have to be re-derived when engage_clock_deg moves.
    #
    # `rotation_deg` is the legacy single relative stroke and still works: it becomes a one-entry
    # sweep at engage_clock_deg + rotation_deg, so the two are one code path rather than two.
    _sw = cc.get('sweep_deg')
    if _sw is None:
        cc_sweep = [eng_clock + np.radians(_num(cc, 'rotation_deg', 90.0))]
    else:
        try:
            cc_sweep = [np.radians(float(v)) for v in _sw]
        except (TypeError, ValueError):
            log.error('assembly.connector_clocking.sweep_deg must be a list of numbers (roll angles '
                      'in deg wrt the target frame), got %r.', _sw)
            return False
        if not cc_sweep:
            log.error('assembly.connector_clocking.sweep_deg is empty -- give it at least one roll '
                      'position, or delete it to use the legacy rotation_deg stroke.')
            return False
    # ROTATIONS FROM THE ENGAGED POSE -- what the strokes are built from. 0.0 joins the span
    # because the arm starts there.
    cc_legs = [th - eng_clock for th in cc_sweep]
    cc_lo, cc_hi = min(cc_legs + [0.0]), max(cc_legs + [0.0])
    # Band centre, the branch every measured clock angle is wrapped onto (_wrap_near). A band
    # wider than a full turn has no unambiguous branch, so it is refused rather than read wrong.
    cc_mid = 0.5 * (cc_lo + cc_hi)
    if cc_hi - cc_lo > 2.0 * np.pi:
        log.error('assembly.connector_clocking.sweep_deg spans %.0f deg from the engaged roll '
                  '(%+.1f deg) -- more than one turn, so a measured clock angle cannot be told '
                  'from itself plus 360. Narrow the sweep or move assembly.engage_clock_deg.',
                  np.degrees(cc_hi - cc_lo), np.degrees(eng_clock))
        return False
    cc_push_m = _num(cc, 'push_mm', 5.0) / 1000.0
    cc_need_m = _num(cc, 'success_advance_mm', 5.0) / 1000.0
    cc_tries = max(1, int(_num(cc, 'max_tries', 3)))
    cc_settle = _num(cc, 'settle_s', settle_shared)
    cc_hold = _num(cc, 'hold_after_s', 0.0)
    cc_open_after = bool(cc.get('open_gripper_after', True))
    cl_off_m = _num(cl, 'collar_offset_mm', 25.0) / 1000.0
    cl_rot = np.radians(_num(cl, 'rotation_deg', 90.0))
    cl_push_m = _num(cl, 'push_mm', 0.0) / 1000.0
    # GRASP CLOCK ANGLE: the ABSOLUTE roll about the socket +X to take the collar at, wrt the
    # TARGET frame -- the same convention connector_clocking.sweep_deg uses. null = the engaged roll.
    # The turn then runs from here to here + rotation_deg. (Replaces prewind_deg, which existed
    # only because the old radial approach reached the ring by orbiting from wherever the sweep
    # left the arm; an axial approach is placed in free space, so the angle is simply stated.)
    _gc = cl.get('grasp_clock_deg')
    cl_grasp_clock = None if _gc is None else np.radians(float(_gc))
    # RETRACT: how far to back straight off along the TARGET connector -X, from wherever the
    # seat push left the arm, before the gripper is pitched onto the axis. Replaces
    # retreat_mm, which measured the reorient station BACKWARD FROM THE COLLAR and so moved
    # with collar_offset_mm; this is a plain relative back-off from the pose we are in.
    cl_retract_m = _num(cl, 'retract_mm', 300.0) / 1000.0
    # RETREAT: how far BEHIND the approach station, along the connector -X, to reorient. Away
    # from the wall, and the furthest-from-it station in the whole maneuver.
    cl_retreat_m = _num(cl, 'retreat_mm', 100.0) / 1000.0
    # WALL STANDOFF: the largest tool0 station along the connector +X any planned pose may take,
    # mm in the TARGET frame (+X points INTO the wall the socket is mounted on, 0 = the mating
    # face). Checked pre-motion against every station. null = no check.
    cl_wall_mm = cl.get('wall_standoff_mm')
    cl_wall_mm = None if cl_wall_mm is None else float(cl_wall_mm)
    cl_settle = _num(cl, 'settle_s', settle_shared)
    cc_v = None if cc.get('speed_translation_mm_s') is None \
        else float(cc['speed_translation_mm_s'])
    cc_w = None if cc.get('speed_rotation_deg_s') is None else float(cc['speed_rotation_deg_s'])
    cl_v = None if cl.get('speed_translation_mm_s') is None \
        else float(cl['speed_translation_mm_s'])
    cl_w = None if cl.get('speed_rotation_deg_s') is None else float(cl['speed_rotation_deg_s'])
    # TARE: default OFF, unlike the insertion. The arm stands in a MATED, loaded pose when
    # clocking starts, so re-zeroing the F/T there would define the mate load as zero and the
    # guard would only see force ADDED by the screw. Off, it sees absolute force, at the cost of
    # the spring yielding to the standing load during warm-up.
    cc_tare = tare if bool(cc.get('tare_before', False)) else None
    cl_tare = tare if bool(cl.get('tare_before', False)) else None
    cl_tilt_deg = _num(cl, 'max_offaxis_tilt_deg', 5.0)
    # WRIST-3 MARGIN: how far inside each end of the usable wrist_3 range the collar
    # turn must stay. The turn is servoed under admittance, so the arm does not track
    # the reference exactly and a start placed hard against the limit can still walk
    # into it.
    cl_w3_margin = np.radians(_num(cl, 'wrist3_margin_deg', 5.0))
    cl_axis_off = [float(v) for v in (cl.get('axis_offset_mm') or [0.0, 0.0, 0.0])]
    if len(cl_axis_off) != 3:
        log.error('assembly.collar_clocking.axis_offset_mm must have 3 entries (connector-frame '
                  'xyz, mm), got %r.', cl.get('axis_offset_mm'))
        return False
    tv = a.get('tug_verify', {}) or {}
    tv_on = bool(tv.get('enabled', True))
    tv_force = _num(tv, 'pull_force_n', 3.0)
    tv_time = _num(tv, 'pull_time_s', 3.0)
    tv_thresh_m = _num(tv, 'displacement_threshold_mm', 3.0) / 1000.0
    tv_extract_m = _num(tv, 'extraction_distance_mm', 50.0) / 1000.0
    # The tug gets its OWN spring: the pull only works if the spring offset
    # (pull_force_n / stiffness) EXCEEDS displacement_threshold_mm. Under that, an unlocked
    # connector could never move far enough to show and the check goes vacuous. Null inherits.
    _tv_phys = dict(cfg.section('compliance'))
    for _k in ('stiffness', 'mass', 'damping_ratio'):
        if tv.get(_k):
            _tv_phys[_k] = tv[_k]
    adm_tug = AdmittanceController(robot.arm, _tv_phys) if tv_on else None
    ground_z = float(cfg.get_path('ground_plane.z_m', -0.76))
    # ---- DISASSEMBLY (optional) -----------------------------------------------------------
    dis = a.get('disassembly', {}) or {}
    dis_on = bool(dis.get('enabled', False))
    dis_unlock = bool(dis.get('unlock_collar', True))
    dis_extract_m = _num(dis, 'extract_mm', 60.0) / 1000.0
    dis_place = dis.get('place', {}) or {}
    dis_place_on = bool(dis_place.get('enabled', True))
    dis_clear_m = _num(dis_place, 'clearance_mm', 25.0) / 1000.0
    dis_rise_m = _num(dis_place, 'retreat_mm', 100.0) / 1000.0
    # CYCLES: repeat the whole localize -> pick -> assemble -> disassemble pass. Only
    # meaningful with disassembly ON, because only disassembly puts the cell back into a
    # state a second pass can start from -- so it is CLAMPED to 1 rather than silently
    # looping a run that would try to insert an already-mated connector.
    n_cycles = max(1, int(_num(dis, 'cycles', 1)))
    if n_cycles > 1 and not (dis_on and dis_place_on):
        log.warning('assembly.disassembly.cycles = %d but %s -- a second pass has nothing to '
                    'pick and nowhere to start from. Running ONE cycle.', n_cycles,
                    'disassembly is off' if not dis_on else 'disassembly.place is off')
        n_cycles = 1
    cycle_ok = []
    if dis_on and not cc_on:
        # Nothing to unwind and, more to the point, no ENGAGED pose to unwind FROM: the
        # clocking maneuvers are what establish the frames disassembly reverses.
        log.error('assembly.disassembly.enabled needs connector_clocking enabled -- the '
                  'disassembly reverses the clocking, and without it there is nothing to '
                  'reverse and no engaged pose to reverse from.')
        return False

    sp = cl.get('seat_push', {}) or {}
    sp_on = bool(sp.get('enabled', True))
    sp_force = _num(sp, 'force_n', 5.0)
    sp_persist = _num(sp, 'persistence_s', 1.0)
    sp_travel_m = _num(sp, 'max_travel_mm', 15.0) / 1000.0
    # A dedicated guard: the push SUCCEEDS by tripping it (force held for the persistence), so
    # its limits are the push spec, not the global safety limits -- those stay on guard_shared.
    guard_push = ForceGuard(robot.arm, dict(cfg.section('force_guard'),
                                            max_force_n=sp_force,
                                            persistence_s=sp_persist)) if sp_on else None
    if cc_on and cl_on and cc_open_after:
        log.error('assembly.collar_clocking needs connector_clocking.open_gripper_after FALSE: '
                  'the seat push presses with the pads still closed on the connector, so the '
                  'sweep must hand it over held. The release happens after the push, before the '
                  'retreat.')
        return False
    if cc_on and cc_tries > 1 and len(cc_sweep) < 2:
        # Not fatal (one position is a legal legacy sweep) but the extra tries become
        # zero-rotation no-ops.
        log.warning('assembly.connector_clocking: max_tries is %d but the sweep has only ONE roll '
                    'position (%+.1f deg). Legs 2..%d have nothing to turn. Give sweep_deg a '
                    'second position to rock across the slot.',
                    cc_tries, np.degrees(cc_sweep[0]), cc_tries)
    if cc_on and cc_need_m <= 0.0:
        log.error('assembly.connector_clocking.success_advance_mm must be > 0 (got %.2f) -- a zero '
                  'early-out threshold trips on the first servo cycle, ending the stroke before '
                  'it has turned anything.', cc_need_m * 1000.0)
        return False

    def _clock_physics(block, name):
        """(AdmittanceController, ForceGuard) for one clocking maneuver -- every compliance key
        inheriting compliance: and every guard key force_guard: when absent or null, the same
        inheritance rule final_insertion uses."""
        comp = dict(cfg.section('compliance'))
        for k in ('stiffness', 'mass', 'damping_ratio'):
            if block.get(k) is not None:
                comp[k] = [float(v) for v in block[k]]
        gsec = dict(cfg.section('force_guard'))
        over = {k: block[k] for k in ('max_force_n', 'max_torque_nm', 'persistence_s')
                if block.get(k) is not None}
        # NOT `enabled` -- that is the MANEUVER's own switch in this block. Reusing it would
        # make "run the screw with no guard" inexpressible.
        if block.get('force_guard_enabled') is not None:
            over['enabled'] = bool(block['force_guard_enabled'])
        g = ForceGuard(robot.arm, {**gsec, **over})
        log.info('%s ON: stiffness %s, guard %.1f N / %.1f Nm.', name, comp.get('stiffness'),
                 g.max_force, g.max_torque)
        return AdmittanceController(robot.arm, comp), g

    adm_cc = guard_cc = adm_cl = guard_cl = None
    if cc_on:
        adm_cc, guard_cc = _clock_physics(cc, 'Connector clocking')
    if cl_on:
        adm_cl, guard_cl = _clock_physics(cl, 'Collar clocking')
    adm_en = guard_en = None
    if ins_mode == 'engage':
        # Its OWN compliance and guard, same override schema as the clocking blocks: the engage
        # wants neither the probing block's stop-on-contact nor a clocking block's jam limit.
        adm_en, guard_en = _clock_physics(en, 'Engage insertion')

    # ---- Target from the SHARED catalogue (the same record estimator_eval assembles to) ----
    tname = a.get('target_frame')
    targets = tool_frames.load_targets(cfg)
    if not tname or tname not in targets:
        log.error('assembly.target_frame %r needs a targets: entry in %s.',
                  tname, tool_frames.frames_path(cfg))
        return False
    T_base_socket = targets[tname]

    # ---- WHERE THE TARGET COMES FROM: assembly.target_source -----------------------------------
    # kinematic  the recorded targets: pose above. It is a reading taken by hand-guiding to a good
    #            mate, and it is only true while nothing moves -- the fixture, the robot base, the
    #            calibration. The 2026-08 hose campaign lost a session to exactly that drift.
    # visual     fiducials bolted around the socket, calibrated once by apps/marker_calibration.
    #            Each marker carries the target in ITS OWN frame, so the rig travels WITH the
    #            fixture: unbolt it, move it, and the run still finds it.
    #
    # THE VISUAL ANSWER IS NOT MORE ACCURATE THAN THE RECORD IT WAS CALIBRATED FROM -- the rig was
    # solved against this same targets: entry, so it inherits every error that reading carried.
    # What it adds is INVARIANCE, and a live check: markers that disagree with each other, or a
    # fixture that has moved further than max_shift_mm, are things the kinematic path cannot
    # notice at all.
    tgt_source = str(a.get('target_source') or 'kinematic').strip().lower()
    if tgt_source not in ('kinematic', 'visual'):
        log.error("assembly.target_source must be 'kinematic' or 'visual', got %r.",
                  a.get('target_source'))
        return False
    vt = a.get('visual_target', {}) or {}
    vt_rig = None
    if tgt_source == 'visual':
        # VALIDATED HERE, at parse time, not after the arm has already homed and driven to a view
        # pose: a missing rig is a config typo, and finding out about it three moves in wastes a
        # run and leaves the operator guessing which name was wrong.
        try:
            rigs = tool_frames.load_marker_rigs(cfg)
        except ValueError as exc:
            log.error('assembly.target_source is visual but %s', exc)
            return False
        vt_rig = rigs.get(tname)
        if vt_rig is None:
            log.error('assembly.target_source is visual but %s has no marker_rigs: entry for %r '
                      '(rigs present: %s). Calibrate one with urlab.apps.marker_calibration.',
                      tool_frames.frames_path(cfg), tname,
                      ', '.join(sorted(rigs)) or 'none')
            return False
        log.info('Target source: VISUAL -- rig %r, markers %s (%s mm).', tname,
                 ', '.join(str(m) for m in sorted(vt_rig['markers'])),
                 ', '.join('%.1f' % (m['size_m'] * 1000.0)
                           for _i, m in sorted(vt_rig['markers'].items())))
    else:
        log.info('Target source: KINEMATIC -- the recorded targets: pose of %r.', tname)

    # ---- ENGAGE CLOCK ANGLE: the roll the connector is MATED at ------------------------------
    # A BNC is free about its own axis until the bayonet pins pick up, so the clock angle it is
    # ENGAGED at is a free parameter. Applied ONCE, here, as a roll of the FRAME: every pose in
    # the app is built from T_base_tconn, so rolling it rolls the trajectory, stand-off,
    # observations, belief reset, screw axis and collar frame rigidly together. Rolling any one of
    # them instead would put the part at one clock angle and its reference at another.
    #
    # It does not place the sweep -- sweep_deg states its positions against the target frame -- so
    # this only decides where the connector starts, and which way the first leg turns. +X is
    # unchanged by a roll about +X, so the insertion axis, push, retract legs and every depth
    # reading are identical at any value.
    T_base_tconn = T_base_socket @ R_clock
    log.info('Clock angles about the socket +X: engage at %+.1f deg, connector sweep visits %s deg '
             'in turn (up to %d leg%s) -- the run works the %+.1f .. %+.1f deg band about the '
             'declared roll of %r.',
             np.degrees(eng_clock), [round(float(np.degrees(t)), 1) for t in cc_sweep],
             cc_tries, '' if cc_tries == 1 else 's',
             np.degrees(eng_clock + cc_lo), np.degrees(eng_clock + cc_hi), tname)
    if abs(eng_clock) > 1e-9:
        if ins_mode == 'estimate':
            # Rolling the frame keeps observations near identity, so they land in the map's
            # coordinates -- but the bayonet slots and keyway are NOT bodies of revolution, so the
            # contact they describe at another clock angle is different contact.
            log.warning('  insertion_mode is ESTIMATE and the clock angle is not 0. The manifold '
                        'was collected at ONE clock angle; the observations will be expressed in '
                        'the rolled frame but the CONTACT they describe is a different part of '
                        'the socket. Re-collect the map at this clock angle, or engage with '
                        'insertion_mode: engage (which matches nothing and is unaffected).')
    # FRAME FOR THE POST-ENGAGEMENT MANEUVERS -- connector clocking, collar clocking, the escape's
    # target-frame leg and the tug all build their axes and stations from T_clk. 'target' binds it
    # to the recorded socket pose; 'believed' rebinds it, when clocking starts, to the
    # estimator-corrected in-hand belief. The ENGAGEMENT itself always uses the target frame.
    pe_frame = str(a.get('post_engage_frame') or 'target').lower()
    if pe_frame not in ('target', 'believed'):
        log.error("assembly.post_engage_frame must be 'target' or 'believed', got %r.",
                  a.get('post_engage_frame'))
        return False
    T_clk = T_base_tconn
    csv_in = urconfig.resolve(cfg, a.get('trajectory_csv', 'assembly_trajectory.csv'))
    mats = traj.load_csv(csv_in, angles_deg=bool(a.get('trajectory_angles_deg', False)))
    dense = traj.resample(mats, float(a.get('translational_resolution_m', 0.001)),
                          float(a.get('rotational_resolution_deg', 1.0)))

    # ---- ANCHORING: the path ENDS on the recorded mate ---------------------------------------
    # Normalises the trajectory so its last row lands exactly on the target, whatever that row
    # says -- the same thing uncertain_sampling does when building the contact map. A no-op for a
    # conforming CSV, and what keeps a non-conforming one (the shipped CSV once carried a +10 mm
    # press in its last row) from silently putting every observation deeper than the map goes.
    # A press belongs in final_insertion.preload_mm, which applies to the COMMIT alone.
    T_base_targetobj = T_base_tconn @ inverse(mats[-1])
    T_base_commit = T_base_targetobj @ translation_matrix([fi_preload_mm / 1000.0, 0.0, 0.0])

    def _anchor_target(T_socket):
        """Rebuild EVERY frame the run plans from, off a (re)measured socket pose.

        These four are the only parse-time products of the target, and each nested maneuver reads
        them through the closure at call time -- so rebinding them here is enough to move the whole
        run onto a new socket pose. Rebinding a subset instead would put the trajectory at one
        place and the clocking axes at another, which is the failure mode this exists to prevent."""
        nonlocal T_base_socket, T_base_tconn, T_clk, T_base_targetobj, T_base_commit
        T_base_socket = T_socket
        T_base_tconn = T_base_socket @ R_clock
        T_clk = T_base_tconn
        T_base_targetobj = T_base_tconn @ inverse(mats[-1])
        T_base_commit = T_base_targetobj @ translation_matrix([fi_preload_mm / 1000.0, 0.0, 0.0])

    _sh_m, _sh_r = pose_error(T_base_targetobj, T_base_tconn)
    if _sh_m * 1000.0 > 1e-6 or np.degrees(_sh_r) > 1e-6:
        log.warning('trajectory_csv\'s last row is NOT identity (off by %.2f mm / %.2f deg) -- '
                    'ANCHORED onto the recorded mate, like uncertain_sampling. A press belongs '
                    'in assembly.final_insertion.preload_mm, not in a trajectory row.',
                    _sh_m * 1000.0, np.degrees(_sh_r))

    # ---- COLLECTION MODE (estimator_eval.eval.collection semantics) ----
    col = a.get('collection', {}) or {}
    col_mode = str(col.get('mode', 'attempts')).strip().lower()
    if col_mode not in ('attempts', 'offset_sweep', 'peck'):
        log.error("assembly.collection.mode %r must be 'attempts', 'offset_sweep' or 'peck'.",
                  col_mode)
        return False
    sweep_offsets = col.get('sweep_offsets')
    if sweep_offsets is None:
        sweep_offsets = [[0.0, 0.0, 0.0, 0.0, float(p), 0.0]
                         for p in np.arange(-4.0, 4.01, 2.0)]
    sweep_offsets = [[float(v) for v in o] for o in sweep_offsets]
    if col_mode == 'offset_sweep' and (not sweep_offsets
                                       or any(len(o) != 6 for o in sweep_offsets)):
        log.error('assembly.collection.sweep_offsets must be a non-empty list of 6-vectors '
                  '[x, y, z (m), roll, pitch, yaw (deg)].')
        return False
    peck_back_m = float(col.get('peck_retract_mm', 5.0)) / 1000.0
    peck_timeout = float(col.get('peck_timeout_s', 30.0))
    if col_mode == 'offset_sweep':
        log.info('Collection: offset_sweep, %d passes per attempt (pitch %s).',
                 len(sweep_offsets), [round(o[4], 1) for o in sweep_offsets])
    elif col_mode == 'peck':
        log.info('Collection: peck, %.1f mm back-off per stop, %.0f s budget.',
                 peck_back_m * 1000.0, peck_timeout)

    # ---- Trajectory noise (same semantics as estimator_eval.eval.trajectory_noise) ----
    tn = a.get('trajectory_noise', {}) or {}
    tn_on = bool(tn.get('enabled', False))
    tn_std = tn.get('std') or [0.0] * 6
    tn_std = [float(v) for v in tn_std]
    if len(tn_std) != 6:
        log.error('assembly.trajectory_noise.std must have 6 entries.')
        return False
    tn_w = max(1, int(tn.get('smooth_window', 10)))
    tn_da, tn_dt = float(tn.get('noise_decay_attempt', 0.0)), \
        float(tn.get('noise_decay_traj', 0.0))
    noise_rng = np.random.default_rng()

    spd = cfg.section('speed')
    scales = spd.get('phase_scale', {}) or {}

    def phase(name):
        robot.arm.set_speed_scale(float(scales.get(name, 1.0)), name)

    g_v = float(spd.get('max_cartesian_translation_mm_s', 3.5))
    g_w = float(spd.get('max_cartesian_rotation_deg_s', 5.0))
    s_asm = float(scales.get('assemble', 1.0))
    s_ret = float(scales.get('retract', 1.0))
    s_std = float(scales.get('standoff', 1.0))
    min_seg_s = 1.0 / adm.rate

    def caps(v=None, w=None):
        """Resolve a maneuver's speed overrides. A null means "no override" = the GLOBAL cap
        times the assemble phase scale. Every duration goes through here, so a null can never
        reach the arithmetic unresolved."""
        return (g_v * s_asm if v is None else v), (g_w * s_asm if w is None else w)

    def seg_time(A, B, v=None, w=None):
        lin_m, ang_rad = pose_error(A, B)
        cv, cw = caps(v, w)
        return _path_time(lin_m * 1000.0, np.degrees(ang_rad), cv, cw, min_seg_s)

    def screw_ramp(adm, ref_at, guard, v, w, ang_deg, label=''):
        """Ramp along the EXACT screw path instead of the straight chord between its endpoints.

        admittance.ramp SLERPs the rotation but LERPs the translation, so one ramp draws a
        STRAIGHT LINE between the two tool0 positions. Harmless at servo-rate steps, badly wrong
        for a whole clocking stroke: a `theta` turn about an axis `r` from tool0 bows into an arc
        and the chord cuts inside it by r(1 - cos(theta/2)) -- 52 mm at 90 deg with r = 178 mm,
        dragging whatever sits on the axis through that excursion. Both ENDPOINTS stay exactly
        right, which is why it hides in every logged pose. Subdividing on the true screw keeps
        each lerp to a fraction of a degree, where the chord error is microns.

        The speed cap is applied to the ARC, which seg_time cannot do -- it measures the chord and
        under-counts a 90 deg turn's real path by ~11%.

        Returns ('done' | 'seated', f), f being the FRACTION of the stroke the reference reached.
        A caller that continues the motion needs it: restarting from the nominal endpoint would
        jump across the arc the guard just refused, and restarting from the MEASURED pose would
        throw away the deflection holding the part loaded."""
        SAMPLES = 64
        coarse = [ref_at(k / SAMPLES) for k in range(SAMPLES + 1)]
        pts = [T[:3, 3] for T in coarse]
        arc_mm = sum(float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(SAMPLES)) * 1000.0
        chord = pts[-1] - pts[0]
        chord_mm = float(np.linalg.norm(chord)) * 1000.0
        sag_mm = 0.0
        if chord_mm > 1e-6:
            u = chord / np.linalg.norm(chord)
            sag_mm = max(float(np.linalg.norm((p - pts[0]) - np.dot(p - pts[0], u) * u))
                         for p in pts) * 1000.0
        cv, cw = caps(v, w)          # a null override must be resolved BEFORE the arithmetic
        dur = _path_time(arc_mm, ang_deg, cv, cw, min_seg_s)
        n = max(1, int(np.ceil(dur * adm.rate)))
        log.info('  %s%.1f deg about a fixed axis: arc %.1f mm in %d steps over %.2f s '
                 '(a single ramp would cut the chord and pull the axis %.1f mm off true)',
                 label, ang_deg, arc_mm, n, dur, sag_mm)
        dt, prev = dur / n, coarse[0]
        for k in range(1, n + 1):
            cur = ref_at(k / n)
            out = adm.ramp(prev, cur, dt, guard)
            prev = cur
            if out == 'seated':
                return 'seated', k / n
        return 'done', 1.0

    retract_m = float(a.get('retract_distance_m', 0.05))
    decim = max(1, int(a.get('log_decimation', 5)))
    tol = a.get('success_tolerance', {}) or {}
    tol_pos_m, tol_rot_rad = float(tol.get('pos_mm', 2.0)) / 1000.0, \
        np.radians(float(tol.get('rot_deg', 3.0)))
    max_attempts = int(a.get('max_attempts', 5))
    accumulate = bool(a.get('accumulate_observations', True))
    dbg = a.get('debug_match', {}) or {}
    dbg_on = bool(dbg.get('enabled', False))
    dbg_live = dbg.get('live', True)
    dbg_live = (os.path.join(cfg.get('data_dir', 'data'), 'experiments')
                if dbg_live is True else (dbg_live or None))

    frames = tool_frames.load_frames(cfg)
    tool_frames.check_drift(frames, cfg)      # warns if frames.yaml and the config sections drift

    # THE BELIEVED IN-HAND POSE, fingertip-relative (the estimator updates it, so it must be the
    # local pose, not the flattened tool0 chain). Preference order:
    #   1. estimation.initial_connector_frame -- a name in the shared frames catalogue, parented
    #      to `fingertip` there. Exists so the value lives in ONE place.
    #   2. estimation.initial_connector_in_fingertip -- the raw pose, kept as an override.
    #   3. cables.yaml junction_in_fingertip -- "wherever the pick put it".
    init = cfg.get_path('estimation.initial_connector_in_fingertip')
    init_name = cfg.get_path('estimation.initial_connector_frame')
    if init_name:
        if init_name not in frames:
            log.error('estimation.initial_connector_frame %r is not in %s.', init_name,
                      tool_frames.frames_path(cfg))
            return False
        T_ftip_conn = inverse(robot.T_tool0_fingertip) @ frames[init_name]
        log.info('Believed in-hand pose from the shared catalogue: %r.', init_name)
        if init:
            # Both given: the catalogue wins, but a silent disagreement is how the two drift.
            d_lin, d_ang = pose_error(from_cfg(init), T_ftip_conn)
            if d_lin * 1000.0 > 0.5 or np.degrees(d_ang) > 0.2:
                log.warning('estimation.initial_connector_in_fingertip disagrees with frame %r by '
                            '%.2f mm / %.2f deg. The FRAME is being used; delete the inline pose '
                            'or align it.', init_name, d_lin * 1000.0, np.degrees(d_ang))
    else:
        T_ftip_conn = from_cfg(init) if init else from_cfg(cfg.section('junction_in_fingertip'))
    # PICKUP PITCH: the frames catalogue declares the SQUARE grip, so a pitched pickup rotates
    # the part in the hand by the same angle. Applied here rather than in frames.yaml, so the
    # declared frame stays the physical truth and the pitch stays a run-time choice.
    _grip_off = fingertip_in_connector(cfg)
    _oxyz, _orpy = matrix_to_xyzrpy(_grip_off)
    # THE RAW NOMINAL, kept before the grasp is folded in. held_belief maps "the frames
    # catalogue's SQUARE grip" to "the grip this run actually takes", so it must always be
    # applied to the catalogue value -- never to a belief that already carries a grasp. Feeding
    # it its own output composes the two grasps: the reorient recovery did exactly that and put
    # the connector down 75 deg off horizontal, nearly axis-vertical.
    T_ftip_conn_catalogue = np.array(T_ftip_conn, dtype=float)
    T_ftip_conn = held_belief(T_ftip_conn, from_cfg(cfg.section('junction_in_fingertip')),
                              _grip_off)
    log.info('fingertip_in_connector xyz %s mm rpy %s deg (target fingertip wrt the detected '
             'connector: +x along the connector, +y across the cable, +z up) -> in-hand '
             'belief derived to match. The pick and the belief come from this ONE transform, '
             'so they cannot drift apart.',
             np.round(_oxyz * 1000.0, 2).tolist(), np.round(np.degrees(_orpy), 2).tolist())
    # THE MEASURED RESIDUAL, applied last and to the BELIEF ONLY: how the part actually seats
    # in the jaws at this approach angle, which no frame can predict. Moves nothing at pickup.
    _bel_off = belief_offset_m(cfg)
    if float(np.linalg.norm(_bel_off)) > 0.0:
        T_ftip_conn = offset_belief(T_ftip_conn, _bel_off)
        log.info('Belief offset %s mm (fingertip frame) applied to the in-hand pose ONLY -- '
                 'the grasp command is unchanged.',
                 np.round(_bel_off * 1000.0, 2).tolist())

    live = a.get('live_plot', True)
    live_path = None
    if live:
        live_path = live if isinstance(live, str) else os.path.join(
            cfg.get('data_dir', 'data'), 'experiments', 'bnc_assembly_live.png')
        os.makedirs(os.path.dirname(live_path) or '.', exist_ok=True)
    out_dir = os.path.join(cfg.get('data_dir', 'data'), 'experiments',
                           f'bnc_assembly_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)
    log.info('Experiment folder: %s', out_dir)

    def end_reset_with_snapshot(label='end reset'):
        """The end-of-run bookend: home (the marker VIEW pose) + one image from that view,
        saved to the experiment folder. The capture is best-effort -- it never fails a run
        that already finished."""
        ok = reset.reset_robot(robot, cfg, label)
        if ok and camera is not None and not robot.arm.dry_run                 and bool(cfg.get('end_view_image', True)):
            try:
                import cv2
                frame = camera.capture()
                path = os.path.join(out_dir, 'end_view.jpg')
                cv2.imwrite(path, frame.color)
                log.info('End-of-run image from the home view: %s', path)
            except Exception as exc:           # noqa: BLE001 -- never fail a finished run
                log.warning('end-of-run image skipped (%s)', exc)
        return ok

    no_prompts = prompts_off(cfg)
    if no_prompts:
        log.warning('skip_prompts: running with NO confirmations -- the reset, the pre-contact '
                    'stand-off and the success calls are all skipped (success falls back to '
                    'the tolerance check). Only the cable labelling still asks.')
    gates_on = cfg.get('confirm_each_step', True) is not False and not no_prompts

    def phase_gate(name, ahead):
        """Continue/abort between two phases. True = go on.

        Placed at the IRREVERSIBLE boundaries -- the screw cams the connector home, the collar
        locks it. Aborting LEAVES THE ARM WHERE IT IS rather than escaping automatically: the
        right recovery depends on how far the mate went and whether it will release, which this
        cannot judge. EOF continues, so a headless run never hangs; dry runs skip the prompt."""
        if robot.arm.dry_run or not gates_on:
            return True
        try:
            ans = input(f'\n[{name}] {ahead}\n    Enter to continue (q to abort): ')
        except EOFError:
            return True
        if ans.strip().lower() in ('q', 'quit', 'n', 'no'):
            log.warning('ABORTED before %s by the user. The arm is LEFT WHERE IT IS -- the '
                        'connector may still be engaged and held. Free it by hand (release the '
                        'gripper / press the unlock button) before commanding any motion.', name)
            return False
        return True

    def retract_from(last_ref, T_tool0_conn, adm_ctl=None):
        """The compliant UN-GUARDED escape along the believed connector's own -X.

        Un-guarded on purpose: a seated or jammed connector is already over the force limit, so a
        guarded retract would block the very motion that frees it (see ForceGuard.disable)."""
        ctl = adm_ctl if adm_ctl is not None else adm
        T_out = _retract_ref(last_ref, T_tool0_conn, retract_m)
        ctl.ramp(last_ref, T_out, seg_time(last_ref, T_out, g_v * s_ret, g_w * s_ret), guard=None)
        robot.arm.servo_stop()
        return T_out

    def run_insertion(adm_ctl, refs, T_tool0_conn, peck=False, guard_ctl=None, settle=None,
                      hold=None, speed=None, pause=None, retract=True):
        """One admittance insertion along refs, collecting observations. Mirrors estimator_eval's
        run_insertion, including peck (a force stop backs off and advances again rather than
        ending the pass) and the un-guarded post-insertion dwell.

        retract=False LEAVES THE ARM AT THE STOP. The insertion meant to SEAT must stay put: a
        retract here pulls the connector back out with it (the gripper holds the cable), and
        connector clocking would then read the retracted pose as its "engaged pose" and anchor the
        whole clocking sequence there. Only an attempt about to be RETRIED should back off, and
        that is the caller's decision (see retract_from).

        Returns (obs, lin, ang, stops, last_ref); `last_ref` is what retract_from needs later."""
        import time as _time
        guard = guard_ctl if guard_ctl is not None else guard_shared
        settle_s = settle if settle is not None else settle_shared
        hold_s = hold if hold is not None else hold_shared
        sv, sw = speed if speed is not None else (None, None)
        obs, cnt = [], [0]

        def log_cb():
            cnt[0] += 1
            if cnt[0] % decim == 0:
                obs.append(_observe(robot, T_tool0_conn, T_base_tconn))

        adm_ctl.reset()
        adm_ctl.warmup(refs[0], tare_fn=tare)
        guard.reset()
        last_ref, i, t0, stops = refs[0], 1, _time.time(), []
        prev = refs[0]
        while i < len(refs):
            res = adm_ctl.ramp(prev, refs[i], seg_time(prev, refs[i], sv, sw), guard,
                               on_step=log_cb)
            last_ref = refs[i]
            if res == 'seated':
                stops.append(float(matrix_to_xyzrpy(
                    inverse(T_base_tconn) @ robot.tool0() @ T_tool0_conn)[0][0] * 1000.0))
                if not peck or _time.time() - t0 > peck_timeout:
                    break
                # back off along the BELIEVED connector's -X, un-guarded (we are at the limit
                # by construction), then keep advancing from where we are.
                T_out = _retract_ref(last_ref, T_tool0_conn, peck_back_m)
                adm_ctl.ramp(last_ref, T_out, seg_time(last_ref, T_out, g_v * s_ret,
                                                       g_w * s_ret), guard=None,
                             on_step=log_cb)
                guard.reset()
                prev = T_out
                i += 1
                continue
            prev = refs[i]
            i += 1
        adm_ctl.hold(last_ref, settle_s, guard, on_step=log_cb)
        if hold_s > 0:                                 # dwell: un-guarded, unlogged
            adm_ctl.hold(last_ref, hold_s, guard=None)
        if pause:
            # Servo RELEASED, so the arm applies no force -- inspection time, not press time.
            adm_ctl.stop()
            log.info('   pausing %.1f s at the seat (servo stopped).', float(pause))
            _time.sleep(float(pause))
        lin, ang = pose_error(robot.tool0() @ T_tool0_conn, T_base_tconn)
        if retract:
            retract_from(last_ref, T_tool0_conn, adm_ctl)
        else:
            # servo_stop only ends the servoL stream; the controller holds the last commanded
            # pose, so the connector keeps its seat.
            robot.arm.servo_stop()
            log.info('   holding the seat (no retract) -- the pose is the engaged pose.')
        return obs, lin, ang, stops, last_ref

    # ====================================================================================
    # POST-MATE CLOCKING. Both maneuvers run only after a successful mate and share one
    # escape. Diagnostics land in clocking.csv rather than estimates.csv, which is already
    # closed by the time these run.
    # ====================================================================================
    clock_rows = []

    def axis_offset_base():
        """collar_clocking.axis_offset_mm as a BASE-frame vector.

        THE OFFSET BELONGS TO THE FIXTURE, NOT THE PLUG'S ROLL -- it is the bench-measured shift
        from the DECLARED socket axis to the barrel centreline the collar turns about, so it is a
        fixed line in space however far round the plug is mated.

        `engage_clock_deg` rolls T_clk about its own +X, so T_clk's Y and Z are no longer the axes
        the number was measured in. Resolving it there would swing a measured -Z correction round
        to -Y at a 90 deg clock angle. Hence the roll-FREE basis. With engage_clock_deg 0 this is
        exactly T_clk's own basis."""
        return (T_clk[:3, :3] @ R_clock[:3, :3].T) @ (np.asarray(cl_axis_off, dtype=float)
                                                      / 1000.0)

    def clocking_retract(label='clocking retract', gripper_leg=True):
        """The post-clocking escape, in two legs.

        First along the GRIPPER's own axis, lifting the open fingers off the connector; then
        along the TARGET frame's axis, backing the arm away from the socket. Guarded straight
        lines, not the compliant `_retract_ref` used between attempts: the part is released by
        now, so there is no held connector to thread back out along its own axis.

        `gripper_leg=False` drops the first leg and backs straight out along the target -X.
        That is the right shape after the COLLAR turn, because there tool0 sits ON the
        connector axis with its Z collinear -- so the gripper -Z leg points essentially where
        the target -X leg already goes, and running both just adds a second, differently
        parametrised move for the same escape."""
        r = a.get('clocking_retract', {}) or {}

        def leg(vec, dist, in_target):
            v = np.asarray(vec, dtype=float)
            n = float(np.linalg.norm(v))
            if n < 1e-9 or abs(float(dist)) < 1e-9:
                return True
            step = v / n * abs(float(dist))
            if in_target:
                # a direction in the TARGET frame -> rotate it into base and left-multiply
                T = translation_matrix(T_clk[:3, :3] @ step) @ robot.tool0()
                what = f'target {np.round(v / n, 3).tolist()}'
            else:
                # a direction in the GRIPPER's own (tool0) frame -> right-multiply
                T = robot.tool0() @ translation_matrix(step)
                what = f'gripper {np.round(v / n, 3).tolist()}'
            return _guarded(robot, guard_shared, lambda: robot.arm.move_l(
                T, label=f'{label} ({what}, {abs(float(dist)) * 1000.0:.0f} mm)'))

        phase('clock_retract')
        if not gripper_leg:
            log.info('  escape: target -X only (tool0 is already on the connector axis, so the '
                     'gripper -Z leg would repeat it).')
        return ((gripper_leg is False
                 or leg(r.get('gripper_axis', [0.0, 0.0, -1.0]),
                        r.get('gripper_distance_m', 0.100), False))
                and leg(r.get('target_axis', [-1.0, 0.0, 0.0]),
                        r.get('target_distance_m', 0.100), True))

    def tug_verify_in_place(hold_after=False):
        """TUG VERIFICATION, from wherever the collar turn ended, WITHOUT letting go.

        collar_clocking returns with the fingers still closed on the locked collar. That is
        already a grip on the assembly, and already on the connector axis -- so the pull happens
        here rather than after a release, a retract, a drive back to the historical engaged pose
        and a blind re-grasp. Each of those was a chance to disturb the very thing being measured,
        and the re-grasp could miss the connector outright.

        A LOCKED bayonet holds the pull; an unlocked one backs out and the displacement says so.
        The pull is a SPRING pull, not a position ramp: the reference is offset along the
        connector -X by pull_force / stiffness, so the force is applied at zero displacement and
        DROPS as the connector comes out -- it can never exceed pull_force_n on one that holds.

        Returns 'verified' (held; released and retracted), 'failed' (backed out; the cable has
        been EXTRACTED, carried home and released), 'terminated' (the GLOBAL guard tripped
        mid-pull -- stop where we are), or 'error'."""
        axn_t = T_clk[:3, 0] / float(np.linalg.norm(T_clk[:3, 0]))
        # NO re-approach and NO re-grip: pull from the pose the turn left, on the collar it is
        # already holding. The axis is still the socket's, so the pull direction is unchanged.
        T_grasp = robot.tool0()
        phase('assemble')
        # THE PULL. Effective axial stiffness of the (diagonal, tool0-frame) spring along the
        # base-frame pull direction: compliances add, 1/S_eff = sum(u_i^2 / S_i).
        u = T_grasp[:3, :3].T @ axn_t
        S_eff = 1.0 / float(np.sum((u ** 2) / adm_tug.S[:3]))
        if tv_force / S_eff <= tv_thresh_m:
            log.warning('TUG VERIFY: the spring offset (%.1f mm at %.0f N/m) does not exceed '
                        'the %.1f mm threshold -- an unlocked connector cannot move past it, so '
                        'this verification cannot fail. Soften tug_verify.stiffness or raise '
                        'pull_force_n.', tv_force / S_eff * 1000.0, S_eff, tv_thresh_m * 1000.0)
        T_pull = translation_matrix(-(tv_force / S_eff) * axn_t) @ T_grasp
        log.info('TUG VERIFY: pulling %.1f N along the connector -X for %.1f s '
                 '(spring %.0f N/m -> %.1f mm reference offset); verified if the connector '
                 'moves <= %.1f mm.', tv_force, tv_time, S_eff, tv_force / S_eff * 1000.0,
                 tv_thresh_m * 1000.0)
        adm_tug.reset()
        adm_tug.warmup(T_grasp, tare_fn=tare)       # tare while gripping and static
        guard_shared.reset()
        phase('assemble')
        res = adm_tug.ramp(T_grasp, T_pull, seg_time(T_grasp, T_pull), guard_shared)
        if res != 'seated':
            res = adm_tug.hold(T_pull, tv_time, guard_shared)
        disp = float(np.dot(T_grasp[:3, 3] - robot.tool0()[:3, 3], axn_t))
        adm_tug.reset()
        adm_tug.stop()
        robot.arm.servo_stop()
        if res == 'seated':
            log.error('TUG VERIFY: the GLOBAL force guard tripped during the pull (%s) -- '
                      'terminating the script where it stands.', guard_shared.tripped_by)
            return 'terminated'
        held = disp <= tv_thresh_m
        clock_rows.append({'maneuver': 'tug_verify', 'try': 1, 'ramp_result': res,
                           'advance_mm': round(-disp * 1000.0, 3),
                           'need_mm': round(tv_thresh_m * 1000.0, 3),
                           'success': bool(held), 'force_stop': False,
                           'state_after': 'locked' if held else 'extracted',
                           'stopped_by': ''})
        if held:
            log.info('TUG VERIFIED -- %.1f N for %.1f s moved the connector %.2f mm '
                     '(<= %.1f mm): the lock holds.', tv_force, tv_time, disp * 1000.0,
                     tv_thresh_m * 1000.0)
            if hold_after:
                # DISASSEMBLY IS NEXT and it starts from exactly this state: fingers closed on
                # the collar, tool0 on the connector axis. Releasing and retracting here only
                # to re-approach and re-grip would add the three free-space moves and the blind
                # re-grasp that doing the tug in place exists to avoid.
                log.info('  holding the grip -- disassembly starts from here.')
                return 'verified_held'
            if not robot.gripper.open('release (tug verified)'):
                log.error('TUG VERIFY: gripper did not release after the tug.')
                return 'error'
            # Straight out along the target -X when disassembly follows: it re-approaches
            # the collar on the axis anyway, so lifting off the gripper -Z first only adds a
            # move in the direction the next leg already travels.
            return ('verified' if clocking_retract('tug retract', gripper_leg=not dis_on)
                    else 'error')
        # FAILED: never locked, and already part-way out in the fingers -- so EXTRACT it fully
        # along the connector -X, carry it home and release. The global guard stays armed, so an
        # extraction that snags terminates rather than tearing at the fixture.
        log.error('TUG VERIFY FAILED -- the connector backed out %.2f mm (> %.1f mm) under a '
                  '%.1f N pull: NOT locked. Extracting the cable.',
                  disp * 1000.0, tv_thresh_m * 1000.0, tv_force)
        phase('retract')
        T_now = robot.tool0()
        T_out = translation_matrix(-tv_extract_m * axn_t) @ T_now
        adm_tug.reset()
        adm_tug.warmup(T_now)
        guard_shared.reset()
        res2 = adm_tug.ramp(T_now, T_out, seg_time(T_now, T_out), guard_shared)
        adm_tug.reset()
        adm_tug.stop()
        robot.arm.servo_stop()
        if res2 == 'seated':
            log.error('TUG: the GLOBAL force guard tripped during the extraction (%s) -- '
                      'terminating the script with the cable still held.',
                      guard_shared.tripped_by)
            return 'terminated'
        phase('reset')
        if not reset.reset_robot(robot, cfg, 'tug failure (carry the cable home)'):
            log.error('TUG: could not go home with the extracted cable -- terminating.')
            return 'terminated'
        robot.gripper.open('release (failed cable, at home)')
        return 'failed'


    def disassembly(state, screw_deg, T_base_conn_d):
        """DISASSEMBLY -- unwind the clocking, pull the connector out, put the cable down.

        THE STATE LADDER RUN BACKWARDS. Assembly walks engaged -> seated -> locked; this walks
        locked -> seated -> engaged -> removed, each rung claimed only by the step that undoes
        it (_retreat_state asserts the rung we are actually on).

        ONE MOTION DROPS TWO RUNGS, and that is a fact about the grasp rather than a shortcut:
        the axial grip closes on the collar AND the connector body together, so the reverse
        turn backs the collar off its lock and carries the bayonet round to its slots in the
        same stroke. Assembly needs two maneuvers there because it grips differently for each
        (the bayonet at the junction, the collar on the ring); disassembly does not.

        WHERE IT STARTS, and why that matters. It runs straight after the tug, which leaves the
        fingers CLOSED on the collar with tool0 ON the connector axis -- the same reason the tug
        itself happens in place. Every rotation below is therefore a wrist twist about the axis
        the arm is already on: no re-approach, no re-grip, no chance to lose the part between
        steps.

        THE GEOMETRY. Both rotations are about the SAME LINE the assembly turned about: the
        socket frame's own +X through its origin (T_clk), which is what connector_clocking
        screwed about and -- with collar_clocking.axis_offset_mm at zero -- what the collar
        turned about too. They are run through screw_ramp for the same reason the assembly is:
        a rotation about a line 178 mm from tool0 bows into an arc that a straight ramp would
        cut across.

        THE DIRECTIONS, and this is the part that is easy to get backwards:
          * the collar was locked by turning +rotation_deg, so the reverse turn is
            -rotation_deg -- or until the torque limit, since the ring runs to a stop at both
            ends of its travel and torque building is the intended end in either direction;
          * that same turn carries the bayonet round with it, because the jaws hold the
            connector body and the collar together -- so there is no separate angle to undo
            and nothing to aim at the mated roll;
          * the pull is along the socket -X, the exact reverse of the insertion axis.

        Returns (ok, state)."""
        axn_d = T_clk[:3, 0] / float(np.linalg.norm(T_clk[:3, 0]))
        if float(np.linalg.norm(np.asarray(cl_axis_off, dtype=float))) > 0.0:
            log.warning('DISASSEMBLY: collar_clocking.axis_offset_mm is non-zero, but the '
                        'unwind turns about the SOCKET axis. Re-check the collar stays on the '
                        'ring through the unlock.')

        # ---- 1. LOCKED/SEATED -> ENGAGED: ONE reverse turn undoes both --------------------
        # THE AXIAL GRASP HOLDS BOTH. Closing on the ring closes on the connector body with
        # it, so the single reverse turn backs the collar off its lock AND carries the bayonet
        # round to where its pins line up with the slots -- there is no second, separate
        # connector rotation to make, and making one would turn a part that is already free.
        # That is why this drops two rungs of the ladder in one motion.
        #
        # The assembly RELEASED and retracted before this, so there is no grip to inherit: the
        # turn drives the SAME approach the lock did (retract along the cable, reorient onto
        # the axis, advance down it, close on the ring) with the rotation negated and the seat
        # push off. Sharing that geometry is what keeps the unlock landing on the ring the lock
        # turned, instead of on a station computed a second, divergent way.
        if dis_unlock and state in ('locked', 'seated'):
            if not phase_gate('UNLOCK',
                              'Re-approach on the axis, close on the collar (which grips the '
                              'connector body with it), and turn %.0f deg BACK -- or until the '
                              'torque limit. That releases the lock and the bayonet together.'
                              % np.degrees(cl_rot)):
                return False, state
            if not collar_clocking(T_base_conn_d, screw_deg, cl_rot=-cl_rot, sp_on=False,
                                   unlocking=True):
                log.error('DISASSEMBLY: the reverse turn did not complete -- stopping with the '
                          'connector still in the socket.')
                return False, state
            if state == 'locked':
                state = _retreat_state(state, 'locked')      # -> seated (the lock is off)
            state = _retreat_state(state, 'seated')          # -> engaged (the bayonet is free)
            log.info('  the reverse turn released the collar AND the bayonet -- the connector '
                     'is ENGAGED only, and free to pull.')
        elif state in ('locked', 'seated'):
            log.warning('DISASSEMBLY: unlock_collar is off, so nothing has released the '
                        'bayonet -- the pull below will be against a %s connector.', state)

        # ---- 2. ENGAGED -> REMOVED: pull it straight out ----------------------------------
        if not phase_gate('EXTRACT',
                          'Pull the connector %.0f mm straight out along the socket -X. The '
                          'bayonet should be free; the global guard stops a pull that snags.'
                          % (dis_extract_m * 1000.0)):
            return False, state
        phase('retract')
        T_now = robot.tool0()
        T_out = translation_matrix(-dis_extract_m * axn_d) @ T_now
        adm_cl.reset()
        adm_cl.warmup(T_now)
        guard_shared.reset()
        res = adm_cl.ramp(T_now, T_out, seg_time(T_now, T_out), guard_shared)
        pulled = float(np.dot(T_now[:3, 3] - robot.tool0()[:3, 3], axn_d))
        adm_cl.reset()
        adm_cl.stop()
        robot.arm.servo_stop()
        clock_rows.append({'maneuver': 'extract', 'try': 1, 'ramp_result': res,
                           'advance_mm': round(-pulled * 1000.0, 3),
                           'need_mm': round(dis_extract_m * 1000.0, 3),
                           'success': res != 'seated', 'force_stop': res == 'seated',
                           'state_after': 'removed' if res != 'seated' else 'engaged',
                           'stopped_by': guard_shared.tripped_by or ''})
        if res == 'seated':
            log.error('EXTRACT: the global guard tripped after %.1f mm (%s) -- the connector '
                      'is still in the socket and still held. Free it by hand.',
                      pulled * 1000.0, guard_shared.tripped_by)
            return False, state
        log.info('EXTRACTED %.1f mm along the socket -X -- the connector is out and in the '
                 'fingers.', pulled * 1000.0)
        state = _retreat_state(state, 'engaged')

        # ---- 3. put the cable down --------------------------------------------------------
        if not dis_place_on:
            log.info('DISASSEMBLY: place is off -- the cable stays in the fingers.')
            return True, state
        T_place = aligned_place_pose(dis_clear_m)
        if not phase_gate('PLACE THE CABLE',
                          'Lay the cable down ALONG THE SOCKET AXIS at xyz %s mm, %.0f mm '
                          'clear of the ground, release and rise.'
                          % (np.round(T_place[:3, 3] * 1000.0, 0).tolist(),
                             dis_clear_m * 1000.0)):
            return False, state
        # AIMED, NOT INHERITED. The old version drove to the pick pose and descended, which
        # laid the part down in whatever attitude the grip happened to have -- and with the
        # COAXIAL grip the connector hangs axis-down, so it was set on its end. The pose is
        # built from the socket heading instead, and the arm is commanded to put the CONNECTOR
        # there via the belief it is currently holding it with.
        T_ftip_place = T_place @ inverse(T_ftip_conn)
        up = translation_matrix([0.0, 0.0, float(dis_rise_m)])
        log.info('PLACE: laying the connector along the socket axis -- heading %+.1f deg '
                 '(socket %+.1f deg), %.0f mm above the ground plane.',
                 np.degrees(np.arctan2(T_place[1, 0], T_place[0, 0])),
                 np.degrees(np.arctan2(T_base_tconn[1, 0], T_base_tconn[0, 0])),
                 (T_place[2, 3] - ground_z) * 1000.0)
        phase('reconfigure')
        if not _guarded(robot, guard_shared,
                        lambda: robot.move_fingertip(up @ T_ftip_place, 'place (above)')):
            log.error('DISASSEMBLY: could not reach the place stand-off.')
            return False, state
        phase('lift')
        if not _guarded(robot, guard_shared,
                        lambda: robot.move_fingertip(T_ftip_place, 'place (down)')):
            log.error('DISASSEMBLY: the descent hit something before the release height.')
            return False, state
        if not robot.gripper.open('release (cable placed)'):
            log.error('DISASSEMBLY: the gripper did not open to release the cable.')
            return False, state
        T_up = translation_matrix([0.0, 0.0, dis_rise_m]) @ robot.tool0()
        if not _guarded(robot, guard_shared,
                        lambda: robot.arm.move_l(T_up, label='rise clear of the cable')):
            log.error('DISASSEMBLY: could not rise after releasing.')
            return False, state
        log.info('CABLE PLACED %.0f mm above the ground plane and released.',
                 dis_clear_m * 1000.0)
        return True, state

    def engage_insertion():
        """ENGAGE -- drive the assembly trajectory home, optionally rocking, stop on axial force.

        Follows configs/assembly_trajectory.csv, extends it by `preload_mm` past the mate, and
        superimposes an oscillation on the way. With every amplitude at 0 it is a plain direct
        insertion, so direct and wiggle are one code path at two settings.

        TWO CLOCKS, deliberately. The PATH advances by DISTANCE (speed_mm_s); the OSCILLATION
        advances by TIME, so its frequency is the frequency configured regardless of how fast the
        path is driven. Pacing both by distance would make the oscillation frequency a function of
        the insert speed. MIND THE DURATION: the insertion lasts path_length / speed_mm_s, so a
        frequency chosen without reference to that can complete less than one cycle and act as a
        constant offset -- the start-up line reports cycles-per-insertion for that reason.

        TERMINATION is the point of the phase: reaching the end of the path is SUCCESS, and
        hitting the axial force limit is a NORMAL end, not a failure -- meeting resistance partway
        is exactly what the bayonet screw is for. That is why the limit is AXIAL (see _AxialForce)
        and can be set low.

        Returns (status, last_ref, depth_mm) with status 'complete' or 'force'."""
        T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn

        # ---- the path: trajectory rows, then the preload, all as connector-wrt-target 6-vecs ----
        rows6 = [np.asarray(vec6_from_mats(r), dtype=float) for r in dense]
        rows6 = [np.concatenate([v[:3] * 1000.0, v[3:]]) for v in rows6]   # -> mm / deg
        if en_pre_mm > 0.0:
            step = float(a.get('translational_resolution_m', 0.001)) * 1000.0
            n_pre = max(1, int(round(en_pre_mm / max(step, 1e-6))))
            base = rows6[-1].copy()
            for k in range(1, n_pre + 1):
                v = base.copy()
                v[0] += en_pre_mm * k / n_pre        # extend along the connector's own +X
                rows6.append(v)
        seg = [float(np.linalg.norm(rows6[i + 1][:3] - rows6[i][:3]))
               for i in range(len(rows6) - 1)]
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total_mm = float(cum[-1])

        v_mm_s = float(en_speed) if en_speed is not None else (g_v * s_asm)
        dt = 1.0 / en_rate

        def path_at(d_mm):
            """The trajectory 6-vec `d_mm` along the path, linearly between rows."""
            d = float(np.clip(d_mm, 0.0, total_mm))
            j = int(np.searchsorted(cum, d, side='right') - 1)
            j = max(0, min(j, len(rows6) - 2))
            span = cum[j + 1] - cum[j]
            f = 0.0 if span <= 1e-12 else (d - cum[j]) / span
            return rows6[j] + f * (rows6[j + 1] - rows6[j])

        def ref_at(t, dur=None):
            """The path point at time t, with the wiggle RIGHT-MULTIPLIED onto it -- so the
            offset acts in the connector's OWN frame and a misaligned part rocks about its own
            axes, which is what the part physically does and what wiggle_sampling collected."""
            T = _corr_to_m(mats_from_vec6(path_at(v_mm_s * t)))
            if en_wig is not None:
                T = T @ en_wig.delta(t, dur)
            return traj_ref(T, T_tool0_conn)

        live = [f'{d} {en_amp[i]:+.2f}@{en_frq[i]:.2f}Hz'
                for i, d in enumerate(DIM_KEYS) if abs(en_amp[i]) > 0.0]
        log.info('--- ENGAGE --- %s over %.1f mm (%.1f mm trajectory + %.1f mm preload) at '
                 '%.2f mm/s -> %.1f s; %s.',
                 'oscillating ' + ', '.join(live) if live else 'DIRECT (no oscillation)',
                 total_mm, total_mm - en_pre_mm, en_pre_mm, v_mm_s, total_mm / max(v_mm_s, 1e-9),
                 f'axial force limit {en_fmax:.1f} N' if en_fmax > 0 else 'NO force limit')
        if live:
            dur = total_mm / max(v_mm_s, 1e-9)
            log.info('  oscillation peak %.2f mm/s / %.2f deg/s%s; the PATH runs at %.2f mm/s, so '
                     'the two add.', en_pv * en_scale, en_pw * en_scale,
                     f' (capped, time scale {en_scale:.3f})' if en_scale < 1.0 else ' (uncapped)',
                     v_mm_s)
            cyc = {d: en_frq[i] * en_scale * dur
                   for i, d in enumerate(DIM_KEYS) if abs(en_amp[i]) > 0.0}
            worst = min(cyc.values()) if cyc else 0.0
            (log.info if worst >= 1.0 else log.warning)(
                '  cycles completed during the %.1f s insertion: %s%s', dur,
                {k: round(v, 2) for k, v in cyc.items()},
                '.' if worst >= 1.0 else ' -- under one full cycle acts as a constant OFFSET, not '
                'a wiggle. Raise the frequency or slow speed_mm_s.')

        first = ref_at(0.0)
        phase('standoff')
        q = robot.arm.ik(first, seed_q)
        if q is None or not _guarded(robot, guard_shared,
                                     lambda: robot.arm.move_j(q, label='engage start')):
            log.error('Could not reach the engage start pose.')
            return 'unreachable', first, float('nan')

        det = _AxialForce(robot, T_tool0_conn, en_fmax, en_fpers)
        combo = _AnyGuard(det, guard_en)
        obs, cnt = [], [0]

        def log_cb():
            cnt[0] += 1
            if cnt[0] % decim == 0:
                obs.append(_observe(robot, T_tool0_conn, T_base_tconn))

        phase('assemble')
        adm_en.reset()
        adm_en.warmup(first, tare_fn=tare)
        combo.reset()
        # REAL ELAPSED TIME, not i*dt. The servo loop does not necessarily cycle at
        # reference_rate_hz -- on 19 Aug it ran at ~332 Hz against a configured 125, which
        # delivered every frequency 2.65x high. Reading the clock makes the delivered frequency
        # the configured one whatever the loop does.
        dur_s = total_mm / max(v_mm_s, 1e-9)
        status, prev, last_ref = 'complete', first, first
        t_wig0 = _t.monotonic()
        while True:
            t = _t.monotonic() - t_wig0
            if t >= dur_s:
                break
            cur = ref_at(t, dur_s)
            res = adm_en.ramp(prev, cur, dt, combo, on_step=log_cb)
            prev = last_ref = cur
            if res != 'seated':
                continue
            status = 'force' if combo.tripped is det else 'guard'
            break

        # SNAPSHOT EVERY CONDITION AT THE MOMENT IT STOPPED, before the settle hold moves
        # anything. Read once, here, rather than per servo cycle.
        t_end = min(t, dur_s)
        w_end = robot.arm.wrench()
        end_state = {
            'elapsed_s': t_end, 'duration_s': dur_s,
            'driven_mm': min(v_mm_s * t_end, total_mm), 'total_mm': total_mm,
            'axial_n': det.axial_n(), 'axial_peak_n': det.peak_n,
            'axial_limit_n': en_fmax, 'axial_persist_s': en_fpers,
            'force_n': float(np.linalg.norm(w_end[:3])),
            'torque_nm': float(np.linalg.norm(w_end[3:])),
        }

        stay = robot.tool0()
        adm_en.reset()
        if settle_shared > 0:
            adm_en.hold(stay, settle_shared, guard=None, on_step=log_cb)
        adm_en.stop()
        robot.arm.servo_stop()

        depth = float(matrix_to_xyzrpy(
            inverse(T_base_tconn) @ (robot.tool0() @ T_tool0_conn))[0][0] * 1000.0)
        _engage_report(status, end_state, depth, det, guard_en, combo)
        if obs:
            _save_observations(os.path.join(out_dir, 'engage_observations.csv'), obs)
        return status, last_ref, depth

    def connector_clocking():
        """CONNECTOR CLOCKING -- the bayonet search, and the belief reset that makes it well posed.

        THE BELIEF RESET: the mate is made, so the connector's pose is known from a PHYSICAL
        CONSTRAINT rather than estimated -- it is AT the target. That replaces the estimated
        belief, which is what makes the screw axis the target's +X exactly instead of inheriting
        the accumulated in-hand error. The pose the arm stands at becomes the engaged pose and
        every measurement below is relative to it.

        THE MOTION IS AN OSCILLATING SCREW. `sweep_deg` lists roll positions about the connector's
        +X -- absolute, wrt the TARGET frame -- visited IN TURN, one per try, while pushing
        `push_mm` along that same axis. Translation ALONG the rotation axis commutes with it, so
        each leg is a true helix. The push target is VIRTUAL: it aims past where the connector can
        go and lets compliance follow whatever path the cams allow.

        WHY IT ROCKS INSTEAD OF TURNING FURTHER. A bayonet pin that missed its slot will not find
        it by turning harder -- it rides the rim and jams. Rocking across the slot under a steady
        axial load is what finds it. Hence the press being established on the first leg and HELD
        across every reversal, and the search ending on measured x-advance rather than any
        commanded angle.

        A LEG THAT JAMS IS NORMAL: the guard stopping one part-way is how the far edge of the slot
        is found. The next leg reverses from where the reference actually stopped (screw_ramp's
        returned fraction), not from the endpoint it never reached.

        SUCCESS IS ASSUMED when the sweep runs. `success_advance_mm` is the EARLY-OUT and
        telemetry, never a post-hoc verdict -- cross-checking against measured advance turned
        soft-spring deflection into false failures. The only not-seated outcome is a sweep in
        which EVERY leg was guard-stopped: nothing swung freely, so the connector is wedged rather
        than searching.

        NO REGRASP -- the retry IS the reversal and the gripper stays closed throughout. (The old
        ratchet existed because a one-way stroke could only be retried by giving back the wrist
        range it had spent; an oscillation never spends range it does not immediately return.)

        Returns (ok, T_tool0_conn, T_base_conn, achieved_deg) -- where the connector ended up,
        read from the MEASURED arm pose, so it is the ACHIEVED roll and not the commanded one."""
        T_tool0_engaged = robot.tool0()
        T_tool0_conn = inverse(T_tool0_engaged) @ T_clk
        moved = matrix_to_xyzrpy(inverse(robot.T_tool0_fingertip @ T_ftip_conn) @ T_tool0_conn)
        log.info('--- CONNECTOR CLOCKING --- belief reset: connector assumed AT the %s frame '
                 '(shifts the in-hand belief by %s mm, %s deg)', pe_frame,
                 np.round(moved[0] * 1000.0, 2).tolist(),
                 np.round(np.degrees(moved[1]), 2).tolist())
        log.info('  oscillating screw: roll to %s deg (wrt the target frame) in turn, up to %d '
                 'leg%s, while pushing %+.1f mm along the connector +X and HOLDING it; early-out '
                 'at %.1f mm of CUMULATIVE advance (a completed sweep = seated).',
                 [round(float(np.degrees(t)), 1) for t in cc_sweep], cc_tries,
                 '' if cc_tries == 1 else 's', cc_push_m * 1000.0, cc_need_m * 1000.0)
        log.info('  starting at the engaged roll of %+.1f deg, so leg 1 turns %+.1f deg (%s) and '
                 'the band worked is %+.1f .. %+.1f deg about the socket +X.',
                 np.degrees(eng_clock), np.degrees(cc_legs[0]),
                 'NEGATIVE' if cc_legs[0] < 0 else 'POSITIVE',
                 np.degrees(eng_clock + cc_lo), np.degrees(eng_clock + cc_hi))

        # Each leg is a BASE-frame motion about the fixed axis line L (engaged connector origin,
        # along its +X) -- independent of the connector-in-gripper belief, and the SAME line every
        # leg, since rotation about +X and push along +X both leave L invariant.
        ref_start = T_tool0_engaged

        def arm_at(th_rel, push):
            """The arm pose at roll `th_rel` (rad from the ENGAGED roll) and axial offset `push`."""
            return (T_clk
                    @ xyzrpy_to_matrix([push, 0.0, 0.0], [th_rel, 0.0, 0.0])
                    @ inverse(T_clk)) @ ref_start

        # ---- REACHABILITY OF THE SWEPT BAND, before anything turns ----------------------------
        # The legs are servoed, not planned, so without this the arm finds out a pose is
        # unreachable by failing at it mid-turn, gripped and cammed part-way home. The band is the
        # union of every leg, so one scan covers all of them however the tries fall.
        #
        # ADVISORY, NOT A GATE: a marginal pose can solve here and not on the arm, so refusing
        # would abort runs that would have completed.
        #
        # THE SEED IS CHAINED from the previous sample, because the solver returns the branch
        # nearest its seed and the servo walks each leg in fractions of a degree. Seeding every
        # sample from the current joints would let the far end resolve onto an unreachable branch
        # and report a range problem that is really a seeding artefact.
        #
        # arm.ik logs 'No IK solution' at ERROR per unsolved pose, so a marginal band prints a
        # burst of them here. They belong to this scan; nothing has moved.
        bad, seed_scan = [], robot.arm.q()
        for _k in range(25):
            _th = cc_lo + (cc_hi - cc_lo) * (_k / 24.0)
            _q = robot.arm.ik(arm_at(_th, cc_push_m), seed_scan)
            if _q is None:
                bad.append(_th)
            else:
                seed_scan = _q
        if bad:
            log.warning('  CONNECTOR CLOCKING: %d of 25 poses sampled across the %.0f deg band have '
                        'no IK solution (first at %+.0f deg wrt the target frame). The sweep will '
                        'be attempted anyway, but it is likely to run out of wrist range there -- '
                        'move assembly.connector_clocking.sweep_deg, or the engage angle, rather than '
                        'letting a leg discover it mid-turn.',
                        len(bad), np.degrees(cc_hi - cc_lo), np.degrees(eng_clock + bad[0]))

        det = _ScrewAdvance(robot, T_tool0_conn, T_clk, cc_need_m)
        combo = _AnyGuard(det, guard_cc)
        # ONE WARM-UP FOR THE WHOLE SWEEP, not one per leg: adm.ramp keeps its integrator across
        # calls, so resetting between legs would dump the axial deflection holding the connector
        # loaded and the press would rebuild on every reversal. Taring is once, here, for the same
        # reason.
        phase('connector_clock')
        adm_cc.reset()
        adm_cc.warmup(ref_start, tare_fn=cc_tare)

        ok = swung = False
        ran = 0                          # legs that actually turned something
        th_at, push_at = 0.0, 0.0        # where the REFERENCE stands: roll from engaged, and push
        for k in range(1, cc_tries + 1):
            th_to = cc_legs[(k - 1) % len(cc_legs)]
            turn = th_to - th_at
            if abs(turn) < 1e-9:
                log.info('  leg %d/%d: the reference is already at %+.1f deg -- nothing to turn.',
                         k, cc_tries, np.degrees(eng_clock + th_to))
                continue
            ran += 1
            combo.reset()
            _a, _pa = th_at, push_at
            # SUBDIVIDED ON THE TRUE SCREW -- see screw_ramp. The push ramps to its full value
            # on the FIRST leg and stays there (_pa == cc_push_m for every leg after), so the
            # axial load is established once and carried through the reversals.
            res, f_done = screw_ramp(
                adm_cc,
                lambda f, _s=_a, _e=th_to, _p=_pa: arm_at(_s + (_e - _s) * f,
                                                          _p + (cc_push_m - _p) * f),
                combo, cc_v, cc_w, abs(np.degrees(turn)),
                label='leg %d/%d (%+.0f -> %+.0f deg) ' % (
                    k, cc_tries, np.degrees(eng_clock + _a), np.degrees(eng_clock + th_to)))
            adv = det.advance_m()
            # `stopped`, not `seated`: ramp's 'seated' means "a guard tripped" and says nothing
            # about the assembly state. WHICH guard fired decides between the two outcomes.
            stopped = res == 'seated'
            early_seat = stopped and combo.tripped is det
            jammed = stopped and not early_seat
            swung = swung or res == 'done'          # at least one leg swung its whole arc
            # CONTINUE FROM WHERE THE REFERENCE ACTUALLY GOT TO -- a jammed leg stopped short,
            # and the next leg reverses from there, not from the endpoint it never reached.
            th_at = _a + turn * f_done
            push_at = _pa + (cc_push_m - _pa) * f_done
            clock_rows.append({'maneuver': 'connector_clocking', 'try': k, 'ramp_result': res,
                               'target_deg': round(float(np.degrees(eng_clock + th_to)), 3),
                               'reached_deg': round(float(np.degrees(eng_clock + th_at)), 3),
                               'advance_mm': round(adv * 1000.0, 3),
                               'peak_advance_mm': round(det.peak_m * 1000.0, 3),
                               'need_mm': round(cc_need_m * 1000.0, 3),
                               'success': bool(early_seat), 'force_stop': bool(jammed),
                               'state_after': 'seated' if early_seat else 'engaged',
                               'stopped_by': combo.tripped_by or ''})
            if early_seat:
                ok = True
                log.info('  leg %d/%d: SEATED at %+.1f deg -- early out: %s (advance %.2f mm)',
                         k, cc_tries, np.degrees(eng_clock + th_at), det.tripped_by,
                         adv * 1000.0)
                break
            if jammed:
                log.warning('  leg %d/%d: the force guard stopped the turn at %+.1f deg of the '
                            '%+.1f deg aimed for (%s); advance %.2f mm -- the next leg reverses '
                            'from there.', k, cc_tries, np.degrees(eng_clock + th_at),
                            np.degrees(eng_clock + th_to), combo.tripped_by, adv * 1000.0)
            else:
                log.info('  leg %d/%d: turned to %+.1f deg without the cams picking up '
                         '(advance %.2f mm of the %.2f mm needed).', k, cc_tries,
                         np.degrees(eng_clock + th_at), adv * 1000.0, cc_need_m * 1000.0)
        if ok:
            verdict = 'early_out'                  # the advance threshold fired mid-leg
        else:
            # SUCCESS IS ASSUMED when the sweep runs -- see the docstring. Not-seated means EVERY
            # leg was cut short by the guard, i.e. a wedge rather than a search.
            ok = swung
            verdict = 'completed' if ok else 'all_legs_guard_stopped'
            log.info('  sweep finished after %d leg%s: %s (peak advance %.2f mm, needed %.2f).',
                     ran, '' if ran == 1 else 's',
                     'SEATED -- at least one leg swung its whole arc, success assumed' if ok
                     else 'NOT SEATED -- every leg was stopped by the force guard',
                     det.peak_m * 1000.0, cc_need_m * 1000.0)
        # ONE SUMMARY ROW on top of the per-leg rows: when the sweep seats by COMPLETION rather
        # than by the early-out, every per-leg row reads success=False and nothing else in the CSV
        # carries the verdict.
        clock_rows.append({'maneuver': 'connector_sweep', 'try': ran, 'ramp_result': verdict,
                           'reached_deg': round(float(np.degrees(eng_clock + th_at)), 3),
                           'advance_mm': round(det.advance_m() * 1000.0, 3),
                           'peak_advance_mm': round(det.peak_m * 1000.0, 3),
                           'need_mm': round(cc_need_m * 1000.0, 3),
                           'success': bool(ok), 'force_stop': bool(not swung),
                           'state_after': 'seated' if ok else 'engaged',
                           'stopped_by': ''})

        # Settle at wherever the sweep actually ended. The integrator is zeroed first so the hold
        # commands the pose the arm is AT rather than that pose plus the deflection already in it.
        last = robot.tool0()
        adm_cc.reset()
        if cc_settle > 0:
            adm_cc.hold(last, cc_settle, guard=None)
        if cc_hold > 0:
            adm_cc.hold(last, cc_hold, guard=None)
        adm_cc.stop()
        robot.arm.servo_stop()

        # det.T_tool0_conn, NOT the local from the belief reset: the detector owns the
        # connector-in-gripper relation, and reading a stale copy here would hand collar clocking
        # a connector pose wrong by exactly whatever the detector has since accounted for.
        T_tool0_conn_now = det.T_tool0_conn
        T_base_conn = robot.tool0() @ T_tool0_conn_now
        got = matrix_to_xyzrpy(inverse(T_clk) @ T_base_conn)
        # WRAPPED ONTO THE SWEEP'S OWN BRANCH. The Euler roll caps at +/-180, and this number is
        # handed to collar clocking as the angle to UNWIND -- so a sign that flipped at the band
        # edge would orbit the open gripper the long way round the part. cc_mid is the centre of
        # the band the roll must lie in, which is the branch to read it on.
        screw_rad = _wrap_near(float(got[1][0]), cc_mid)
        log.info('  achieved: %+.2f mm along +X, %+.2f deg about it from the engaged roll -- so '
                 '%+.1f deg wrt the target frame (aimed for %s). Peak advance %.2f mm.',
                 got[0][0] * 1000.0, np.degrees(screw_rad),
                 np.degrees(eng_clock + screw_rad),
                 [round(float(np.degrees(t)), 1) for t in cc_sweep], det.peak_m * 1000.0)
        if robot.arm.dry_run and not ok:
            # arm.fk is a fixed stand-in offline, so tcp_pose never moves and advance is
            # unmeasurable by construction -- pass it so the rest of the sequence is exercised.
            log.info('  dry run: advance is unmeasurable (tcp_pose is a fixed stand-in); '
                     'treating the screw as successful to exercise the rest of the sequence.')
            ok = True
        if cc_open_after:
            phase('retract')
            if not robot.gripper.open('release (post connector clocking)'):
                log.error('Gripper did not open after connector clocking.')
                return False, T_tool0_conn_now, T_base_conn
        return ok, T_tool0_conn_now, T_base_conn, float(np.degrees(screw_rad))

    def collar_clocking(T_base_conn, screw_deg=None, cl_rot=cl_rot, sp_on=sp_on,
                        unlocking=False):
        """COLLAR CLOCKING -- grasp the locking collar AXIALLY and twist the wrist.

        RUNS IN BOTH DIRECTIONS. With `unlocking` the identical approach is driven with the
        turn NEGATED and the seat push off, which is what disassembly uses: the station, the
        wall check, the wrist-3 budget and the IK scan are all the same geometry, so sharing
        them is what keeps the unlock landing on the ring the lock turned. The wrist-margin
        arithmetic below already reads the SIGN of the rotation (sorted() and max(0, -rot)),
        so a negative turn is budgeted correctly rather than accidentally.

        The gripper is brought onto the connector axis POINTING ALONG IT, fingers parallel to the
        cable, so tool0 sits on the axis 183 mm back and its Z is collinear with it. Turning the
        collar about that axis is then a rotation about tool0's own Z -- a WRIST TWIST, with the
        flange stationary.

        WHY, and the number that decides it: the socket is wall-mounted. The old radial grasp
        (fingertip frame ON the collar frame, jaws closing across the ring from the side) put
        tool0 183 mm OFF the axis at +25 mm past the mating face, and a 90 deg turn swept it
        through a 258 mm arc at that depth -- straight across the wall. Axially tool0 sits at
        -158 mm and travels 0 mm for the same turn. Same grasp on the ring (the jaws still close
        across a diameter), 183 mm more wall clearance, and no lateral sweep at all.

        WHAT IT COSTS: the approach is no longer a pure orbit from where the connector sweep left the
        arm. Radial and axial differ by a 90 deg pitch about the jaw-closing axis, which cannot be
        done with the open fingers still around the cable -- so the sequence retreats along the
        connector -X first, reorients in clear space, and only then advances back down the axis.
        Every station is measured along the axis and gated against `wall_standoff_mm`.

        THE CABLE THREADS THE JAW. Advancing along +X with the gripper on the axis runs the cable
        down between the open fingers. That is the intended motion, and it is the one part of this
        that geometry cannot verify: the advance is guarded so a snag stops it, but a cable stiff
        enough to hold itself off the axis will be pushed rather than threaded. Jog it once with
        the phase gate before trusting it.

        A force-guard stop during the TURN is NOT failure. A collar that has reached its lock
        stops turning, which is the intended end state and is indistinguishable here from jamming;
        the achieved angle is logged for the operator to judge.

        TODO: grasp verification and failure recovery on the close -- a missed collar currently
        turns an empty gripper."""
        # THE AXIS IS THE SOCKET'S, NOT THE ARM'S. T_base_conn is rebuilt at RUNTIME from
        # robot.tool0(), so it carries every deviation the sweep accumulated -- spring yield, an
        # advance that stopped short. Turning about it means turning about where the ARM ended up
        # rather than about the part.
        #
        # What the measurement DOES legitimately know is how far the bayonet cammed the connector
        # in along its own axis. Keep that (project the measured origin onto the true axis) and
        # discard the lateral and angular drift, which the socket forbids.
        axis = T_clk[:3, 0]
        axn = axis / float(np.linalg.norm(axis))
        cammed = float(np.dot(T_base_conn[:3, 3] - T_clk[:3, 3], axn))
        point = T_clk[:3, 3] + cammed * axn
        # COLLAR AXIS OFFSET (collar_clocking.axis_offset_mm, SOCKET-frame xyz). The declared
        # frame origin is the mating-face reference, not necessarily on the barrel centreline the
        # collar turns about. Shifts the LINE the whole maneuver works about; Y/Z move the line, X
        # only slides the reference point along it. Scoped to collar clocking -- the connector sweep
        # still turns about the unoffset axis. Resolved in the ROLL-FREE basis, see
        # axis_offset_base.
        off_conn = np.asarray(cl_axis_off, dtype=float) / 1000.0
        if float(np.linalg.norm(off_conn)) > 0.0:
            point = point + axis_offset_base()
            log.info('  collar axis OFFSET by %s mm (socket frame, NOT rolled by the %+.1f deg '
                     'engage clock angle) -> the line moves %.2f mm laterally.',
                     np.round(cl_axis_off, 2).tolist(), np.degrees(eng_clock),
                     float(np.linalg.norm(off_conn - np.dot(off_conn, [1.0, 0.0, 0.0])
                                          * np.array([1.0, 0.0, 0.0]))) * 1000.0)
        T_base_axis = T_clk.copy()
        T_base_axis[:3, 3] = point
        here = robot.tool0()

        # THE CONNECTOR MUST BE STRAIGHT IN THE SOCKET before anything is threaded down its axis.
        # This is the gate max_offaxis_tilt_deg now guards: the measured connector frame's tilt
        # from the socket axis. A cocked connector means the cable does not lie on the line the
        # gripper is about to advance along, so the jaw would meet it side-on instead of
        # swallowing it. (It used to gate the ARM's orientation, which only meant anything while
        # the approach was an orbit from wherever the sweep ended.)
        _d = T_base_conn[:3, 3] - point
        _lat = float(np.linalg.norm(_d - np.dot(_d, axn) * axn))
        _tilt = float(np.degrees(np.arccos(np.clip(
            abs(float(np.dot(T_base_conn[:3, 0] / np.linalg.norm(T_base_conn[:3, 0]), axn))),
            0.0, 1.0))))
        log.info('  axis: %s, advanced %+.2f mm by the sweep. The measured connector frame sits '
                 '%.2f mm lateral / %.2f deg tilted from it (gate %.1f deg).',
                 'SOCKET (target frame)' if pe_frame == 'target' else 'BELIEVED connector',
                 cammed * 1000.0, _lat * 1000.0, _tilt, cl_tilt_deg)
        if _tilt > cl_tilt_deg:
            log.error('COLLAR CLOCKING: the connector is %.1f deg off the socket axis, over the '
                      '%.1f deg gate (collar_clocking.max_offaxis_tilt_deg). The cable will not '
                      'lie on the line the gripper advances down -- refusing to thread it.',
                      _tilt, cl_tilt_deg)
            return False

        # THE COLLAR SITS ON THE CONNECTOR AXIS, collar_offset_mm along it from the connector
        # frame ORIGIN. Nothing here is back-calculated from how the part sits in the fingers: the
        # belief reset put the connector AT T_clk, so the ring's station is a property of the
        # CONNECTOR, and T_base_axis already carries that axis and the cammed origin.
        #
        # Deriving it through the FINGERTIP (collar = the live pad station + offset) is what put
        # the target 45.7 mm behind where it belongs: that 45.7 mm is where the PADS sit relative
        # to the mating face -- a fact about the grasp, not about the collar -- and it dragged the
        # ring's station with it.
        collar_x = cl_off_m
        # WHERE TO GRASP, as an ABSOLUTE roll about the socket +X (wrt the target frame), the same
        # convention connector_clocking.sweep_deg uses. The collar is a body of revolution, so any
        # angle grips the same ring -- what this decides is where the TURN starts and ends, and
        # therefore how much wrist range it needs. null = the engaged roll.
        #
        # This replaces prewind_deg. A pre-wind existed because the radial approach could only
        # reach the ring by orbiting from wherever the sweep left the arm, so the start angle was
        # whatever that orbit could afford. The axial approach is placed in free space, so the
        # start angle is simply stated.
        # ---- THE APPROACH, IN FOUR LEGS -------------------------------------------------------
        # 1. REALIGN with the ENGAGEMENT POSE -- the flange attitude the connector actually went
        #    in at, not the one the oscillating sweep happened to stop on.
        # 2. RETRACT `retract_mm` along the TARGET connector -X from there, straight away from the
        #    wall, the open fingers sliding ALONG the cable rather than across it.
        # 3. REORIENT onto the axis: tool0 +Z along the connector +X (so the fingertip axis IS the
        #    connector axis) and tool0 -Y on the connector -Z (the roll that leaves the wrist).
        # 4. ADVANCE back down the axis onto the collar, threading the cable into the open jaw.
        #
        # WHY LEG 1 EXISTS. Every station below is measured from where the arm stood when the
        # connector MATED. `robot.tool0()` at this point is wherever the last sweep leg stopped:
        # rolled by up to sweep_deg, and by a compliant stroke that yields to contact. Retracting
        # from THAT carries the sweep's attitude into the reorientation and makes the retract
        # distance mean something slightly different on every run. The engagement pose is a clean,
        # known attitude on the axis that the operator has already watched go in.
        #
        # IT IS RECONSTRUCTED, NOT STORED. connector_clocking captured the in-hand belief at
        # engagement (`_T_tool0_conn` = the connector expressed in tool0 THERE), and T_clk is that
        # same connector in base -- so the flange pose that produced it is T_clk @ inverse(belief).
        T_engaged = T_clk @ inverse(_T_tool0_conn)
        T_withdraw = translation_matrix(-cl_retract_m * axn) @ T_engaged
        ftip_p = (T_withdraw @ robot.T_tool0_fingertip)[:3, 3]

        _dp = ftip_p - point
        ftip_lat = float(np.linalg.norm(_dp - np.dot(_dp, axn) * axn))
        (log.info if ftip_lat < 1e-3 else log.warning)(
            '  fingertip sits %.2f mm off the connector axis while gripping. The axial '
            'pose puts it back ON the axis, so this is corrected rather than carried%s.',
            ftip_lat * 1000.0,
            '' if ftip_lat < 1e-3 else ' -- the pads close on a cable coaxial with the connector, '
            'so a non-zero value here is a FRAME error (compare frames.yaml fingertip against '
            'the connector frame), not a property of the part')

        # THE AXIAL POSE, BUILT DIRECTLY -- an END STATE, not a path.
        #
        # What this maneuver needs is: tool0 +Z along the connector +X, the fingertip ON the axis
        # at the collar station, so the jaws close across the ring and the turn becomes a wrist
        # twist. Constraining HOW the arm gets there -- one pitch about the connector -Y, pivoting
        # on the fingertip -- adds a requirement the frames need not satisfy. A rotation about a
        # single fixed axis carries z_now onto the connector axis ONLY when the two make the same
        # angle with that axis, and the attitude the sweep leaves has no reason to. A large
        # residual there is not a frame error; it is the path being over-specified.
        #
        # So state the target and let IK find the joints. The reorient is a move_j through free
        # space with the fingers open and the arm a retract's length back, so it is free to take
        # whatever combined rotation it likes -- nothing downstream depends on the route, only on
        # the pose it ends at, which is checked below.
        G_axial = (xyzrpy_to_matrix([0.0, 0.0, 0.0], [0.0, -np.pi / 2.0, 0.0])
                   @ inverse(robot.T_tool0_fingertip))

        def at(station_m):
            """Fingertip at `station_m` along the connector axis, ON it, tool0 +Z along it."""
            return T_base_axis @ translation_matrix([station_m, 0.0, 0.0]) @ G_axial

        # Reorient at the station the withdraw reached, so the threading leg is exactly the
        # retract's length and no more.
        ftip_x = float((inverse(T_base_axis) @ T_withdraw @ robot.T_tool0_fingertip)[0, 3])
        T_reorient = at(ftip_x)
        adv_m = collar_x - ftip_x             # the guarded axial leg that threads the cable
        T_grip0 = at(collar_x)                # the grasp at roll 0

        # THE END STATE IS VERIFIED, since the path no longer proves it by construction. These are
        # exact by build, so a non-zero reading means T_base_axis or the fingertip frame is wrong.
        _rel = inverse(T_base_axis) @ T_grip0
        _zdot = float(np.dot(_rel[:3, 2], [1.0, 0.0, 0.0]))
        _ftip_off = float(np.linalg.norm(
            (inverse(T_base_axis) @ T_grip0 @ robot.T_tool0_fingertip)[1:3, 3])) * 1000.0
        log.info('  axial pose: tool0 +Z . connector +X = %.6f, tool0 %.4f mm off the axis, '
                 'fingertip %.4f mm off it, %.1f mm to thread.',
                 _zdot, float(np.linalg.norm(_rel[1:3, 3])) * 1000.0, _ftip_off, adv_m * 1000.0)
        if abs(_zdot - 1.0) > 1e-6 or _ftip_off > 1e-3:
            log.error('COLLAR CLOCKING: the axial grasp pose does not come out axial -- tool0 +Z '
                      'is %.3f deg off the connector +X and the fingertip %.3f mm off the axis. '
                      'That is a FRAME problem (frames.yaml fingertip, or the connector frame '
                      'rpy), not a reachability one.',
                      np.degrees(np.arccos(np.clip(_zdot, -1.0, 1.0))), _ftip_off)
            return False

        # ---- THE GRASP CLOCK ANGLE, and why it is worth solving for ---------------------------
        # The reorientation from the lift-off pose (radial, where the sweep left the wrist) to the
        # axial retreat pose is the largest single rotation in the app, and its size depends
        # ENTIRELY on this angle: measured on the shipped fixture it runs 180 deg at 0 and 90 deg
        # at 180. A 180 deg tool reorientation is where the analytic IK's nearest-branch answer
        # stops being reachable from the seed -- which is what "the axial retreat station is
        # unreachable" looks like.
        #
        # The collar is a body of revolution, so this angle is FREE: any of them grips the same
        # ring. So spend it. null = pick the cheapest angle that actually solves; a number pins it
        # (absolute, wrt the target frame, like connector_clocking.sweep_deg).
        #
        # The tool0 station along the axis does NOT depend on the angle -- every candidate is a
        # rotation ABOUT the axis -- so this cannot trade wall clearance for reach.
        def roll(T, th):
            """`T` rolled `th` about the connector axis. The collar is a body of revolution, so
            this is the one freedom the pitch leaves and the only thing the prewind spends."""
            return rotate_about_axis(T, axis, point, th)

        def _reachable(th):
            """True when the reorient, grasp and turn-end all IK-solve at roll `th`."""
            _grip = roll(T_grip0, th)
            return all(robot.arm.ik(_T, seed_c) is not None for _T in (
                roll(T_reorient, th), _grip,
                translation_matrix(cl_push_m * axn)
                @ rotate_about_axis(_grip, axis, point, cl_rot)))

        seed_c = robot.arm.ik(T_withdraw, robot.arm.q()) or robot.arm.q()

        # ---- THE CLOCK ANGLE IS A WRIST-3 OFFSET, EXACTLY -------------------------------------
        # After the pitch, tool0 stands ON the collar axis with its Z collinear (the fingertip
        # frame is a pure 183 mm translation along tool0 Z, so nothing puts the flange off the
        # line). Rolling the grasp about that axis therefore leaves the FLANGE ORIGIN fixed and
        # turns it about its own Z, which is the joint-6 axis. Checked numerically across the
        # band: the tool0 origin stays 0.0000 mm off the line and tool0_Z . axis = 1.000000.
        #
        # Therefore JOINTS 1..5 ARE IDENTICAL AT EVERY CLOCK ANGLE, and q6(th) = q6(0) + th. The
        # station picks the first five joints; the clock angle picks the sixth and nothing else.
        #
        # That turns the angle from something to search for into something to CALCULATE -- and
        # into a PREWIND, because what the turn actually needs is wrist_3 range: rotation_deg of
        # it, from wherever the grasp starts. Spending that range is the whole job of this angle,
        # and it has to be spent BEFORE the arm threads itself down the axis. Discovering it
        # afterwards means finding out with the collar clamped in the fingers.
        #
        # (The old code scanned 24 candidate angles at 3 IK calls each -- 72 calls to explore a
        # family that is one joint. Two calls answer it, and a 15 deg grid could miss a feasible
        # window narrower than its own step.)
        q_grip0 = robot.arm.ik(T_grip0, seed_c)
        if q_grip0 is None:
            log.error('COLLAR CLOCKING: the collar grasp station does not IK-solve at all. That '
                      'is joints 1-5, i.e. REACH -- no clock angle can help, because the angle '
                      'only moves wrist_3. Reduce collar_clocking.retreat_mm (now %.0f mm) or '
                      'move the fixture.', cl_retreat_m * 1000.0)
            return False

        # THE USABLE WRIST-3 RANGE, asked of the CONTROLLER at that exact configuration rather
        # than assumed to be the model's +/-360 (see _wrist3_window).
        w_lo, w_hi = _wrist3_window(robot.arm, q_grip0)
        need = abs(cl_rot) + 2.0 * cl_w3_margin
        if (w_hi - w_lo) < need:
            log.error('COLLAR CLOCKING: wrist_3 has %.0f deg of range in this configuration '
                      '(%+.0f .. %+.0f deg) but the turn needs %.0f (rotation_deg %.0f plus '
                      '2 x %.1f deg of margin). NO grasp angle can fit it -- lower '
                      'collar_clocking.rotation_deg.',
                      np.degrees(w_hi - w_lo), np.degrees(w_lo), np.degrees(w_hi),
                      np.degrees(need), np.degrees(cl_rot), np.degrees(cl_w3_margin))
            return False

        # THE FEASIBLE CLOCK-ANGLE WINDOW. The turn runs q6 -> q6 + rotation_deg, so the START
        # must sit a full rotation inside whichever end it travels toward -- which is what makes
        # this a prewind rather than a reachability check.
        th_lo = (w_lo + cl_w3_margin) - q_grip0[5] + max(0.0, -cl_rot)
        th_hi = (w_hi - cl_w3_margin) - q_grip0[5] - max(0.0, cl_rot)

        # ---- THE ROLL: tool0 -Y ONTO THE CONNECTOR -Z -----------------------------------------
        # G_axial fixes two of the three rotational freedoms -- tool0 +Z along the connector +X --
        # and the roll ABOUT that axis is the third. The collar is a body of revolution, so every
        # roll grips the same ring; what the angle decides is the attitude the gripper arrives in
        # and how much wrist_3 the turn has left.
        #
        # Spend it on tool0 -Y || connector -Z. That is the same rule the reorientation before the
        # sweep uses, so the wrist arrives the way up the operator has already seen, and on the
        # shipped fixture it is also the CHEAPER reorientation (about 90 deg of tool swing against
        # 180 at zero roll -- and a 180 deg tool reorientation is where the analytic IK's
        # nearest-branch answer stops being reachable from the seed).
        #
        # CLOSED FORM, not a search. tool0 +Y at zero roll is perpendicular to the connector axis
        # by construction (it is a column of a rotation whose +Z IS the axis), so a single atan2
        # gives the angle about +X that carries it onto the connector +Z exactly. On the shipped
        # frames the build puts tool0 -Y on the connector +Z -- the wrong way up -- and this comes
        # out at 180 deg. _fit_turn below still shifts it by whole turns onto a wrist_3 branch
        # with room for the turn.
        #
        # A pinned collar_clocking.grasp_clock_deg overrides, stated as an ABSOLUTE roll about the
        # socket +X wrt the target frame (the convention connector_clocking.sweep_deg uses), for
        # when the attitude must be fixed rather than derived.
        v0 = G_axial[:3, 1]                        # tool0 +Y at zero roll, in connector coords
        want = np.array([0.0, 0.0, 1.0])           # connector +Z -- tool0 -Y then lands on the -Z
        th_want = float(np.arctan2(float(np.dot([1.0, 0.0, 0.0], np.cross(v0, want))),
                                   float(np.dot(v0, want))))
        pinned = cl_grasp_clock is not None
        if pinned:
            # _fit_turn already shifts by whole turns, so no branch-wrapping is needed here.
            th_now = float(matrix_to_xyzrpy(inverse(T_clk) @ T_reorient)[1][0])
            th_want = cl_grasp_clock - eng_clock - th_now
        else:
            log.info('  roll %+.1f deg about the connector axis lays tool0 -Y on the connector -Z '
                     '(collar_clocking.grasp_clock_deg is null, so it is derived).',
                     np.degrees(th_want))
        th_grasp, clamped = _fit_turn(th_want, th_lo, th_hi)

        q6_start = q_grip0[5] + th_grasp
        log.info('--- WRIST-3 PREWIND --- grasp at %+.1f deg wrt the target frame puts wrist_3 at '
                 '%+.1f deg, ending at %+.1f after the %+.1f deg turn. The controller allows '
                 '%+.1f .. %+.1f deg; margin %.1f deg at each end.',
                 np.degrees(eng_clock + th_grasp), np.degrees(q6_start),
                 np.degrees(q6_start + cl_rot), np.degrees(cl_rot),
                 np.degrees(w_lo), np.degrees(w_hi), np.degrees(cl_w3_margin))
        if clamped:
            log.warning('  the %s roll (%+.1f deg) leaves no room for the turn on ANY whole-turn '
                        'branch, so it was CLAMPED to %+.1f deg. The ring gripped is the same; '
                        'the approach attitude is not -- check the reorientation looks sane.',
                        'pinned collar_clocking.grasp_clock_deg' if pinned
                        else 'derived (tool0 -Y on the connector -Z)',
                        np.degrees(th_want), np.degrees(th_grasp))
        elif abs(th_grasp - th_want) > np.radians(0.5):
            log.info('  (rolled %+.0f deg off the -Y/-Z attitude onto a wrist_3 branch with '
                     'room -- the collar is a body of revolution, so the grasp on the ring is '
                     'identical, but the gripper arrives turned by that much.)',
                     np.degrees(th_grasp - th_want))

        # WRIST_3 now has room. Whether the STATIONS solve in joints 1..5 is a different question,
        # and the only one left -- so a failure here names the knob that actually moves it.
        if not _reachable(th_grasp):
            log.error('COLLAR CLOCKING: wrist_3 has room at a %+.1f deg roll, but the reorient, '
                      'grasp or turn-end pose does not IK-solve there. That is joints 1-5 -- '
                      'REACH, not wrist range -- so the knob is collar_clocking.retract_mm (now '
                      '%.0f mm) or the fixture position, NOT the roll.',
                      np.degrees(th_grasp), cl_retract_m * 1000.0)
            return False
        T_retreat = roll(T_reorient, th_grasp)      # backed off, already pitched onto the axis
        T_grip = roll(T_grip0, th_grasp)            # fingertip ON the collar
        T_end = translation_matrix(cl_push_m * axn) @ rotate_about_axis(
            T_grip, axis, point, cl_rot)
        log.info('--- COLLAR CLOCKING (axial) --- collar %.1f mm along the connector +X from the '
                 'ORIGIN, on the axis. Realign with the engagement pose, retract %.0f mm along '
                 'the connector -X, reorient onto the axis (tool0 +Z on the connector +X, tool0 '
                 '-Y on the -Z), advance %+.1f mm down the axis onto the ring (the fingertip '
                 'lands at the %+.1f mm station), grasp, then TWIST the wrist %+.1f deg while '
                 'pushing %+.1f mm.',
                 cl_off_m * 1000.0, cl_retract_m * 1000.0, adv_m * 1000.0, ftip_x * 1000.0,
                 np.degrees(cl_rot), cl_push_m * 1000.0)

        # ---- WALL CLEARANCE, checked before anything moves ------------------------------------
        # The socket is wall-mounted, so the number that matters is how far along the connector +X
        # the FLANGE gets: +X points into the wall, and tool0 is the bulkiest thing on the arm.
        # Every pose below is a planned station on the axis, so the whole maneuver's approach to
        # the wall is known up front rather than discovered by driving into it.
        stations = (('withdraw', T_withdraw), ('reorient', T_retreat),
                    ('collar grasp', T_grip), ('turn end', T_end))
        x_tool = {lab: float((inverse(T_clk) @ T)[0, 3]) * 1000.0 for lab, T in stations}
        worst_lab = max(x_tool, key=x_tool.get)
        log.info('  tool0 station along the connector +X (larger = closer to the wall): %s. '
                 'Closest: %s at %+.1f mm.',
                 ', '.join('%s %+.1f' % (k, v) for k, v in x_tool.items()),
                 worst_lab, x_tool[worst_lab])
        log.info('  (the sweep left the arm at %+.1f mm and the engagement pose is at %+.1f mm '
                 '-- both are INHERITED, poses the arm has already occupied rather than stations '
                 'chosen here, so they are reported and not gated.)',
                 float((inverse(T_clk) @ here)[0, 3]) * 1000.0,
                 float((inverse(T_clk) @ T_engaged)[0, 3]) * 1000.0)
        if cl_wall_mm is not None and x_tool[worst_lab] > cl_wall_mm:
            log.error('COLLAR CLOCKING: the %s pose puts tool0 at %+.1f mm along the connector '
                      '+X, past the %+.1f mm wall standoff (collar_clocking.wall_standoff_mm). '
                      'Refusing to move toward the wall.',
                      worst_lab, x_tool[worst_lab], cl_wall_mm)
            return False
        # REACHABILITY of every station, before anything grips: discovering mid-turn that the far
        # end is unreachable leaves the collar clamped in a stalled gripper.
        for lab, T_chk in stations[2:]:
            if robot.arm.ik(T_chk, robot.arm.q()) is None:
                log.error('COLLAR CLOCKING: the %s pose is unreachable. The twist needs %.0f deg '
                          'of wrist range from a grasp at %+.1f deg; adjust '
                          'collar_clocking.rotation_deg or grasp_clock_deg.', lab,
                          np.degrees(cl_rot), np.degrees(eng_clock + th_grasp))
                return False

        # ---- SEAT PUSH, done WHERE THE ARM ALREADY IS -----------------------------------------
        # Re-grip the junction and PRESS the connector deeper along +X until force_n is SUSTAINED
        # for persistence_s; the guard TRIP is the SUCCESS. Completing the travel without ever
        # building the force means it slid in freely, which also ends deeper.
        #
        # It runs FIRST, before the retreat, because the pads are already around the cable at the
        # junction -- exactly where the push wants them. (The radial sequence had to centre on the
        # axis first for the collar's sake and pressed from there, a few mm off where the grasp
        # actually held.) The press moves the CONNECTOR, so the collar stations are shifted by the
        # measured travel afterwards: the ring rides the connector.
        d_push = 0.0
        if sp_on:
            # NO RE-GRIP. The sweep left the pads closed on the connector at the junction --
            # exactly where the push wants them -- so opening and re-closing here would only
            # release a part that is already held, risk a worse bite, and cost the grasp check
            # a chance to abort a run that is going fine. connector_clocking.open_gripper_after
            # must therefore be FALSE; the release happens after the push instead, because the
            # retreat and the advance are the legs that genuinely need open fingers.
            if True:
                T_a = robot.tool0()
                # The spring must STRETCH force/S to apply force_n, so the reference travel has
                # to cover that stretch on top of any real seating motion.
                u_p = T_a[:3, :3].T @ axn
                S_p = 1.0 / float(np.sum((u_p ** 2) / adm_cl.S[:3]))
                if sp_travel_m < sp_force / S_p:
                    log.warning('SEAT PUSH: max_travel_mm (%.1f) cannot stretch the %.0f N/m '
                                'spring to %.1f N (needs %.1f mm) -- the push cannot reach its '
                                'force.', sp_travel_m * 1000.0, S_p, sp_force,
                                sp_force / S_p * 1000.0)
                T_b = translation_matrix(sp_travel_m * axn) @ T_a
                adm_cl.reset()
                adm_cl.warmup(T_a, tare_fn=tare)      # tare while gripping and static
                guard_push.reset()
                res_p = adm_cl.ramp(T_a, T_b, seg_time(T_a, T_b), guard_push)
                if res_p != 'seated':
                    # the reference has stopped; give the guard its full persistence window
                    res_p = adm_cl.hold(T_b, sp_persist + 0.5, guard_push)
                d_push = float(np.dot(robot.tool0()[:3, 3] - T_a[:3, 3], axn))
                adm_cl.reset()
                adm_cl.stop()
                robot.arm.servo_stop()
                pressed = res_p == 'seated'
                clock_rows.append({'maneuver': 'seat_push', 'try': 1, 'ramp_result': res_p,
                                   'advance_mm': round(d_push * 1000.0, 3),
                                   'success': True, 'force_stop': bool(pressed),
                                   'state_after': 'seated',
                                   'stopped_by': guard_push.tripped_by or ''})
                (log.info if pressed else log.warning)(
                    'SEAT PUSH: %s -- connector moved %+.2f mm.',
                    'held %.1f N for %.1f s (pressed home)' % (sp_force, sp_persist) if pressed
                    else 'never built %.1f N over %.1f mm of reference travel'
                         % (sp_force, sp_travel_m * 1000.0),
                    d_push * 1000.0)
                if not robot.gripper.open('release (seat push)'):
                    log.error('SEAT PUSH: gripper did not release -- cannot retreat with the '
                              'junction clamped.')
                    return False
                if abs(d_push) > 1e-6:
                    # the connector (and its collar) moved deeper -- keep the ring in the sights.
                    # T_engaged and T_withdraw ride along too: the realign is what the retract is
                    # measured from, so it has to track the PART rather than the history.
                    T_engaged = translation_matrix(d_push * axn) @ T_engaged
                    T_withdraw = translation_matrix(d_push * axn) @ T_withdraw
                    T_retreat = translation_matrix(d_push * axn) @ T_retreat
                    T_grip = translation_matrix(d_push * axn) @ T_grip
                    T_end = translation_matrix(d_push * axn) @ T_end

        # PACED BY ITS OWN PHASE SCALE. The two legs below are FREE SPACE -- the fingers are
        # open and clear, nothing is being inserted -- but they used to run at the `standoff`
        # scale, which is contact-approach pacing (the global 25 mm/s / 30 deg/s at 1.0x). The
        # reorient is the longest single move in the app: a ~90 deg swing plus a few hundred mm of
        # travel, and at 25 mm/s the translation cap alone puts it near arm.move_timeout_s.
        # `collar_approach` exists so it can be paced like the escape it is. The axial ADVANCE
        # below deliberately does NOT use it -- that one threads the cable and stays slow.
        phase('collar_approach')
        # ---- 1. REALIGN WITH THE ENGAGEMENT POSE ----------------------------------------------
        # Undo the sweep. The difference between where the last leg stopped and the engagement
        # pose is essentially a roll about the connector axis, and a straight move_l would cut its
        # chord -- at 60 deg with tool0 183 mm off the axis the fingertip dips about 25 mm off
        # true, which with OPEN fingers around the cable is a swipe rather than a slide.
        #
        # So the path is parametrised BY THE FINGERTIP, not by the flange: the fingertip walks the
        # straight line between its two stations (both ON the axis, differing only by whatever the
        # seat push drove in) while the attitude slerps, and the gripper body swings around the
        # cable instead of through it. Compliant and guarded, like every other leg with something
        # in front of it.
        _now = robot.tool0()
        _ft_off = robot.T_tool0_fingertip[:3, 3]
        _p0 = (_now @ robot.T_tool0_fingertip)[:3, 3]
        _p1 = (T_engaged @ robot.T_tool0_fingertip)[:3, 3]

        def _to_engaged(f):
            """tool0 at fraction `f` of the realign, with the FINGERTIP on the straight line
            between its start and end stations."""
            T = slerp_matrix(_now, T_engaged, f)
            T[:3, 3] = (_p0 + f * (_p1 - _p0)) - T[:3, :3] @ _ft_off
            return T

        _re_lin, _re_ang = pose_error(_now, T_engaged)
        if unlocking:
            # UNLOCKING DOES NOT REALIGN. The realign exists to undo the bayonet sweep so the
            # collar stations are measured from the mated pose -- but on the way OUT the
            # connector is already mated and the arm is already clear of it (the escape ran
            # before this), so driving the OPEN gripper back onto the connector would be a
            # pointless approach to a pose we only want to leave again. The collar is turned
            # from wherever it sits: the station below is absolute (built from the axis
            # frame), so the approach still lands on the ring without it.
            log.info('  UNLOCKING -- skipping the realign with the engagement pose (%.1f deg / '
                     '%.1f mm away); the collar station is absolute, so the approach does not '
                     'need it.', np.degrees(_re_ang), _re_lin * 1000.0)
        elif _re_ang > np.radians(0.5) or _re_lin > 1e-4:
            adm_cl.reset()
            adm_cl.warmup(_now)
            guard_shared.reset()
            res_re, f_re = screw_ramp(adm_cl, _to_engaged, guard_shared, cl_v, cl_w,
                                      float(np.degrees(_re_ang)),
                                      label='realign with the engagement pose ')
            adm_cl.reset()
            adm_cl.stop()
            robot.arm.servo_stop()
            if res_re == 'seated':
                log.error('COLLAR CLOCKING: the force guard tripped (%s) %.0f%% of the way through '
                          'the %.1f deg realign with the engagement pose. The fingers are OPEN and '
                          'still around the cable, so nothing is clamped -- something is fouling '
                          'the swing.', guard_shared.tripped_by or 'unknown', f_re * 100.0,
                          np.degrees(_re_ang))
                return False
            log.info('  realigned with the engagement pose (%.1f deg, %.1f mm of flange travel).',
                     np.degrees(_re_ang), _re_lin * 1000.0)
        else:
            log.info('  already at the engagement pose (%.2f deg, %.2f mm) -- nothing to realign.',
                     np.degrees(_re_ang), _re_lin * 1000.0)

        # ---- 2. RETRACT ALONG THE CABLE, straight away from the wall --------------------------
        # A pure translation along the connector -X with the open fingers still around the cable:
        # they slide ALONG it rather than across it, so nothing is swept, and every millimetre is
        # away from the wall. It also puts the reorient that follows as far from the wall as the
        # maneuver ever gets.
        if cl_retract_m > 1e-6:
            if not _guarded(robot, guard_shared, lambda: robot.arm.move_l(
                    T_withdraw, label='collar retract (connector -X)')):
                log.error('Could not retract %.0f mm along the connector -X from the engagement '
                          'pose (collar_clocking.retract_mm).', cl_retract_m * 1000.0)
                return False

        # ---- 3. REORIENT ONTO THE AXIS ---------------------------------------------------------
        # One move_j to the stated end pose. A joint move rather than a straight line because it
        # is the only large reorientation in the maneuver, and it happens a retract's length back
        # with the fingers clear, so the route it takes does not matter -- only the pose it
        # reaches, which was verified above.
        #
        # TWO FAILURES, REPORTED APART. "Could not reach" used to cover both an unreachable pose
        # and a move that started and did not finish, which are opposite problems: one is fixed by
        # moving the station, the other by pacing or the pendant.
        q_ret = robot.arm.ik(T_retreat, robot.arm.q())
        if q_ret is None:
            # UNREACHABLE. Say so, and say which retract distances DO solve -- that turns a dead
            # end into a number to put in the config. (IK is seeded from the current joints, so
            # this also catches a branch the arm cannot get to from where it stands.)
            ok_mm = [d for d in (100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 500.0)
                     if robot.arm.ik(
                         translation_matrix((cl_retract_m - d / 1000.0) * axn) @ T_retreat,
                         robot.arm.q()) is not None]
            log.error('COLLAR CLOCKING: the pitched (axial) pose is UNREACHABLE -- no IK solution '
                      'with the fingertip at %+.1f mm along the connector +X (tool0 %+.1f mm, on '
                      'the axis, pointing at the socket). This is REACH or a joint limit, not '
                      'speed: collar_clocking.retract_mm (%.0f) is what puts the flange that far '
                      'back. %s',
                      ftip_x * 1000.0, x_tool['reorient'], cl_retract_m * 1000.0,
                      ('Retract distances that DO solve from here: %s mm.'
                       % ', '.join('%.0f' % d for d in ok_mm)) if ok_mm else
                      'NO retract distance from 100 to 500 mm solves -- the axial orientation '
                      'itself is out of reach at this fixture pose, not just the distance.')
            return False
        if not _guarded(robot, guard_shared, lambda: robot.arm.move_j(
                q_ret, label='collar pitch onto the axis')):
            log.error('COLLAR CLOCKING: the pitch onto the axis did not finish. The pose IS '
                      'reachable (IK solved), so this is the force guard tripping on the way, the '
                      'controller rejecting the move, or arm.move_timeout_s (%.0f s) running out '
                      '-- raise speed.phase_scale.collar_approach (now %.2fx) or move_timeout_s.',
                      robot.arm.move_timeout, float(scales.get('collar_approach', 1.0)))
            return False

        # ---- 4. ADVANCE DOWN THE AXIS, threading the cable into the open jaw ------------------
        # A PURE TRANSLATION along the connector +X: no rotation, so a straight move is exactly
        # the right path and there is no chord to cut. Guarded, because this is the leg that runs
        # the cable between the open fingers -- a snag must stop it rather than push through.
        # Compliant, for the same reason, and back on the SLOW standoff pacing: this is the one
        # leg of the approach with something in front of it.
        phase('standoff')
        adm_cl.reset()
        adm_cl.warmup(T_retreat)
        guard_shared.reset()
        res_adv = adm_cl.ramp(T_retreat, T_grip,
                              seg_time(T_retreat, T_grip, g_v * s_std, g_w * s_std),
                              guard_shared)
        adm_cl.stop()
        robot.arm.servo_stop()
        if res_adv == 'seated':
            log.error('COLLAR CLOCKING: the force guard tripped during the axial advance (%s) '
                      'after %.1f of %.1f mm. The cable most likely did not thread into the open '
                      'jaw -- the fingers are still clear of the ring, so nothing is clamped.',
                      guard_shared.tripped_by or 'unknown',
                      float(np.dot(robot.tool0()[:3, 3] - T_retreat[:3, 3], axn)) * 1000.0,
                      adv_m * 1000.0)
            return False

        # ---- DID THE PREWIND SURVIVE? --------------------------------------------------
        # The prewind was computed at the PLANNED grip pose; the arm reached the real one through
        # a compliant advance that yields to contact, so wrist_3 is where the servo left it, not
        # necessarily where the plan put it. Checked HERE, with the fingers still open and the
        # collar not yet clamped, because this is the last moment a refusal is free.
        q6_now = float(robot.arm.q()[5])
        turn_lo, turn_hi = sorted((q6_now, q6_now + cl_rot))
        if turn_lo < w_lo + cl_w3_margin or turn_hi > w_hi - cl_w3_margin:
            log.error('COLLAR CLOCKING: wrist_3 is at %+.1f deg after the advance, so the %+.1f '
                      'deg turn would run %+.1f .. %+.1f and leave the usable range '
                      '%+.1f .. %+.1f (margin %.1f). The prewind aimed for %+.1f. Refusing to '
                      'turn -- the fingers are still OPEN, so nothing is clamped.',
                      np.degrees(q6_now), np.degrees(cl_rot), np.degrees(turn_lo),
                      np.degrees(turn_hi), np.degrees(w_lo), np.degrees(w_hi),
                      np.degrees(cl_w3_margin), np.degrees(q6_start))
            return False
        log.info('  wrist_3 at %+.1f deg after the advance (prewound to %+.1f) -- the %+.1f deg '
                 'turn fits with %.1f deg to spare.', np.degrees(q6_now), np.degrees(q6_start),
                 np.degrees(cl_rot),
                 min(turn_lo - w_lo, w_hi - turn_hi) * 180.0 / 3.141592653589793)

        # ---- TARE BEFORE THE GRASP -------------------------------------------------------
        # The arm is standing at the collar station with the fingers still OPEN and clear of the
        # ring: the only genuinely unloaded moment in the maneuver. The tare used to happen
        # inside the warmup BELOW, i.e. after the close, which folds in whatever the jaws
        # preload against the captive ring -- jaw-on-jaw force cancels at the wrist, but an
        # off-centre close reacts through the connector into the socket and does not. At
        # max_torque_nm 1.0 that offset is a large fraction of the whole limit, so the turn could
        # trip on the grasp rather than on the lock.
        #
        # An IDLE tare (settle=True), unlike the servo-active one warmup performs: the gripper
        # close blocks for ~1 s, and streaming servoL around a blocking call is exactly what
        # warmup's mid-hold tare exists to avoid. What that trades away is the idle-vs-servo
        # offset, which is the uncompensated tool weight -- and robot.payload IS configured here,
        # so it is already subtracted. settle=True also re-checks the residual and warns if it
        # is not.
        if cl_tare is not None:
            robot.arm.zero_ft(settle=True)
        if not robot.gripper.close('grasp collar'):
            log.error('Gripper did not close on the collar.')
            return False
        phase('collar_clock')
        start = robot.tool0()
        adm_cl.reset()
        adm_cl.warmup(start)          # NOT tare_fn=cl_tare -- zeroed above, with open fingers
        guard_cl.reset()
        # THE TWIST. Still written as a rotation about the connector axis LINE, unchanged from the
        # radial version -- but tool0 now sits ON that line, so it resolves to a rotation about
        # tool0's own Z and the flange stays put. screw_ramp still subdivides it: with push_mm set
        # the turn is a SCREW (the axial press keeps the collar's lugs on their ramps), and a
        # translation along the rotation axis commutes with it, so the composed path is exact.
        res, _f_turn = screw_ramp(
            adm_cl,
            lambda f: (translation_matrix(cl_push_m * f * axn)
                       @ rotate_about_axis(start, axis, point, cl_rot * f)),
            guard_cl, cl_v, cl_w, abs(np.degrees(cl_rot)), label='twist ')
        _lin, turned = pose_error(start, robot.tool0())
        adm_cl.reset()
        if cl_settle > 0:
            adm_cl.hold(robot.tool0(), cl_settle, guard=None)
        adm_cl.stop()
        robot.arm.servo_stop()
        stopped = res == 'seated'            # ramp's word for a guard trip -- not the state
        clock_rows.append({'maneuver': 'collar_clocking', 'try': 1, 'ramp_result': res,
                           'grasp_clock_deg': round(float(np.degrees(eng_clock + th_grasp)), 3),
                           'turned_deg': round(float(np.degrees(turned)), 3),
                           'commanded_deg': round(float(np.degrees(cl_rot)), 3),
                           'tool0_x_mm': round(x_tool['collar grasp'], 3),
                           'success': True, 'force_stop': bool(stopped),
                           'state_after': 'seated' if unlocking else 'locked',
                           'stopped_by': guard_cl.tripped_by or ''})
        _word = 'UNLOCKED' if unlocking else 'LOCKED'
        if stopped:
            # SYMMETRIC WITH THE LOCK: the collar runs to a stop at BOTH ends of its travel, so
            # the torque building is the intended termination in either direction -- turn the
            # full rotation_deg or until the ring stops turning, whichever comes first.
            log.info('  %s -- the collar stopped on the force guard (%s) after %.1f of %.1f '
                     'deg, which is what reaching the end of its travel looks like; check it.',
                     _word, guard_cl.tripped_by, np.degrees(turned),
                     abs(np.degrees(cl_rot)))
        else:
            log.info('  %s -- collar turned %.1f deg (commanded %.1f). The flange moved '
                     '%.1f mm: a wrist twist, not an arm swing.', _word, np.degrees(turned),
                     np.degrees(cl_rot), _lin * 1000.0)
        return True


    def locate_target_visually(q_return):
        """Drive to the view pose, sweep the markers, and re-anchor the run on what they say.

        RUNS BEFORE THE PICK, with the gripper empty and the arm at home: the camera has a clear
        view of the fixture, and nothing is being carried that a view sweep could disturb. After
        the mate the socket is behind a connector and a gripper, so this is the only moment the
        markers are worth looking at."""
        from ..perception import ArucoDetector

        if camera is None:
            log.error('assembly.target_source is visual but the app has no camera.')
            return False
        try:
            plan = mloc.ViewPlan(cfg.section('marker_views'))
        except ValueError as exc:
            log.error('marker_views: %s', exc)
            return False
        # The per-marker servo refinement is OPTIONAL at run time: visual_target.servo_refine
        # switches it off WITHOUT touching the shared marker_views.servo parameters, which
        # must stay identical to marker_calibration's for the runs where it is on.
        if not bool(vt.get('servo_refine', True)):
            plan.servo.enabled = False
            log.info('VISUAL TARGET: servo refinement OFF (visual_target.servo_refine) -- '
                     'sweep views only.')
        detector = ArucoDetector(cfg, sizes_m=tool_frames.marker_sizes(vt_rig))

        # Gate BEFORE the first sweep move: the camera is about to drive a ring of views a few
        # hundred mm off the fixture, and the whole run is anchored on what it measures.
        if not phase_gate('VISUAL LOCALIZATION',
                          'The camera will sweep the markers from the home view%s. The run is '
                          'then anchored on the pose they measure.'
                          % (' and servo to each one' if plan.servo.enabled else '')):
            return False

        phase('visual_localize')
        q_view = vt.get('view_joints_deg')
        if q_view is not None:
            log.info('VISUAL TARGET: driving to the view pose %s deg.',
                     list(np.round(np.asarray(q_view, dtype=float), 1)))
            if not robot.arm.move_j(list(np.radians(np.asarray(q_view, dtype=float))),
                                    label='marker view pose'):
                log.error('VISUAL TARGET: could not reach '
                          'assembly.visual_target.view_joints_deg.')
                return False
        else:
            log.info('VISUAL TARGET: sweeping from the HOME pose (reset.home_joints_deg IS the '
                     'marker view pose; set visual_target.view_joints_deg to sweep from '
                     'somewhere else).')

        # Every image the localization estimated from, annotated + indexed under
        # <experiment>/marker_images/ -- the only record of what the run actually saw.
        images = mloc.MarkerImageWriter(out_dir, detector,
                                        enabled=getattr(plan, 'save_images', True))
        T_vis = mloc.locate(robot, camera, detector, vt_rig, plan,
                            on_view=images.sweep_view, on_servo_view=images.servo_view)
        _vis_xyz, _vis_rpy = ((None, None) if T_vis is None else matrix_to_xyzrpy(T_vis))
        images.finish(
            ['visual target localization -- rig %r' % tname, '']
            + (['NO TARGET POSE -- the markers did not produce a fused answer.']
               if T_vis is None else
               ['located target in base_link:',
                '  xyz %+8.2f %+8.2f %+8.2f mm' % tuple(v * 1000.0 for v in _vis_xyz),
                '  rpy %+7.2f %+7.2f %+7.2f deg'
                % tuple(float(np.degrees(v)) for v in _vis_rpy)]))
        if q_return is not None and vt.get('return_home_after', True):
            # BACK TO HOME BEFORE THE PICK, whatever the localisation decided: the scan, the grasp
            # geometry and every retry offset are written from the home pose, and leaving the arm
            # parked at a view pose would silently change where the pick starts.
            phase('reset')
            if not robot.arm.move_j(q_return, label='home after the marker sweep'):
                log.error('VISUAL TARGET: could not return home after the sweep.')
                return False
        if T_vis is None:
            log.error('VISUAL TARGET: the markers did not produce a target pose. Nothing has '
                      'moved and the recorded mate is untouched -- re-run with '
                      'assembly.target_source: kinematic to use it, or fix the rig.')
            return False

        # HOW FAR THE FIXTURE HAS APPARENTLY MOVED. Not an error by itself -- the whole point of
        # the rig is that the fixture MAY move -- but a shift of the wrong ORDER (a metre, a
        # quarter turn) is a stale calibration or a marker on the wrong fixture, and driving an
        # insertion trajectory at it would be the expensive way to find out.
        d_lin, d_ang = pose_error(targets[tname], T_vis)
        lim_mm = vt.get('max_shift_mm', 25.0)
        lim_deg = vt.get('max_shift_deg', 10.0)
        over = ((lim_mm is not None and d_lin * 1000.0 > float(lim_mm))
                or (lim_deg is not None and np.degrees(d_ang) > float(lim_deg)))
        (log.error if over else log.info)(
            'VISUAL TARGET: the markers put %r %.2f mm / %.2f deg from the recorded mate%s.',
            tname, d_lin * 1000.0, np.degrees(d_ang),
            ' -- OVER assembly.visual_target.max_shift_mm/_deg (%s mm / %s deg)'
            % (lim_mm, lim_deg) if over else '')
        if over:
            log.error('  Refusing to plan an insertion at it. If the fixture really did move that '
                      'far, raise the limit or re-record the targets: entry; otherwise the rig is '
                      'stale or a marker is on the wrong fixture.')
            return False
        _anchor_target(T_vis)
        log.info('VISUAL TARGET: the run is now anchored on the MEASURED socket pose.')
        return True

    def aligned_place_pose(extra_clearance_m=0.0):
        """Where the cable is set down, in base_link -- LYING ALONG THE SOCKET AXIS.

        THE CONNECTOR +X COMES OUT PARALLEL TO THE TARGET CONNECTOR +X. That is the whole
        requirement and it is easy to lose: a place that simply descends from the pick pose
        inherits whatever attitude the grip has, and with a COAXIAL grip the connector hangs
        with its axis along the tool -- i.e. pointing at the floor. The part then lands on its
        end instead of on its side.

        So the orientation here is BUILT, not inherited: heading from the socket, roll and
        pitch zero. `extra_clearance_m` lifts the release point above the resting height.

        DEFINED RELATIVE TO THE TARGET CONNECTOR, because the whole point is to leave the cable
        ALIGNED WITH THE SOCKET: yaw 0 wrt the target puts the connector axis on the same
        compass heading the socket has, which is the one heading a coaxial approach is known to
        reach. Setting it down on an arbitrary heading would fail the same way again.

        ROLL AND PITCH ARE NOT FREE -- the part is going onto a flat bench, so its axis ends up
        horizontal and its z up, whatever attitude the socket itself has. Only the HEADING
        carries over. (The recorded socket is level to within half a degree, so here the two
        readings agree to well under a millimetre; the code does not lean on that.)"""
        r = a.get('reorient_recovery', {}) or {}
        off = r.get('place_offsets', {}) or {}
        z_mm = float(off.get('z_mm', -400.0))
        d = np.array([float(off.get('x_mm', 0.0)), float(off.get('y_mm', 0.0)), z_mm],
                     dtype=float) / 1000.0
        p = T_base_tconn[:3, 3] + T_base_tconn[:3, :3] @ d      # offsets in the TARGET frame
        yaw = (float(np.arctan2(T_base_tconn[1, 0], T_base_tconn[0, 0]))
               + float(np.radians(float(off.get('yaw_deg', 0.0)))))
        T = xyzrpy_to_matrix([0.0, 0.0, 0.0], [0.0, 0.0, yaw])   # flat: roll = pitch = 0
        # THE GROUND DECIDES z, by default. A resting connector's AXIS sits one barrel-radius
        # up -- a property of the part, not a number to type twice -- and "place it on the
        # ground plane" is the actual requirement. The configured z_mm is still reported,
        # because a large gap between the two means one of the two is wrong.
        axis_z = ground_z + connector_axis_height_m(cfg) + float(extra_clearance_m)
        if bool(r.get('snap_to_ground', True)):
            log.info('REORIENT PLACE: z_mm %+.0f puts the connector axis %+.0f mm above the '
                     'bench; snapping to the ground plane at %+.0f mm instead (one barrel '
                     'radius, %.1f mm, up).', z_mm, (p[2] - ground_z) * 1000.0,
                     (axis_z - ground_z) * 1000.0, connector_axis_height_m(cfg) * 1000.0)
            p[2] = axis_z
        T[:3, 3] = p
        return T

    def reorient_recovery():
        """PICK IT SQUARE, SET IT DOWN ALIGNED, TRY AGAIN.

        THE FAILURE THIS ANSWERS. The coaxial grasp reaches the connector from along its own
        axis, so whether the arm can get there at all depends on which way the cable happens to
        be lying. A cable pointing the wrong way has NO collision-free path to that grasp, and
        no number of retries produces one -- the geometry refused, not the attempt.

        A SQUARE grasp (pitch 0, straight down) has no such dependence: the approach is
        vertical whatever the heading. That is exactly why it is the fallback. So this picks
        the cable the easy way, sets it down pointing where the socket points, and hands back
        to the scan -- which now sees a cable the coaxial grasp CAN reach.

        ONCE PER PICK. If the coaxial grasp is still unreachable after the cable has been
        squared up, its heading was not the problem and repeating this would only shuffle the
        part around the bench."""
        r = a.get('reorient_recovery', {}) or {}
        if not bool(r.get('enabled', True)):
            log.error('The coaxial grasp is unreachable and reorient_recovery is off.')
            return False
        if not phase_gate('REORIENT THE CABLE',
                          'The coaxial grasp has no collision-free path to this cable. Next: '
                          'pick it SQUARE (pitch 0), set it down aligned with the socket, and '
                          're-scan.'):
            return False

        # THE SQUARE GRASP, WHOLE POSE, for this manoeuvre only -- restored in the finally,
        # so a recovery that fails half way cannot leave the run picking with the wrong
        # geometry. It overrides xyz as well as rpy: the bite point that suits a coaxial
        # approach is not the one that suits a vertical one, and the same override is what the
        # in-hand belief for the PLACE is derived from below, so the two cannot disagree.
        keep = cfg.get_path('pickup.fingertip_in_connector')
        square = dict(r.get('fingertip_in_connector')
                      or {'xyz_mm': [5.0, 0.0, 0.0], 'rpy_deg': [0.0, 0.0, 0.0]})
        try:
            cfg.set_path('pickup.fingertip_in_connector', square)
            log.info('REORIENT: picking SQUARE (xyz %s mm, rpy %s deg) instead of the coaxial '
                     '(xyz %s, rpy %s).', square.get('xyz_mm'), square.get('rpy_deg'),
                     (keep or {}).get('xyz_mm'), (keep or {}).get('rpy_deg'))
            # NO NEW SCAN AND NO SECOND PROMPT. The cable has not moved since the coaxial
            # attempt was refused -- only the way the arm means to approach it has changed --
            # so the detection and the cable the operator already identified both still hold.
            phase('scan')
            if _pick(cfg, robot, scanner, geom, check, recovery, grasp, confirm, recorder,
                     T_conn=getattr(geom, 'T_base_detection', None)) != 'ok':
                log.error('REORIENT: the square pick failed too -- the cable heading is not '
                          'what is wrong here. Stopping.')
                return False
            if not grasp.lift(robot, geom, 'reorient lift',
                              position_guard=lambda mv: _guarded(robot, guard_shared, mv)):
                return False

            # WHERE IT GOES. The belief for the SQUARE grasp (cfg is still overridden here)
            # turns "put the CONNECTOR there" into a fingertip pose.
            # RELEASED A LITTLE ABOVE THE RESTING HEIGHT, so the pads are not pressing the
            # cable into the bench when they open; the part drops the last few millimetres.
            T_place = aligned_place_pose(
                float(r.get('release_clearance_mm', 10.0)) / 1000.0)
            # THE BELIEF FOR THE SQUARE GRASP -- cfg is still overridden here, so this is
            # derived from the very transform the pick was commanded with. That is the whole
            # reason the override wraps the place as well as the pick: a place computed from
            # the COAXIAL belief would set the part down rotated by the difference.
            T_ftip_sq = held_belief(T_ftip_conn_catalogue,
                                    from_cfg(cfg.section('junction_in_fingertip')),
                                    fingertip_in_connector(cfg))
            T_ftip_target = T_place @ inverse(T_ftip_sq)
            log.info('REORIENT PLACE: connector to xyz %s mm on heading %+.1f deg (the socket '
                     'heading %+.1f deg + yaw offset).',
                     np.round(T_place[:3, 3] * 1000.0, 1).tolist(),
                     np.degrees(np.arctan2(T_place[1, 0], T_place[0, 0])),
                     np.degrees(np.arctan2(T_base_tconn[1, 0], T_base_tconn[0, 0])))

            above = translation_matrix(
                [0.0, 0.0, float(r.get('approach_mm', 120.0)) / 1000.0])
            for tag, ph, T in (('above', 'reconfigure', above @ T_ftip_target),
                               ('down', 'lift', T_ftip_target)):
                phase(ph)
                if not _guarded(robot, guard_shared,
                                lambda _T=T, _t=tag: robot.move_fingertip(
                                    _T, 'reorient place (%s)' % _t)):
                    return False
            if not robot.gripper.open('release (cable reoriented)'):
                return False
            phase('retract')
            if not _guarded(robot, guard_shared,
                            lambda: robot.move_fingertip(above @ T_ftip_target,
                                                         'reorient place (clear)')):
                return False
            if hasattr(scanner, 'reselect'):
                scanner.reselect()      # it is not where it was scanned from any more
            log.info('REORIENT COMPLETE -- the cable now points where the socket does. '
                     'Re-scanning and retrying the coaxial grasp.')
            return True
        finally:
            cfg.set_path('pickup.fingertip_in_connector', keep)

    # ---- RESET + PICK + slip-checked LIFT (identical to cable_pick_estimate_assemble) ----
    phase('reset')
    if not reset.reset_robot(robot, cfg, 'start reset'):
        return False
    # THE GROUND-COLLISION MODEL IS CHECKED ONCE, HERE, against the controller's own forward
    # kinematics -- it is built from a DH chain written in this repo, and a typo there would
    # make every clearance it reports confidently wrong. On a dry run it reports UNVERIFIED
    # rather than pretending to have checked. Never fatal: an unchecked path is worse than a
    # guarded one, but better than refusing to run at all.
    grasp.verify_collision_model(robot)
    # GRIPPER WARM-UP instead of a plain open: full stroke, two partial cycles, end open.
    if not robot.gripper.warmup():
        return False
    q_home = robot.arm.q()
    # ==================================================================================
    # THE CYCLE. One pass is localize -> pick -> assemble -> clock -> verify -> take
    # apart -> place the cable back -- i.e. the cell ends each pass in the state it
    # started, which is the whole reason a second pass is possible. So the loop only runs
    # when disassembly is enabled; without it the connector stays mated and there is
    # nothing to assemble a second time.
    #
    # EVERYTHING INSIDE REPEATS, deliberately: the marker sweep re-measures the socket
    # (the fixture is allowed to move between cycles -- that is what the rig buys), and
    # the scan re-finds the cable, which after a place is NOT where it was picked from.
    # ==================================================================================
    T_ftip_conn_nominal = np.array(T_ftip_conn, dtype=float)
    for cycle in range(1, n_cycles + 1):
        if n_cycles > 1:
            log.info('=' * 78)
            log.info('CYCLE %d/%d', cycle, n_cycles)
            log.info('=' * 78)
        if cycle > 1:
            # A FRESH PICK HAS A FRESH IN-HAND ERROR. The estimator spent the last cycle
            # correcting the belief for the PREVIOUS grasp; carrying that correction into a
            # new grasp would start the next insertion from a confidently wrong pose.
            T_ftip_conn = np.array(T_ftip_conn_nominal, dtype=float)
            # The cable was PLACED, so it is not where it was picked from and the cached
            # junction selection is stale -- the same reason the slip recovery re-prompts.
            if hasattr(scanner, 'reselect'):
                scanner.reselect()
        if tgt_source == 'visual' and not locate_target_visually(q_home):
            return False
        # THE PICK POSE. Home is the marker VIEW pose (the sweep above, and the end-of-run
        # image); the scan, the grasp geometry and every retry offset are written from HERE.
        q_pick = q_home
        _pick_deg = cfg.get('pick_joints_deg')
        if _pick_deg is not None:
            q_pick = list(np.radians(np.asarray(_pick_deg, dtype=float)))
            # Gate BEFORE reconfiguring: this is a large joint move away from the marker view and
            # into the pick pose, and everything downstream (scan, grasp, retry offsets) is
            # written from where it lands.
            if not phase_gate('RECONFIGURE FOR PICKUP',
                              'The arm will leave the marker view and move to the pick pose %s deg.'
                              % list(np.round(np.asarray(_pick_deg, dtype=float), 1))):
                return False
            # Its OWN phase, not 'reset': this is a large free-space traverse with an EMPTY
            # gripper and nothing near the workpiece, so it has no reason to be paced like a
            # move that ends in contact.
            phase('reconfigure')
            if not robot.arm.move_j(q_pick, label='pick pose'):
                log.error('Could not reach pick_joints_deg.')
                return False


        attempt = 0
        reoriented = False
        reoriented = False
        runner = StepRunner(log, confirm=confirm is not None)
        while True:
            phase('scan')
            result = _pick(cfg, robot, scanner, geom, check, recovery, grasp, confirm, recorder,
                           offset_x_m=retry_offset_x(attempt, check.retry_perturb_x_m))
            if result == 'ok':
                status = {}

                def do_lift(_s=status):
                    _s['r'] = grasp.lift_verified(
                        robot, geom, check, 'lift',
                        position_guard=lambda mv: _guarded(robot, guard_shared, mv))
                    return _s['r'] == 'ok'

                if runner.run([('lift (slip-checked)', do_lift)]):
                    break
                result = status.get('r')
                if result != 'slipped':
                    return False
            if result == 'abort':
                return False
            # NO COLLISION-FREE PATH TO THE COAXIAL GRASP. Retrying the identical approach
            # cannot help -- the geometry, not the attempt, is what refused. Square the cable
            # up once and let the scan try again on a heading that works.
            if result == 'unreachable':
                if reoriented:
                    log.error('The coaxial grasp is still unreachable after the cable was '
                              'squared up -- its heading was not the problem. Aborting.')
                    return False
                reoriented = True
                if not reorient_recovery():
                    return False
                continue
            if attempt >= check.max_retries:
                log.error('Grasp failed on all %d attempts; aborting.', check.max_retries + 1)
                return False
            attempt += 1
            log.warning('Grasp %s -- recovering (attempt %d/%d).', result, attempt + 1,
                        check.max_retries + 1)
            if result == 'slipped' and hasattr(scanner, 'reselect'):
                phase('scan')
                T_up = translation_matrix([0.0, 0.0, check.slip_raise_m]) @ robot.tool0()
                if not (robot.gripper.open('drop')
                        and _guarded(robot, guard_shared,
                                     lambda: robot.arm.move_l(T_up, label='slip recovery (up)'))):
                    return False
                scanner.reselect()
            else:
                phase('reset')
                if not (robot.gripper.open('drop')
                        and robot.arm.move_j(q_pick, label='pick pose')):
                    return False

        # ---- Stand-off, held check, and the unconditional human gate before contact ----
        st = a.get('standoff', {}) or {}
        T_standoff_row = translation_matrix(
            np.asarray(st.get('axis', [-1, 0, 0]), dtype=float)
            * float(st.get('distance_m', 0.01))) @ mats[0]

        def traj_ref(row, T_tool0_conn, commit=False):
            """A TRAJECTORY row -> a tool0 reference, anchored so the path ends on the mate.

            ANCHORED (T_base_targetobj = the mate with the CSV's last row normalised away), which is
            why a preload must never be written as a trajectory row: it would be normalised out here
            and driven everywhere else.

            `commit=True` adds the final insertion's preload, so the press applies to the attempt
            meant to SEAT and to nothing that collects observations."""
            return (T_base_commit if commit else T_base_targetobj) @ row @ inverse(T_tool0_conn)

        phase('standoff')
        seed_q = robot.arm.q()
        T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
        q = robot.arm.ik(traj_ref(T_standoff_row, T_tool0_conn), seed_q)
        if q is None or not _guarded(robot, guard_shared,
                                     lambda: robot.arm.move_j(q, label='stand-off')):
            return False
        seed_q = q
        if not verify_cable_held(robot, check, 'stand-off'):
            return False
        # Asked even with --yes -- this is the boundary between free space and contact. Only
        # skip_prompts (--no-prompts) silences it.
        if not robot.arm.dry_run and not no_prompts:
            try:
                ans = input('\n[stand-off] Ready to ENGAGE (contact ahead). '
                            'Enter to continue (q to abort): ')
            except EOFError:
                ans = ''
            if ans.strip().lower() in ('q', 'quit', 'n', 'no'):
                log.info('Aborted at the stand-off by the user.')
                return False

        # ---- The collect / estimate / update loop ----
        est_rows, success = [], False
        acc = np.zeros((0, 12))
        T_cum = np.eye(4)
        trackc, trackr, trackg = [np.zeros(len(estimator.estimate_dims))], [], []
        try:
            if ins_mode == 'engage':
                # ENGAGE replaces the estimate loop and the commit. A force stop is an ordinary
                # outcome, so both endings continue to the clocking sequence.
                en_status, _last_ref_e, en_depth = engage_insertion()
                success = en_status in ('complete', 'force')
                est_rows.append({'attempt': 'engage', 'status': en_status,
                                 'depth_mm': en_depth, 'success': bool(success)})
            for it in (range(1, max_attempts + 1) if ins_mode == 'estimate' else ()):
                T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
                e_xyz, e_rpy = matrix_to_xyzrpy(T_ftip_conn)
                log.info('--- attempt %d/%d --- in-hand estimate xyz=%s mm rpy=%s deg', it,
                         max_attempts, np.round(e_xyz * 1000, 2).tolist(),
                         np.round(np.degrees(e_rpy), 2).tolist())

                # COLLECTION PASSES: the sweep commands one insertion per deliberate offset with
                # the belief held FIXED across passes, so their evidence fuses exactly.
                passes = ([list(o) for o in sweep_offsets] if col_mode == 'offset_sweep'
                          else [None])
                obs, lin, ang = [], 0.0, 0.0
                for pi, poff in enumerate(passes):
                    if tn_on or poff is not None:
                        rows_t = traj.noised(dense, noise_rng, tn_std if tn_on else [0.0] * 6,
                                             tn_w, tn_dt, (1.0 - tn_da) ** (it - 1), poff)
                    else:
                        rows_t = dense
                    refs = [traj_ref(row, T_tool0_conn) for row in rows_t]
                    phase('standoff')
                    label = (f'attempt {it}'
                             + (f' sweep {pi + 1}/{len(passes)}' if poff is not None else '')
                             + ' start')
                    q = robot.arm.ik(refs[0], seed_q)
                    if q is None or not _guarded(robot, guard_shared,
                                                 lambda: robot.arm.move_j(q, label=label)):
                        log.error('Could not reach the pass start; aborting.')
                        return False
                    seed_q = q
                    phase('assemble')
                    # Intermediate passes MUST back off -- the next realigns to a different offset's
                    # start. The LAST pass stays at its stop, so a mate the operator calls successful
                    # leaves the arm AT the seat for clocking to anchor on.
                    obs_i, lin, ang, stops, last_ref = run_insertion(
                        adm, refs, T_tool0_conn, peck=(col_mode == 'peck'),
                        retract=(pi < len(passes) - 1))
                    obs.extend(obs_i)
                    if poff is not None:
                        log.info('  sweep %d/%d (pitch %+.1f deg, z %+.1f mm): %d obs, stop %s mm.',
                                 pi + 1, len(passes), poff[4], poff[2] * 1000.0, len(obs_i),
                                 [round(s, 1) for s in stops])
                    if not verify_cable_held(robot, check, f'attempt {it} pass {pi + 1}'):
                        return False

                log.info('check: believed connector vs target: %.2f mm, %.2f deg (tol %.2f mm, '
                         '%.2f deg)', lin * 1000, np.degrees(ang), tol_pos_m * 1000,
                         np.degrees(tol_rot_rad))
                _save_observations(os.path.join(out_dir, f'attempt_{it:02d}_observations.csv'), obs)
                row = {'attempt': it, 'n_observations': len(obs), 'n_passes': len(passes),
                       'check_pos_mm': lin * 1000.0, 'check_rot_deg': float(np.degrees(ang))}

                # SUCCESS is the operator's call -- they can see the physical mate; the kinematic
                # numbers only see the belief. A dry run has no operator.
                if robot.arm.dry_run or no_prompts:
                    row['success'] = bool(lin <= tol_pos_m and ang <= tol_rot_rad)
                else:
                    try:
                        ans = input(f'[check attempt {it}] Was the assembly SUCCESSFUL? '
                                    '(y = done / Enter = retry / q = abort): ').strip().lower()
                    except EOFError:
                        ans = ''
                    if ans in ('q', 'quit'):
                        est_rows.append(row)
                        return False
                    row['success'] = ans in ('y', 'yes')
                if row['success']:
                    est_rows.append(row)
                    # NO retract: the connector is mated and the arm stays on it, which is what makes
                    # this pose the ENGAGED pose the clocking maneuvers anchor to.
                    log.info('ASSEMBLY COMPLETE on attempt %d -- holding the seat (connector ENGAGED).',
                             it)
                    success = True
                    break
                # Not successful: NOW back off, because the next thing (another attempt, or the final
                # insertion) approaches its own start under position control and needs the clearance.
                phase('retract')
                retract_from(last_ref, T_tool0_conn)
                if it == max_attempts:
                    est_rows.append(row)
                    log.error('Attempt limit reached (%d) without a successful mate.', max_attempts)
                    break

                # ---- ESTIMATE. Accumulated rows are re-projected into the belief after every
                # update, exactly as in estimator_eval, so old evidence stays valid.
                obs_arr = np.asarray(obs, dtype=float).reshape(-1, 12)
                full = np.vstack([acc, obs_arr]) if accumulate and len(acc) else obs_arr
                vec6, w6 = estimator.prepare_observations(full[:, :6], full[:, 6:9], full[:, 9:12])
                if commit == 'argmin':
                    T_corr_mm, info, land_pack = _argmin_estimate(estimator, vec6, w6)
                else:
                    T_corr_mm, info = estimator.estimate(vec6, w6)
                    land_pack = None
                if T_corr_mm is None:
                    log.warning('Estimation skipped (%s) -- retrying with the UNCHANGED belief.',
                                info)
                    row['estimate'] = f'skipped: {info}'
                    est_rows.append(row)
                    trackc.append(trackc[-1])
                    trackr.append(float('nan'))
                    trackg.append(np.zeros(0))
                    if accumulate:
                        acc = full
                    continue
                log.info('belief correction: %s  (residual %.3f, %d obs, %s)',
                         {k: round(v, 3) for k, v in info['theta_corr'].items()},
                         info['final_residual'], info['n_observations'], commit)
                if dbg_on:
                    try:
                        manifold_debug.figures(
                            estimator, vec6, w6, dict(info['theta_corr']), None,
                            os.path.join(out_dir, f'attempt_{it:02d}_match.png'),
                            title=f'attempt {it} (no ground truth)',
                            max_rows=int(dbg.get('max_rows', 250)),
                            grid_n=int(dbg.get('grid_points', 41)), live_dir=dbg_live)
                    except Exception as exc:               # noqa: BLE001 -- never fatal
                        log.warning('match diagnostics skipped (%s)', exc)

                T_ftip_conn = T_ftip_conn @ _corr_to_m(T_corr_mm)      # believed @ corr ~= true
                T_cum = T_cum @ np.asarray(T_corr_mm, dtype=float)
                if accumulate:
                    from .estimator_eval import _rebase_rows
                    acc = _rebase_rows(full, T_corr_mm) if len(full) else full
                trackc.append(vec6_from_mats(T_cum)[estimator.idx])
                trackr.append(float(info['final_residual']))
                trackg.append(np.asarray(info['res_hist'], dtype=float)[:, -1]
                              if info.get('res_hist') is not None else np.zeros(0))
                _plot_run(os.path.join(out_dir, 'run_corrections.png'), estimator.estimate_dims,
                          trackc, trackr, trackg,
                          f'attempt {it}/{max_attempts} | check {lin * 1000:.1f} mm', live_path)
                row.update({f'corr_{k}': v for k, v in info['theta_corr'].items()})
                row.update({'icp_residual': info['final_residual'], 'commit': commit})
                est_rows.append(row)

            # ---- FINAL INSERTION -- the COMMIT, from the final corrected belief, zero noise ----
            if fi_on and not success and ins_mode == 'estimate':
                log.info('FINAL INSERTION from the corrected belief (zero noise).')
                T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
                rows_f = (traj.noised(dense, noise_rng, fi_noise_std, fi_noise_w, 0.0, 1.0)
                          if fi_noise_on else dense)
                refs = [traj_ref(row_, T_tool0_conn, commit=True) for row_ in rows_f]
                phase('standoff')
                q = robot.arm.ik(refs[0], seed_q)
                if q is None or not _guarded(robot, guard_shared,
                                             lambda: robot.arm.move_j(q, label='final start')):
                    log.warning('IK/approach failed for the final insertion.')
                else:
                    seed_q = q
                    phase('assemble')
                    # retract=False: this is the attempt meant to SEAT. Backing out of it would undo
                    # the mate before the operator can judge it and would leave the clocking maneuvers
                    # anchored on a retracted pose.
                    obs_f, lin, ang, _stops, last_ref_f = run_insertion(
                        adm_final, refs, T_tool0_conn, guard_ctl=guard_final, settle=fi_settle,
                        hold=fi_hold, speed=(fi_v, fi_wr) if (fi_v or fi_wr) else None,
                        pause=fi_pause, retract=False)
                    _save_observations(os.path.join(out_dir, 'final_insertion_observations.csv'),
                                       obs_f)
                    frow = {'attempt': 'final_insertion', 'n_observations': len(obs_f),
                            'check_pos_mm': lin * 1000.0,
                            'check_rot_deg': float(np.degrees(ang))}
                    if robot.arm.dry_run or no_prompts:
                        frow['success'] = bool(lin <= tol_pos_m and ang <= tol_rot_rad)
                        success = success or frow['success']
                    else:
                        try:
                            ansf = input('[final insertion] SUCCESSFUL? (y/n): ').strip().lower()
                        except EOFError:
                            ansf = ''
                        frow['success'] = ansf in ('y', 'yes')
                        success = success or frow['success']
                    est_rows.append(frow)
                    if not success:
                        # Only NOW back off -- a failed commit has nothing to hold on to, and the
                        # release/escape tail below expects clearance.
                        phase('retract')
                        retract_from(last_ref_f, T_tool0_conn, adm_final)
        finally:
            robot.arm.servo_stop()
            if est_rows:
                keys = sorted({k for r in est_rows for k in r}, key=str)
                with open(os.path.join(out_dir, 'estimates.csv'), 'w', newline='') as fh:
                    w = _csv.DictWriter(fh, fieldnames=keys)
                    w.writeheader()
                    w.writerows(est_rows)
                log.info('Per-attempt log: %s', os.path.join(out_dir, 'estimates.csv'))

        if not success:
            return False

        # ---- POST-MATE: CONNECTOR CLOCKING, then COLLAR CLOCKING, then the shared escape -------------
        # Both paths end in the same retract. The mate itself has already succeeded by here, so a
        # clocking failure is reported without undoing it.
        if cc_on:
            cc_ok = ret_ok = False
            tug_res = None
            state = 'engaged'                    # the initial assembly mated it; that is where we are
            if pe_frame == 'believed':
                # The believed connector frozen in base coordinates NOW, while the arm still grips
                # it -- the maneuvers need one consistent frame, not one re-derived per use.
                T_clk = robot.tool0() @ T_tool0_conn
                _pl, _pa = pose_error(T_clk, T_base_tconn)
                log.info('Post-engage frame: BELIEVED connector -- %.2f mm / %.2f deg from the '
                         'recorded target frame.', _pl * 1000.0, np.degrees(_pa))
            else:
                T_clk = T_base_tconn
                log.info('Post-engage frame: TARGET connector (recorded socket pose).')
            if not phase_gate('CONNECTOR CLOCKING (insert)',
                              'The connector is ENGAGED. Next is the bayonet screw, which cams it '
                              'HOME -- check the engagement looks right first.'):
                robot.arm.servo_stop()
                return False
            try:
                cc_ok, _T_tool0_conn, T_base_conn, cc_screw_deg = connector_clocking()
                if cc_ok:
                    state = _advance_state(state, 'engaged')          # -> seated
                    if cl_on and not phase_gate(
                            'COLLAR CLOCKING (lock)',
                            'The connector is SEATED. Next is the collar turn, which LOCKS it -- the '
                            'gripper will re-grasp the collar and rotate. Check the seat first.'):
                        robot.arm.servo_stop()
                        return False
                    if cl_on and collar_clocking(T_base_conn, cc_screw_deg):
                        state = _advance_state(state, 'seated')       # -> locked
                    elif cl_on:
                        cc_ok = False
                else:
                    log.error('CONNECTOR CLOCKING FAILED after %d tr%s (every stroke was stopped by the '
                              'force guard) -- the connector is ENGAGED but NOT SEATED. The mate '
                              'itself succeeded; skipping collar clocking and retracting.',
                              cc_tries, 'y' if cc_tries == 1 else 'ies')
                # ---- TUG VERIFICATION, IN PLACE and BEFORE the escape ---------------------------
                # collar_clocking returns with the fingers still CLOSED on the locked collar, which is
                # already a grip on the assembly and already on the connector axis -- so the pull can
                # happen right here. The old order released, retracted, drove back to the historical
                # engaged pose and re-gripped, which put three free-space moves and a blind re-grasp
                # between the lock and the test, every one of them a chance to disturb what it was
                # meant to measure (and the re-grip could miss entirely).
                if tv_on and state == 'locked':
                    if phase_gate('TUG VERIFY',
                                  'The collar is LOCKED and still HELD. Next: pull %.1f N along the '
                                  'connector -X for %.1f s without letting go -- a locked bayonet '
                                  'holds, an unlocked one backs out.' % (tv_force, tv_time)):
                        tug_res = tug_verify_in_place()
                        if tug_res == 'terminated':
                            return False              # the finally still writes clocking.csv
                    else:
                        tug_res = 'skipped'
                        log.warning('Tug verification skipped by the user -- the assembly is '
                                    'UNVERIFIED.')
                # ---- ESCAPE, ONLY IF THE TUG HAS NOT ALREADY DONE IT ----------------------------
                # tug_verify_in_place ends both of its outcomes with the arm already clear: 'verified'
                # releases and runs clocking_retract itself, 'failed' extracts the cable, drives HOME
                # and releases there. Running the escape again on top of either put two more retract
                # legs after a retract that had already happened -- and after a 'failed' it fired them
                # from the home pose, nowhere near the socket. So the escape belongs to the paths that
                # still have the arm at the connector: a skipped or disabled tug, or a run that never
                # reached 'locked'.
                if tug_res in ('verified', 'failed'):
                    ret_ok = True
                    log.info('Escape not needed -- the tug verification already left the arm clear '
                             '(%s). Going straight home.',
                             'released and retracted' if tug_res == 'verified'
                             else 'cable extracted and carried home')
                elif phase_gate('ESCAPE',
                                'Clocking done. Next: RELEASE the gripper, then the two-leg retract '
                                '(the gripper backs off its own -Z, then away along the target -X).'):
                    # RELEASE FIRST: the fingers are still CLOSED on the collar, and the retract's
                    # first leg is written for OPEN fingers. Idempotent where it is already open.
                    if robot.gripper.open('release before escape'):
                        ret_ok = clocking_retract()
                    else:
                        log.error('Gripper did not open before the escape -- leaving the arm in '
                                  'place rather than dragging the locked assembly with clamped '
                                  'fingers.')
                else:
                    log.warning('Escape skipped by the user -- the arm is still at the connector '
                                'with the gripper in whatever state clocking left it.')

                # ---- DISASSEMBLY, after the assembly has fully let go ------------------------
                # It runs HERE, once the escape has released the connector and backed the arm off,
                # so the two halves are cleanly separated: the assembly ends with the connector
                # mated and the arm clear, exactly as a run without disassembly would leave it, and
                # the disassembly starts by re-approaching and re-gripping like any other maneuver.
                # Nothing it does depends on a grip inherited from the assembly.
                if dis_on and ret_ok and tug_res != 'failed' and state in ('locked', 'seated'):
                    dis_ok, state = disassembly(state, cc_screw_deg, T_base_conn)
                    if not dis_ok:
                        log.error('DISASSEMBLY did not finish -- the arm is LEFT WHERE IT IS and '
                                  'the part may still be held or still in the socket. Free it by '
                                  'hand before commanding motion.')
                        return False
                    log.info('DISASSEMBLED -- the connector is %s.',
                             'out and placed on the ground' if dis_place_on
                             else 'out and still in the fingers')
                elif dis_on and tug_res == 'failed':
                    log.warning('DISASSEMBLY skipped -- the tug already extracted the cable and '
                                'carried it home, so there is nothing left to take apart.')
                elif dis_on and not ret_ok:
                    log.warning('DISASSEMBLY skipped -- the escape did not complete, so the arm is '
                                'not in a known clear state to re-approach from.')
            finally:
                robot.arm.servo_stop()
                if clock_rows:
                    keys = sorted({k for r in clock_rows for k in r}, key=str)
                    with open(os.path.join(out_dir, 'clocking.csv'), 'w', newline='') as fh:
                        w = _csv.DictWriter(fh, fieldnames=keys)
                        w.writeheader()
                        w.writerows(clock_rows)
                    log.info('Clocking log: %s', os.path.join(out_dir, 'clocking.csv'))
            # 'assembled' is claimed only at 'locked': a seated but unlocked BNC can still back out,
            # so a run with collar clocking disabled succeeds without being called assembled.
            if state == 'locked' and tug_res == 'failed':
                log.error('Collar clocking reported LOCKED but the TUG pulled the connector back '
                          'out -- NOT assembled. The cable has been extracted and released at home.')
            elif state == 'locked':
                log.info('ASSEMBLED -- connector ENGAGED -> SEATED -> LOCKED%s.',
                         ', TUG-VERIFIED' if tug_res == 'verified' else
                         ' (tug verification %s)' % ('skipped' if tug_res == 'skipped' else
                                                     'disabled' if tug_res is None else 'ERRORED'))
            elif state == 'seated':
                log.warning('Connector SEATED but NOT LOCKED (collar clocking %s) -- not assembled.',
                            'disabled' if not cl_on else 'FAILED')
            else:
                log.error('Connector ENGAGED only -- neither seated nor locked.')
            phase('reset')
            rst = end_reset_with_snapshot()                       # always, even after a failed screw
            ok_cycle = bool(cc_ok and ret_ok and rst
                            and tug_res in (None, 'skipped', 'verified'))
            cycle_ok.append(ok_cycle)
            if not ok_cycle:
                log.error('CYCLE %d/%d did not complete cleanly -- stopping the loop here '
                          'rather than starting another pass on an unknown cell.',
                          cycle, n_cycles)
                return False
            if cycle < n_cycles:
                log.info('CYCLE %d/%d complete. The cable is back on the ground and the arm '
                         'is home -- next cycle re-localizes and re-picks.', cycle, n_cycles)
                continue
            log.info('ALL %d CYCLE%s COMPLETE.', n_cycles, '' if n_cycles == 1 else 'S')
            return True

    d_out = float(a.get('release_retract_distance_m', 0.08))
    back = -(robot.tool0() @ (robot.T_tool0_fingertip @ T_ftip_conn))[:3, 0] * d_out

    def release_escape():
        return _guarded(robot, guard_shared,
                        lambda: robot.arm.move_l(translation_matrix(back) @ robot.tool0(),
                                                 label='retract (connector -X)'))

    phase('retract')
    ok = runner.run([('open gripper (release)', robot.gripper.open),
                     ('retract (connector -X)', release_escape)])
    phase('reset')
    return ok and end_reset_with_snapshot()


def main():
    run_app('BNC assembly: pick + estimator_eval-tuned manifold estimation',
            'bnc_assembly', build_and_run, needs_camera=True)


if __name__ == '__main__':
    main()
