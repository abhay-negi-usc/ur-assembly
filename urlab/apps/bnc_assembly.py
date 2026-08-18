"""BNC ASSEMBLY -- cable_pick_estimate_assemble's pipeline with estimator_eval's estimator.

Same shape as cable_pick_estimate_assemble (scan, grasp, slip-checked lift, stand-off, then an
assemble/estimate loop that corrects the in-hand belief), but everything downstream of the
grasp is estimator_eval's, because that is where the 2026-08 BNC campaign did its tuning:

  * ESTIMATOR: the same `estimation:` block, including `commit: argmin` (take the dense
    landscape's minimum and skip the multi-start ICP entirely), the per-candidate wrench
    re-basing, per-block kNN bandwidths and the per-channel weights. `_argmin_estimate` and
    `_landscape` are IMPORTED from estimator_eval rather than reimplemented, so the two apps
    cannot drift.
  * COLLECTION: `assembly.collection.mode` = attempts | offset_sweep | peck, with the same
    semantics -- offset_sweep commands one insertion per deliberate offset with the belief held
    FIXED across passes (so their evidence fuses exactly), peck keeps advancing past each force
    stop. Deeper contact was the strongest measured predictor of a good estimate.
  * COMPLIANCE / GUARD: read from the TOP-LEVEL `compliance:` and `force_guard:` blocks, the
    same place and the same names estimator_eval uses, so tuned values copy across verbatim.
  * FINAL INSERTION: `assembly.final_insertion`, the same grouped block -- stiffness, mass,
    damping, settle, dwell and the guard overrides all in one place, each inheriting the shared
    value when null. This is the attempt meant to SEAT, so it gets its own physics.

STATE VOCABULARY -- the four words this app reports progress in, in order:

    ENGAGED     the initial assembly mated the connector. Where the estimate/insert loop ends.
    SEATED      cable clocking succeeded: the bayonet cams pulled the connector home.
    LOCKED      collar clocking succeeded: the locking collar has been turned.
    ASSEMBLED   all of the above -- the connector/cable is done.

Each maneuver advances the state by exactly one step and nothing skips (`CLOCK_STATES` below is the
progression, and the code walks it rather than setting flags, so the log and the CSV cannot disagree
about where a run got to). A run that stops early reports the last state it actually reached.

CAREFUL -- 'seated' IS OVERLOADED, and the two meanings are unrelated. `AdmittanceController.ramp`
returns the string 'seated' to mean "a guard tripped and I stopped early"; that is the ROBOT
layer's word and it says nothing about the assembly state. In the clocking code that return is
therefore read into a local named `stopped`, and `seated` is only ever the state above.

WHAT THIS APP ADDS BEYOND THE MATE -- two operations that run only once the connector is ENGAGED
and the operator has called the assembly successful, each with its own compliance, force guard and
speed scale (`assembly.cable_clocking`, `assembly.collar_clocking`; a failed screw and a finished
collar turn share one escape, `assembly.clocking_retract`):

  * CABLE CLOCKING. A screw about the connector's +X -- rotate while pushing along the same axis,
    aimed at a VIRTUAL target past where the connector can physically go, so compliance follows
    whatever path the bayonet cams allow. Success is MEASURED (the connector must advance a set
    distance along +X) rather than commanded, and it terminates the motion the moment it is
    reached. A rotation that finishes without it RETRIES AS A REGRASP, never an unscrew: open,
    take the gripper back to the saved engaged pose, re-grip, repeat the identical stroke. The
    connector is captive and keeps its progress, so tries accumulate like a ratchet -- which is
    why advance is tracked cumulatively across the regrasp rather than per try. Then the gripper
    opens.
  * COLLAR CLOCKING. Only if the screw succeeded. The opened gripper aligns its CLOSED fingertip
    frame with the collar (a fixed offset along the connector's +X from the junction), closes, and
    turns about the believed connector's +X -- about an axis fixed in space, so the fingers orbit
    the collar rather than scrubbing across it.

THE BELIEF RESET at the start of cable clocking is the load-bearing idea. Everything before it
estimates where the connector is in the hand; once the mate is made, the connector's pose is known
from a PHYSICAL CONSTRAINT -- it is at the target -- so that replaces the estimate and the screw
axis becomes the target's +X exactly instead of inheriting the accumulated in-hand error.

WHAT IS NECESSARILY DIFFERENT. estimator_eval fixtures the part and injects a KNOWN belief
error, so it can score every estimate against ground truth. Here the part is really picked and
the true in-hand pose is unknown: there is no injected error, no err_before/after, and no
truth to draw. Success is the OPERATOR's call at the check (a dry run falls back to the
kinematic tolerance), exactly as in cable_pick_estimate_assemble. The match diagnostics still
render -- with the committed estimate marked and no truth line.

Units: robot poses are metres/radians; the manifold space is mm/deg. The conversion happens
only at the observation/correction boundary, as in the app this is derived from.
"""

import csv as _csv
import os
from datetime import datetime

import time as _t

import numpy as np
from scipy.spatial.transform import Rotation

from .. import config as urconfig
from .. import log as urlog
from .. import tool_frames
from ..log import StepRunner
from ..robot import AdmittanceController, ForceGuard
from ..skills import manifold_debug, reset
from ..skills import trajectory as traj
from ..skills.manifold import mats_from_vec6, vec6_from_mats
from ..skills.pick import (GraspCheck, GraspController, GraspGeometry, GraspImageRecorder,
                           GraspRecovery, retry_offset_x, verify_cable_held)
from ..skills.solution_check import CheckedManifoldEstimator
from ..transforms import (from_cfg, inverse, matrix_to_xyzrpy, pose_error, rotate_about_axis,
                          translation_matrix, xyzrpy_to_matrix)
from ._cable import build_scanner, make_confirm
from ._runner import run_app
from .cable_pick_assemble import _guarded, _pick
from .cable_pick_estimate_assemble import (_corr_to_m, _observe, _plot_run, _save_observations)
from .estimator_eval import _argmin_estimate, _landscape
from .uncertain_sampling import _retract_ref

log = urlog.get('bnc-assembly')

# The assembly state progression (see the module docstring). Walked in order -- cable clocking
# advances engaged -> seated, collar clocking seated -> locked -- so a state can never be reported
# without the step that earns it having actually succeeded.
CLOCK_STATES = ('engaged', 'seated', 'locked')

# The six pose axes, in the order every 6-vector in this app uses (mm, mm, mm, deg, deg, deg).
DIM_KEYS = ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')


def _num(block, key, default):
    """A number from a config block, treating an EXPLICIT null the same as absent.

    The config's own convention is that null inherits, and the comments invite nulling keys out --
    so float(None) must not be how that gets discovered."""
    v = (block or {}).get(key)
    return float(default) if v is None else float(v)


def _path_time(lin_mm, ang_deg, v_mm_s, w_deg_s, min_s):
    """Seconds for a segment of `lin_mm` translation and `ang_deg` rotation under both caps.

    THE CAPS MUST ALREADY BE RESOLVED. A None reaching here means a caller skipped its own
    defaulting -- which is exactly how a `null` speed override in the config once got as far as
    this arithmetic and raised a TypeError partway into a clocking stroke, with the gripper
    closed on the part. Raising with the values named beats comparing None to a number.

    Module level, and not a closure inside build_and_run, SO IT CAN BE TESTED. The closure form
    is why the null path was never exercised by the suite -- nothing outside a live robot run
    could reach it."""
    if v_mm_s is None or w_deg_s is None:
        raise ValueError(
            f'_path_time needs resolved speed caps, got v={v_mm_s!r} w={w_deg_s!r}. A null '
            f'override in the config must be defaulted by the caller before it gets here.')
    return max((lin_mm / v_mm_s) if v_mm_s > 0 else 0.0,
               (ang_deg / w_deg_s) if w_deg_s > 0 else 0.0, min_s)


def _clocking_plan(cable, collar):
    """Which post-mate maneuvers to run, from their `enabled` flags in configs/bnc_assembly.yaml.

    BOTH ARE OPTIONAL and independently switchable (`assembly.cable_clocking.enabled`,
    `assembly.collar_clocking.enabled`). An absent block or an absent key means OFF, so neither
    maneuver can start by accident -- a config that predates them behaves exactly as it did, taking
    the plain release-and-escape tail. With cable clocking off the run ends ENGAGED; with collar
    clocking off it ends SEATED.

    ONE DEPENDENCY: collar clocking requires cable clocking, and the combination is REJECTED rather
    than silently reinterpreted. Two independent reasons, either sufficient:
      * collar clocking has no connector pose to place the collar frame against -- it uses the one
        cable clocking's belief reset establishes;
      * locking a connector that was never SEATED would turn the collar on a part still proud of
        the socket, which is worse than not locking it at all.

    Returns (cable_on, collar_on); raises ValueError carrying the operator-facing reason.

    This is a pure function so the rule is testable. A config rule that only exists inside the
    robot routine can only be verified by running the robot, which means in practice it is not
    verified at all."""
    cable_on = bool((cable or {}).get('enabled', False))
    collar_on = bool((collar or {}).get('enabled', False))
    if collar_on and not cable_on:
        raise ValueError(
            'assembly.collar_clocking.enabled is true but assembly.cable_clocking.enabled is '
            'false. Collar clocking only runs on a SEATED connector and takes the connector pose '
            'it places the collar against from cable clocking. Enable cable clocking, or disable '
            'collar clocking.')
    return cable_on, collar_on


def _advance_state(state, expected):
    """The next state after `expected`, asserting that is where we actually are.

    Cheap, but it is the reason the vocabulary is trustworthy: a future edit that calls collar
    clocking without cable clocking, or advances twice, fails here instead of quietly logging
    'locked' for a connector that was never seated."""
    if state != expected:
        raise AssertionError(f'cannot advance from {state!r}: expected {expected!r}')
    return CLOCK_STATES[CLOCK_STATES.index(state) + 1]


class _ScrewAdvance:
    """Progress detector for a clocking screw, shaped like ForceGuard so that
    AdmittanceController.ramp can terminate on it (ramp returns 'seated' the cycle check() first
    returns True).

    That early termination is the point, not a convenience: the bayonet either cams the connector
    forward or it does not, and once it has moved far enough there is nothing left to gain by
    finishing the rotation -- so success ENDS the motion instead of being read off afterwards.

    Advance is measured from the MEASURED arm pose, never from the commanded reference. Under
    admittance the two differ by exactly the compliant deflection, and here that deflection IS
    the signal: the reference is deliberately a virtual target the connector cannot reach."""

    def __init__(self, robot, T_tool0_conn, T_base_conn_engaged, threshold_m):
        self.robot = robot
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)
        self._inv_engaged = inverse(np.asarray(T_base_conn_engaged, dtype=float))
        self.threshold_m = float(threshold_m)
        self.peak_m = 0.0                  # best advance seen across ALL tries (never reset)
        self.tripped_by = None

    def rebase(self, T_tool0_conn):
        """Adopt a new connector-in-gripper relationship, keeping the engaged reference frame.

        Needed after a REGRASP. Releasing the connector breaks the relationship the detector was
        built with: the connector stays put in the socket while the gripper travels back, so the
        two afterwards differ by exactly the progress made. Rebasing with the last observed
        connector pose keeps advance measured from the ORIGINAL engaged pose, so it accumulates
        across regrasps instead of re-zeroing at each one."""
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

    ForceGuard watches |f|, which is the right question for "did we hit something unexpected"
    and the wrong one for "is the insertion being resisted": a lateral graze against the socket
    rim raises |f| without opposing the push at all, so a magnitude limit tight enough to catch
    real resistance also stops on every glancing touch. Projecting onto the insertion axis
    separates the two, which is what lets the ENGAGE limit be set low without becoming a
    hair-trigger.

    Shaped like ForceGuard (check/reset/tripped_by) so _AnyGuard can OR it with the others.
    Persistence has the same meaning: the limit must hold CONTINUOUSLY for that long, so a
    single-sample spike on first contact does not end the phase."""

    def __init__(self, robot, T_tool0_conn, max_force_n, persistence_s=0.0):
        self.robot = robot
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)
        self.max_force_n = float(max_force_n)
        self.persistence_s = float(persistence_s)
        self.peak_n = 0.0                  # best axial force seen; never reset, for the log
        self.tripped_by = None
        self._over_since = None

    def axial_n(self):
        """|force along the connector +X|, in newtons.

        Magnitude, not signed: compression is what an insertion produces, but a sign convention
        that silently inverted would turn this guard off rather than make it noisy, and there is
        no legitimate large TENSION during an engage either."""
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
    mean opposite things -- one ends the maneuver satisfied, the other ends it jammed -- so which
    of them fired has to survive the call. `ramp` only reports 'seated'."""

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
    # Pacing, dwell and noise for the COMMIT, in the same grouped block. The commit is
    # deliberately ZERO-NOISE by default -- the jitter exists to gather varied contact while
    # probing and has no place in the attempt meant to seat -- so enabling
    # final_insertion.trajectory_noise is an explicit opt-out of that.
    fi_v = None if fi.get('speed_translation_mm_s') is None \
        else float(fi['speed_translation_mm_s'])
    fi_wr = None if fi.get('speed_rotation_deg_s') is None \
        else float(fi['speed_rotation_deg_s'])
    fi_pause = float(fi.get('pause_s', 0.0) or 0.0)
    # PRELOAD -- the press, on the COMMIT only. Drives the commit's reference this far PAST the
    # mate along the connector's +X: the part stops at the mate, the reference keeps going, and
    # the admittance spring turns the leftover travel into contact force.
    #
    # It used to be an extra +10 mm row in configs/assembly_trajectory.csv, the one place it
    # cannot work -- apps/uncertain_sampling anchors that path, so the contact map this app's
    # estimator matches against was collected with the press REMOVED while this app drove it,
    # putting every observation 10 mm deeper than anything the map held. Named here it survives
    # anchoring and lands only on the insertion meant to SEAT; the probing passes end on the mate.
    #
    # What it buys is the break-in SPIKE (a 10 mm preload peaks in the tens-to-low-hundreds of N
    # against a stiff jam, then decays as the spring yields over D/S ~ 6-9 s). What it can HOLD is
    # only stiffness x preload, so raise final_insertion.stiffness if a sustained press is what
    # the connector needs. 0 disables.
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

    # ---- INSERTION MODE: estimate | engage ----------------------------------------------------
    # ESTIMATE  the original: insert, observe, fit the contact manifold, correct the in-hand
    #           belief, repeat. Needs a map that covers the contact states production actually
    #           visits, which is exactly what the 2026-08-17 diagnosis found it did not.
    # WIGGLE    no estimation at all. Drive straight at a fixed target pose and OSCILLATE about it
    #           in the connector's own axes, letting compliance find the mate the way a person
    #           jiggles a plug in. It needs no map, no belief correction and no landscape, so it is
    #           immune to every failure mode of the estimator -- and it is the honest fallback while
    #           the map and the collection are being reconciled.
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
    # OSCILLATION SPEED CAP. speed_mm_s paces the PATH; the oscillation adds its own velocity on
    # top (amplitude x 2*pi*f) that the path speed does not bound. The cap derives a time dilation
    # of the oscillation clock only -- amplitude (the search area) and the frequency RATIO (the
    # orbit shape) are untouched, so it costs wall-clock and nothing else. null = uncapped.
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
        # EFFECTIVE frequency: the speed cap dilates the oscillation clock, so checking the raw
        # value would reject a configuration that samples perfectly well.
        f_live = [en_frq[i] * en_scale for i in range(6) if abs(en_amp[i]) > 0.0]
        if f_live and en_rate < 4.0 * max(f_live):
            log.error('assembly.engage.sample_rate_hz %.1f Hz is too coarse for a %.2f Hz '
                      'component (need >= 4x) -- the sampled sine would alias.',
                      en_rate, max(f_live))
            return False
        if en_fmax <= 0:
            log.warning('assembly.engage.max_axial_force_n is 0 -- the engage will run the whole '
                        'trajectory no matter how hard it presses.')

    # ---- CLOCKING (post-mate): CABLE clocking, then COLLAR clocking --------------------------
    # Two maneuvers that run only after a mate the operator called successful. Each gets its OWN
    # compliance and force guard, and unlike final_insertion the guard override is not optional in
    # practice: the shared force_guard: is tuned for a light probing insertion (5 N), and a
    # deliberate press-and-twist exceeds that on the first cycle. Inheriting it would mean the
    # screw never runs.
    cc = a.get('cable_clocking', {}) or {}
    cl = a.get('collar_clocking', {}) or {}

    try:
        cc_on, cl_on = _clocking_plan(cc, cl)
    except ValueError as exc:
        log.error('%s', exc)
        return False
    log.info('Post-mate clocking: cable clocking %s, collar clocking %s (a run therefore ends %s '
             'at best).', 'ON' if cc_on else 'off', 'ON' if cl_on else 'off',
             'LOCKED/ASSEMBLED' if cl_on else ('SEATED' if cc_on else 'ENGAGED'))
    cc_rot = np.radians(_num(cc, 'rotation_deg', 90.0))
    cc_push_m = _num(cc, 'push_mm', 5.0) / 1000.0
    cc_need_m = _num(cc, 'success_advance_mm', 5.0) / 1000.0
    cc_tries = max(1, int(_num(cc, 'max_tries', 3)))
    cc_settle = _num(cc, 'settle_s', settle_shared)
    cc_hold = _num(cc, 'hold_after_s', 0.0)
    cc_open_after = bool(cc.get('open_gripper_after', True))
    cl_off_m = _num(cl, 'collar_offset_mm', 25.0) / 1000.0
    cl_rot = np.radians(_num(cl, 'rotation_deg', 90.0))
    # PRE-WIND: how far to unwind the OPEN gripper before the turn. null = rotation_deg,
    # i.e. exactly the range the turn is about to spend. 0 disables.
    _pw = cl.get('prewind_deg')
    cl_prewind = cl_rot if _pw is None else np.radians(float(_pw))
    cl_settle = _num(cl, 'settle_s', settle_shared)
    cc_v = None if cc.get('speed_translation_mm_s') is None \
        else float(cc['speed_translation_mm_s'])
    cc_w = None if cc.get('speed_rotation_deg_s') is None else float(cc['speed_rotation_deg_s'])
    cl_v = None if cl.get('speed_translation_mm_s') is None \
        else float(cl['speed_translation_mm_s'])
    cl_w = None if cl.get('speed_rotation_deg_s') is None else float(cl['speed_rotation_deg_s'])
    # TARE: default OFF for both maneuvers, unlike the insertion. The arm is standing in a MATED,
    # loaded pose when clocking starts, so re-zeroing the F/T there would define the mate load as
    # zero and the guard would only ever see force ADDED by the screw. Left off, the guard sees
    # absolute force (safer) at the cost of the spring yielding slightly to the standing load at
    # warm-up. Turn it on per block if that transient matters more than the absolute limit.
    cc_tare = tare if bool(cc.get('tare_before', False)) else None
    cl_tare = tare if bool(cl.get('tare_before', False)) else None
    cl_app_mm = cl.get('approach_mm')     # None = the fingertip's station at that moment
    cl_tilt_deg = _num(cl, 'max_offaxis_tilt_deg', 5.0)
    if cc_on and cl_on and not cc_open_after:
        log.error('assembly.collar_clocking needs cable_clocking.open_gripper_after true: the '
                  'collar is grasped by the same gripper, which must release the cable first.')
        return False
    if cc_on and cc_tries > 1 and not cc_open_after:
        # A retry is a regrasp, so it needs the gripper. (open_gripper_after governs the FINAL
        # release; the per-retry release is unconditional, but a config that says "never open"
        # while asking for retries is contradictory and worth catching before the robot moves.)
        log.error('assembly.cable_clocking.max_tries > 1 needs open_gripper_after true: a retry '
                  'releases, realigns to the engaged pose and re-grips -- it never unscrews.')
        return False
    if cc_on and cc_need_m <= 0.0:
        log.error('assembly.cable_clocking.success_advance_mm must be > 0 (got %.2f) -- there '
                  'would be no way to tell the screw worked.', cc_need_m * 1000.0)
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
        # NOT `enabled`: that key is the MANEUVER's own on/off switch in this block, so reusing it
        # for the guard would make "run the screw with no guard" inexpressible and silently couple
        # two unrelated decisions. The guard gets its own name.
        if block.get('force_guard_enabled') is not None:
            over['enabled'] = bool(block['force_guard_enabled'])
        g = ForceGuard(robot.arm, {**gsec, **over})
        log.info('%s ON: stiffness %s, guard %.1f N / %.1f Nm.', name, comp.get('stiffness'),
                 g.max_force, g.max_torque)
        return AdmittanceController(robot.arm, comp), g

    adm_cc = guard_cc = adm_cl = guard_cl = None
    if cc_on:
        adm_cc, guard_cc = _clock_physics(cc, 'Cable clocking')
    if cl_on:
        adm_cl, guard_cl = _clock_physics(cl, 'Collar clocking')
    adm_en = guard_en = None
    if ins_mode == 'engage':
        # Its OWN compliance and guard, same override schema as the clocking blocks. The engage
        # presses along one axis while (optionally) rocking across it, so it wants neither the
        # probing block's stop-on-contact nor a clocking block's jam limit.
        adm_en, guard_en = _clock_physics(en, 'Engage insertion')

    # ---- Target from the SHARED catalogue (the same record estimator_eval assembles to) ----
    tname = a.get('target_frame')
    targets = tool_frames.load_targets(cfg)
    if not tname or tname not in targets:
        log.error('assembly.target_frame %r needs a targets: entry in %s.',
                  tname, tool_frames.frames_path(cfg))
        return False
    T_base_tconn = targets[tname]
    csv_in = urconfig.resolve(cfg, a.get('trajectory_csv', 'assembly_trajectory.csv'))
    mats = traj.load_csv(csv_in, angles_deg=bool(a.get('trajectory_angles_deg', False)))
    dense = traj.resample(mats, float(a.get('translational_resolution_m', 0.001)),
                          float(a.get('rotational_resolution_deg', 1.0)))

    # ---- ANCHORING: the path ENDS on the recorded mate ---------------------------------------
    # traj.anchor_target normalises the trajectory so its last row lands exactly on the target,
    # whatever that row says. apps/uncertain_sampling -- which builds the contact map this app's
    # estimator matches against -- has always done this; this app applied rows straight to the
    # target instead. Identical for a conforming trajectory, and NOT identical for one carrying a
    # deliberate press in its last row, which the shipped CSV did (+10 mm): the map was collected
    # with that press removed and this app drove it, so every observation sat 10 mm deeper than
    # anything the map contained. The press now lives in final_insertion.preload_mm, applied to
    # the COMMIT alone, and this anchoring keeps the probing passes where the map is.
    #
    # With the CSV fixed this is a no-op (inverse(identity)); it stays because it is the thing
    # that makes a future non-conforming CSV harmless instead of silently 10 mm wrong.
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
        """Resolve a maneuver's speed overrides. A null in the config means "no override", which
        is the GLOBAL cap times the assemble phase scale -- every duration in this app goes
        through here so a null can never reach the arithmetic unresolved again."""
        return (g_v * s_asm if v is None else v), (g_w * s_asm if w is None else w)

    def seg_time(A, B, v=None, w=None):
        lin_m, ang_rad = pose_error(A, B)
        cv, cw = caps(v, w)
        return _path_time(lin_m * 1000.0, np.degrees(ang_rad), cv, cw, min_seg_s)

    def screw_ramp(adm, ref_at, guard, v, w, ang_deg, label=''):
        """Ramp along the EXACT screw path instead of the straight chord between its endpoints.

        WHY THIS EXISTS. admittance.ramp interpolates with slerp_matrix, which SLERPs the rotation
        but LERPs the translation -- so one ramp draws a STRAIGHT LINE between the two tool0
        positions. Harmless for the servo-rate steps engage takes (a fraction of a degree each),
        badly wrong for a clocking stroke taken in a single call: a `theta` turn about an axis `r`
        away from tool0 has a true path that bows out into an arc, and the chord cuts inside it by
        r(1 - cos(theta / 2)). At the 90 deg strokes used here, r = 178 mm, so the chord passes
        52 mm inside the arc and DRAGS whatever sits on the axis through that excursion. Both
        endpoints are still exactly right, which is why this hides in every logged pose -- but
        mid-stroke the turn behaves as though its axis were ~18 mm CLOSER to the tool
        (r sin(theta/2) / (theta/2) instead of r), which is the symptom it presents as.

        The fix is the shape engage already uses: subdivide at the servo rate with every waypoint
        placed on the true screw by `ref_at(s)`, so each lerp spans a fraction of a degree and its
        chord error is microns.

        The speed cap is applied to the ARC rather than the chord, which seg_time cannot do -- it
        measures the straight line between endpoints and so under-counts a 90 deg turn's real path
        by about 11%, quietly running the stroke that much fast."""
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
                return 'seated'
        return 'done'

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

    # The shared frame catalogue -- this app previously loaded only targets:, but the believed
    # in-hand pose is now declared there too (see estimation.initial_connector_frame).
    frames = tool_frames.load_frames(cfg)
    tool_frames.check_drift(frames, cfg)      # warns if frames.yaml and the config sections drift

    # THE BELIEVED IN-HAND POSE, fingertip-relative (the estimator updates this pose, so it has
    # to be the local one, not the flattened tool0 chain).
    #
    # Preference order, and the first one exists so the value lives in ONE place:
    #   1. estimation.initial_connector_frame -- a name in the shared frames catalogue. Parented
    #      to `fingertip` there, so inverse(T_tool0_fingertip) @ frames[name] recovers exactly the
    #      local pose that was declared.
    #   2. estimation.initial_connector_in_fingertip -- the raw pose, kept as an override.
    #   3. cables.yaml junction_in_fingertip -- the grasp geometry, i.e. "wherever the pick put it".
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

    gates_on = cfg.get('confirm_each_step', True) is not False

    def phase_gate(name, ahead):
        """Continue/abort between two phases. True = go on.

        Placed at the IRREVERSIBLE boundaries: the screw cams the connector home and the collar
        locks it, so each is a point of no easy return and worth a look first.

        Aborting leaves the arm exactly where it is rather than escaping automatically. After the
        engage the connector is in the socket with the gripper still on it; the right recovery
        depends on how far it went and whether the mate will release, which is a judgement this
        function cannot make. EOF continues, so a headless run never hangs on a prompt nobody can
        answer, and dry runs skip the prompt entirely."""
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

        retract=False LEAVES THE ARM AT THE STOP. It used to retract unconditionally, which was
        wrong twice over: the operator was asked whether the mate succeeded AFTER the gripper had
        already backed 30 mm out along the connector's own -X (pulling the connector with it, since
        the gripper holds the cable), and cable clocking then read that retracted pose as its
        "engaged pose" -- so the screw axis and the whole clocking sequence were anchored 30 mm off.
        The insertion meant to SEAT must stay put; only an attempt that is about to be retried
        should back off, and that is now the caller's decision (see retract_from).

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
            # PAUSE: servo released, the arm just stands still. Unlike the dwell above it
            # applies no force -- inspection/measurement time, not press time.
            adm_ctl.stop()
            log.info('   pausing %.1f s at the seat (servo stopped).', float(pause))
            _time.sleep(float(pause))
        lin, ang = pose_error(robot.tool0() @ T_tool0_conn, T_base_tconn)
        if retract:
            retract_from(last_ref, T_tool0_conn, adm_ctl)
        else:
            # STAY at the stop. servo_stop only ends the servoL stream -- the controller holds the
            # last commanded pose, so the connector keeps its seat.
            robot.arm.servo_stop()
            log.info('   holding the seat (no retract) -- the pose is the engaged pose.')
        return obs, lin, ang, stops, last_ref

    # ====================================================================================
    # POST-MATE CLOCKING. Both maneuvers run only after a successful mate and share one
    # escape. Diagnostics land in clocking.csv rather than estimates.csv, which is already
    # closed by the time these run.
    # ====================================================================================
    clock_rows = []

    def clocking_retract(label='clocking retract'):
        """The post-clocking escape, in two legs.

        First straight back along the GRIPPER's own axis, which lifts the open fingers off the
        connector; then along the TARGET CONNECTOR frame's axis, which backs the arm away from the
        socket. Guarded straight lines, not the compliant `_retract_ref` used between attempts:
        the part has been released by now, so there is no held connector to thread back out along
        its own axis, and the escape belongs to the gripper and the fixture instead."""
        r = a.get('clocking_retract', {}) or {}

        def leg(vec, dist, in_target):
            v = np.asarray(vec, dtype=float)
            n = float(np.linalg.norm(v))
            if n < 1e-9 or abs(float(dist)) < 1e-9:
                return True
            step = v / n * abs(float(dist))
            if in_target:
                # a direction in the TARGET frame -> rotate it into base and left-multiply
                T = translation_matrix(T_base_tconn[:3, :3] @ step) @ robot.tool0()
                what = f'target {np.round(v / n, 3).tolist()}'
            else:
                # a direction in the GRIPPER's own (tool0) frame -> right-multiply
                T = robot.tool0() @ translation_matrix(step)
                what = f'gripper {np.round(v / n, 3).tolist()}'
            return _guarded(robot, guard_shared, lambda: robot.arm.move_l(
                T, label=f'{label} ({what}, {abs(float(dist)) * 1000.0:.0f} mm)'))

        phase('clock_retract')
        return (leg(r.get('gripper_axis', [0.0, 0.0, -1.0]),
                    r.get('gripper_distance_m', 0.100), False)
                and leg(r.get('target_axis', [-1.0, 0.0, 0.0]),
                        r.get('target_distance_m', 0.100), True))

    def engage_insertion():
        """ENGAGE -- drive the assembly trajectory home, optionally rocking, stop on axial force.

        Unlike wiggle_insertion (which ignores the trajectory and drives at ONE fixed target),
        this follows configs/assembly_trajectory.csv, extends it by `preload_mm` past the mate,
        and superimposes an oscillation on the way. With every amplitude at 0 it is a plain
        direct insertion, so `direct` and `wiggle` are the same behaviour at two settings rather
        than two code paths that can drift apart.

        THE REFERENCE, in the connector-w.r.t.-target frame:

            v(t) = path(distance = speed * t)  +  SUM amplitude_i * sin(2*pi*frequency_i*t)

        TWO CLOCKS, deliberately. The PATH advances by DISTANCE (speed_mm_s), so the insertion
        takes the time its length says regardless of how the oscillation is tuned; the
        OSCILLATION advances by TIME, so its frequency is the frequency configured regardless of
        how fast the path is driven. Pacing both by distance -- the way seg_time paces every
        other ramp in this app -- would make the oscillation frequency a function of the insert
        speed.

        MIND THE DURATION. The insertion lasts path_length / speed_mm_s, so a frequency chosen
        without reference to that can complete less than one cycle and act as a constant offset.
        The start-up line reports cycles-per-insertion for exactly this reason.

        TERMINATION, and this is the point of the phase: reaching the end of the path (mate plus
        preload) is SUCCESS, and hitting the axial force limit is a NORMAL, EXPECTED end -- not a
        failure of the run. The connector meeting resistance partway is exactly what the bayonet
        screw is for, so the phase reports where it stopped and the sequence carries on. That is
        why the limit is AXIAL (see _AxialForce) and can be set low: a magnitude limit tight
        enough to catch real resistance would also stop on every lateral graze.

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

        def ref_at(t):
            v = path_at(v_mm_s * t).copy()
            for i in range(6):
                if abs(en_amp[i]) > 0.0 and en_frq[i] > 0.0:
                    # t * en_scale: the oscillation clock is dilated by the speed cap, the PATH
                    # clock above is not. Scaling the argument keeps a co-prime frequency pair
                    # co-prime.
                    v[i] += en_amp[i] * np.sin(2.0 * np.pi * en_frq[i] * float(t) * en_scale)
            return traj_ref(_corr_to_m(mats_from_vec6(v)), T_tool0_conn)

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
        nstep = int(np.ceil((total_mm / max(v_mm_s, 1e-9)) * en_rate))
        status, prev, last_ref = 'complete', first, first
        for i in range(1, nstep + 1):
            cur = ref_at(i * dt)
            res = adm_en.ramp(prev, cur, dt, combo, on_step=log_cb)
            prev = last_ref = cur
            if res != 'seated':
                continue
            status = 'force' if combo.tripped is det else 'guard'
            break

        stay = robot.tool0()
        adm_en.reset()
        if settle_shared > 0:
            adm_en.hold(stay, settle_shared, guard=None, on_step=log_cb)
        adm_en.stop()
        robot.arm.servo_stop()

        depth = float(matrix_to_xyzrpy(
            inverse(T_base_tconn) @ (robot.tool0() @ T_tool0_conn))[0][0] * 1000.0)
        if status == 'complete':
            log.info('  ENGAGE COMPLETE -- drove the full %.1f mm; depth %.2f mm past the mate, '
                     'peak axial force %.1f N.', total_mm, depth, det.peak_n)
        elif status == 'force':
            log.info('  ENGAGE stopped on the AXIAL FORCE LIMIT (%s) at depth %.2f mm. This is a '
                     'normal end, not a failure: the clocking screw is what drives the rest.',
                     det.tripped_by, depth)
        else:
            log.warning('  ENGAGE stopped on the general force guard (%s) at depth %.2f mm -- '
                        'that is a JAM, not the axial limit.', combo.tripped_by, depth)
        if obs:
            _save_observations(os.path.join(out_dir, 'engage_observations.csv'), obs)
        return status, last_ref, depth

    def cable_clocking():
        """CABLE CLOCKING -- the bayonet screw, and the belief reset that makes it well posed.

        The mate is made, so for the first time in the run the connector's pose is known from a
        PHYSICAL CONSTRAINT rather than estimated: it is AT the target. That replaces the
        estimated belief here, which is what lets the screw axis be the target's +X exactly
        instead of inheriting the accumulated in-hand error. The pose the arm is standing at is
        kept as the engaged pose and every measurement below is relative to it.

        The motion is a screw about the connector's +X: rotate `rotation_deg` while translating
        `push_mm` along that same axis. Because the translation is ALONG the rotation axis the two
        commute, so the reference is a true helix and the order they are composed in does not
        matter. The push target is VIRTUAL -- it aims past where the connector can actually go and
        lets compliance follow whatever path the bayonet cams allow.

        SUCCESS is measured, not commanded: the connector must advance `success_advance_mm` along
        the engaged frame's +X, detected mid-ramp so it ends the motion (see _ScrewAdvance).

        A RETRY IS A REGRASP, NOT AN UNSCREW. The gripper has rotated with the cable, so the stroke
        cannot simply be repeated -- but undoing it would give back whatever the cams gained. So:
        open the gripper, take it back to the SAVED ENGAGED POSE, re-grip, and repeat the identical
        stroke. The connector is captive in the socket and keeps its progress while the gripper
        travels, exactly like backing a ratchet handle off and taking a fresh bite.

        Two consequences worth being explicit about, because both are easy to get silently wrong:

          * THE STROKE IS THE SAME EVERY TRY. It is written as a base-frame screw about the fixed
            axis LINE through the engaged connector origin along its +X. Since the rotation is
            about +X and the push is ALONG +X, that line is invariant under the screw -- so no
            per-try accumulation is needed, and the form is independent of the connector-in-gripper
            belief that the regrasp invalidates.
          * ADVANCE STAYS CUMULATIVE. Releasing breaks the connector-in-gripper relationship: the
            connector stays put while the gripper moves back, so the two differ afterwards by
            exactly the progress made. The detector is REBASED across the regrasp with the last
            observed connector pose, so advance keeps being measured from the ORIGINAL engaged
            pose and progress accumulates across tries instead of resetting to zero.

        Returns (ok, T_tool0_conn, T_base_conn): the connector in the gripper (rebased across any
        regrasps) and where it actually ended up -- read from the MEASURED arm pose, so it is the
        ACHIEVED screw and not the commanded one."""
        T_tool0_engaged = robot.tool0()
        T_tool0_conn = inverse(T_tool0_engaged) @ T_base_tconn
        moved = matrix_to_xyzrpy(inverse(robot.T_tool0_fingertip @ T_ftip_conn) @ T_tool0_conn)
        log.info('--- CABLE CLOCKING --- belief reset: connector assumed AT the target '
                 '(shifts the in-hand belief by %s mm, %s deg)',
                 np.round(moved[0] * 1000.0, 2).tolist(),
                 np.round(np.degrees(moved[1]), 2).tolist())
        log.info('  screw: %+.1f deg about the connector +X while pushing %+.1f mm along it; '
                 'success = %.1f mm of CUMULATIVE advance, up to %d tr%s (a retry regrasps at the '
                 'engaged pose and repeats the stroke -- it never unscrews)',
                 np.degrees(cc_rot), cc_push_m * 1000.0, cc_need_m * 1000.0, cc_tries,
                 'y' if cc_tries == 1 else 'ies')

        # The stroke as a BASE-frame motion about the fixed axis line L (engaged connector origin,
        # along its +X). Independent of the connector-in-gripper belief, and the SAME every try.
        # Built at each FRACTION of the stroke rather than only at its end, so the commanded path
        # stays on that axis line instead of cutting the chord between the endpoints -- see
        # screw_ramp for why a single ramp across 90 deg does not.
        ref_start = T_tool0_engaged

        det = _ScrewAdvance(robot, T_tool0_conn, T_base_tconn, cc_need_m)
        combo = _AnyGuard(det, guard_cc)
        ok = False
        for k in range(1, cc_tries + 1):
            phase('cable_clock')
            adm_cc.reset()
            adm_cc.warmup(ref_start, tare_fn=cc_tare)
            combo.reset()
            # SUBDIVIDED ON THE TRUE SCREW -- see screw_ramp. A single ramp here interpolates
            # the tool0 POSITION in a straight line, which for this 90 deg orbit about an axis
            # 178 mm away drags the connector 52 mm sideways off its own axis at mid-stroke.
            res = screw_ramp(
                adm_cc,
                lambda f: (T_base_tconn
                           @ xyzrpy_to_matrix([cc_push_m * f, 0.0, 0.0], [cc_rot * f, 0.0, 0.0])
                           @ inverse(T_base_tconn)) @ ref_start,
                combo, cc_v, cc_w, abs(np.degrees(cc_rot)), label='screw ')
            adv = det.advance_m()
            # `stopped`, not `seated`: ramp's 'seated' means "a guard tripped", which is the robot
            # layer's word and unrelated to the assembly state. WHICH guard fired is what decides
            # between the two outcomes.
            stopped = res == 'seated'
            is_seated = stopped and combo.tripped is det
            jammed = stopped and not is_seated
            clock_rows.append({'maneuver': 'cable_clocking', 'try': k, 'ramp_result': res,
                               'advance_mm': round(adv * 1000.0, 3),
                               'peak_advance_mm': round(det.peak_m * 1000.0, 3),
                               'need_mm': round(cc_need_m * 1000.0, 3),
                               'success': bool(is_seated), 'force_stop': bool(jammed),
                               'state_after': 'seated' if is_seated else 'engaged',
                               'stopped_by': combo.tripped_by or ''})
            if is_seated:
                ok = True
                log.info('  try %d/%d: SEATED -- %s', k, cc_tries, det.tripped_by)
                break
            if jammed:
                log.warning('  try %d/%d: NOT SEATED, force guard stopped the screw (%s); '
                            'advance %.2f mm', k, cc_tries, combo.tripped_by, adv * 1000.0)
            else:
                log.warning('  try %d/%d: NOT SEATED, rotation completed but advance is %.2f mm '
                            '(need %.2f)', k, cc_tries, adv * 1000.0, cc_need_m * 1000.0)
            if k == cc_tries:
                break

            # ---- REGRASP RATCHET: release, return the GRIPPER to the engaged pose, re-grip ----
            # The connector keeps whatever it gained. Its pose is captured BEFORE releasing --
            # while the gripper still holds it -- because it is unobservable once the fingers open.
            C_conn = robot.tool0() @ det.T_tool0_conn
            adm_cc.stop()
            robot.arm.servo_stop()
            phase('retract')
            if not robot.gripper.open(f'release for clocking retry {k + 1}'):
                log.error('Gripper did not open for the clocking retry.')
                break
            phase('standoff')
            if not _guarded(robot, guard_shared, lambda: robot.arm.move_l(
                    ref_start, label=f'realign to the engaged pose (retry {k + 1})')):
                log.error('Could not realign to the saved engaged pose.')
                break
            if not robot.gripper.close(f'regrasp for clocking retry {k + 1}'):
                log.error('Gripper did not close on the regrasp.')
                break
            if not verify_cable_held(robot, check, f'clocking regrasp {k + 1}'):
                log.error('The regrasp missed the cable -- screwing again would turn nothing.')
                break
            # Rebase: the connector sat still while the gripper travelled, so the relationship
            # between them changed by exactly the progress made. Without this, advance would be
            # re-zeroed at every regrasp and a cumulative threshold could never be reached.
            det.rebase(inverse(robot.tool0()) @ C_conn)
            log.info('  regrasped at the engaged pose, carrying %.2f mm of advance into try %d',
                     det.advance_m() * 1000.0, k + 1)

        # Settle at wherever the screw actually ended. The integrator is zeroed first so the hold
        # commands the pose the arm is AT rather than that pose plus the deflection already in it.
        last = robot.tool0()
        adm_cc.reset()
        if cc_settle > 0:
            adm_cc.hold(last, cc_settle, guard=None)
        if cc_hold > 0:
            adm_cc.hold(last, cc_hold, guard=None)
        adm_cc.stop()
        robot.arm.servo_stop()

        # det.T_tool0_conn, NOT the local from the belief reset: a regrasp rebases it, and reading
        # the stale one here would hand collar clocking a connector pose wrong by exactly the
        # progress the ratchet made -- placing the collar grasp that far off.
        T_tool0_conn_now = det.T_tool0_conn
        T_base_conn = robot.tool0() @ T_tool0_conn_now
        got = matrix_to_xyzrpy(inverse(T_base_tconn) @ T_base_conn)
        log.info('  achieved screw: %+.2f mm along +X, %+.2f deg about +X '
                 '(commanded %+.2f mm x %d tr%s, %+.1f deg); peak advance %.2f mm',
                 got[0][0] * 1000.0, np.degrees(got[1][0]), cc_push_m * 1000.0, cc_tries,
                 'y' if cc_tries == 1 else 'ies', np.degrees(cc_rot), det.peak_m * 1000.0)
        if robot.arm.dry_run and not ok:
            # arm.fk is a fixed stand-in offline, so tcp_pose never moves and advance is
            # unmeasurable by construction -- pass it so the rest of the sequence is exercised.
            log.info('  dry run: advance is unmeasurable (tcp_pose is a fixed stand-in); '
                     'treating the screw as successful to exercise the rest of the sequence.')
            ok = True
        if cc_open_after:
            phase('retract')
            if not robot.gripper.open('release (post cable clocking)'):
                log.error('Gripper did not open after cable clocking.')
                return False, T_tool0_conn_now, T_base_conn
        return ok, T_tool0_conn_now, T_base_conn

    def collar_clocking(T_base_conn):
        """COLLAR CLOCKING -- grasp the locking collar and turn it.

        The collar sits `collar_offset_mm` along the connector's +X from the cable junction (which
        is what the connector frame is). The OPEN gripper is positioned so its CLOSED fingertip
        frame coincides with the collar frame, closes on the collar, then turns `rotation_deg`
        about the BELIEVED connector frame's +X -- a rotation about an axis fixed IN SPACE, not
        about the tool, so the fingers orbit the collar instead of scrubbing across it.

        A force-guard stop is NOT treated as failure. A collar that has reached its lock stops
        turning, which is the intended end state and is indistinguishable here from jamming; the
        achieved angle is logged for the operator to judge.

        TODO: grasp verification and failure recovery on the close, as agreed -- a missed collar
        currently turns an empty gripper."""
        # THE AXIS IS THE SOCKET'S, NOT THE ARM'S.
        #
        # Cable clocking turns about T_base_tconn -- the fixed, hand-measured socket pose -- and
        # that is the right frame here too: the connector is captive in a socket that does not
        # move. T_base_conn is a different animal. It is rebuilt at RUNTIME as
        # robot.tool0() @ T_tool0_conn_now, so it carries every deviation the screw accumulated:
        # the spring's yield under load, an advance that stopped short, a regrasp rebase. Turning
        # about it means turning about where the ARM ended up rather than about the part, and no
        # amount of following the commanded arc fixes an arc drawn round the wrong line.
        #
        # What the runtime measurement DOES legitimately know is how far the bayonet cammed the
        # connector IN along its own axis. Keep exactly that -- project the measured origin onto
        # the true axis -- and discard the lateral and angular drift, which the socket forbids.
        axis = T_base_tconn[:3, 0]
        axn = axis / float(np.linalg.norm(axis))
        cammed = float(np.dot(T_base_conn[:3, 3] - T_base_tconn[:3, 3], axn))
        point = T_base_tconn[:3, 3] + cammed * axn
        T_base_axis = T_base_tconn.copy()
        T_base_axis[:3, 3] = point
        T_base_collar = T_base_axis @ translation_matrix([cl_off_m, 0.0, 0.0])
        # Report the drift that was being used as an axis -- it is the direct measure of how far
        # the screw's accumulated error would have thrown this turn.
        _d = T_base_conn[:3, 3] - point
        _lat = float(np.linalg.norm(_d - np.dot(_d, axn) * axn))
        _tilt = float(np.degrees(np.arccos(np.clip(
            abs(float(np.dot(T_base_conn[:3, 0] / np.linalg.norm(T_base_conn[:3, 0]), axn))),
            0.0, 1.0))))
        log.info('  axis: SOCKET (assembly.target_frame), advanced %+.2f mm by the screw. The '
                 'measured connector frame sits %.2f mm lateral / %.2f deg tilted from it -- that '
                 'drift is discarded, not turned about.', cammed * 1000.0, _lat * 1000.0, _tilt)
        here = robot.tool0()
        # THE GRASP ROLL IS FREE FOR THE GRIPPER AND NOT FREE FOR THE ARM.
        #
        # A parallel jaw is symmetric under a 180 deg roll about its APPROACH axis (fingertip Z):
        # it swaps which pad is which and the grasp is identical. So the collar can be taken at the
        # nominal alignment or at that alignment rolled 180 deg -- same grasp, either way.
        #
        # But the nominal alignment is defined by making the fingertip frame coincide with the
        # COLLAR frame, and the collar frame inherits the CONNECTOR's orientation, which sits
        # 180 deg rolled from the fingertip's own (estimation.initial_connector_in_fingertip
        # carries rpy [0, 0, 180]). Taking that literally demands a 180 deg wrist roll on the way
        # in -- and that roll is about an axis 90 deg OFF the connector axis, so the approach stops
        # being an orbit and becomes a wide swing through the part. Measured, arm -> pre-wound:
        #     as defined    180.0 deg about an axis 90.0 deg off the connector axis
        #     jaws swapped   90.0 deg about an axis  0.0 deg off it  -- a PURE ORBIT
        # So pick the roll that keeps the approach on the axis. Nothing about the grasp changes.
        grip = min((inverse(robot.T_tool0_fingertip),
                    xyzrpy_to_matrix([0.0, 0.0, 0.0], [0.0, 0.0, np.pi])
                    @ inverse(robot.T_tool0_fingertip)),
                   key=lambda G: pose_error(here, T_base_collar @ G)[1])
        # The NOMINAL alignment: the closed fingertip frame coincides with the collar frame.
        T_nominal = T_base_collar @ grip
        # PRE-WIND. The turn is a rigid orbit about the connector's +X, so starting it from the
        # nominal alignment spends `rotation_deg` of wrist range on whichever side the arm happens
        # to be on -- and there may not be that much left. Unwinding by the same amount FIRST puts
        # the whole stroke on the reachable side, and the turn then LANDS on the nominal alignment
        # instead of finishing 90 deg past it.
        #
        # It has to happen with the gripper OPEN. Unwinding while gripping would turn the collar
        # backwards -- undoing the lock rather than making room to apply it -- so this move sits
        # strictly between the align and the close.
        #
        # The collar sits ON the rotation axis (it is offset along +X, the axis direction), so
        # orbiting about that line keeps the fingertip on the collar's own circle; the pre-wind
        # slides the grip around the ring without moving off it.
        T_start = rotate_about_axis(T_nominal, axis, point, -cl_prewind)
        T_end = rotate_about_axis(T_start, axis, point, cl_rot)
        # THE COLLAR IS AHEAD OF THE GRASP, so the gripper cannot close where it stands. The pads
        # close BEHIND the cable junction (-45.7 mm along the connector +X on this geometry) and
        # the collar sits AHEAD of it (+collar_offset_mm), so 70.7 mm separates the open fingers
        # from the ring the moment the cable is released. The gripper unwinds at that near station
        # -- clear of the collar, which is what makes the roll safe -- and only then ADVANCES along
        # the connector +X onto the ring.
        app_x = (float((inverse(T_base_axis) @ here @ robot.T_tool0_fingertip)[0, 3])
                 if cl_app_mm is None else float(cl_app_mm) / 1000.0)
        adv_m = cl_off_m - app_x
        T_nom_back = T_base_axis @ translation_matrix([app_x, 0.0, 0.0]) @ grip
        T_start_back = rotate_about_axis(T_nom_back, axis, point, -cl_prewind)
        if adv_m < 0.0:
            log.warning('COLLAR CLOCKING: the approach station (%+.1f mm) is AHEAD of the collar '
                        '(%+.1f mm), so the advance runs backwards along the connector +X. Check '
                        'collar_clocking.approach_mm.', app_x * 1000.0, cl_off_m * 1000.0)
        log.info('--- COLLAR CLOCKING --- collar is %.1f mm along the connector +X. '
                 'Centre on the axis, unwind to %+.1f deg of pre-wind by ORBITING the '
                 'connector +X at the %+.1f mm station (gripper OPEN and clear of the ring, so '
                 'the collar stays put), advance %+.1f mm onto it, grasp, then turn %+.1f deg '
                 'about the same axis.',
                 cl_off_m * 1000.0, np.degrees(-cl_prewind), app_x * 1000.0, adv_m * 1000.0,
                 np.degrees(cl_rot))
        # REACHABILITY of BOTH ends, before anything grips. Discovering mid-turn that the far end
        # is unreachable leaves the collar clamped in a stalled gripper, which is the one failure
        # this maneuver must not have -- and it is exactly what the pre-wind exists to prevent, so
        # a failure here should say so rather than surfacing as a generic move error.
        for lab, T_chk in (('approach station', T_nom_back), ('pre-wound approach', T_start_back),
                           ('collar grasp', T_start), ('turn end', T_end)):
            if robot.arm.ik(T_chk, robot.arm.q()) is None:
                log.error('COLLAR CLOCKING: the %s pose is unreachable. The turn needs '
                          '%.0f deg of range about the connector +X from a start unwound '
                          '%.0f deg; reduce collar_clocking.rotation_deg, adjust prewind_deg, '
                          'or reposition the fixture.', lab, np.degrees(cl_rot),
                          np.degrees(cl_prewind))
                return False
        phase('standoff')
        # 1+2. ALIGN AND UNWIND ARE ONE ORBIT about the connector axis.
        #
        # The arm is standing where the cable screw left it: same station along the axis, a
        # different clock angle about it. With the roll chosen above, that pose and the pre-wound
        # pose differ by a rotation about the connector axis AND NOTHING ELSE -- so a single orbit
        # lands on it exactly, and no part of the approach is a straight-line swing.
        #
        # The angle is SOLVED rather than assumed (the arm's clock angle depends on what the screw
        # actually achieved, not on what it commanded), and the solution is checked: if the two
        # poses do not in fact differ by a pure rotation about the axis, orbiting would miss, so
        # that is reported instead of quietly driving somewhere wrong.
        # A. CENTRE ON THE AXIS -- a PURE TRANSLATION at the current clock angle.
        #
        # While the cable was gripped the pads sat 7.5 mm to one side of the connector axis
        # (T_ftip_conn carries a Z offset), and a collar grasp needs the fingertip ON the axis.
        # That much travel is real geometry, not error, and it is NOT a rotation -- so a straight
        # move is exactly the right path for it and there is no chord to cut. Doing it first is
        # what leaves the rest of the approach a pure orbit.
        rel = here @ inverse(T_nom_back)
        th_now = float(np.dot(Rotation.from_matrix(rel[:3, :3]).as_rotvec(), axn))
        T_centred = rotate_about_axis(T_nom_back, axis, point, th_now)
        c_lin, c_ang = pose_error(here, T_centred)
        # ALWAYS LOGGED, gated only above max_offaxis_tilt_deg. The number is the diagnostic: a
        # value that repeats run to run is a declared-frame orientation error; one that varies is
        # tilt the cable screw's spring left behind. Raising the gate hides neither, because this
        # line is printed either way.
        log.info('  out-of-axis tilt at the end of the screw: %.2f deg (gate %.1f deg)',
                 np.degrees(c_ang), cl_tilt_deg)
        if np.degrees(c_ang) > cl_tilt_deg:
            log.error('COLLAR CLOCKING: the arm is %.1f deg away from the connector axis frame in '
                      'a way no rotation about that axis explains, over the %.1f deg gate '
                      '(collar_clocking.max_offaxis_tilt_deg). The pose the cable screw left, the '
                      'connector belief and the collar geometry disagree -- refusing to swing '
                      'blindly.', np.degrees(c_ang), cl_tilt_deg)
            return False
        if c_lin * 1000.0 > 1e-3:
            if not _guarded(robot, guard_shared, lambda: robot.arm.move_l(
                    T_centred, label='collar centre on the axis')):
                log.error('Could not centre on the connector axis (%.1f mm).', c_lin * 1000.0)
                return False
        # B. ORBIT to the pre-wound pose -- about the connector axis and nothing else. The angle
        #    is SOLVED, not assumed: the arm's clock angle depends on what the cable screw actually
        #    achieved, not on what it commanded.
        theta = -cl_prewind - th_now
        r_lin, r_ang = pose_error(rotate_about_axis(T_centred, axis, point, theta), T_start_back)
        if r_lin * 1000.0 > 1e-6 or np.degrees(r_ang) > 1e-6:
            log.error('COLLAR CLOCKING: orbiting from the centred pose would miss the pre-wound '
                      'pose by %.3f mm / %.3f deg. Refusing to swing blindly.',
                      r_lin * 1000.0, np.degrees(r_ang))
            return False
        # THE UNWIND IS AN ORBIT, NOT A STRAIGHT MOVE TO THE PRE-WOUND POSE.
        #
        # At the nominal alignment the FINGERTIP SITS ON THE ROTATION AXIS -- the collar frame is
        # the connector frame slid along its own +X, so it is on the axis by construction (measured
        # radial offset 0.000 mm). A true orbit about that axis therefore leaves the grip point
        # exactly where it is and only ROLLS the gripper about the collar. That is what the
        # pre-wind is: a fresh bite at a different clock angle on the same ring, with the wrist
        # carried back around its own 183 mm circle. Nothing needs to translate, because the collar
        # is a body of revolution about that same axis.
        #
        # A straight move to the pre-wound pose does not do that. It interpolates the tool0
        # POSITION linearly while slerping the orientation, and tool0 is 183 mm off the axis, so
        # the pose cuts inside the arc and carries the fingertip 53.6 mm OFF the collar on the way
        # -- driving the open fingers through the connector they are supposed to be rolling around.
        #
        # Compliant rather than kinematic, and guarded, because this threads open fingers around a
        # part -- a graze should yield and stop, not push through. Paced by the standoff scale,
        # matching the move it replaces.
        if abs(theta) > 1e-9:
            adm_cl.reset()
            adm_cl.warmup(T_centred)
            guard_shared.reset()
            res_uw = screw_ramp(
                adm_cl,
                lambda f: rotate_about_axis(T_centred, axis, point, theta * f),
                guard_shared, g_v * s_std, g_w * s_std, abs(np.degrees(theta)),
                label='unwind ')
            adm_cl.stop()
            robot.arm.servo_stop()
            if res_uw == 'seated':
                log.error('COLLAR CLOCKING: the force guard tripped during the unwind (%s). The '
                          'open fingers hit something on the way around the ring -- the collar '
                          'is still free, so nothing is clamped.',
                          guard_shared.tripped_by or 'unknown')
                return False
        # 3. ADVANCE onto the ring along the connector +X. A PURE TRANSLATION -- no rotation, so
        #    a straight move IS the right path here and there is no chord to cut. Guarded, because
        #    this slides open fingers over a part: a mis-estimated collar should stop the move
        #    rather than be pushed through.
        if abs(adv_m) > 1e-6:
            if not _guarded(robot, guard_shared, lambda: robot.arm.move_l(
                    T_start, label='collar advance onto the ring')):
                log.error('Could not advance onto the collar (%+.1f mm along the connector +X). '
                          'The fingers are still clear of the ring, so nothing is clamped.',
                          adv_m * 1000.0)
                return False
        if not robot.gripper.close('grasp collar'):
            log.error('Gripper did not close on the collar.')
            return False
        phase('collar_clock')
        start = robot.tool0()
        adm_cl.reset()
        adm_cl.warmup(start, tare_fn=cl_tare)
        guard_cl.reset()
        # SUBDIVIDED ON THE TRUE ARC -- see screw_ramp. Same 90 deg orbit about the same axis
        # as the cable screw, but here the fingers are CLOSED on the collar, so the chord's 52 mm
        # excursion would be applied straight into the ring.
        res = screw_ramp(adm_cl, lambda f: rotate_about_axis(start, axis, point, cl_rot * f),
                         guard_cl, cl_v, cl_w, abs(np.degrees(cl_rot)), label='turn ')
        _lin, turned = pose_error(start, robot.tool0())
        adm_cl.reset()
        if cl_settle > 0:
            adm_cl.hold(robot.tool0(), cl_settle, guard=None)
        adm_cl.stop()
        robot.arm.servo_stop()
        stopped = res == 'seated'            # ramp's word for a guard trip -- not the state
        clock_rows.append({'maneuver': 'collar_clocking', 'try': 1, 'ramp_result': res,
                           'prewind_deg': round(float(np.degrees(cl_prewind)), 3),
                           'turned_deg': round(float(np.degrees(turned)), 3),
                           'commanded_deg': round(float(np.degrees(cl_rot)), 3),
                           'success': True, 'force_stop': bool(stopped),
                           'state_after': 'locked',
                           'stopped_by': guard_cl.tripped_by or ''})
        if stopped:
            log.info('  LOCKED -- the collar stopped on the force guard (%s) after %.1f deg, '
                     'which is what reaching the lock looks like; check it.',
                     guard_cl.tripped_by, np.degrees(turned))
        else:
            log.info('  LOCKED -- collar turned %.1f deg (commanded %.1f).', np.degrees(turned),
                     np.degrees(cl_rot))
        return True

    # ---- RESET + PICK + slip-checked LIFT (identical to cable_pick_estimate_assemble) ----
    phase('reset')
    if not reset.reset_robot(robot, cfg, 'start reset'):
        return False
    q_home = robot.arm.q()
    attempt = 0
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
            if not (robot.gripper.open('drop') and robot.arm.move_j(q_home, label='home')):
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
        and driven everywhere else. There used to be a second helper for poses stated DIRECTLY
        wrt the recorded mate; the standalone wiggle was its only caller and both are gone, so
        every reference in this app is now an anchored trajectory row.

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
    # UNCONDITIONAL, unlike the gates below: this is the boundary between free space and
    # contact, so it is asked even with --yes.
    if not robot.arm.dry_run:
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
            # ENGAGE replaces the estimate loop and the commit. A force stop is
            # an ordinary outcome, so both endings continue to the clocking sequence; the gate
            # before cable clocking is where a person decides whether the depth reached is good.
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
                # Intermediate passes MUST back off -- the next one realigns to a different
                # offset's start. The LAST pass stays at its stop, so that if the operator calls
                # the mate successful the arm is still AT the seat and clocking can anchor on it.
                # The retract for a retry happens after the verdict instead (see below).
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
            if robot.arm.dry_run:
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
                if not robot.arm.dry_run:
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

    # ---- POST-MATE: CABLE CLOCKING, then COLLAR CLOCKING, then the shared escape -------------
    # Both paths end in the same retract, so a failed screw and a completed collar turn leave the
    # arm in the same place. The mate itself already succeeded by the time we get here, so a
    # clocking failure is reported without retracting that: the return value says the requested
    # sequence did not complete, and the log says which part of it did.
    if cc_on:
        cc_ok = ret_ok = False
        state = 'engaged'                    # the initial assembly mated it; that is where we are
        if not phase_gate('CABLE CLOCKING (insert)',
                          'The connector is ENGAGED. Next is the bayonet screw, which cams it '
                          'HOME -- check the engagement looks right first.'):
            robot.arm.servo_stop()
            return False
        try:
            cc_ok, _T_tool0_conn, T_base_conn = cable_clocking()
            if cc_ok:
                state = _advance_state(state, 'engaged')          # -> seated
                if cl_on and not phase_gate(
                        'COLLAR CLOCKING (lock)',
                        'The connector is SEATED. Next is the collar turn, which LOCKS it -- the '
                        'gripper will re-grasp the collar and rotate. Check the seat first.'):
                    robot.arm.servo_stop()
                    return False
                if cl_on and collar_clocking(T_base_conn):
                    state = _advance_state(state, 'seated')       # -> locked
                elif cl_on:
                    cc_ok = False
            else:
                log.error('CABLE CLOCKING FAILED after %d tr%s (peak advance below the %.1f mm '
                          'threshold) -- the connector is ENGAGED but NOT SEATED. The mate itself '
                          'succeeded; skipping collar clocking and retracting.',
                          cc_tries, 'y' if cc_tries == 1 else 'ies', cc_need_m * 1000.0)
            if phase_gate('ESCAPE',
                           'Clocking done. Next is the two-leg retract: the gripper backs off '
                           'its own -Z, then away along the target -X.'):
                ret_ok = clocking_retract()
            else:
                log.warning('Escape skipped by the user -- the arm is still at the connector.')
        finally:
            robot.arm.servo_stop()
            if clock_rows:
                keys = sorted({k for r in clock_rows for k in r}, key=str)
                with open(os.path.join(out_dir, 'clocking.csv'), 'w', newline='') as fh:
                    w = _csv.DictWriter(fh, fieldnames=keys)
                    w.writeheader()
                    w.writerows(clock_rows)
                log.info('Clocking log: %s', os.path.join(out_dir, 'clocking.csv'))
        # Report the state actually REACHED. 'assembled' is claimed only at 'locked': a seated but
        # unlocked BNC can still back out, so a run with collar clocking disabled succeeds (its
        # configured sequence finished) without being called assembled.
        if state == 'locked':
            log.info('ASSEMBLED -- connector ENGAGED -> SEATED -> LOCKED.')
        elif state == 'seated':
            log.warning('Connector SEATED but NOT LOCKED (collar clocking %s) -- not assembled.',
                        'disabled' if not cl_on else 'FAILED')
        else:
            log.error('Connector ENGAGED only -- neither seated nor locked.')
        phase('reset')
        rst = reset.reset_robot(robot, cfg, 'end reset')      # always, even after a failed screw
        return bool(cc_ok and ret_ok and rst)

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
    return ok and reset.reset_robot(robot, cfg, 'end reset')


def main():
    run_app('BNC assembly: pick + estimator_eval-tuned manifold estimation',
            'bnc_assembly', build_and_run, needs_camera=True)


if __name__ == '__main__':
    main()
