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

import numpy as np

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
    fin = fi.get('trajectory_noise', {}) or {}
    fi_noise_on = bool(fin.get('enabled', False))
    fi_noise_std = [float(v) for v in (fin.get('std') or [0.0] * 6)]
    fi_noise_w = max(1, int(fin.get('smooth_window', 10)))
    if fi_noise_on and len(fi_noise_std) != 6:
        log.error('final_insertion.trajectory_noise.std must have 6 entries.')
        return False
    for nm, v in (('speed_translation_mm_s', fi_v), ('speed_rotation_deg_s', fi_wr),
                  ('pause_s', fi_pause)):
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

    # ---- CLOCKING (post-mate): CABLE clocking, then COLLAR clocking --------------------------
    # Two maneuvers that run only after a mate the operator called successful. Each gets its OWN
    # compliance and force guard, and unlike final_insertion the guard override is not optional in
    # practice: the shared force_guard: is tuned for a light probing insertion (5 N), and a
    # deliberate press-and-twist exceeds that on the first cycle. Inheriting it would mean the
    # screw never runs.
    cc = a.get('cable_clocking', {}) or {}
    cl = a.get('collar_clocking', {}) or {}

    def _num(block, key, default):
        """A number from the block, treating an EXPLICIT null the same as absent -- the config's
        own "null inherits" convention. float(None) would otherwise raise on a key the comments
        invite you to null out."""
        v = block.get(key)
        return float(default) if v is None else float(v)

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
    min_seg_s = 1.0 / adm.rate

    def seg_time(A, B, v=None, w=None):
        v = g_v * s_asm if v is None else v
        w = g_w * s_asm if w is None else w
        lin_m, ang_rad = pose_error(A, B)
        return max((lin_m * 1000.0 / v) if v > 0 else 0.0,
                   (np.degrees(ang_rad) / w) if w > 0 else 0.0, min_seg_s)

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

    init = cfg.get_path('estimation.initial_connector_in_fingertip')
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

    def run_insertion(adm_ctl, refs, T_tool0_conn, peck=False, guard_ctl=None, settle=None,
                      hold=None, speed=None, pause=None):
        """One admittance insertion along refs, collecting observations; then the compliant
        UN-GUARDED retract along the believed connector's own -X. Mirrors estimator_eval's
        run_insertion, including peck (a force stop backs off and advances again rather than
        ending the pass) and the un-guarded post-insertion dwell."""
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
        T_out = _retract_ref(last_ref, T_tool0_conn, retract_m)
        adm_ctl.ramp(last_ref, T_out, seg_time(last_ref, T_out, g_v * s_ret, g_w * s_ret),
                     guard=None)
        robot.arm.servo_stop()
        return obs, lin, ang, stops

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

        screw = xyzrpy_to_matrix([cc_push_m, 0.0, 0.0], [cc_rot, 0.0, 0.0])
        # The stroke as a BASE-frame motion about the fixed axis line L (engaged connector origin,
        # along its +X). Independent of the connector-in-gripper belief, and the SAME every try.
        ref_start = T_tool0_engaged
        ref_goal = (T_base_tconn @ screw @ inverse(T_base_tconn)) @ T_tool0_engaged

        det = _ScrewAdvance(robot, T_tool0_conn, T_base_tconn, cc_need_m)
        combo = _AnyGuard(det, guard_cc)
        ok = False
        for k in range(1, cc_tries + 1):
            phase('cable_clock')
            adm_cc.reset()
            adm_cc.warmup(ref_start, tare_fn=cc_tare)
            combo.reset()
            res = adm_cc.ramp(ref_start, ref_goal, seg_time(ref_start, ref_goal, cc_v, cc_w),
                              combo)
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
        T_base_collar = T_base_conn @ translation_matrix([cl_off_m, 0.0, 0.0])
        T_ref = T_base_collar @ inverse(robot.T_tool0_fingertip)
        log.info('--- COLLAR CLOCKING --- collar is %.1f mm along the connector +X; aligning the '
                 'CLOSED fingertip frame with it, then %+.1f deg about the connector +X',
                 cl_off_m * 1000.0, np.degrees(cl_rot))
        phase('standoff')
        if not _guarded(robot, guard_shared,
                        lambda: robot.arm.move_l(T_ref, label='collar align')):
            log.error('Could not reach the collar frame.')
            return False
        if not robot.gripper.close('grasp collar'):
            log.error('Gripper did not close on the collar.')
            return False
        phase('collar_clock')
        start = robot.tool0()
        end = rotate_about_axis(start, T_base_conn[:3, 0], T_base_conn[:3, 3], cl_rot)
        adm_cl.reset()
        adm_cl.warmup(start, tare_fn=cl_tare)
        guard_cl.reset()
        res = adm_cl.ramp(start, end, seg_time(start, end, cl_v, cl_w), guard_cl)
        _lin, turned = pose_error(start, robot.tool0())
        adm_cl.reset()
        if cl_settle > 0:
            adm_cl.hold(robot.tool0(), cl_settle, guard=None)
        adm_cl.stop()
        robot.arm.servo_stop()
        stopped = res == 'seated'            # ramp's word for a guard trip -- not the state
        clock_rows.append({'maneuver': 'collar_clocking', 'try': 1, 'ramp_result': res,
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

        screw = xyzrpy_to_matrix([cc_push_m, 0.0, 0.0], [cc_rot, 0.0, 0.0])
        # The stroke as a BASE-frame motion about the fixed axis line L (engaged connector origin,
        # along its +X). Independent of the connector-in-gripper belief, and the SAME every try.
        ref_start = T_tool0_engaged
        ref_goal = (T_base_tconn @ screw @ inverse(T_base_tconn)) @ T_tool0_engaged

        det = _ScrewAdvance(robot, T_tool0_conn, T_base_tconn, cc_need_m)
        combo = _AnyGuard(det, guard_cc)
        ok = False
        for k in range(1, cc_tries + 1):
            phase('cable_clock')
            adm_cc.reset()
            adm_cc.warmup(ref_start, tare_fn=cc_tare)
            combo.reset()
            res = adm_cc.ramp(ref_start, ref_goal, seg_time(ref_start, ref_goal, cc_v, cc_w),
                              combo)
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
        T_base_collar = T_base_conn @ translation_matrix([cl_off_m, 0.0, 0.0])
        T_ref = T_base_collar @ inverse(robot.T_tool0_fingertip)
        log.info('--- COLLAR CLOCKING --- collar is %.1f mm along the connector +X; aligning the '
                 'CLOSED fingertip frame with it, then %+.1f deg about the connector +X',
                 cl_off_m * 1000.0, np.degrees(cl_rot))
        phase('standoff')
        if not _guarded(robot, guard_shared,
                        lambda: robot.arm.move_l(T_ref, label='collar align')):
            log.error('Could not reach the collar frame.')
            return False
        if not robot.gripper.close('grasp collar'):
            log.error('Gripper did not close on the collar.')
            return False
        phase('collar_clock')
        start = robot.tool0()
        end = rotate_about_axis(start, T_base_conn[:3, 0], T_base_conn[:3, 3], cl_rot)
        adm_cl.reset()
        adm_cl.warmup(start, tare_fn=cl_tare)
        guard_cl.reset()
        res = adm_cl.ramp(start, end, seg_time(start, end, cl_v, cl_w), guard_cl)
        _lin, turned = pose_error(start, robot.tool0())
        adm_cl.reset()
        if cl_settle > 0:
            adm_cl.hold(robot.tool0(), cl_settle, guard=None)
        adm_cl.stop()
        robot.arm.servo_stop()
        stopped = res == 'seated'            # ramp's word for a guard trip -- not the state
        clock_rows.append({'maneuver': 'collar_clocking', 'try': 1, 'ramp_result': res,
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

    def tool0_ref(row, T_tool0_conn):
        return T_base_tconn @ row @ inverse(T_tool0_conn)

    phase('standoff')
    seed_q = robot.arm.q()
    T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
    q = robot.arm.ik(tool0_ref(T_standoff_row, T_tool0_conn), seed_q)
    if q is None or not _guarded(robot, guard_shared,
                                 lambda: robot.arm.move_j(q, label='stand-off')):
        return False
    seed_q = q
    if not verify_cable_held(robot, check, 'stand-off'):
        return False
    if not robot.arm.dry_run:
        try:
            ans = input('\n[stand-off] Ready to ASSEMBLE (contact ahead). '
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
        for it in range(1, max_attempts + 1):
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
                refs = [tool0_ref(row, T_tool0_conn) for row in rows_t]
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
                obs_i, lin, ang, stops = run_insertion(adm, refs, T_tool0_conn,
                                                       peck=(col_mode == 'peck'))
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
                log.info('ASSEMBLY COMPLETE on attempt %d.', it)
                success = True
                break
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
        if fi_on and not success:
            log.info('FINAL INSERTION from the corrected belief (zero noise).')
            T_tool0_conn = robot.T_tool0_fingertip @ T_ftip_conn
            rows_f = (traj.noised(dense, noise_rng, fi_noise_std, fi_noise_w, 0.0, 1.0)
                      if fi_noise_on else dense)
            refs = [tool0_ref(row_, T_tool0_conn) for row_ in rows_f]
            phase('standoff')
            q = robot.arm.ik(refs[0], seed_q)
            if q is None or not _guarded(robot, guard_shared,
                                         lambda: robot.arm.move_j(q, label='final start')):
                log.warning('IK/approach failed for the final insertion.')
            else:
                seed_q = q
                phase('assemble')
                obs_f, lin, ang, _ = run_insertion(
                    adm_final, refs, T_tool0_conn, guard_ctl=guard_final, settle=fi_settle,
                    hold=fi_hold, speed=(fi_v, fi_wr) if (fi_v or fi_wr) else None,
                    pause=fi_pause)
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
        try:
            cc_ok, _T_tool0_conn, T_base_conn = cable_clocking()
            if cc_ok:
                state = _advance_state(state, 'engaged')          # -> seated
                if cl_on and collar_clocking(T_base_conn):
                    state = _advance_state(state, 'seated')       # -> locked
                elif cl_on:
                    cc_ok = False
            else:
                log.error('CABLE CLOCKING FAILED after %d tr%s (peak advance below the %.1f mm '
                          'threshold) -- the connector is ENGAGED but NOT SEATED. The mate itself '
                          'succeeded; skipping collar clocking and retracting.',
                          cc_tries, 'y' if cc_tries == 1 else 'ies', cc_need_m * 1000.0)
            ret_ok = clocking_retract()
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


    # ---- POST-MATE: CABLE CLOCKING, then COLLAR CLOCKING, then the shared escape -------------
    # Both paths end in the same retract, so a failed screw and a completed collar turn leave the
    # arm in the same place. The mate itself already succeeded by the time we get here, so a
    # clocking failure is reported without retracting that: the return value says the requested
    # sequence did not complete, and the log says which part of it did.
    if cc_on:
        cc_ok = ret_ok = False
        state = 'engaged'                    # the initial assembly mated it; that is where we are
        try:
            cc_ok, _T_tool0_conn, T_base_conn = cable_clocking()
            if cc_ok:
                state = _advance_state(state, 'engaged')          # -> seated
                if cl_on and collar_clocking(T_base_conn):
                    state = _advance_state(state, 'seated')       # -> locked
                elif cl_on:
                    cc_ok = False
            else:
                log.error('CABLE CLOCKING FAILED after %d tr%s (peak advance below the %.1f mm '
                          'threshold) -- the connector is ENGAGED but NOT SEATED. The mate itself '
                          'succeeded; skipping collar clocking and retracting.',
                          cc_tries, 'y' if cc_tries == 1 else 'ies', cc_need_m * 1000.0)
            ret_ok = clocking_retract()
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
