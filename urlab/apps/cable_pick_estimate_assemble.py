"""Cable pick, then ESTIMATE-while-ASSEMBLING -- cable_pick_assemble + the contact manifold.

The PICK is exactly the cable_pick_assemble pipeline (scan, grasp, check, recovery).  The
assembly differs: the robot KNOWS the target connector pose (assembly.target_frame names a
configs/frames.yaml frame whose targets: entry is the recorded mate) and holds an ESTIMATE of
the connector-in-hand (fingertip -> connector, initialised from the grasp geometry) that
carries in-hand error.  Each attempt runs the assembly trajectory under software admittance,
yielding to contact and LOGGING observations (believed connector-wrt-target pose + wrench in
the believed connector frame):

    [pick] -> lift (slip-checked) -> stand-off (held check) ->
        LOOP (max assembly.max_attempts):
            assemble (admittance, guarded, observing)
            check    (operator decides; believed pose vs target logged for the record)
            retract  (linear, back along the connector's own -X)
            held check (the insertion/retract can strip the part out of the fingers)
            estimate (ICP of the observations against the CONTACT MANIFOLD, skills/manifold.py)
            update   (T_fingertip_connector <- T_fingertip_connector @ T_corr)
            realign  (recompute the trajectory references from the new estimate)

WHY THE CHECK WORKS: the check uses the BELIEVED pose, but under admittance a wrong belief
cannot fake success -- if the part jams short of the mate the arm DEFLECTS off the reference,
so the believed connector pose lags the target and the check fails.  The failed attempt's
observations are exactly what the manifold estimator needs to correct the belief.

Units: robot poses are metres/radians (repo convention); the manifold space is mm/deg -- the
conversions happen only at the observation/correction boundary in this file.
"""

import csv as _csv
import os

import numpy as np

from .. import behaviors as bt
from .. import config as urconfig
from .. import log as urlog
from .. import tool_frames
from ..robot import AdmittanceController, ForceGuard
from ..skills import reset
from ..skills import trajectory as traj
from ..skills.manifold import (FORCE_COLS, ManifoldEstimator, POSE_COLS, TORQUE_COLS,
                               vec6_from_mats)
from ..skills.pick import (GraspCheck, GraspController, GraspGeometry, GraspImageRecorder,
                           GraspRecovery, belief_offset_m, fingertip_in_connector,
                           held_belief, offset_belief, retry_offset_x, verify_cable_held)
from ..transforms import from_cfg, inverse, matrix_to_xyzrpy, pose_error, translation_matrix
from ._cable import build_scanner, make_confirm
from ._common import experiment_dir, prompts_off, seg_time
from ._estimate_plots import plot_estimate as _plot_estimate
from ._estimate_plots import plot_run as _plot_run
from ._runner import run_app
from .cable_pick_assemble import _guarded, _pick
from .uncertain_sampling import _retract_ref

log = urlog.get('cable-est-assemble')


def _corr_to_m(T_corr_mm):
    """The estimator's correction (translation in mm) -> a metre-based transform."""
    T = np.array(T_corr_mm, dtype=float)
    T[:3, 3] /= 1000.0
    return T


def _observe(robot, T_tool0_conn, T_base_tconn):
    """One observation row: believed connector-wrt-target [mm, deg 6-vec] + raw wrench in the
    believed connector frame [N, Nm]."""
    T_base_tool0 = robot.tool0()
    T_base_conn = T_base_tool0 @ T_tool0_conn
    rel = inverse(T_base_tconn) @ T_base_conn
    xyz, rpy = matrix_to_xyzrpy(rel)
    # Flange pose handed over: wrench_in moves the moment's reference point off the flange,
    # and re-reading the pose there would pair the wrench with a different cycle.
    w = robot.arm.wrench_in(T_base_conn, T_base_tool0)
    return list(xyz * 1000.0) + list(np.degrees(rpy)) + list(w)


def _save_observations(path, rows):
    with open(path, 'w', newline='') as fh:
        w = _csv.writer(fh)
        w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
        w.writerows(rows)


def _parse_traj_noise(a):
    """assembly.trajectory_noise -> a normalised dict (enabled/std/window/decays), or None on
    a malformed std vector.  Same semantics as estimator_eval's eval.trajectory_noise."""
    tn = a.get('trajectory_noise', {}) or {}
    std = [float(v) for v in (tn.get('std') or [0.0005] * 3 + [0.5] * 3)]
    if len(std) != 6:
        log.error('assembly.trajectory_noise.std must have 6 entries [x,y,z (m), r,p,y (deg)].')
        return None
    return {'enabled': bool(tn.get('enabled', False)), 'std': std,
            'window': max(1, int(tn.get('smooth_window', 25))),
            'decay_attempt': float(tn.get('noise_decay_attempt', 0.0)),
            'decay_traj': float(tn.get('noise_decay_traj', 0.0))}


class _AssemblyTask:
    """State + phase actions for one pick-and-assemble run; build_and_run wires these into the
    behavior tree."""

    def __init__(self, cfg, robot, camera, estimator):
        a = cfg.section('assembly')
        self.cfg = cfg
        self.robot = robot
        self.a = a
        self.estimator = estimator

        self.scanner, _detector, _estimator = build_scanner(cfg, robot, camera)
        self.geom = GraspGeometry(cfg)
        self.check = GraspCheck(cfg)
        self.recovery = GraspRecovery(cfg)
        self.grasp = GraspController(cfg)
        self.recorder = GraspImageRecorder(cfg)
        self.guard = ForceGuard(robot.arm, a.get('force_guard', {}))
        self.adm = AdmittanceController(robot.arm, a.get('compliance', {}))
        self.confirm = make_confirm(cfg)
        self._gate = self.confirm if self.confirm is not None else None

        # The recorded mate, from the shared frames catalogue.
        self.T_base_tconn = None                     # set by build_and_run after validation

        # The ideal insertion path (connector wrt the target connector, last row = the mate).
        csv_in = urconfig.resolve(cfg, a.get('trajectory_csv', 'assembly_trajectory.csv'))
        self.mats = traj.load_csv(csv_in,
                                  angles_deg=bool(a.get('trajectory_angles_deg', False)))
        if float(np.abs(self.mats[-1] - np.eye(4)).max()) > 1e-6:
            log.warning('trajectory last row is not identity -- rows are still applied '
                        'relative to the recorded connector target.')
        self.dense = traj.resample(self.mats,
                                   float(a.get('translational_resolution_m', 0.001)),
                                   float(a.get('rotational_resolution_deg', 1.0)))

        # The in-hand ESTIMATE (fingertip -> connector). The grasp geometry is the initial
        # belief; estimation.initial_connector_in_fingertip (m/rad) overrides it when set.
        init = cfg.get_path('estimation.initial_connector_in_fingertip')
        self.T_ftip_conn = from_cfg(init) if init \
            else from_cfg(cfg.section('junction_in_fingertip'))
        # A pitched pickup rotates the part in the hand by the same angle (see skills/pick).
        grip_off = fingertip_in_connector(cfg)
        oxyz, orpy = matrix_to_xyzrpy(grip_off)
        self.T_ftip_conn = held_belief(
            self.T_ftip_conn, from_cfg(cfg.section('junction_in_fingertip')), grip_off)
        log.info('fingertip_in_connector xyz %s mm rpy %s deg (target fingertip wrt the '
                 'connector) -> in-hand belief derived to match.',
                 np.round(oxyz * 1000.0, 2).tolist(), np.round(np.degrees(orpy), 2).tolist())
        # The measured seating residual -- BELIEF ONLY, nothing moves at pickup.
        bel_off = belief_offset_m(cfg)
        if float(np.linalg.norm(bel_off)) > 0.0:
            self.T_ftip_conn = offset_belief(self.T_ftip_conn, bel_off)
            log.info('Belief offset %s mm (CONNECTOR frame) applied to the in-hand pose ONLY.',
                     np.round(bel_off * 1000.0, 2).tolist())

        # Speeds: ONE global speed: block, each phase applying its own scale to all four
        # limits (speed.phase_scale.<phase>).  Free-space moves inherit the scale from
        # arm.set_speed_scale; the compliant ramps are paced by seg_time below.
        spd = cfg.section('speed')
        self.scales = spd.get('phase_scale', {}) or {}
        self.g_v = float(spd.get('max_cartesian_translation_mm_s', 3.5))
        self.g_w = float(spd.get('max_cartesian_rotation_deg_s', 5.0))
        self.s_asm = float(self.scales.get('assemble', 1.0))
        self.s_ret = float(self.scales.get('retract', 1.0))
        self.min_seg_s = 1.0 / self.adm.rate

        comp = a.get('compliance', {}) or {}
        self.settle_s = float(comp.get('settle_s', 0.5))
        self.tare = (lambda: robot.arm.zero_ft(settle=False)) \
            if bool(comp.get('tare_before', True)) else None
        self.retract_m = float(a.get('retract_distance_m', 0.05))
        self.decim = max(1, int(a.get('log_decimation', 5)))
        tol = a.get('success_tolerance', {}) or {}
        self.tol_pos_m = float(tol.get('pos_mm', 2.0)) / 1000.0
        self.tol_rot_rad = np.radians(float(tol.get('rot_deg', 3.0)))
        self.max_attempts = int(a.get('max_attempts', 5))

        self.noise = None                            # set by build_and_run (may refuse)
        self.noise_rng = np.random.default_rng()

        # LIVE run figure: one fixed path outside the experiment folder, atomically
        # overwritten after every estimate. true = data/experiments/cable_pick_live.png;
        # a string = explicit path; false disables.
        live = a.get('live_plot', True)
        self.live_path = None
        if live:
            self.live_path = live if isinstance(live, str) else os.path.join(
                cfg.get('data_dir', 'data'), 'experiments', 'cable_pick_live.png')
            os.makedirs(os.path.dirname(self.live_path) or '.', exist_ok=True)

        self.out_dir = experiment_dir(cfg, 'cable_pick_estimate_assemble')

        standoff = a.get('standoff', {}) or {}
        axis = np.asarray(standoff.get('axis', [-1, 0, 0]), dtype=float)
        self.T_standoff_row = translation_matrix(
            axis * float(standoff.get('distance_m', 0.01))) @ self.mats[0]

        self.q_home = None
        self.seed = {'q': None}
        self.est_rows = []
        self.success = False

    # ---- shared little helpers ---------------------------------------------------------------
    def phase(self, name):
        self.robot.arm.set_speed_scale(float(self.scales.get(name, 1.0)), name)

    def seg_time(self, A, B, v=None, w=None):
        return seg_time(A, B, self.g_v * self.s_asm if v is None else v,
                        self.g_w * self.s_asm if w is None else w, self.min_seg_s)

    def tool0_ref(self, row, T_tool0_conn):
        return self.T_base_tconn @ row @ inverse(T_tool0_conn)

    # ---- phases ------------------------------------------------------------------------------
    def start_reset(self):
        self.phase('reset')
        if not reset.reset_robot(self.robot, self.cfg, 'start reset'):
            return False
        self.q_home = self.robot.arm.q()
        return True

    def pick_and_lift(self):
        """The pick with grasp-check retry.  A cable that slips out during the LIFT restarts
        the whole scan->grasp; each full retry perturbs the grasp along the junction +/-x
        (retry_offset_x), because a deterministic scan->grasp->fail loop is a fixed point."""
        attempt = 0
        while True:
            self.phase('scan')
            result = _pick(self.cfg, self.robot, self.scanner, self.geom, self.check,
                           self.recovery, self.grasp, self.confirm, self.recorder,
                           offset_x_m=retry_offset_x(attempt, self.check.retry_perturb_x_m))
            if result == 'ok':
                status = {}

                def do_lift(_s=status):
                    _s['r'] = self.grasp.lift_verified(
                        self.robot, self.geom, self.check, 'lift',
                        position_guard=lambda mv: _guarded(self.robot, self.guard, mv))
                    return _s['r'] == 'ok'

                if bt.run_tree(bt.Action('lift (slip-checked)', do_lift,
                                         confirm=self._gate), log):
                    return True
                result = status.get('r')
                if result != 'slipped':
                    return False       # a move failed, or the user aborted at the step gate
            if result == 'abort':
                return False
            if attempt >= self.check.max_retries:
                log.error('Grasp failed on all %d attempts; aborting.',
                          self.check.max_retries + 1)
                return False
            attempt += 1
            log.warning('Grasp %s -- recovering (attempt %d/%d).',
                        result, attempt + 1, self.check.max_retries + 1)
            if result == 'slipped' and hasattr(self.scanner, 'reselect'):
                # SLIP RECOVERY -- lighter than a full reset: the cable fell somewhere below,
                # so open, rise straight up, and let the next scan re-image and re-prompt from
                # this vantage (reselect() drops the now-stale cached junction selection).
                self.phase('scan')
                T_up = translation_matrix([0.0, 0.0, self.check.slip_raise_m]) \
                    @ self.robot.tool0()
                if not (self.robot.gripper.open('drop')
                        and self.robot.move_cartesian(T_up, interpolation='lin',
                                                      label='slip recovery (up)',
                                                      guard=self.guard)):
                    return False
                self.scanner.reselect()
            else:
                self.phase('reset')
                if not (self.robot.gripper.open('drop')
                        and self.robot.move_joints(self.q_home, label='home')):
                    return False

    def payload_check(self):
        """Informational: the stalled counts -> held width through the calibrated gripper
        model, in units a human can check against the datasheet with calipers."""
        d_conn = self.cfg.get_path('grasp_check.connector_diameter_mm')
        if d_conn and not self.robot.arm.dry_run:
            w_mm = self.robot.gripper.held_width_m() * 1000.0
            log.info('Payload width: %.2f mm (expected connector %.2f-%.2f mm).',
                     w_mm, min(d_conn), max(d_conn))

    def goto_standoff(self):
        """Approach the stand-off (beyond the trajectory START, target frame)."""
        self.phase('standoff')
        self.seed['q'] = self.robot.arm.q()
        T_tool0_conn = self.robot.T_tool0_fingertip @ self.T_ftip_conn
        return self.robot.move_cartesian(self.tool0_ref(self.T_standoff_row, T_tool0_conn),
                                         label='stand-off', seed=self.seed,
                                         guard=self.guard)

    # ---- the assemble / check / retract / estimate loop --------------------------------------
    def _attempt_refs(self, it, T_tool0_conn):
        tn = self.noise
        if tn['enabled']:
            rows = traj.noised(self.dense, self.noise_rng, tn['std'], tn['window'],
                               tn['decay_traj'], (1.0 - tn['decay_attempt']) ** (it - 1))
        else:
            rows = self.dense
        return [self.tool0_ref(row, T_tool0_conn) for row in rows]

    def _assemble(self, refs, log_cb):
        """Drive the references under admittance until done or the guard trips ('seated'),
        then settle.  Returns the last commanded reference."""
        self.adm.reset()
        self.adm.warmup(refs[0], tare_fn=self.tare)
        self.guard.reset()
        last_ref = refs[0]
        for i in range(1, len(refs)):
            res = self.adm.ramp(refs[i - 1], refs[i], self.seg_time(refs[i - 1], refs[i]),
                                self.guard, on_step=log_cb)
            last_ref = refs[i]
            if res == 'seated':
                log.info('Contact limit at waypoint %d/%d -- stopped advancing.',
                         i, len(refs) - 1)
                break
        self.adm.hold(last_ref, self.settle_s, self.guard, on_step=log_cb)
        return last_ref

    def _judge(self, it, row):
        """The SUCCESS DECISION is the OPERATOR's -- they can see the physical mate; the
        kinematic numbers only see the belief.  A dry run has no operator, so it falls back
        to the tolerance check.  Returns 'done' | 'retry' | 'abort'."""
        if self.robot.arm.dry_run or prompts_off(self.cfg):
            row['success'] = bool(row['check_pos_mm'] / 1000.0 <= self.tol_pos_m
                                  and np.radians(row['check_rot_deg']) <= self.tol_rot_rad)
            return 'done' if row['success'] else 'retry'
        try:
            ans = input(f'[check attempt {it}] Was the assembly SUCCESSFUL? '
                        '(y = done / Enter = retry / q = abort): ').strip().lower()
        except EOFError:
            ans = ''
        if ans in ('q', 'quit'):
            log.info('Aborted at the check by the user.')
            return 'abort'
        row['success'] = ans in ('y', 'yes')
        return 'done' if row['success'] else 'retry'

    def _estimate_and_update(self, it, obs, trackc, trackr, trackg, row, lin, ang):
        """One manifold-ICP estimate from this attempt's observations; applies the correction
        to the in-hand belief and refreshes the run figures."""
        obs_arr = np.asarray(obs, dtype=float)
        vec6, w6 = self.estimator.prepare_observations(
            obs_arr[:, :6], obs_arr[:, 6:9], obs_arr[:, 9:12]) if len(obs) else \
            (np.zeros((0, 6)), np.zeros((0, 6)))
        T_corr_mm, info = self.estimator.estimate(vec6, w6)
        if T_corr_mm is None:
            log.warning('Estimation skipped (%s) -- retrying with the UNCHANGED estimate.',
                        info)
            row['estimate'] = f'skipped: {info}'
            self.est_rows.append(row)
            trackc.append(trackc[-1])          # belief unchanged this attempt
            trackr.append(float('nan'))
            trackg.append(np.zeros(0))
            _plot_run(os.path.join(self.out_dir, 'run_corrections.png'),
                      self.estimator.estimate_dims, trackc, trackr, trackg,
                      f'attempt {it}/{self.max_attempts} (estimate skipped)', self.live_path)
            return None
        log.info('estimated belief correction: %s  (inliers %d/%d, residual %.3f, %d obs)',
                 {k: round(v, 3) for k, v in info['theta_corr'].items()},
                 info['inliers'], info['guesses'], info['final_residual'],
                 info['n_observations'])
        _plot_estimate(os.path.join(self.out_dir, f'attempt_{it:02d}_estimate.png'),
                       self.estimator.estimate_dims, info)
        self.T_ftip_conn = self.T_ftip_conn @ _corr_to_m(T_corr_mm)   # believed @ corr ~= true
        self._T_cum = self._T_cum @ np.asarray(T_corr_mm, dtype=float)
        trackc.append(vec6_from_mats(self._T_cum)[self.estimator.idx])
        trackr.append(float(info['final_residual']))
        trackg.append(np.asarray(info['res_hist'], dtype=float)[:, -1])
        _plot_run(os.path.join(self.out_dir, 'run_corrections.png'),
                  self.estimator.estimate_dims, trackc, trackr, trackg,
                  f'attempt {it}/{self.max_attempts} | check {lin * 1000:.1f} mm / '
                  f'{np.degrees(ang):.1f} deg', self.live_path)
        row.update({f'corr_{k}': v for k, v in info['theta_corr'].items()})
        row.update({'icp_inliers': info['inliers'], 'icp_residual': info['final_residual']})
        self.est_rows.append(row)
        return True

    def assembly_loop(self):
        try:
            # Run-level tracks for the figure: cumulative applied correction (row 0 = the
            # initial in-hand estimate) + per-attempt residuals.
            self._T_cum = np.eye(4)
            trackc = [np.zeros(len(self.estimator.estimate_dims))]
            trackr, trackg = [], []

            for it in range(1, self.max_attempts + 1):
                T_tool0_conn = self.robot.T_tool0_fingertip @ self.T_ftip_conn
                e_xyz, e_rpy = matrix_to_xyzrpy(self.T_ftip_conn)
                log.info('--- attempt %d/%d --- in-hand estimate xyz=%s mm rpy=%s deg',
                         it, self.max_attempts, np.round(e_xyz * 1000, 2).tolist(),
                         np.round(np.degrees(e_rpy), 2).tolist())
                refs = self._attempt_refs(it, T_tool0_conn)

                # Realign with the START of the (re-estimated) trajectory -- stiff, free
                # space, guarded.
                self.phase('standoff')
                if not self.robot.move_cartesian(refs[0], label=f'align start {it}',
                                                 seed=self.seed, guard=self.guard):
                    log.error('Could not reach the trajectory start; aborting.')
                    return False

                # ASSEMBLE under admittance, collecting observations.
                obs, cnt = [], [0]

                def log_cb(_obs=obs, _cnt=cnt, _Ttc=T_tool0_conn):
                    _cnt[0] += 1
                    if _cnt[0] % self.decim == 0:
                        _obs.append(_observe(self.robot, _Ttc, self.T_base_tconn))

                last_ref = self._assemble(refs, log_cb)

                # CHECK: kinematic numbers logged for the record; the decision is the
                # operator's (_judge).
                T_conn_now = self.robot.tool0() @ T_tool0_conn
                lin, ang = pose_error(T_conn_now, self.T_base_tconn)
                log.info('check: believed connector vs target: %.2f mm, %.2f deg (reference '
                         'tol %.2f mm, %.2f deg)', lin * 1000, np.degrees(ang),
                         self.tol_pos_m * 1000, np.degrees(self.tol_rot_rad))
                _save_observations(
                    os.path.join(self.out_dir, f'attempt_{it:02d}_observations.csv'), obs)

                row = {'attempt': it, 'n_observations': len(obs),
                       'check_pos_mm': lin * 1000.0,
                       'check_rot_deg': float(np.degrees(ang))}
                verdict = self._judge(it, row)
                if verdict == 'abort':
                    self.est_rows.append(row)
                    return False
                if verdict == 'done':
                    self.est_rows.append(row)
                    log.info('Within tolerance -- ASSEMBLY COMPLETE on attempt %d.', it)
                    self.success = True
                    break

                # RETRACT: linear escape along the connector's own -X (compliant,
                # un-guarded), at the 'retract' phase scale.
                T_out = _retract_ref(last_ref, T_tool0_conn, self.retract_m)
                self.adm.ramp(last_ref, T_out,
                              self.seg_time(last_ref, T_out, self.g_v * self.s_ret,
                                            self.g_w * self.s_ret), guard=None)
                self.robot.arm.servo_stop()

                # CABLE-IN-GRIPPER check: an insertion/retract can strip the part out of the
                # fingers (it may even be left IN the socket) -- without it, further attempts
                # and this attempt's estimate are meaningless.
                if not verify_cable_held(self.robot, self.check, f'attempt {it} retract'):
                    row['cable_held'] = False
                    self.est_rows.append(row)
                    return False

                if it == self.max_attempts:
                    self.est_rows.append(row)
                    log.error('Attempt limit reached (%d) without a successful mate.',
                              self.max_attempts)
                    break

                self._estimate_and_update(it, obs, trackc, trackr, trackg, row, lin, ang)
        finally:
            self.robot.arm.servo_stop()
            self._write_estimates()
        return self.success

    def _write_estimates(self):
        if not self.est_rows:
            return
        keys = sorted({k for r in self.est_rows for k in r}, key=str)
        path = os.path.join(self.out_dir, 'estimates.csv')
        with open(path, 'w', newline='') as fh:
            w = _csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(self.est_rows)
        log.info('Per-attempt log: %s', path)

    # ---- release + escape --------------------------------------------------------------------
    def release_escape(self):
        """Escape along the CONNECTOR's OWN -X (peg-in-hole: the direction comes from the
        connector frame at the mate, executed as a pure world translation)."""
        d_out = float(self.a.get('release_retract_distance_m', 0.08))
        T_tool0_conn = self.robot.T_tool0_fingertip @ self.T_ftip_conn
        T_conn_final = self.robot.tool0() @ T_tool0_conn
        back = -T_conn_final[:3, 0] * d_out              # connector -X, in base coordinates
        T_new = translation_matrix(back) @ self.robot.tool0()
        return self.robot.move_cartesian(T_new, interpolation='lin',
                                         label='retract (connector -X)', guard=self.guard)

    def end_reset(self):
        self.phase('reset')
        return reset.reset_robot(self.robot, self.cfg, 'end reset')


def build_and_run(cfg, robot, camera, args):
    a = cfg.section('assembly')

    # Build the ESTIMATOR first -- a missing/stale manifold CSV must fail before the robot
    # moves.  (Module-level name on purpose: the *_check variant swaps the class.)
    estimator = ManifoldEstimator(cfg.section('estimation'))

    # The KNOWN target: assembly.target_frame names a configs/frames.yaml frame whose
    # targets: entry is the recorded mate (base_link <- connector, one per socket).
    tname = a.get('target_frame')
    if not tname:
        log.error('assembly.target_frame is required -- name a %s frame whose targets: entry '
                  'records the mate (hand-guide to a good mate, read `base_link <- <frame>` '
                  'off the monitor, paste it under targets:).', tool_frames.frames_path(cfg))
        return False
    targets = tool_frames.load_targets(cfg)
    if tname not in targets:
        log.error('assembly.target_frame %r has no targets: entry in %s.',
                  tname, tool_frames.frames_path(cfg))
        return False
    task = _AssemblyTask(cfg, robot, camera, estimator)
    task.T_base_tconn = targets[tname]
    task.noise = _parse_traj_noise(a)
    if task.noise is None:
        return False
    if task.noise['enabled']:
        log.info('Trajectory noise ON: std %s, smooth %d, decay/attempt %.2f, decay/traj '
                 '%.2f.', task.noise['std'], task.noise['window'],
                 task.noise['decay_attempt'], task.noise['decay_traj'])
    log.info('Experiment folder: %s', task.out_dir)

    gate = task._gate
    root = bt.sequence(
        'cable-pick-estimate-assemble',
        bt.Action('start reset', task.start_reset),
        bt.Action('pick + slip-checked lift', task.pick_and_lift),
        bt.Action('payload width check', task.payload_check),
        bt.Action('approach stand-off', task.goto_standoff),
        # The transit from the lift can lose the part without any force signature.
        bt.VerifyHeld(robot, task.check, 'stand-off'),
        # UNCONDITIONAL pause: the next motion drives the held part into contact, so a human
        # confirms the scene is ready regardless of confirm_each_step.
        bt.OperatorGate(robot, '\n[stand-off] Ready to ASSEMBLE (contact ahead). '
                               'Enter to continue (q to abort): ', label='stand-off gate',
                        skip=prompts_off(cfg)),
        bt.Action('assemble / estimate / retry loop', task.assembly_loop),
        bt.Action('set retract speed', lambda: task.phase('retract')),
        bt.Action('open gripper (release)', lambda: robot.gripper.open(), confirm=gate),
        bt.Action('retract (connector -X)', task.release_escape, confirm=gate),
        bt.Action('end reset', task.end_reset))
    return bt.run_tree(root, log)


def main():
    run_app('Cable pick + estimate-while-assemble (contact-manifold ICP)',
            'cable_pick_estimate_assemble', build_and_run, needs_camera=True)


if __name__ == '__main__':
    main()
