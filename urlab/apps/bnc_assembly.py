"""BNC ASSEMBLY -- cable_pick_estimate_assemble's pipeline with estimator_eval's estimator.

Same shape as cable_pick_estimate_assemble (scan, grasp, slip-checked lift, stand-off, then an
assemble/estimate loop that corrects the in-hand belief), but everything downstream of the grasp
is estimator_eval's:

  * ESTIMATOR: the same `estimation:` block, including `commit: argmin`. `_argmin_estimate` and
    are IMPORTED from estimator_eval rather than reimplemented, so the two apps
    cannot drift.
  * COLLECTION: `collection.mode` = attempts | offset_sweep | peck. offset_sweep
    commands one insertion per deliberate offset with the belief held FIXED across passes (so
    their evidence fuses exactly); peck keeps advancing past each force stop.
  * COMPLIANCE / GUARD: read from the TOP-LEVEL `compliance:` and `force_guard:` blocks, the same
    names estimator_eval uses, so tuned values copy across verbatim.
  * FINAL INSERTION: `final_insertion` -- stiffness, mass, damping, settle, dwell and the
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
speed scale (`connector_clocking`, `collar_clocking`; a failed screw and a finished
collar turn share one escape, `clocking_retract`):

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

  * ABSOLUTE roll about the socket +X, wrt the TARGET frame -- `run.engage_clock_deg` and
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


import numpy as np

from .. import log as urlog
from ..log import StepRunner
from ..skills import manifold_debug, reset
from ..skills import trajectory as traj
from ..skills.manifold import vec6_from_mats
from ..skills.pick import retry_offset_x, verify_cable_held
from ..transforms import inverse, matrix_to_xyzrpy, translation_matrix
from ._runner import run_app
from ._common import guarded as _guarded
from .. import domain as _domain
from ..domain import BncSpec
from ..domain import advance_state as _advance_state
from ._estimate_plots import plot_run as _plot_run
from .cable_pick_assemble import _pick
from ..skills import bnc as bnc_skills
from ..skills.estimate import (corr_to_m as _corr_to_m,
                               save_observations as _save_observations)
from .estimator_eval import _argmin_estimate

log = urlog.get('bnc-assembly')

# Re-exported from urlab.domain for existing importers (tests included); the walk lives there.

# The assembly state progression, walked in order (see the module docstring).
# DISASSEMBLY walks the same ladder DOWNWARD and then one rung below the bottom. 'removed' is
# not a clocking state -- it means the connector is out of the socket and in the fingers, which
# is the only state from which placing it down is meaningful.



# The six pose axes, in the order every 6-vector in this app uses (mm, mm, mm, deg, deg, deg).



def build_and_run(cfg, robot, camera, args):
    spec = BncSpec.from_config(cfg)
    asm = _domain.BncAssembly(robot, camera, cfg, spec)
    if not bnc_skills.setup(asm):
        return False
    # ---- the run's tuning and rig, read ONCE -- the algorithm below stays plain -----------
    estimator, commit = asm.estimator, asm.commit
    scanner, geom, check = asm.scanner, asm.geom, asm.check
    recovery, grasp, recorder, confirm = asm.recovery, asm.grasp, asm.recorder, asm.confirm
    adm, guard_shared = asm.adm, asm.guard
    adm_final, guard_final = asm.adm_final, asm.guard_final
    frames_t, dense, mats = asm.frames, asm.dense, asm.mats
    out_dir, no_prompts, clock_rows = asm.out_dir, asm.no_prompts, asm.clock_rows
    phase = asm.pace.phase
    n_cycles, cc_on, cl_on = asm.n_cycles, asm.cc_on, asm.cl_on
    noise_rng, col_mode, sweep_offsets = asm.noise_rng, asm.col_mode, asm.sweep_offsets
    live_path = asm.live_path
    dbg, dbg_on, dbg_live = asm.debug_match
    _sfi, _es, _stn = spec.final_insertion, spec.engage, spec.trajectory_noise
    fi_on = bool(_sfi.enabled)
    fi_settle = None if _sfi.settle_s is None else float(_sfi.settle_s)
    fi_hold = (None if _sfi.hold_after_insertion_s is None
               else float(_sfi.hold_after_insertion_s))
    fi_v, fi_wr = _sfi.speed_translation_mm_s, _sfi.speed_rotation_deg_s
    fi_pause = float(_sfi.pause_s or 0.0)
    fi_noise_on = bool(_sfi.trajectory_noise.enabled)
    fi_noise_std = [float(v) for v in (_sfi.trajectory_noise.std or [0.0] * 6)]
    fi_noise_w = max(1, int(_sfi.trajectory_noise.smooth_window))
    ins_mode = str(spec.run.insertion_mode or 'estimate').strip().lower()
    tgt_source = str(spec.run.target_source or 'kinematic').strip().lower()
    pe_frame = str(spec.run.post_engage_frame or 'target').lower()
    max_engage_misses = max(0, int(_es.max_misses))
    en_travel_mm, en_timeout_s = float(_es.travel_mm), float(_es.timeout_s)
    en_fmax = float(_es.max_axial_force_n or 0.0)
    en_fpers = float(_es.persistence_s or 0.0)
    cc_tries = max(1, int(spec.connector_clocking.max_tries))
    tv_on = bool(spec.tug_verify.enabled)
    tv_force = float(spec.tug_verify.pull_force_n)
    tv_time = float(spec.tug_verify.pull_time_s)
    dis_on = bool(spec.disassembly.enabled)
    dis_place_on = bool(spec.disassembly.place.enabled)
    tn_on = bool(_stn.enabled)
    tn_std = [float(v) for v in (_stn.std or [0.0] * 6)]
    tn_w = max(1, int(_stn.smooth_window))
    tn_da, tn_dt = float(_stn.noise_decay_attempt), float(_stn.noise_decay_traj)
    tol_pos_m = float(spec.run.success_pos_mm) / 1000.0
    tol_rot_rad = np.radians(float(spec.run.success_rot_deg))
    max_attempts = int(spec.run.max_attempts)
    accumulate = bool(spec.run.accumulate_observations)
    cycle_ok = []

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
    T_ftip_conn_nominal = np.array(asm.T_ftip_conn, dtype=float)
    # A WHILE, NOT A FOR, so a missed engagement can start the cycle over without consuming one
    # of the production cycles the operator asked for. `engage_misses` is the budget across the
    # whole run -- a cell that keeps missing is a setup problem, and retrying it forever just
    # wears the part.
    cycle, engage_misses, repick_pending = 0, 0, False
    while cycle < n_cycles:
        cycle += 1
        engage_missed = False
        if n_cycles > 1:
            log.info('=' * 78)
            log.info('CYCLE %d/%d', cycle, n_cycles)
            log.info('=' * 78)
        if cycle > 1 or repick_pending:
            # A FRESH PICK HAS A FRESH IN-HAND ERROR. The estimator spent the last cycle
            # correcting the belief for the PREVIOUS grasp; carrying that correction into a
            # new grasp would start the next insertion from a confidently wrong pose.
            asm.T_ftip_conn = np.array(T_ftip_conn_nominal, dtype=float)
            # The cable was PLACED, so it is not where it was picked from and the cached
            # junction selection is stale -- the same reason the slip recovery re-prompts.
            if hasattr(scanner, 'reselect'):
                scanner.reselect()
            repick_pending = False
        if tgt_source == 'visual' and not bnc_skills.locate_target_visually(asm, q_home):
            return False
        # THE PICK POSE. Home is the marker VIEW pose (the sweep above, and the end-of-run
        # image); the scan, the grasp geometry and every retry offset are written from HERE.
        q_pick = asm.q_pick = q_home
        _pick_deg = cfg.get('pick_joints_deg')
        if _pick_deg is not None:
            q_pick = asm.q_pick = list(np.radians(np.asarray(_pick_deg, dtype=float)))
            # Gate BEFORE reconfiguring: this is a large joint move away from the marker view and
            # into the pick pose, and everything downstream (scan, grasp, retry offsets) is
            # written from where it lands.
            if not bnc_skills.phase_gate(asm, 'RECONFIGURE FOR PICKUP',
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
        runner = StepRunner(log, confirm=confirm is not None)
        if not bnc_skills.phase_gate(asm, 'PICK THE CABLE',
                          'Next the arm localizes the cable and grasps it.'):
            return False
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
                if not bnc_skills.reorient_recovery(asm):
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
        st = spec.trajectory.standoff or {}
        T_standoff_row = translation_matrix(
            np.asarray(st.get('axis', [-1, 0, 0]), dtype=float)
            * (float(st.get('distance_mm', 10.0)) / 1000.0)) @ mats[0]


        phase('standoff')
        asm.seed_q = robot.arm.q()
        T_tool0_conn = robot.T_tool0_fingertip @ asm.T_ftip_conn
        q = robot.arm.ik(bnc_skills.traj_ref(asm, T_standoff_row, T_tool0_conn), asm.seed_q)
        if q is None or not _guarded(robot, guard_shared,
                                     lambda: robot.arm.move_j(q, label='stand-off')):
            return False
        asm.seed_q = q
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
                # ENGAGE replaces the estimate loop and the commit.
                #
                # A FORCE STOP IS THE SUCCESS. Meeting the socket is what the phase is for, so
                # 'force' continues to the bayonet search. 'complete' means the whole path ran --
                # trajectory AND preload -- without ever reaching the axial limit, i.e. the
                # connector never touched anything. That is a MISS, not a completion, and clocking
                # from there would turn a connector that is not in a socket.
                #
                # UNLESS NO LIMIT IS SET: with max_axial_force_n at 0 the axial condition can never
                # fire, so 'complete' is the only outcome possible and treating it as failure would
                # fail every run. _engage_report already says the condition is dead; here it just
                # means the miss cannot be detected.
                en_status, _last_ref_e, en_depth = bnc_skills.engage_insertion(asm)
                # SUCCESS IS POSITIVE EVIDENCE ONLY: the part travelled, or it pushed back hard
                # enough for long enough. Everything else -- the path running out, the timeout,
                # a jam, a failed seat confirmation -- is a miss, and they all take the same
                # recovery: put the cable down and measure everything again.
                if en_status == 'aborted':
                    log.warning('ENGAGE aborted by the operator at the contact gate -- the arm is '
                                'LEFT WHERE IT IS with the cable held and touching the socket. '
                                'No recovery motion is attempted; that is what abort means.')
                    success = False
                elif en_status in ('complete', 'timeout', 'guard'):
                    log.error('ENGAGE FAILED (%s): neither %.1f mm of travel nor %.1f N held '
                              '%.2f s was reached, so the connector is not seated. Treating this '
                              'as a FAILED engagement.',
                              {'complete': 'the path ran out',
                               'timeout': 'timed out after %.0f s' % en_timeout_s,
                               'guard': 'the general force guard tripped -- a JAM'}[en_status],
                              en_travel_mm, en_fmax, en_fpers)
                    engage_missed = True
                    success = False
                elif en_status == 'unconfirmed':
                    # The axial limit WAS met, so something resisted -- but the confirmation says
                    # it does not behave like a socket. Same recovery as a clean miss: the cable
                    # goes down and everything is measured again. Continuing would clock a
                    # connector that is only leaning on something.
                    log.error('ENGAGE UNCONFIRMED: the connector met resistance but failed the '
                              'seat confirmation, so it is most likely against a face rather '
                              'than in the socket. Treating this as a FAILED engagement.')
                    engage_missed = True
                    success = False
                else:
                    success = en_status in ('force', 'travel')
                est_rows.append({'attempt': 'engage', 'status': en_status,
                                 'depth_mm': en_depth, 'success': bool(success)})
            for it in (range(1, max_attempts + 1) if ins_mode == 'estimate' else ()):
                T_tool0_conn = robot.T_tool0_fingertip @ asm.T_ftip_conn
                e_xyz, e_rpy = matrix_to_xyzrpy(asm.T_ftip_conn)
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
                    refs = [bnc_skills.traj_ref(asm, row, T_tool0_conn) for row in rows_t]
                    phase('standoff')
                    label = (f'attempt {it}'
                             + (f' sweep {pi + 1}/{len(passes)}' if poff is not None else '')
                             + ' start')
                    q = robot.arm.ik(refs[0], asm.seed_q)
                    if q is None or not _guarded(robot, guard_shared,
                                                 lambda: robot.arm.move_j(q, label=label)):
                        log.error('Could not reach the pass start; aborting.')
                        return False
                    asm.seed_q = q
                    phase('assemble')
                    # Intermediate passes MUST back off -- the next realigns to a different offset's
                    # start. The LAST pass stays at its stop, so a mate the operator calls successful
                    # leaves the arm AT the seat for clocking to anchor on.
                    obs_i, lin, ang, stops, last_ref = bnc_skills.run_insertion(
                        asm, adm, refs, T_tool0_conn, peck=(col_mode == 'peck'),
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
                bnc_skills.retract_from(asm, last_ref, T_tool0_conn)
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

                asm.T_ftip_conn = asm.T_ftip_conn @ _corr_to_m(T_corr_mm)      # believed @ corr ~= true
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
                T_tool0_conn = robot.T_tool0_fingertip @ asm.T_ftip_conn
                rows_f = (traj.noised(dense, noise_rng, fi_noise_std, fi_noise_w, 0.0, 1.0)
                          if fi_noise_on else dense)
                refs = [bnc_skills.traj_ref(asm, row_, T_tool0_conn, commit=True) for row_ in rows_f]
                phase('standoff')
                q = robot.arm.ik(refs[0], asm.seed_q)
                if q is None or not _guarded(robot, guard_shared,
                                             lambda: robot.arm.move_j(q, label='final start')):
                    log.warning('IK/approach failed for the final insertion.')
                else:
                    asm.seed_q = q
                    phase('assemble')
                    # retract=False: this is the attempt meant to SEAT. Backing out of it would undo
                    # the mate before the operator can judge it and would leave the clocking maneuvers
                    # anchored on a retracted pose.
                    obs_f, lin, ang, _stops, last_ref_f = bnc_skills.run_insertion(
                        asm, adm_final, refs, T_tool0_conn, guard_ctl=guard_final,
                        settle=fi_settle,
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
                        bnc_skills.retract_from(asm, last_ref_f, T_tool0_conn, adm_final)
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
            # A MISSED ENGAGEMENT IS RECOVERABLE; anything else is not. Put the cable down, go
            # back to the pick view, and run the cycle again from there -- re-localizing the
            # target and re-grasping, so every input to the engagement is measured afresh.
            if engage_missed and engage_misses < max_engage_misses:
                engage_misses += 1
                log.warning('ENGAGE MISS %d/%d -- placing the cable and starting this cycle over '
                            'from the pick view.', engage_misses, max_engage_misses)
                if not bnc_skills.place_after_failed_engage(asm, _last_ref_e,
                                                 robot.T_tool0_fingertip @ asm.T_ftip_conn):
                    log.error('Could not put the cable down after the missed engagement -- the '
                              'arm is LEFT WHERE IT IS and the part may still be held.')
                    return False
                cycle -= 1                      # this attempt does not count as a cycle
                repick_pending = True
                continue
            if engage_missed:
                log.error('ENGAGE missed %d times (budget %d) -- stopping. The target pose or the '
                          'in-hand belief is wrong; re-running will not fix it.',
                          engage_misses, max_engage_misses)
            return False

        # ---- POST-MATE: CONNECTOR CLOCKING, then COLLAR CLOCKING, then the shared escape -------------
        # Both paths end in the same retract. The mate itself has already succeeded by here, so a
        # clocking failure is reported without undoing it.
        if cc_on:
            cc_ok = ret_ok = False
            tug_res = None
            state = 'engaged'                    # the initial assembly mated it; that is where we are
            # ALWAYS REPORTED, whichever frame the maneuvers then use. At the mate the part is
            # physically IN the socket, so the believed connector and the target frame describe
            # the same thing -- and any gap between them is in-hand belief error, measured
            # exactly where it matters. Decomposed in the CONNECTOR's own axes because that is
            # where the fix lives: along-axis is insertion depth or belief_offset x, lateral and
            # vertical are the grasp, and roll is the clock angle.
            _believed = robot.tool0() @ T_tool0_conn
            _d = inverse(frames_t.T_base_tconn) @ _believed
            _dx, _dr = matrix_to_xyzrpy(_d)
            log.info('BELIEF vs TARGET at the mate (connector frame): along-axis %+.2f mm, '
                     'lateral %+.2f mm, vertical %+.2f mm | roll %+.2f, pitch %+.2f, yaw %+.2f '
                     'deg. Zero means the belief agreed with the socket; anything else is the '
                     'in-hand error the insertion had to absorb.',
                     _dx[0] * 1000.0, _dx[1] * 1000.0, _dx[2] * 1000.0,
                     *np.degrees(_dr))
            if pe_frame == 'believed':
                # The believed connector frozen in base coordinates NOW, while the arm still grips
                # it -- the maneuvers need one consistent frame, not one re-derived per use.
                frames_t.T_clk = _believed
                log.info('Post-engage frame: BELIEVED connector.')
            else:
                frames_t.T_clk = frames_t.T_base_tconn
                log.info('Post-engage frame: TARGET connector (recorded socket pose).')
            if not bnc_skills.phase_gate(asm, 'CONNECTOR CLOCKING (insert)',
                              'The connector is ENGAGED. Next is the bayonet screw, which cams it '
                              'HOME -- check the engagement looks right first.'):
                robot.arm.servo_stop()
                return False
            try:
                cc_ok, _T_tool0_conn, T_base_conn, cc_screw_deg = \
                    bnc_skills.connector_clocking(asm)
                asm.T_tool0_conn_engaged = _T_tool0_conn
                if cc_ok:
                    state = _advance_state(state, 'engaged')          # -> seated
                    if cl_on and not bnc_skills.phase_gate(
                            asm,
                            'COLLAR CLOCKING (lock)',
                            'The connector is SEATED. Next is the collar turn, which LOCKS it -- the '
                            'gripper will re-grasp the collar and rotate. Check the seat first.'):
                        robot.arm.servo_stop()
                        return False
                    if cl_on and bnc_skills.collar_clocking(asm, T_base_conn, cc_screw_deg):
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
                    if bnc_skills.phase_gate(asm, 'TUG VERIFY',
                                  'The collar is LOCKED and still HELD. Next: pull %.1f N along the '
                                  'connector -X for %.1f s without letting go -- a locked bayonet '
                                  'holds, an unlocked one backs out.' % (tv_force, tv_time)):
                        tug_res = bnc_skills.tug_verify_in_place(asm)
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
                elif bnc_skills.phase_gate(asm, 'ESCAPE',
                                'Clocking done. Next: RELEASE the gripper, then the two-leg retract '
                                '(the gripper backs off its own -Z, then away along the target -X).'):
                    # RELEASE FIRST: the fingers are still CLOSED on the collar, and the retract's
                    # first leg is written for OPEN fingers. Idempotent where it is already open.
                    if robot.gripper.open('release before escape'):
                        ret_ok = bnc_skills.clocking_retract(asm)
                    else:
                        log.error('Gripper did not open before the escape -- leaving the arm in '
                                  'place rather than dragging the locked assembly with clamped '
                                  'fingers.')
                else:
                    log.warning('Escape skipped by the user -- the arm is still at the connector '
                                'with the gripper in whatever state clocking left it.')

                # ---- CELEBRATE, between a verified assembly and taking it apart --------------
                # Here and not earlier: the escape has released the connector and backed the arm
                # off, so the flourish cannot drag or disturb what it is celebrating. Every
                # condition -- success, end state, and how often to do it -- is decided inside
                # celebrate(). Never fatal: a failed flourish must not fail a good assembly.
                try:
                    bnc_skills.celebrate(asm, state, cycle, tug_res, ret_ok)
                except Exception as exc:                       # noqa: BLE001 -- cosmetic only
                    log.warning('CELEBRATE raised (%s) -- ignored; the assembly stands.', exc)

                # ---- DISASSEMBLY, after the assembly has fully let go ------------------------
                # It runs HERE, once the escape has released the connector and backed the arm off,
                # so the two halves are cleanly separated: the assembly ends with the connector
                # mated and the arm clear, exactly as a run without disassembly would leave it, and
                # the disassembly starts by re-approaching and re-gripping like any other maneuver.
                # Nothing it does depends on a grip inherited from the assembly.
                if dis_on and ret_ok and tug_res != 'failed' and state in ('locked', 'seated'):
                    if not bnc_skills.phase_gate(asm, 'DISASSEMBLE',
                                      'The assembly is complete and the arm is clear. Next it '
                                      're-approaches, re-grips and takes it apart.'):
                        return False
                    dis_ok, state = bnc_skills.disassembly(asm, state, cc_screw_deg,
                                                          T_base_conn)
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
            rst = bnc_skills.end_reset_with_snapshot(asm)                       # always, even after a failed screw
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

    d_out = float(spec.run.release_retract_distance_mm) / 1000.0
    back = -(robot.tool0() @ (robot.T_tool0_fingertip @ asm.T_ftip_conn))[:3, 0] * d_out

    def release_escape():
        return _guarded(robot, guard_shared,
                        lambda: robot.arm.move_l(translation_matrix(back) @ robot.tool0(),
                                                 label='retract (connector -X)'))

    phase('retract')
    ok = runner.run([('open gripper (release)', robot.gripper.open),
                     ('retract (connector -X)', release_escape)])
    phase('reset')
    return ok and bnc_skills.end_reset_with_snapshot(asm)


def main():
    run_app('BNC assembly: pick + estimator_eval-tuned manifold estimation',
            'bnc_assembly', build_and_run, needs_camera=True)


if __name__ == '__main__':
    main()
