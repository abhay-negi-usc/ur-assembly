"""TILE ASSEMBLY EXPERIMENT -- assemble one tile onto the fixture over and over, recording what
compliance absorbs. The operator loads the tile into the coupler and resets the fixture by hand.

    python -m urlab.apps.tile_assembly_experiment
    python -m urlab.apps.tile_grasp_plot data/experiments/tile_assembly_experiment_<stamp>

Every trial is apps/tile_assembly's cycle with the autonomous PICK replaced by a manual LOAD:

    fixed view -> LOCATE the fixture (every trial, arm empty -- the held tile would occlude it)
    -> load pose -> open the coupler -> operator places the tile, Enter -> LOCK (verified)
    -> payload -> operator lets go, Enter -> PREDRIVE the bolt -> TARE (tile hanging free)
    -> assembly joints -> approach path -> INSERT (compliant, preload) -> SEATED pose recorded
    -> FASTEN (compliant) -> release -> payload back -> WITHDRAW (compliant)
    -> INSPECT: from the inspection pose, sweep the assembly from several angles and estimate the
       tile against the hub -> home
    -> operator: was the assembly successful? (a label on the trial)
    -> operator unbolts and removes the tile, Enter -> next trial

WHAT IS RECORDED: every servo cycle of the insertion (with its preload), the fasten hold and the
withdraw -- measured / reference / commanded coupler pose, the admittance yield, and the wrench at
the mating point in coupler axes (see apps/tile_grasp_experiment, whose recorder this reuses).
Poses are also stored RELATIVE TO THE SEATED POSE -- the coupler pose measured once the insertion
preload holds -- so every trial converges to zero there, and the reference shows where the camera
put the seat. trials.csv carries the camera-located goal's error against that seat. Only trials
you label successful are plotted by default.

THE INSPECTION measures the ASSEMBLY ERROR by sight, the way urlab.apps.marker_assembly_
calibration measured the assembled state in the first place: one sweep (experiment.inspection
.views, several viewpoints re-aimed at the markers) over BOTH parts' markers, per-marker fusion,
multi-view refinement, then for each part the marker vote + joint PnP:

    T_base_goal = the hub's markers through the calibration -- where the tile's mating frame
                  SHOULD be (the calibrated assembled state; goal_offset is not applied)
    T_base_tile = the tile's markers through objects.yaml -- where its mating frame IS
    error       = T_goal^-1 @ T_tile, as x y z mm / roll pitch yaw deg in the goal frame

Both come from the same images, so the robot's kinematics and the hand-eye cancel: zero error
means the tile sits on the hub exactly as it did when the assembly was calibrated. Each trial's
error, the per-marker poses and the images go into its .npz / trials.csv / inspection_images/, and
assembly_error_summary.csv gives the mean, sigma, min and max over the labelled-successful trials.
`tile_vs_seated` compares the camera's tile with the coupler's seated pose (kinematic): what the
fasten and the release moved it by -- plus any hand-eye error.

TARING: once per trial, after the operator lets go and the bolt is predriven -- the tile hanging
free, nothing touching. The insertion starts from the approach path's end with nothing touching,
and tares there again (the cycle's own); the fasten hold and the withdraw start in contact and do
not.

THE COUPLER DOES NOT CONSTRAIN YAW about its own axis: a pick commands the calibrated yaw, a hand
load does not. Seat the tile at a consistent, marked angle, or that scatter is in the data.
"""

import numpy as np

from .. import behaviors as bt
from .. import log as urlog
from .. import tool_frames
from ..robot.coupler import Coupler
from ..skills import marker_localize as mloc
from ..transforms import inverse, matrix_to_xyzrpy, pose_error
from ._common import ask, experiment_dir, prompts_off
from ._runner import run_app
from .coupler_pick_place import object_rig, step_gate
from .tile_assembly import TileAssemblyCycle, marker_to_marker, resolve_marker_assembly
from .tile_grasp_experiment import TrialRecorder, apply_speed_scale, relative_pose6

log = urlog.get('tile-asm-exp')

PHASE_ORDER = ('insert', 'fasten', 'withdraw')
ERROR_COMPONENTS = ('dx_mm', 'dy_mm', 'dz_mm', 'droll_deg', 'dpitch_deg', 'dyaw_deg')
_HEADER = ['trial', 'success', 'outcome', 'samples', 'goal_err_mm', 'goal_err_deg',
           'goal_dx_mm', 'goal_dy_mm', 'goal_dz_mm', 'insert_peak_n', 'fasten_peak_n',
           'withdraw_peak_n', 'assy_err_mm', 'assy_err_deg'] + \
    [f'assy_{c}' for c in ERROR_COMPONENTS] + \
    ['tile_vs_seated_mm', 'tile_vs_seated_deg', 'inspect_hub_markers', 'inspect_tile_markers']


def assembly_phase(what):
    """The cycle's leg label -> the phase name the data and plots use. Pure."""
    if what.startswith('fasten'):
        return 'fasten'
    return {'assemble the object': 'insert', 'withdraw from the object': 'withdraw'}.get(what, what)


def assembly_error(T_goal, T_tile):
    """(6,) [x y z mm, roll pitch yaw deg]: the tile's mating frame in the goal frame -- zero when
    it sits on the hub exactly as calibrated. NaN when either is missing. Pure."""
    if T_goal is None or T_tile is None:
        return np.full(6, np.nan)
    return relative_pose6(T_goal, [T_tile])[0]


def assembly_summary(trial, success, outcome, a, poses, rec=None):
    """One trials.csv row: the camera-located goal's error against the seat (what the insertion
    absorbed), the peak force of each phase, and the INSPECTED assembly error. Pure."""
    row = {'trial': trial, 'success': int(bool(success)), 'outcome': outcome,
           'samples': len(a['t'])}
    T_goal, T_seat = poses.get('goal'), poses.get('seated')
    if T_goal is not None and T_seat is not None:
        lin, ang = pose_error(T_seat, T_goal)
        d = (inverse(T_seat) @ T_goal)[:3, 3] * 1000.0
        row.update(goal_err_mm=round(lin * 1000.0, 3), goal_err_deg=round(np.degrees(ang), 3),
                   goal_dx_mm=round(d[0], 3), goal_dy_mm=round(d[1], 3), goal_dz_mm=round(d[2], 3))
    for phase in PHASE_ORDER:
        sel = a['phase'] == phase
        if np.any(sel):
            row[f'{phase}_peak_n'] = round(float(np.max(np.linalg.norm(a['wrench'][sel, :3],
                                                                      axis=1))), 3)
    T_ig, T_it = poses.get('inspect_goal'), poses.get('inspect_tile')
    if T_ig is not None and T_it is not None:
        lin, ang = pose_error(T_ig, T_it)
        row.update(assy_err_mm=round(lin * 1000.0, 3), assy_err_deg=round(np.degrees(ang), 3))
        for c, v in zip(ERROR_COMPONENTS, assembly_error(T_ig, T_it)):
            row[f'assy_{c}'] = round(float(v), 3)
    if T_it is not None and T_seat is not None:
        lin, ang = pose_error(T_seat, T_it)
        row.update(tile_vs_seated_mm=round(lin * 1000.0, 3),
                   tile_vs_seated_deg=round(np.degrees(ang), 3))
    extras = getattr(rec, 'extras', {}) if rec is not None else {}
    for key, col in (('inspect_hub_ids', 'inspect_hub_markers'),
                     ('inspect_tile_ids', 'inspect_tile_markers')):
        if key in extras:
            row[col] = ' '.join(str(int(m)) for m in extras[key])
    return row


def inspect_assembly(robot, camera, detector, plan, fixed_rig, held_rig, images=None):
    """Sweep the assembled pair and estimate it -- the calibration's own measurement:
    {'T_goal', 'T_tile', 'hub': {id: T_base_marker}, 'tile': {...}, 'views'}.

    One sweep over BOTH parts' markers (and the servo, if the plan has it), per-marker fusion,
    multi-view refinement, then the vote + joint PnP of each part's rig on the shared corners."""
    corner_views = []
    wanted = set(fixed_rig['markers']) | set(held_rig['markers'])
    T_overview = robot.camera()
    seen = mloc.sweep(robot, camera, detector, plan, wanted=wanted,
                      on_view=images.sweep_view if images else None, corner_log=corner_views)
    if plan.servo.enabled and seen:
        mloc.merge_refined(seen, mloc.servo_refine(
            robot, camera, detector, plan, seen, T_overview=T_overview,
            on_view=images.servo_view if images else None,
            on_iteration=images.servo_iteration_view if images else None,
            corner_log=corner_views))
    fused = mloc.fuse_markers(seen, plan) if seen else {}
    if fused and getattr(plan, 'multiview_refine', True):
        sizes = dict(tool_frames.marker_sizes(fixed_rig))
        sizes.update(tool_frames.marker_sizes(held_rig))
        for mid, (T_ref, _rms, _n) in mloc.refine_markers_multiview(
                fused, corner_views, sizes, plan).items():
            fused[mid] = (T_ref,) + tuple(fused[mid][1:])
    hub = {m: f for m, f in fused.items() if m in fixed_rig['markers']}
    tile = {m: f for m, f in fused.items() if m in held_rig['markers']}
    T_goal = mloc.estimate_target(fixed_rig, hub, corner_views, plan) if hub else None
    T_tile = mloc.estimate_target(held_rig, tile, corner_views, plan) if tile else None
    return {'T_goal': T_goal, 'T_tile': T_tile, 'views': len(corner_views),
            'hub': {m: f[0] for m, f in hub.items()}, 'tile': {m: f[0] for m, f in tile.items()}}


def inspection_settings(cfg, masm_view_joints=None):
    """`experiment.inspection` -> {enabled, joints (rad) or None, via [q (rad)], save_images,
    plan}. The pose defaults to the calibration's view (both parts were in frame there), then
    fixed_view_joints; the views to `inspection.views`, else marker_views. Pure."""
    i = dict((cfg.section('experiment') or {}).get('inspection') or {})
    q = i.get('joints_deg')
    if q is not None:
        q = np.radians(np.asarray(q, dtype=float))
    elif masm_view_joints is not None:
        q = np.asarray(masm_view_joints, dtype=float)
    elif cfg.get('fixed_view_joints_deg') is not None:
        q = np.radians(np.asarray(cfg['fixed_view_joints_deg'], dtype=float))
    if q is not None and np.asarray(q).shape != (6,):
        raise ValueError('experiment.inspection.joints_deg must be six angles')
    via = np.asarray(i.get('via_joints') or np.zeros((0, 6)), dtype=float)
    if via.ndim == 1:
        via = via[None, :]
    if via.ndim != 2 or via.shape[1] != 6:
        raise ValueError('experiment.inspection.via_joints must be a list of six-angle poses')
    return {'enabled': bool(i.get('enabled', True)), 'joints': q,
            'via': [np.radians(v) for v in via],
            'save_images': bool(i.get('save_images', True)),
            'plan': mloc.ViewPlan(i.get('views') or cfg.section('marker_views'))}


def assembly_error_stats(run_dir, include_failed=False):
    """{component: (mean, std, min, max, n)} of the inspected assembly error over the run's
    trials (labelled-successful ones unless include_failed), NaNs skipped."""
    import glob
    import json
    errs = []
    for path in sorted(glob.glob(f'{run_dir}/attempt_*.npz')):
        with np.load(path) as z:
            if 'assy_err6' not in z.files:
                continue
            meta = json.loads(str(z['meta']))
            if meta.get('success') or include_failed:
                errs.append(np.asarray(z['assy_err6'], dtype=float))
    out = {}
    E = np.array(errs).reshape(-1, 6)
    for k, c in enumerate(ERROR_COMPONENTS):
        col = E[:, k][np.isfinite(E[:, k])]
        out[c] = ((float(col.mean()), float(col.std(ddof=1)) if len(col) > 1 else float('nan'),
                   float(col.min()), float(col.max()), len(col)) if len(col) else
                  (float('nan'),) * 4 + (0,))
    return out


def write_error_summary(run_dir, stats):
    """assembly_error_summary.csv: one row per component."""
    import csv
    path = f'{run_dir}/assembly_error_summary.csv'
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['component', 'mean', 'std', 'min', 'max', 'n'])
        for c, (mean, std, lo, hi, n) in stats.items():
            w.writerow([c, round(mean, 4), round(std, 4), round(lo, 4), round(hi, 4), n])
    return path


def experiment_settings(cfg):
    """`experiment:` -> {trials, speed_scale, load_joints (rad) or None}. Pure."""
    e = dict(cfg.section('experiment') or {})
    trials = int(e.get('trials', 10))
    if trials < 1:
        raise ValueError('experiment.trials must be at least 1')
    speed_scale = float(e.get('speed_scale') or 1.0)
    if not speed_scale > 0.0:
        raise ValueError(f'experiment.speed_scale must be positive, got {speed_scale}')
    q = e.get('load_joints_deg')
    if q is None:
        q = cfg.get('pick_view_joints_deg')
    load = None
    if q is not None:
        load = np.radians(np.asarray(q, dtype=float))
        if load.shape != (6,):
            raise ValueError(f'experiment.load_joints_deg must be six angles, got {q!r}')
    return {'trials': trials, 'speed_scale': speed_scale, 'load_joints': load,
            'plot_at_end': bool(e.get('plot_at_end', True))}


def ask_label(cfg, robot):
    """'y' / 'n' = the assembly was / was not successful, 'q' = stop. Unattended = 'y'."""
    if robot.arm.dry_run or prompts_off(cfg):
        return 'y'
    while True:
        try:
            ans = input('  Was the ASSEMBLY successful? [y / n / q = stop the experiment]: ')
        except EOFError:
            return 'y'
        ans = ans.strip().lower()
        if ans in ('y', 'yes', 'n', 'no', 'q', 'quit'):
            return ans[0]


class AssemblyTrialCycle(TileAssemblyCycle):
    """tile_assembly's cycle, recording, with a manual load in place of the pick."""

    RUN_NAME = 'tile_assembly_experiment'

    def load_tile(self):
        """Open the coupler, let the operator seat the tile, lock it and check -- and again on a
        refused lock, until it locks or the operator stops. Then the payload, and wait for the
        operator to let go. Unattended: lock straight away."""
        unattended = self.robot.arm.dry_run or prompts_off(self.cfg)

        def stop():
            self.recorder.event('stopped')
            return False
        while True:
            if not self.prepare_coupler():
                return False
            if not unattended and not ask('  Place the TILE in the coupler (at its marked yaw) and '
                                          'hold it -- Enter LOCKS it (q to stop): '):
                return stop()
            if self.lock():
                break
            if unattended or not ask('  The lock was refused. Enter to open and try again (q to '
                                     'stop): '):
                return stop()
        self.recorder.event('loaded (locked)')
        if not self.take_payload():
            return False
        if not unattended and not ask('  Locked. LET GO of the tile and stand clear -- Enter to '
                                      'start the assembly (q to stop): '):
            return stop()
        return True

    def tare_free(self):
        """Tare with the tile hanging free (settle_after_lift's tare), and note it."""
        ok = self.settle_after_lift()
        self.recorder.event('tare (tile hanging free)',
                            wrench=np.asarray(self.robot.arm.wrench(), float).tolist())
        return ok

    def locate_goal(self):
        """Locate the fixture (the cycle's own fixed-view trip) and take its goal as the target --
        there is no pick, so nothing else sets T_place."""
        if not self.locate_fixed():
            return False
        self.T_place = self.T_goal
        self.recorder.pose('goal', self.T_goal)
        return True

    def record_seated(self):
        """The SEATED pose -- the coupler pose now, with the insertion preload holding -- is the
        zero of the recorded poses."""
        self.recorder.pose('seated', self.robot.arm.tcp_pose() @ self.T_tool0_coupler)
        self.recorder.event('seated (preload held)')
        return True

    def record_released(self):
        self.recorder.pose('released', self.robot.arm.tcp_pose() @ self.T_tool0_coupler)
        return True

    # Set by build_and_run: inspection_settings(), the tile's rig and the detector for both parts.
    inspection = None
    held_rig = None
    inspect_detector = None
    inspect_images = None

    def inspect(self):
        """Drive to the inspection pose and measure the assembly by sight. A missing estimate is
        recorded (NaN) and warned about, never fatal: the trial is still labelled and saved. Only a
        move that does not finish fails the step."""
        ins = self.inspection
        if not ins or not ins['enabled']:
            return True
        if ins['joints'] is not None:
            moves = [(f'inspection via {k}/{len(ins["via"])}', q)
                     for k, q in enumerate(ins['via'], start=1)]
            for label, q in moves + [('inspection pose', ins['joints'])]:
                if not self.robot.move_joints(q, label=label, guard=self.guard,
                                              caps=self._joint_caps()):
                    log.error('Could not reach the %s.', label)
                    return False
        self.recorder.event('inspection start')
        res = inspect_assembly(self.robot, self.camera, self.inspect_detector, ins['plan'],
                               self.fixed_rig, self.held_rig, self.inspect_images)
        rec = self.recorder
        rec.pose('inspect_goal', res['T_goal'])
        rec.pose('inspect_tile', res['T_tile'])
        err = assembly_error(res['T_goal'], res['T_tile'])
        rec.extras.update(
            assy_err6=err,
            tile_vs_seated6=(relative_pose6(rec.poses['seated'], [res['T_tile']])[0]
                             if res['T_tile'] is not None and rec.poses.get('seated') is not None
                             else np.full(6, np.nan)),
            inspect_hub_ids=np.array(sorted(res['hub']), dtype=int),
            inspect_hub_T=np.array([res['hub'][m] for m in sorted(res['hub'])]).reshape(-1, 4, 4),
            inspect_tile_ids=np.array(sorted(res['tile']), dtype=int),
            inspect_tile_T=np.array([res['tile'][m] for m in sorted(res['tile'])]
                                    ).reshape(-1, 4, 4),
            inspect_views=res['views'])
        if self.inspect_images is not None:
            self.inspect_images.finish(['ASSEMBLY ERROR (tile in goal frame) mm / deg: '
                                        + ' '.join(f'{v:+.2f}' for v in err)])
        if np.all(np.isfinite(err)):
            log.info('ASSEMBLY ERROR (tile in the calibrated goal frame): xyz %s mm, rpy %s deg '
                     '(%d hub / %d tile marker(s), %d view(s)).', np.round(err[:3], 2).tolist(),
                     np.round(err[3:], 2).tolist(), len(res['hub']), len(res['tile']),
                     res['views'])
        else:
            log.warning('The assembly could not be measured -- hub markers seen %s, tile markers '
                        'seen %s. The trial is saved without it.', sorted(res['hub']),
                        sorted(res['tile']))
        rec.event('inspection done')
        return True


def run_trial(cfg, robot, job, rec, trial, q_load, q_home, step):
    """One trial. Returns 'done' (assembled, released, home), 'locate_failed', 'stopped' or
    'error'."""
    rec.start(trial)
    if not bt.run_tree(bt.sequence(f'trial {trial}: locate',
                                   bt.Action('locate the fixture', job.locate_goal)), log):
        return 'locate_failed'
    steps = []
    if q_load is not None:
        steps.append(bt.Action('to the load pose',
                               lambda: robot.move_joints(q_load, label='load pose',
                                                         guard=job.guard)))
    steps.append(bt.Action('load the tile', job.load_tile))
    steps += list(job.steps_after_lift(step))           # the predrive
    steps.append(bt.Action('tare with the tile hanging free', job.tare_free))
    steps += [bt.Action('carry to the assembly standoff', job.carry, confirm=step),
              bt.Action('insert the tile', job.set_down, confirm=step),
              bt.Action('record the seated pose', job.record_seated)]
    steps += list(job.steps_after_insertion(step))      # the fasten
    if job.secures_target:
        steps.append(bt.Action(job.SECURE_LABEL, job.secure_target, confirm=step))
    steps += [bt.Action('release the coupler', job.unlock, confirm=step),
              bt.Action('record the released pose', job.record_released),
              bt.Action('payload back to the tool alone', job.drop_payload),
              bt.Action('withdraw', job.withdraw, confirm=step),
              bt.Action('inspect the assembly', job.inspect),
              bt.Action('home', lambda: robot.move_joints(q_home, label='home',
                                                          guard=job.guard))]
    if not bt.run_tree(bt.sequence(f'trial {trial}: assemble', *steps), log):
        return 'stopped' if rec.events and rec.events[-1]['name'] == 'stopped' else 'error'
    return 'done'


def build_and_run(cfg, robot, camera, args):
    try:
        settings = experiment_settings(cfg)
        masm_name, entry = resolve_marker_assembly(cfg)
        objects = tool_frames.load_objects(cfg)
        plan = mloc.ViewPlan(cfg.section('marker_views'))
    except (ValueError, KeyError) as exc:
        log.error('%s', exc.args[0] if exc.args else exc)
        return False
    held = entry['held_object']
    if cfg.get('object_name') and cfg['object_name'] != held:
        log.error('%r was calibrated for %r, not object_name=%r.', masm_name, held,
                  cfg['object_name'])
        return False
    cfg['object_name'] = held
    if held not in objects:
        log.error('%r assembles %r, which is not in %s.', masm_name, held,
                  tool_frames.objects_path(cfg))
        return False
    obj = objects[held]
    for fid in sorted(entry['markers']):
        for hid in sorted(obj['markers']):
            xyz, rpy = matrix_to_xyzrpy(marker_to_marker(entry['markers'][fid]['T_marker_goal'],
                                                         obj['markers'][hid]['T_marker_grasp']))
            log.info('  marker %d -> %d: xyz %s mm  rpy %s deg', fid, hid,
                     np.round(xyz * 1000.0, 2).tolist(), np.round(np.degrees(rpy), 2).tolist())

    from ..perception import ArucoDetector
    detector = ArucoDetector(cfg, sizes_m={int(m): v['size_m'] for m, v in obj['markers'].items()})
    coupler = Coupler(cfg)
    try:
        job = AssemblyTrialCycle(cfg, robot, camera, detector, plan, held, obj, coupler)
    except (ValueError, KeyError) as exc:
        log.error('%s', exc)
        coupler.close()
        return False
    run_dir = job.out_dir or experiment_dir(cfg, AssemblyTrialCycle.RUN_NAME)
    job.out_dir = run_dir
    rec = TrialRecorder(robot, job.T_tool0_coupler, phase_of=assembly_phase, zero='seated',
                        zero_label='seated',
                        pose_names=('goal', 'seated', 'released', 'inspect_goal', 'inspect_tile'),
                        summarize=assembly_summary, header=_HEADER)
    try:
        job.inspection = inspection_settings(cfg, entry['view_joints'])
    except ValueError as exc:
        log.error('%s', exc)
        job.teardown()
        coupler.close()
        return False
    job.held_rig = object_rig(obj, cfg.get_path('aruco.dictionary'))
    sizes = dict(tool_frames.marker_sizes(job.fixed_rig))
    sizes.update(tool_frames.marker_sizes(job.held_rig))
    job.inspect_detector = ArucoDetector(cfg, sizes_m=sizes)
    job.recorder, job.on_servo_step = rec, rec.on_servo_step
    q_home = np.asarray(robot.arm.q(), dtype=float)
    q_load = settings['load_joints']
    step = step_gate(cfg, robot)
    meta = {'marker_assembly': masm_name, 'object': held,
            'start_joints_deg': np.degrees(q_home).tolist(),
            'load_joints_deg': None if q_load is None else np.degrees(q_load).tolist(),
            'speed_scale': settings['speed_scale'],
            'compliant_speed_mm_s': cfg.get('compliant_speed_mm_s'),
            'goal_offset': cfg.section('goal_offset'),
            'approach_path': cfg.get('approach_path'),
            'compliance': {k: cfg.section(k) for k in ('compliance', 'compliance_loaded',
                                                       'compliance_insert') if cfg.section(k)},
            'assembly_preload': cfg.section('assembly_preload'),
            'fastening': cfg.section('fastening'),
            'inspection': {'joints_deg': (None if job.inspection['joints'] is None
                                          else np.degrees(job.inspection['joints']).tolist()),
                           'views': job.inspection['plan'].describe(),
                           'assy_err6': 'tile mating frame in the calibrated goal frame '
                                        '[mm, deg]; zero = assembled exactly as calibrated'},
            'tare': {'insert': 'once per trial with the tile hanging free (after the load and '
                               'predrive), and again at the insertion standoff, nothing touching',
                     'fasten': 'none -- in contact', 'withdraw': 'none -- in contact'},
            'frames': {'pose_rel': 'coupler pose in the SEATED pose frame [mm, deg]',
                       'wrench': 'at the mating point, coupler axes [N, Nm]',
                       'delta_mm_deg': 'admittance yield, tool0 = coupler axes [mm, deg]'}}
    log.info('TILE ASSEMBLY EXPERIMENT: %d trial(s) of %r onto %r, loaded by hand at %s. Data -> '
             '%s', settings['trials'], held, masm_name,
             'the start pose' if q_load is None else 'the load pose', run_dir)

    done = ok_count = 0
    ok_run = True
    previous_scale = apply_speed_scale(robot, settings['speed_scale'])
    try:
        while done < settings['trials']:
            trial = done + 1
            log.info('==== TRIAL %d of %d ====', trial, settings['trials'])
            if job.fixed_images is not None:
                # One image folder per trial: the writer names views by index.
                job.fixed_images = mloc.MarkerImageWriter(
                    run_dir, job.fixed_detector, subdir=f'fixed_marker_images/trial_{trial:02d}')
            job.inspect_images = (mloc.MarkerImageWriter(
                run_dir, job.inspect_detector, subdir=f'inspection_images/trial_{trial:02d}')
                if job.inspection['enabled'] and job.inspection['save_images'] else None)
            outcome = run_trial(cfg, robot, job, rec, trial, q_load, q_home, step)
            robot.arm.servo_stop()
            if outcome == 'locate_failed':
                log.error('Trial %d: the fixture was not located; nothing has moved toward it.',
                          trial)
                if (robot.arm.dry_run or prompts_off(cfg)
                        or not ask('  Enter to retry (q to stop): ')):
                    ok_run = False
                    break
                continue
            if outcome != 'done':
                rec.save(run_dir, False, outcome, meta)
                ok_run = outcome == 'stopped'
                (log.info if ok_run else log.error)(
                    'Trial %d ended with %r -- stopping the experiment. Whatever the coupler holds '
                    'is still held.', trial, outcome)
                break
            label = ask_label(cfg, robot)
            verdict = {'y': 'ok', 'n': 'FAILED'}.get(label, 'stop')
            rec.event(f'operator: assembly {verdict}')
            rec.save(run_dir, label == 'y', 'success' if label == 'y' else 'failed', meta)
            done += 1
            ok_count += label == 'y'
            if label == 'q' or done >= settings['trials']:
                break
            if (not robot.arm.dry_run and not prompts_off(cfg)
                    and not ask('  UNBOLT and remove the tile from the fixture -- Enter for the '
                                'next trial (q to stop): ')):
                break
        log.info('EXPERIMENT DONE: %d trial(s), %d labelled successful. Data: %s', done, ok_count,
                 run_dir)
        if done and job.inspection['enabled']:
            stats = assembly_error_stats(run_dir)
            path = write_error_summary(run_dir, stats)
            log.info('ASSEMBLY ERROR over the labelled-successful trials (%s):', path)
            for c, (mean, std, lo, hi, n) in stats.items():
                log.info('  %-11s mean %+8.3f  sigma %7.3f  [%+.3f, %+.3f]  n=%d', c, mean, std,
                         lo, hi, n)
        if ok_run and done:
            job.park_coupler()
        if settings['plot_at_end'] and ok_count:
            try:
                from .tile_grasp_plot import plot_run
                for path in plot_run(run_dir):
                    log.info('Plot: %s', path)
            except Exception as exc:                # noqa: BLE001 -- plots never fail a run
                log.warning('Could not plot the run (%s). Plot it later with: python -m '
                            'urlab.apps.tile_grasp_plot %s', exc, run_dir)
        return ok_run and done >= settings['trials']
    finally:
        robot.arm.servo_stop()
        robot.arm.set_speed_scale(previous_scale, 'restored')
        job.teardown()
        coupler.close()


def main():
    run_app('Tile assembly experiment: hand-loaded tile, repeated assembly, recording pose, wrench '
            'and compliance', 'tile_assembly_experiment', build_and_run, with_gripper=False,
            needs_camera=True)


if __name__ == '__main__':
    main()
