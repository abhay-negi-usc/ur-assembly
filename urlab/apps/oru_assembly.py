"""ORU ASSEMBLY -- pick the ORU with the coupler, follow a CSV trajectory to the mate, put it back.

    python -m urlab.apps.oru_assembly                  # mini_ORU into its taught `cleat` assembly
    python -m urlab.apps.oru_assembly --set object_name=ORU_v8 --set assembly_name=<taught>

Two existing apps, end to end, with their behaviour unchanged:

    1. PLAN (apps/kinematic_assembly) -- load the CSV, anchor it on the object's taught assembly,
       and IK every waypoint from where the run starts. Done FIRST, so an unreachable trajectory
       or a missing assembly stops the run before anything has been picked up.
    2. PICK (apps/coupler_pick_place) -- locate the ORU by its markers, power and open the
       coupler, align, mate compliantly with preload, lock, set the payload, lift, tare. The same
       steps from the same CouplerCycle, not a copy of them.
    3. ASSEMBLE (apps/kinematic_assembly) -- [stand-off ->] follow the trajectory (plain position
       moves, or forceMode under `control_mode: admittance`) -> mate -> PRELOAD -> disassemble in
       reverse.
    4a. MOUNT IT (`cleat_toolchanger.enabled`) -- the cleat, opened right before the trajectory's
       LAST waypoint (the final move into the seat), CLAMPS the part and is re-checked; only then
       does the end-effector coupler release, the payload drop and the arm retract by
       `mount_retract`. The part is left mounted on the cleat, so
       there is no disassembly and no put-back. (The order is apps/coupler_pick_assemble's: held
       by both for a moment is harmless, held by neither drops it.) Then, motor off, OUT ALONG
       THE ROUTE -- `exit_along_route`: the approach waypoints in reverse, every one raised by
       mount_retract, so the empty coupler leaves the way the part came in -- and HOME.
    4b. OR PUT IT BACK (cleat off, `return_to_pick`) -- the place half of the same
       CouplerCycle, with the place pose = the pose it was picked from: carry to the place
       standoff, compliant set-down pushing to `place_preload`, release, drop the payload,
       withdraw by `motion.final_retract`, motor off, HOME. With return_to_pick off it goes home
       holding the part, as kinematic_assembly does.

HOME is `home_joints_deg` when set, otherwise the joints the run was STARTED from (read before
anything moves -- in practice the pick view). Logged at startup.

THE PRELOAD (`assembly_preload`), at the mate: under the compliance law, step the reference along
the INSERTION direction -- the CSV's last segment, second-last row -> last row -- until `force_n`
is measured along it and holds for `persistence_s`. Reaching the last waypoint proves the arm got where the taught pose says, not that
the part is seated against anything; the measured force is that proof. No re-tare: the zero taken
with the part hanging free after the lift is what every newton is read against.

TOGETHER WITH IT, THE TORQUE PRELOAD (`assembly_torque_preload`): turn the reference about `axis`
(the tool/coupler frame at the mate, through the part's mating point; its sign is the direction of
the twist). Both advance in the same steps, each until its own target is met, and the run moves on
only when the force AND the torque hold together -- see combined_preload.
The torque is read with arm.wrench_in AT the mating point -- read at the flange instead, the push
alone would add (force x the coupler_mate offset) of lever-arm moment about any axis across it:
5 N x 144 mm = 0.7 Nm, more than a typical threshold.

THE STAND-OFF IS FOR A TRAJECTORY THAT IS ONLY THE MATE. kinematic_assembly builds it off the LAST
row -- right next to the mate -- which suits a short insertion. A CSV that is a real route already
says how to come in, and a stand-off beside the mate would send the part there first, past the
very structure the route goes around. So `use_standoff: auto` (the default) uses one only for a
single-row trajectory; true / false force it.

WHERE IT GOES COMES FROM configs/objects.yaml, not from this config. `assembly_name` names an
entry under the object's `assemblies:` (taught by apps/coupler_assembly_calibration): the pose of
the ORU's MATING FRAME in base_link at the assembled position. Once the ORU is locked on, its
mating frame IS frames.yaml's `coupler_mate`, so the held pose is that frame too -- the two poses
kinematic_assembly asks to be entered by hand are both already measured. The CSV rows are the
mating frame's poses relative to the assembled one (last row identity), so

    T_base_target = T_base_assembly @ inverse(last_row)
    tool0         = T_base_target @ row @ inverse(T_tool0_coupler_mate)

THE ONLY RELEASES ARE THE MOUNT AND THE PUT-BACK. Every other ending -- a failed step, a declined
prompt, Ctrl-C -- leaves the part on the end-effector coupler, so configs/oru_assembly.yaml sets
`toolchanger.latch: true`: an unlatched port resets the board on close, and the board then
re-decides the clamp from one sensor reading. The cleat is latched for the same reason: after a
mount, closing its port must not unclamp the part it is holding.
"""

import time

import numpy as np

from .. import behaviors as bt
from .. import config as urconfig
from .. import log as urlog
from .. import tool_frames
from ..robot import ForceGuard
from ..robot.coupler import Coupler
from ..skills import marker_localize as mloc
from ..skills import trajectory as traj
from ..transforms import fmt_pose, inverse, translation_matrix
from ._cable import make_confirm
from ._common import prompts_off
from ._runner import run_app
from .coupler_pick_place import CouplerCycle, offset_pose, parse_offset, parse_preload, step_gate

log = urlog.get('oru-assembly')


class PickAndReturn(CouplerCycle):
    """CouplerCycle with the place pose = the pick pose: the pick half takes the part off the
    bench, the assembly runs in between, and the place half puts it back exactly where it was."""

    RUN_NAME = 'oru_assembly'

    def _target_pose(self, T_pick):
        return T_pick


# ---------------------------------------------------------------------------- the assembly
def resolve_assembly(cfg, name, obj):
    """The object's taught assembly named by `assembly_name`. Raises with what IS taught -- there
    is no default destination."""
    assembly = cfg.get('assembly_name')
    taught = obj.get('assemblies') or {}
    listed = ', '.join(sorted(taught)) or '(none)'
    if not assembly or assembly not in taught:
        raise KeyError(f'{name!r} has no assembly {assembly!r} in '
                       f'{tool_frames.objects_path(cfg)} (taught: {listed}). Teach it with '
                       f'urlab.apps.coupler_assembly_calibration --set object_name={name} '
                       f'--set assembly_name={assembly or "<name>"}')
    return assembly, taught[assembly]


def use_standoff(cfg, n_rows):
    """Whether to go via the stand-off: `use_standoff` true / false, or `auto` (the default) --
    only when the trajectory is the mate alone (one row). Pure."""
    value = cfg.get('use_standoff', 'auto')
    if isinstance(value, bool):
        return value
    if str(value).lower() == 'auto':
        return n_rows <= 1
    raise ValueError(f'use_standoff must be true, false or auto, not {value!r}')


def insertion_back_axis(mats, standoff_axis):
    """The unit direction that BACKS OUT of the mate, in the seat's (last row's) frame: towards
    the second-last row -- or along standoff_axis for a single-row trajectory. Reversed, it is the
    direction the preload pushes. Pure."""
    back = (inverse(mats[-1]) @ mats[-2])[:3, 3] if len(mats) >= 2 else \
        np.asarray(standoff_axis, dtype=float)
    n = float(np.linalg.norm(back))
    if n < 1e-9:
        raise ValueError('the last two trajectory rows are at the same point, so there is no '
                         'insertion direction to preload along')
    return back / n


_TORQUE_KEYS = {'enabled': True, 'torque_nm': 0.5, 'axis': [0.0, 0.0, 1.0],
                'max_rotation_deg': 10.0, 'step_deg': 0.5, 'settle_s': 0.2, 'persistence_s': 0.5}


def parse_torque_preload(block):
    """`assembly_torque_preload:` -> {torque_nm, axis (unit, seat frame), max_rotation, step (rad),
    settle_s, persistence_s}, or None when it is absent or off. Pure."""
    b = {k: v for k, v in dict(block or {}).items() if not urconfig._is_derived(v)}
    if not b or not bool(b.get('enabled', True)):
        return None
    unknown = sorted(set(b) - set(_TORQUE_KEYS))
    if unknown:
        raise ValueError(f'assembly_torque_preload has unknown key(s) {unknown} -- allowed: '
                         f'{", ".join(_TORQUE_KEYS)}')
    v = dict(_TORQUE_KEYS, **b)
    axis = np.asarray(v['axis'], dtype=float)
    if axis.shape != (3,) or float(np.linalg.norm(axis)) < 1e-9:
        raise ValueError(f'assembly_torque_preload.axis must be a non-zero direction of three '
                         f'numbers, got {axis.tolist()}')
    out = {'torque_nm': float(v['torque_nm']), 'axis': axis / float(np.linalg.norm(axis)),
           'max_rotation': np.radians(float(v['max_rotation_deg'])),
           'step': np.radians(float(v['step_deg'])), 'settle_s': float(v['settle_s']),
           'persistence_s': float(v['persistence_s'])}
    for key in ('torque_nm', 'max_rotation', 'step', 'settle_s'):
        if not out[key] > 0.0:
            raise ValueError(f'assembly_torque_preload: {key} must be positive')
    if out['persistence_s'] < 0.0:
        raise ValueError('assembly_torque_preload.persistence_s must not be negative')
    return out


def rotation_about(axis, angle):
    """A 4x4 rotation by `angle` about unit `axis` (Rodrigues), no translation. Pure."""
    k = np.asarray(axis, dtype=float)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return T


def _joined(*parts):
    """The non-empty parts, joined with ' + ' -- 'N', 'Nm' or 'N + Nm' for a log line."""
    return ' + '.join(p for p in parts if p)


def combined_preload(job, robot, adm, pre, tq, T_at, insert_dir):
    """Preload FORCE along `insert_dir` and TORQUE about tq['axis'] SIMULTANEOUSLY, inside an
    active servo session. `pre` / `tq` may be None to preload only the other. True / False.

    Every step moves the coupler reference from T_at by BOTH a translation along insert_dir and a
    rotation about the torque axis (through the mating point), each measured in T_at's frame:

        T_ref = T_at @ translation(insert_dir * travelled) @ rotation(axis, turned)

    Each one advances only while its own reading is below target, and starts again if the other's
    progress pushes it back under -- the two couple wherever the contact is not at the mating
    point. Success is BOTH held together for the longer persistence_s; running out of travel on
    either, or a guard trip, is failure.

    Force and torque come from ONE arm.wrench_in sample at the measured coupler pose: at the mating
    point, in coupler axes, negated so that pressing / twisting against something reads positive."""
    ref = lambda T: T @ inverse(job.T_tool0_coupler)            # noqa: E731 -- one expression
    d = np.asarray(insert_dir, dtype=float)
    axis = tq['axis'] if tq else np.array([0.0, 0.0, 1.0])
    f_goal = pre['force_n'] if pre else None
    t_goal = tq['torque_nm'] if tq else None
    settle = max(pre['settle_s'] if pre else 0.0, tq['settle_s'] if tq else 0.0, 0.1)
    persist = max(pre['persistence_s'] if pre else 0.0, tq['persistence_s'] if tq else 0.0)

    def reading():
        w = robot.arm.wrench_in(robot.arm.tcp_pose() @ job.T_tool0_coupler)
        return -float(np.dot(w[:3], d)), -float(np.dot(w[3:], axis))

    def pose(travelled, turned):
        return T_at @ translation_matrix(d * travelled) @ rotation_about(axis, turned)

    travelled = turned = 0.0
    T_ref = T_at
    f, t = reading()
    limit = 4 * (int((pre['max_travel_m'] / pre['step_m']) if pre else 0)
                 + int((tq['max_rotation'] / tq['step']) if tq else 0) + 1)
    for _ in range(limit):
        f_low = f_goal is not None and f < f_goal
        t_low = t_goal is not None and t < t_goal
        if not f_low and not t_low:
            # BOTH MET -- hold the reference and require them to STAY met together.
            worst = {'f': f, 't': t}

            def sample():
                fs, ts = reading()
                worst['f'], worst['t'] = min(worst['f'], fs), min(worst['t'], ts)
            if persist > 0.0:
                if adm.hold(ref(T_ref), persist, job.guard, on_step=sample) == 'seated':
                    log.error('The force guard tripped while holding the preload -- stopping.')
                    return False
            f_low = f_goal is not None and worst['f'] < f_goal
            t_low = t_goal is not None and worst['t'] < t_goal
            if not f_low and not t_low:
                log.info('  PRELOAD held: %s after %.2f mm / %.2f deg of reference motion.',
                         _joined(f'{worst["f"]:.2f} N' if f_goal is not None else '',
                                 f'{worst["t"]:.2f} Nm' if t_goal is not None else ''),
                         travelled * 1000.0, np.degrees(turned))
                return True
            log.info('  preload dipped while holding (%.2f N, %.2f Nm) -- advancing again.',
                     worst['f'], worst['t'])
        if f_low and travelled >= pre['max_travel_m'] - 1e-12:
            log.error('The force preload never held %.2f N (%.2f N) within %.1f mm of travel.',
                      f_goal, f, pre['max_travel_m'] * 1000.0)
            return False
        if t_low and turned >= tq['max_rotation'] - 1e-12:
            log.error('The torque preload never held %.2f Nm (%.2f Nm) within %.1f deg.',
                      t_goal, t, np.degrees(tq['max_rotation']))
            return False
        if f_low:
            travelled = min(travelled + pre['step_m'], pre['max_travel_m'])
        if t_low:
            turned = min(turned + tq['step'], tq['max_rotation'])
        T_next = pose(travelled, turned)
        if adm.ramp(ref(T_ref), ref(T_next), settle, job.guard) == 'seated':
            log.error('The force guard tripped during the preload -- stopping.')
            return False
        T_ref = T_next
        f, t = reading()
    log.error('The preload did not settle (%.2f N, %.2f Nm) -- the force and the torque keep '
              'undoing each other. Stiffen the compliance or reduce one of the targets.', f, t)
    return False


def home_joints(cfg, q_start):
    """Home: `home_joints_deg` (six angles) when set, else the joints the run started from."""
    q = cfg.get('home_joints_deg')
    if q is None:
        return np.asarray(q_start, dtype=float)
    q = np.radians(np.asarray(q, dtype=float))
    if q.shape != (6,):
        raise ValueError(f'home_joints_deg must be six joint angles in degrees, got {q.tolist()}')
    return q


def mount_retract_leg(cfg):
    """The retract off the cleat after a mount, and the lift of the way out: `mount_retract` (a
    motion-leg block), else motion.final_retract. Separate because the cleat can sit far out in
    the workspace, where final_retract's length -- tuned for picks -- runs past the reach."""
    block = cfg.get('mount_retract')
    if block is None:
        return parse_offset(cfg.get_path('motion.final_retract'), 'motion.final_retract', 100.0)
    return parse_offset(block, 'mount_retract', 200.0)


def raised_route(T_base_target, mats, lift, T_tool0_held):
    """The route's waypoints in REVERSE, the last (the seat) left out, each coupler pose shifted
    by the base-frame vector `lift` -- as tool0 poses. Pure."""
    out = []
    for m in reversed(mats[:-1]):
        T_c = np.array(T_base_target @ m, dtype=float)
        T_c[:3, 3] = T_c[:3, 3] + np.asarray(lift, dtype=float)
        out.append(T_c @ inverse(T_tool0_held))
    return out


def plan(cfg, robot, T_base_assembly):
    """kinematic_assembly's planning, anchored on a taught assembly: {q_home, q_standoff,
    waypoint_q, T_task}, or None. `T_base_assembly` is the held object's mating frame at the
    mate; the held frame is `coupler_mate`. q_standoff is None when there is no stand-off."""
    try:
        q_home = home_joints(cfg, robot.arm.q())
    except ValueError as exc:
        log.error('%s', exc)
        return None
    log.info('HOME: %s deg (%s).', np.round(np.degrees(q_home), 1).tolist(),
             'home_joints_deg' if cfg.get('home_joints_deg') is not None
             else 'the joints the run started from')
    T_tool0_held = tool_frames.coupler_mate(cfg)

    csv_path = urconfig.resolve(cfg, cfg.get('trajectory_csv', 'oru_assembly_trajectory.csv'))
    mats = traj.load_csv(csv_path, angles_deg=bool(cfg.get('trajectory_angles_deg', False)))
    if len(mats) < 1:
        log.error('Trajectory %s has no usable rows.', csv_path)
        return None
    log.info('Loaded %d trajectory rows from %s.', len(mats), csv_path)

    # The taught pose IS the held frame at the mate, so the target is it with the last row undone
    # (traj.anchor_target does the same from a tool0 pose: assembled @ held @ inverse(last)).
    T_base_targetobj = np.asarray(T_base_assembly, dtype=float) @ inverse(mats[-1])
    poses = [traj.tool0_at(T_base_targetobj, m, T_tool0_held) for m in mats]

    q_standoff = None
    if use_standoff(cfg, len(mats)):
        standoff_axis = np.asarray(cfg.get('standoff_axis', [0, 0, 1]), dtype=float)
        standoff_dist = float(cfg.get('standoff_distance_m', 0.2))
        T_standoff_held = translation_matrix(standoff_axis * standoff_dist) @ mats[-1]
        standoff_pose = traj.tool0_at(T_base_targetobj, T_standoff_held, T_tool0_held)
        q_standoff = robot.arm.ik(standoff_pose, q_home)
        if q_standoff is None:
            log.error('Stand-off pose is unreachable.')
            return None
    else:
        log.info('No stand-off (use_standoff: %s, %d-row trajectory): the trajectory is entered '
                 'at its first row and left the same way.', cfg.get('use_standoff', 'auto'),
                 len(mats))
    waypoint_q = traj.ik_chain(robot.arm, poses, q_standoff if q_standoff is not None
                               else q_home)
    if waypoint_q is None:
        return None
    try:
        back = insertion_back_axis(mats, cfg.get('standoff_axis', [0, 0, 1]))
    except ValueError as exc:
        log.error('%s', exc)
        return None

    # THE WAY OUT after a cleat mount, solved NOW so an unreachable exit stops the run before the
    # pick rather than with the part mounted: the route's waypoints in reverse (the seat itself
    # excluded -- the retract is the raised seat), every one shifted by the retract.
    exit_q = []
    retract = mount_retract_leg(cfg)
    if cfg.get_path('cleat_toolchanger.enabled', False):
        # THE RETRACT ITSELF FIRST: it runs after every mount, exit route or not, and a cleat far
        # out in the workspace can put it past the reach.
        T_seat = T_base_targetobj @ mats[-1]
        T_out = offset_pose(T_seat, retract)
        q_out = robot.arm.ik(T_out @ inverse(T_tool0_held), waypoint_q[-1])
        if q_out is None:
            log.error('The retract off the cleat -- the seat raised %.0f mm along %s (%s frame) '
                      '-- is outside the reach / safety limits. Shorten mount_retract.',
                      retract['distance_m'] * 1000.0, np.round(retract['axis'], 2).tolist(),
                      retract['frame'])
            return None
        if cfg.get('exit_along_route', True):
            poses_out = raised_route(T_base_targetobj, mats, T_out[:3, 3] - T_seat[:3, 3],
                                     T_tool0_held)
            exit_q = traj.ik_chain(robot.arm, poses_out, q_out)
            if exit_q is None:
                log.error('The way out -- the route reversed and raised %.0f mm -- is '
                          'unreachable. Shorten mount_retract, or set exit_along_route: false '
                          'to go straight home instead.', retract['distance_m'] * 1000.0)
                return None
    return {'q_home': q_home, 'q_standoff': q_standoff, 'waypoint_q': waypoint_q,
            'exit_q': exit_q, 'retract_leg': retract,
            'T_task': traj.tool0_at(T_base_targetobj, mats[-1], T_tool0_held),
            'T_seat': T_base_targetobj @ mats[-1],      # the coupler frame at the mate
            'insert_back_axis': back}


def preload_at_mate(cfg, robot, job, p):
    """`assembly_preload` (force along the insertion) and `assembly_torque_preload` (torque about
    its axis) applied SIMULTANEOUSLY -- see combined_preload. True on success or when both are
    off. One compliant session that starts where the arm IS (rebase, no tare) and ends with the
    servo stopped."""
    pre = parse_preload(cfg.section('assembly_preload'), 'assembly_preload')
    pre = pre if pre['enabled'] else None
    tq = parse_torque_preload(cfg.section('assembly_torque_preload'))
    if pre is None and tq is None:
        return True
    if robot.arm.dry_run:
        log.info('DRY RUN: the assembly preload needs contact -- skipped.')
        return True
    adm = job.adm_insert
    ref = lambda T: T @ inverse(job.T_tool0_coupler)            # noqa: E731 -- one expression
    T_at = adm.rebase(robot.arm.tcp_pose() @ job.T_tool0_coupler)
    adm.warmup(ref(T_at))
    job.guard.reset()
    job._active_law = adm
    log.info('ASSEMBLY PRELOAD, simultaneous: %s.', _joined(
        f'{pre["force_n"]:.1f} N along {np.round(-p["insert_back_axis"], 3).tolist()} '
        f'(up to {pre["max_travel_m"] * 1000.0:.1f} mm)' if pre else '',
        f'{tq["torque_nm"]:.2f} Nm about {np.round(tq["axis"], 3).tolist()} '
        f'(up to {np.degrees(tq["max_rotation"]):.1f} deg)' if tq else ''))
    try:
        return combined_preload(job, robot, adm, pre, tq, T_at, -p['insert_back_axis'])
    finally:
        job._active_law = None
        robot.arm.servo_stop()


def assemble(cfg, robot, p, confirm, preload=None, go_home=True, end_at_mate=False,
             before_last=None):
    """kinematic_assembly's run: stand-off -> trajectory -> mate -> [preload] -> disassemble /
    return home. `preload()` runs at the mate, before anything backs out; False stops there.
    go_home=False ends after the disassembly (and the stand-off, if any) -- the put-back follows.
    end_at_mate=True ends right after the preload, part seated -- the cleat mount follows.
    `before_last()` runs right before the LAST waypoint's move (the cleat opening); False stops
    there, the part held just short of the seat."""
    q_home, q_standoff, waypoint_q = p['q_home'], p['q_standoff'], p['waypoint_q']
    compliant = str(cfg.get('control_mode', 'position')).lower() == 'admittance'
    guard = ForceGuard(robot.arm, {'max_force_n': cfg.get_path('admittance.max_force_n', 30.0),
                                   'max_torque_nm': cfg.get_path('admittance.max_torque_nm', 5.0)})

    # 1. Traverse to the stand-off under position control (a free-space move) -- if there is one.
    #    Without it the first waypoint move below is the traverse.
    if q_standoff is not None:
        if confirm and not confirm('move to stand-off'):
            return False
        if not robot.arm.move_j(q_standoff, label='stand-off'):
            return False
        time.sleep(cfg.get('settle_s', 0.2))

    # 2. Follow the trajectory. Under compliance, forceMode + per-waypoint moves; otherwise plain
    #    position moves. The force guard stops the insert when the part seats.
    ok = True
    try:
        if compliant:
            robot.arm.zero_ft()
            robot.arm.force_mode(p['T_task'], [1, 1, 1, 1, 1, 1], [0.0] * 6,
                                 [0.05] * 3 + [0.17] * 3)
        for i, q in enumerate(waypoint_q):
            if guard.check():
                log.info('Contact limit reached at waypoint %d -- part seated.', i)
                break
            if before_last is not None and i == len(waypoint_q) - 1:
                if not before_last():
                    ok = False
                    break
            if confirm and not confirm(f'waypoint {i + 1}/{len(waypoint_q)}'):
                ok = False
                break
            guard.reset()
            robot.arm.add_guard(guard)
            moved = robot.arm.move_j(q, label=f'waypoint {i + 1}')
            robot.arm.clear_guards()
            if not moved:
                if guard.tripped_by:
                    log.info('Guard tripped (%s) -- seated.', guard.tripped_by)
                    break
                ok = False
                break
    finally:
        robot.arm.end_force_mode()
    if not ok:
        return False

    # 2b. Preload at the mate -- proof of seating, before anything backs out.
    if preload is not None:
        if confirm and not confirm('preload at the mate'):
            return False
        if not preload():
            log.error('The assembly preload was not reached -- stopping AT THE MATE, part held.')
            return False
    if end_at_mate:
        return True

    # 3. Disassemble (reverse) or return home.
    if cfg.get('disassemble_after', True):
        for i in range(len(waypoint_q) - 2, -1, -1):
            if not robot.arm.move_j(waypoint_q[i], label=f'disassemble {i}'):
                return False
    elif q_standoff is None:
        # No stand-off to back out to, and a joint move home from the mate would drag the held
        # part straight out of its fixture. Stop here, assembled.
        log.info('disassemble_after is off and there is no stand-off -- stopping AT THE MATE, with '
                 'the part still held.')
        return True
    if q_standoff is not None and not robot.arm.move_j(q_standoff, label='stand-off'):
        return False
    if not go_home:
        return True
    return robot.arm.move_j(q_home, label='home')


# ---------------------------------------------------------------------------- the cleat
def open_cleat(cleat):
    """Power the cleat and open its jaws, before the part is anywhere near it. A cleat that booted
    with something in front of its probe comes up CLAMPED, and cannot be entered."""
    if not cleat.prepare_to_mate():
        log.error('The CLEAT could not be powered and opened, so the part would be driven into a '
                  'clamp that is already shut. Nothing has moved toward it.')
        return False
    if not cleat.latched:
        log.warning('The cleat is NOT latched: closing its port at the end RESETS its board, which '
                    're-decides the clamp from one sensor reading -- the part this run mounts can '
                    'be dropped. Set cleat_toolchanger.latch: true.')
    return True


def clamp_cleat(cleat, name, opened=True):
    """Clamp the cleat onto the part while the end-effector coupler still holds it, and re-check.
    A refusal stops the run with the part still on the coupler -- the state it can be backed out
    of -- rather than releasing it into a clamp that is gripping nothing. `opened` False (the
    trajectory ended before its last waypoint, so the cleat was never opened) refuses outright."""
    if not opened:
        log.error('The cleat was never OPENED -- the trajectory ended before its last waypoint -- '
                  'so whatever seated %r did so against a cleat in an unknown state. Not '
                  'clamping, and the coupler is NOT releasing.', name)
        return False
    if not cleat.hold():
        log.error('The CLEAT would not clamp %r. The end-effector coupler is NOT releasing; the '
                  'part is still held at the mate.', name)
        return False
    again = cleat.verify()
    if again is False:
        log.error('The cleat confirmed the clamp and then its sensor disagreed. %r is not reliably '
                  'held by the cleat; refusing to let go of it.', name)
        return False
    if again is None:
        log.warning('The cleat reports clamped but NOTHING VERIFIED IT (no board, or the sensor is '
                    'bypassed). The release below assumes the cleat has the part.')
    log.info('The cleat is holding %r -- held by BOTH now, the one safe moment to let go.', name)
    return True


def retract_from_seat(job, p):
    """Back the released coupler out of the part by mount_retract, compliantly, starting from
    where the arm IS (it is still pressed in by the preload)."""
    T_seat = p['T_seat']
    return job._compliant(T_seat, offset_pose(T_seat, p['retract_leg']),
                          'retract from the mounted part', in_contact=True)


def exit_along_route(robot, p, confirm, guard):
    """Leave by the raised route (plan()'s exit_q), each move asked for like the approach."""
    n = len(p['exit_q'])
    for k, q in enumerate(p['exit_q'], start=1):
        label = f'exit {k}/{n}: waypoint {n - k + 1}, raised'
        if confirm and not confirm(label):
            return False
        if not robot.move_joints(q, label=label, guard=guard):
            log.error('%s did not finish.', label)
            return False
    return True


# ---------------------------------------------------------------------------- entry point
def build_and_run(cfg, robot, camera, args):
    name = cfg.get('object_name')
    try:
        catalogue = tool_frames.load_objects(cfg)
        view_plan = mloc.ViewPlan(cfg.section('marker_views'))
    except ValueError as exc:
        log.error('%s', exc)
        return False
    if name not in catalogue:
        log.error('object_name %r is not in %s. Catalogued: %s.', name,
                  tool_frames.objects_path(cfg), ', '.join(sorted(catalogue)) or '(none)')
        return False
    obj = catalogue[name]
    try:
        assembly, entry = resolve_assembly(cfg, name, obj)
    except KeyError as exc:
        log.error('%s', exc.args[0])
        return False
    meta = entry.get('meta') or {}
    log.info('Assembly %r of %r: %s (taught %s from %s approach(es), %s mm / %s deg).',
             assembly, name, fmt_pose(entry['T_base_assembly']), meta.get('measured', '?'),
             meta.get('approaches', '?'), meta.get('residual_mm', '?'),
             meta.get('residual_deg', '?'))

    # PLAN FIRST: an unreachable trajectory should cost nothing, not a picked-up ORU.
    p = plan(cfg, robot, entry['T_base_assembly'])
    if p is None:
        return False

    # THE PICK'S SWEEP: marker_views.offsets, camera moves relative to the pose the run starts at
    # (camera optical frame: x right, y down, z forward -- negative z backs away).
    log.info('Pick sweep: %s.', view_plan.describe())
    for i, T in enumerate(view_plan.offsets, start=1):
        log.info('  view %d: %s mm from the start pose (camera frame)', i,
                 np.round(T[:3, 3] * 1000.0, 1).tolist())

    from ..perception import ArucoDetector
    detector = ArucoDetector(cfg, sizes_m={int(mid): m['size_m']
                                           for mid, m in obj['markers'].items()})
    coupler = Coupler(cfg)
    step = step_gate(cfg, robot)
    try:
        job = PickAndReturn(cfg, robot, camera, detector, view_plan, name, obj, coupler)
    except (ValueError, KeyError) as exc:
        log.error('%s', exc)
        coupler.close()
        return False

    # MOUNT ON THE CLEAT, or else PUT IT BACK once it is OUT of the fixture.
    mount = bool(cfg.get_path('cleat_toolchanger.enabled', False))
    cleat = Coupler(cfg, section='cleat_toolchanger', label='cleat') if mount else None
    if mount:
        log.info('MOUNT: at the mate the cleat clamps %r, then the coupler releases and the arm '
                 'retracts %.0f mm (mount_retract). disassemble_after / return_to_pick do not '
                 'apply.', name, p['retract_leg']['distance_m'] * 1000.0)
    put_back = not mount and bool(cfg.get('return_to_pick', True))
    if put_back and not cfg.get('disassemble_after', True):
        log.warning('return_to_pick needs disassemble_after -- the part stays in the fixture, so '
                    'the run ends at the mate, holding it.')
        put_back = False
    if put_back:
        log.info('After the disassembly %r goes BACK to where it was picked: place standoff %.0f '
                 'mm, set-down to %.1f N, release, withdraw %.0f mm.', name,
                 job.legs[job.TARGET_STANDOFF]['distance_m'] * 1000.0,
                 job.preload_place['force_n'], job.legs['final_retract']['distance_m'] * 1000.0)

    confirm = make_confirm(cfg)
    steps = [
        bt.Action('locate the object', job.locate, confirm=step),
        bt.Action('power and open the coupler', job.prepare_coupler, confirm=step)]
    cleat_open = {'done': False}

    def go_home():
        """Guarded joint move to HOME -- free space, nothing held any more."""
        log.info('Home: %s deg.', np.round(np.degrees(p['q_home']), 1).tolist())
        return robot.move_joints(p['q_home'], label='home', guard=job.guard)

    def open_before_last():
        """Right before the final move into the seat: open the cleat so the part can enter."""
        if confirm and not confirm('power and OPEN the cleat'):
            return False
        cleat_open['done'] = open_cleat(cleat)
        return cleat_open['done']

    steps += [
        bt.OperatorGate(robot, 'About to approach and MATE. Clear of the arm? (Enter / q): ',
                        label='before the mate', skip=prompts_off(cfg) or step is not None),
        bt.Action('align at the mate standoff', job.approach, confirm=step),
        bt.Action('mate with the coupling feature', job.descend_and_mate, confirm=step),
        bt.Action('lock the coupler', job.lock, confirm=step),
        bt.Action('take the payload', job.take_payload),
        bt.Action('retract with the object', job.lift, confirm=step),
        bt.Action('tare with the object clear', job.settle_after_lift),
        # The trajectory asks for itself (stand-off, each waypoint) -- no second gate here.
        bt.Action('assemble along the trajectory',
                  lambda: assemble(cfg, robot, p, confirm,
                                   preload=lambda: preload_at_mate(cfg, robot, job, p),
                                   go_home=not put_back, end_at_mate=mount,
                                   before_last=open_before_last if mount else None))]
    if mount:
        # BEFORE the release, never after: held twice for a moment is harmless, held by nothing
        # drops it.
        steps += [
            bt.Action('CLAMP the cleat onto the part',
                      lambda: clamp_cleat(cleat, name, opened=cleat_open['done']), confirm=step),
            bt.Action('release the coupler', job.unlock, confirm=step),
            bt.Action('drop the payload', job.drop_payload),
            bt.Action('retract from the mounted part', lambda: retract_from_seat(job, p),
                      confirm=step),
            bt.Action('switch the coupler motor off', job.park_coupler)]
        if p['exit_q']:
            # Each exit move asks for itself, as the approach waypoints do -- no second gate.
            steps.append(bt.Action('retrace the route out (raised)',
                                   lambda: exit_along_route(robot, p, confirm, job.guard)))
        steps.append(bt.Action('move home', go_home, confirm=step))
    elif put_back:
        steps += [
            bt.Action('carry back to the pick position', job.carry, confirm=step),
            bt.Action('place the object', job.set_down, confirm=step),
            bt.Action('release the coupler', job.unlock, confirm=step),
            bt.Action('drop the payload', job.drop_payload),
            bt.Action('withdraw', job.withdraw, confirm=step),
            bt.Action('switch the coupler motor off', job.park_coupler),
            bt.Action('report', job.report),
            bt.Action('move home', go_home, confirm=step)]
    root = bt.sequence('oru-assembly', *steps)
    try:
        return bt.run_tree(root, log)
    finally:
        robot.arm.servo_stop()
        # NOTHING HERE RELEASES. Unless the mount or the put-back completed, the part is still on
        # the end-effector coupler; after a mount the cleat must stay clamped (latched).
        if not coupler.latched and not coupler.manual:
            log.warning('The toolchanger is NOT latched, so closing its port resets the board -- '
                        'and the board re-decides the clamp from a single sensor reading. '
                        'Support %r before this exits, or set toolchanger.latch: true.', name)
        coupler.close()
        if cleat is not None:
            if not cleat.latched and not cleat.manual:
                log.warning('Closing the cleat\'s port now resets its board, which re-decides the '
                            'clamp from one sensor reading. Support %r if it must not fall.', name)
            cleat.close()


def main():
    # with_gripper=False: the coupler does the holding.
    run_app('ORU assembly: pick the ORU with the coupler, then follow a CSV trajectory to the mate',
            'oru_assembly', build_and_run, with_gripper=False, needs_camera=True)


if __name__ == '__main__':
    main()
