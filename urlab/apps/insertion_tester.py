"""INSERTION TESTER -- does the insertion still seat when the connector is deliberately misaligned?

calibration_check asks "is the recorded mate where the socket is?" and probes a NOMINAL approach.
This asks the next question: how far off can the connector be before the insertion stops working,
and does a different insertion STRATEGY widen that margin? It is the success-basin experiment run
on hardware -- offset in, seated/not-seated out -- with no estimation anywhere in the loop.

Per cycle:
    offset     the next entry of `offsets` (repeated `repeats` times each, optionally shuffled).
               It is injected exactly the way estimator_eval injects a belief error --
               T_believed = T_true @ delta -- so the robot plans a PERFECT insertion against a
               WRONG belief and the connector physically arrives misaligned. The part is fixtured
               between closed fingers, so T_true is exact and the resulting physical pose is known
               rather than estimated; both numbers land in the CSV (see `_physical_offset`).
    standoff   backed off standoff_distance_m along the BELIEVED connector's -X (free space)
    insert     one of:
                 direct  a straight advance along +X to the mate, guarded -- calibration_check's
                         motion, so a zero-offset direct run reproduces that script's probe.
                 wiggle  drive at a fixed target PAST the mate and oscillate about it in the
                         connector's own axes, letting compliance find the lead-in the way a
                         person jiggles a plug in (bnc_assembly's maneuver, same Lissajous
                         reasoning: co-prime frequencies so the orbit sweeps the rectangle
                         instead of retracing one line through it).
    verdict    SEATED is measured, never assumed: the TRUE connector pose w.r.t. the recorded
               mate at the end of the motion, against seat_tolerance (mm / deg). A force-guard
               trip short of the mate is a JAM, and is recorded as one.
    retract    compliant, un-guarded, back to the standoff -- a seated part is already over the
               guard limit, so a guarded retract would block the motion that frees it.

Output: data/experiments/insertion_tester_<stamp>/insertions.csv + a per-offset success table.
The CSV is one row per cycle: the injected offset, the physical offset it produced, the seat pose,
the wrench, how the motion ended, and the duration.

Run:  python -m urlab.apps.insertion_tester --config configs/insertion_tester.yaml
      python -m urlab.apps.insertion_tester --set assembly.insertion_mode=wiggle
"""

import csv as _csv
import os
import time
from datetime import datetime

import numpy as np

from .. import log as urlog
from .. import tool_frames
from ..robot import AdmittanceController, ForceGuard
from ..skills.manifold import mats_from_vec6
from ..transforms import inverse, matrix_to_xyzrpy, pose_error
from ._runner import run_app
# The maneuver pieces come from the apps that own them rather than being re-implemented here:
# a second copy of the advance detector or the wiggle geometry would drift from the one the
# production app actually runs, and then this tester would be measuring the wrong thing.
from .bnc_assembly import _AnyGuard, _ScrewAdvance
from .calibration_check import line_rows
from .estimator_eval import _corr_to_m

log = urlog.get('insertion-tester')

MODES = ('direct', 'wiggle')
_DIMS = ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')


def _physical_offset(delta):
    """The connector's ACTUAL misalignment at the mate, given an injected BELIEF error `delta`.

    The robot builds every reference as `T_base_tconn @ row @ inverse(T_believed)` and the part
    really sits at T_true from tool0, so the connector's true pose w.r.t. the target works out to
    `row @ inverse(delta)`. At the mate (row = identity) that is inverse(delta): injecting a
    +2 mm belief error puts the part 2 mm the OTHER way.

    Both numbers are logged because they answer different questions -- the injected one makes a
    run comparable with estimator_eval's trials, the physical one is the misalignment the
    insertion actually had to overcome, and it is the x-axis of any success basin drawn from
    this data. They are NOT interchangeable, and for rotations they are not even equal in
    magnitude once the offsets get large."""
    xyz, rpy = matrix_to_xyzrpy(inverse(_corr_to_m(np.asarray(delta, dtype=float))))
    return list(xyz * 1000.0) + list(np.degrees(rpy))


def _parse_offsets(raw, repeats, rng, shuffle):
    """`offsets` -> a flat list of (label, 6-vector mm/deg) cycles, or None on a bad entry."""
    if not raw:
        raw = [[0.0] * 6]                          # no offsets = a nominal probe, like calib-check
    out = []
    for idx, row in enumerate(raw):
        try:
            v = [float(x) for x in row]
        except (TypeError, ValueError):
            log.error('offsets[%d] is not a list of 6 numbers: %r', idx, row)
            return None
        if len(v) != 6:
            log.error('offsets[%d] must have 6 entries [x, y, z (mm), roll, pitch, yaw (deg)], '
                      'got %d.', idx, len(v))
            return None
        lab = ('nominal' if not any(abs(c) > 1e-12 for c in v)
               else ' '.join(f'{n.split("_")[0]}{c:+g}' for n, c in zip(_DIMS, v)
                             if abs(c) > 1e-12))
        out.extend([(lab, v)] * repeats)
    if shuffle:
        # ORDER MATTERS when a fixture creeps: running every repeat of one offset back-to-back
        # confounds "this offset is hard" with "the fixture had drifted by then". Shuffling
        # spreads each offset across the run so a drift shows up as scatter, not as a false
        # difference between offsets.
        order = rng.permutation(len(out))
        out = [out[i] for i in order]
    return out


def build_and_run(cfg, robot, camera, args):
    held_name = cfg.get('held_frame')
    if not held_name:
        log.error('held_frame is required.')
        return False
    frames = tool_frames.load_frames(cfg)
    targets = tool_frames.load_targets(cfg)
    if held_name not in frames or held_name not in targets:
        log.error('held_frame %r needs BOTH a frames: and a targets: entry in %s.',
                  held_name, tool_frames.frames_path(cfg))
        return False
    T_true = frames[held_name]                     # tool0 -> held connector, EXACT (it is fixtured)
    T_base_tconn = targets[held_name]              # base_link <- target connector (the mate)

    a = cfg.section('insertion')
    mode = str(a.get('insertion_mode', 'direct')).strip().lower()
    if mode not in MODES:
        log.error('insertion.insertion_mode %r must be one of %s.', mode, list(MODES))
        return False                               # bad values fail HERE, pre-motion

    repeats = int(a.get('repeats', 1))
    seed = int(a.get('random_seed', 0))
    rng = np.random.default_rng(seed if seed > 0 else None)
    shuffle = bool(a.get('shuffle', True))
    if repeats < 1:
        log.error('insertion.repeats must be >= 1 (got %d).', repeats)
        return False
    cycles = _parse_offsets(a.get('offsets'), repeats, rng, shuffle)
    if cycles is None:
        return False

    standoff_m = float(a.get('standoff_distance_m', 0.030))
    res_m = float(a.get('resolution_m', 0.001))
    overshoot_m = float(a.get('overshoot_m', 0.002))
    seat_tol_mm = float(a.get('seat_tolerance_mm', 1.0))
    seat_tol_deg = float(a.get('seat_tolerance_deg', 2.0))
    if standoff_m <= 0 or res_m <= 0 or overshoot_m < 0:
        log.error('standoff_distance_m and resolution_m must be > 0, overshoot_m >= 0.')
        return False
    if seat_tol_mm <= 0 or seat_tol_deg <= 0:
        log.error('seat_tolerance_mm and seat_tolerance_deg must be > 0.')
        return False

    v_mm_s = float(cfg.get_path('speed.approach_translation_mm_s', 2.0))
    rv_mm_s = float(cfg.get_path('speed.retract_translation_mm_s', 20.0))
    settle_s = float(cfg.get_path('compliance.settle_s', 1.0))

    adm = AdmittanceController(robot.arm, cfg.section('compliance'))
    guard = ForceGuard(robot.arm, cfg.section('force_guard'))
    tare = (lambda: robot.arm.zero_ft(settle=False)) \
        if bool(cfg.get_path('compliance.tare_before', True)) else None
    min_seg_s = 1.0 / adm.rate

    # ---- WIGGLE parameters, validated pre-motion (the same checks bnc_assembly makes) ----
    wg = a.get('wiggle', {}) or {}
    adm_wg, guard_wg = adm, guard
    wg_target = [float(v) for v in (wg.get('target') or [5.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    wg_amp = [float((wg.get('amplitude') or {}).get(d, 0.0)) for d in _DIMS]
    wg_frq = [float((wg.get('frequency_hz') or {}).get(d, 0.0)) for d in _DIMS]
    wg_rate = float(wg.get('sample_rate_hz', 25.0))
    wg_max_s = float(wg.get('max_duration_s', 60.0))
    wg_engage_mm = float(wg.get('engage_advance_mm', 4.0))
    if mode == 'wiggle':
        if len(wg_target) != 6:
            log.error('insertion.wiggle.target must be a 6-vector [x, y, z (mm), roll, pitch, '
                      'yaw (deg)].')
            return False
        if wg_rate <= 0 or wg_max_s <= 0:
            log.error('insertion.wiggle.sample_rate_hz and max_duration_s must be > 0.')
            return False
        if not any(abs(v) > 0.0 for v in wg_amp):
            log.error('insertion.wiggle: every amplitude is 0 -- nothing would oscillate.')
            return False
        for i, d in enumerate(_DIMS):
            if abs(wg_amp[i]) > 0.0 and wg_frq[i] <= 0.0:
                log.error('insertion.wiggle.frequency_hz.%s must be > 0 when its amplitude is '
                          'non-zero.', d)
                return False
        # NYQUIST, with margin. A reference rebuilt at sample_rate_hz that is not several times
        # the highest commanded frequency is ALIASED into a slower wiggle -- and it would look
        # like it ran correctly, just with a frequency nobody chose.
        f_max = max((wg_frq[i] for i in range(6) if abs(wg_amp[i]) > 0.0), default=0.0)
        if f_max > 0.0 and wg_rate < 4.0 * f_max:
            log.error('insertion.wiggle.sample_rate_hz %.1f Hz is too coarse for a %.2f Hz '
                      'component (need >= 4x = %.1f Hz) -- the sampled sine would alias.',
                      wg_rate, f_max, 4.0 * f_max)
            return False
        if wg_engage_mm > wg_target[0]:
            log.warning('insertion.wiggle.engage_advance_mm (%.1f) exceeds the target x (%.1f) '
                        '-- the success threshold sits beyond where the reference ever pushes, '
                        'so the wiggle can only ever time out.', wg_engage_mm, wg_target[0])
        # The wiggle presses and rocks, so it usually wants its own (stiffer along x, softer
        # across) compliance and a looser guard meaning "genuinely jammed" rather than "touching".
        comp_wg = dict(cfg.section('compliance'))
        for key in ('stiffness', 'mass', 'damping_ratio'):
            if wg.get(key) is not None:
                comp_wg[key] = [float(v) for v in wg[key]]
        adm_wg = AdmittanceController(robot.arm, comp_wg)
        over = {k: wg[k] for k in ('max_force_n', 'max_torque_nm', 'persistence_s')
                if wg.get(k) is not None}
        guard_wg = ForceGuard(robot.arm, {**dict(cfg.section('force_guard')), **over}) \
            if over else guard

    def seg_time(A, B, v):
        lin_m, _ = pose_error(A, B)
        return max(lin_m * 1000.0 / max(v, 1e-6), min_seg_s)

    def true_pose6():
        """The TRUE connector w.r.t. the recorded mate, mm / deg. Exact: the part is fixtured, so
        this is forward kinematics through a known frame, not an estimate."""
        xyz, rpy = matrix_to_xyzrpy(inverse(T_base_tconn) @ robot.tool0() @ T_true)
        return list(xyz * 1000.0) + list(np.degrees(rpy))

    log.info('Insertion tester: mode %s, %d cycles (%d offset(s) x %d repeat(s)%s), '
             '%.0f mm standoff, guard %.0f N (persistence %.2f s).', mode.upper(), len(cycles),
             len(cycles) // max(repeats, 1), repeats, ', shuffled' if shuffle else '',
             standoff_m * 1000.0, guard.max_force, guard.persistence_s)
    log.info('Seated means the TRUE connector lands within %.2f mm / %.2f deg of the recorded '
             'mate -- measured at the end of every cycle, never inferred from the guard.',
             seat_tol_mm, seat_tol_deg)
    if mode == 'wiggle':
        log.info('  wiggle: target %s, amplitude %s, frequency %s Hz, engage at %.1f mm, '
                 'up to %.0f s at %.0f Hz.', wg_target,
                 [wg_amp[i] for i in range(6) if abs(wg_amp[i]) > 0],
                 [wg_frq[i] for i in range(6) if abs(wg_amp[i]) > 0], wg_engage_mm,
                 wg_max_s, wg_rate)

    out_dir = os.path.join(cfg.get('data_dir', 'data'), 'experiments',
                           f'insertion_tester_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)
    fout = open(os.path.join(out_dir, 'insertions.csv'), 'w', newline='')
    writer = _csv.writer(fout)
    writer.writerow(['cycle', 'mode', 'offset_label']
                    + [f'inj_{d}' for d in _DIMS]      # the BELIEF error that was injected
                    + [f'phys_{d}' for d in _DIMS]     # the misalignment it physically produced
                    + [f'seat_{d}' for d in _DIMS]     # where the connector actually ended up
                    + ['seated', 'outcome', 'force_n', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz',
                       'advance_mm', 'tripped_by', 'duration_s'])
    log.info('Output: %s', out_dir)

    q_home = robot.arm.q()
    seed_q = q_home
    rows, ok = [], True
    try:
        for cyc, (label, off6) in enumerate(cycles, start=1):
            t0 = time.time()
            # THE INJECTION, estimator_eval's convention exactly: corrupt the BELIEF, never the
            # geometry. Everything downstream plans against T_believed, so the misalignment is
            # realised by the robot's own motion rather than by moving the target.
            delta = mats_from_vec6(np.asarray(off6, dtype=float))
            T_believed = T_true @ _corr_to_m(delta)
            phys6 = _physical_offset(delta)
            log.info('--- cycle %d/%d --- offset %s\n'
                     '      injected  xyz=[%+6.2f, %+6.2f, %+6.2f] mm  rpy=[%+6.2f, %+6.2f, %+6.2f] deg\n'
                     '      physical  xyz=[%+6.2f, %+6.2f, %+6.2f] mm  rpy=[%+6.2f, %+6.2f, %+6.2f] deg',
                     cyc, len(cycles), label, *off6, *phys6)

            if mode == 'direct':
                seated_by, advance_mm, last_ref, reached = _direct(
                    robot, adm, guard, T_base_tconn, T_believed, T_true, standoff_m,
                    overshoot_m, res_m, v_mm_s, seg_time, tare, seed_q)
            else:
                seated_by, advance_mm, last_ref, reached = _wiggle(
                    robot, adm_wg, guard_wg, T_base_tconn, T_believed, T_true, wg_target,
                    wg_amp, wg_frq, wg_rate, wg_max_s, wg_engage_mm, tare, seed_q)
            if last_ref is None:                   # IK/approach failed -- reported by the helper
                ok = False
                break
            seed_q = robot.arm.q()

            adm_ctl = adm if mode == 'direct' else adm_wg
            adm_ctl.hold(last_ref, settle_s, guard=None)   # settle AT the stop, no re-trip races
            seat6 = true_pose6()
            w = list(robot.arm.wrench())
            fmag = float(np.linalg.norm(w[:3]))
            # SEATED is the measured pose, not the reason the motion ended. A guard trip AT the
            # mate is a good seat; completing the path 3 mm short is not.
            seated = bool(abs(seat6[0]) <= seat_tol_mm
                          and max(abs(v) for v in seat6[1:3]) <= seat_tol_mm
                          and max(abs(v) for v in seat6[3:]) <= seat_tol_deg)
            outcome = ('seated' if seated
                       else 'jammed' if seated_by == 'force'
                       else 'timeout' if seated_by == 'timeout'
                       else 'short')
            rows.append(dict(label=label, seated=seated, seat6=seat6, phys6=phys6,
                             outcome=outcome))
            writer.writerow([cyc, mode, label]
                            + [f'{v:.4f}' for v in off6]
                            + [f'{v:.4f}' for v in phys6]
                            + [f'{v:.4f}' for v in seat6]
                            + [int(seated), outcome, f'{fmag:.2f}']
                            + [f'{v:.3f}' for v in w]
                            + [f'{advance_mm:.3f}',
                               (guard if mode == 'direct' else guard_wg).tripped_by or '',
                               f'{time.time() - t0:.1f}'])
            fout.flush()
            log.info('   %s -- seat xyz [%+6.2f, %+6.2f, %+6.2f] mm  rpy [%+6.2f, %+6.2f, '
                     '%+6.2f] deg  |f| %.1f N%s', outcome.upper(), *seat6, fmag,
                     '' if reached else '  [motion ended early]')

            # RETRACT: compliant, UN-guarded. A seated or jammed connector is already over the
            # limit, so an armed guard would refuse the very motion that frees it.
            T_out = T_base_tconn @ line_rows(standoff_m, 0.0, res_m)[0] @ inverse(T_believed)
            adm_ctl.ramp(last_ref, T_out, seg_time(last_ref, T_out, rv_mm_s), guard=None)
            adm_ctl.stop()
    except Exception:                              # noqa: BLE001
        ok = False
        log.exception('Insertion tester error:')
    finally:
        robot.arm.servo_stop()
        fout.close()

    _summarise(rows, seat_tol_mm, seat_tol_deg, robot.arm.dry_run)
    if ok:
        robot.arm.move_j(q_home, label='home')
    log.info('Insertion tester complete: %s', out_dir)
    return ok and len(rows) == len(cycles)


def _direct(robot, adm, guard, T_base_tconn, T_believed, T_true, standoff_m, overshoot_m,
            res_m, v_mm_s, seg_time, tare, seed_q):
    """DIRECT insertion -- calibration_check's straight advance, planned against the BELIEF.

    With a zero offset this is byte-for-byte that script's probe, which is the point: it is the
    control the wiggle has to beat, and it is already the motion the production apps drive."""
    refs = [T_base_tconn @ r @ inverse(T_believed)
            for r in line_rows(standoff_m, overshoot_m, res_m)]
    q = robot.arm.ik(refs[0], seed_q)
    if q is None or not robot.arm.move_j(q, label='insertion standoff'):
        log.error('IK/approach failed for the standoff.')
        return None, 0.0, None, False
    adm.reset()
    adm.warmup(refs[0], tare_fn=tare)
    guard.reset()
    last_ref, tripped = refs[0], False
    for i in range(1, len(refs)):
        res = adm.ramp(refs[i - 1], refs[i], seg_time(refs[i - 1], refs[i], v_mm_s), guard)
        last_ref = refs[i]
        if res == 'seated':
            tripped = True
            break
    # Advance = the TRUE connector's x w.r.t. the mate at the stop, so it means the same
    # thing as the wiggle's peak advance and the two modes can be compared directly.
    # Measured from the ARM, not the reference: under admittance they differ by exactly
    # the compliant deflection, and that deflection is the part that did not go in.
    adv = float((inverse(T_base_tconn) @ robot.tool0() @ T_true)[0, 3]) * 1000.0
    return ('force' if tripped else 'end'), adv, last_ref, not tripped


def _wiggle(robot, adm_wg, guard_wg, T_base_tconn, T_believed, T_true, target, amp, frq,
            rate, max_s, engage_mm, tare, seed_q):
    """WIGGLE insertion -- bnc_assembly's maneuver, planned against the BELIEF.

    Drives at ONE fixed target PAST the mate and rocks about it in the connector's own axes:

        v(t) = target + SUM over axes of  amplitude_i * sin(2*pi*frequency_i*t)

    The reference is rebuilt at `rate` and each consecutive pair ramped over exactly one sample
    period, so the commanded motion follows the intended TIME law -- pacing a wiggle by distance
    the way the rest of this file does would change its frequency.

    Engagement is MEASURED every servo cycle (_ScrewAdvance, the detector bnc_assembly's clocking
    screw uses) so a success ends the motion the instant it happens; a force trip ends it as a
    jam. Both outcomes are distinguished through _AnyGuard, because `ramp` only reports 'seated'
    and the two mean opposite things."""
    def ref_at(t):
        v = np.array(target, dtype=float)
        for i in range(6):
            if abs(amp[i]) > 0.0 and frq[i] > 0.0:
                v[i] += amp[i] * np.sin(2.0 * np.pi * frq[i] * float(t))
        # inverse(T_believed): the robot commands the connector IT THINKS it holds
        # to pose v w.r.t. the target, which is what realises the injected offset.
        return T_base_tconn @ _corr_to_m(mats_from_vec6(v)) @ inverse(T_believed)

    first = ref_at(0.0)
    q = robot.arm.ik(first, seed_q)
    if q is None or not robot.arm.move_j(q, label='wiggle start'):
        log.error('IK/approach failed for the wiggle start pose.')
        return None, 0.0, None, False

    # Advance is measured on the TRUE frame: the question is where the part physically got to,
    # and the belief is deliberately wrong here.
    # T_true, not T_believed: the question is where the part PHYSICALLY got to, and the
    # belief is deliberately wrong here. Engaged reference is the recorded mate, so
    # advance is the true connector's x w.r.t. it -- negative outside, 0 at the mate.
    det = _ScrewAdvance(robot, T_true, T_base_tconn, engage_mm / 1000.0)
    combo = _AnyGuard(det, guard_wg)
    dt, nstep = 1.0 / rate, int(round(max_s * rate))
    adm_wg.reset()
    adm_wg.warmup(first, tare_fn=tare)
    combo.reset()
    prev, ended = first, 'timeout'
    for i in range(1, nstep + 1):
        cur = ref_at(i * dt)
        res = adm_wg.ramp(prev, cur, dt, combo)
        prev = cur
        if res != 'seated':
            continue
        if combo.tripped is det:
            ended = 'advance'
            log.info('   engaged after %.1f s of wiggle -- %s', i * dt, det.tripped_by)
        else:
            ended = 'force'
            log.info('   wiggle stopped on the force guard after %.1f s (%s); peak advance '
                     '%.2f mm.', i * dt, combo.tripped_by, det.peak_m * 1000.0)
        break
    else:
        log.info('   wiggle ran the full %.0f s without reaching %.1f mm; peak advance %.2f mm.',
                 max_s, engage_mm, det.peak_m * 1000.0)
    # Hold the pose the arm is AT, not that pose plus the deflection already in it.
    stay = robot.tool0()
    adm_wg.reset()
    return ended, det.peak_m * 1000.0, stay, ended == 'advance'


def _summarise(rows, seat_tol_mm, seat_tol_deg, dry_run=False):
    """Success rate per offset -- the table this whole script exists to produce."""
    if not rows:
        return
    if dry_run:
        # tcp_pose is a fixed stand-in without a robot, so every seat pose below is that
        # constant and every verdict is meaningless. Say so once, loudly, rather than
        # letting the 0%% table read as a finding.
        log.warning('DRY RUN: tcp_pose is a fixed stand-in, so the seat poses and the '
                    'seated/not-seated verdicts below are NOT measurements. This run '
                    'exercises the sequence and the CSV shape only.')
    log.info('--- insertion summary over %d cycles (seated = within %.2f mm / %.2f deg) ---',
             len(rows), seat_tol_mm, seat_tol_deg)
    order, seen = [], set()
    for r in rows:                                 # preserve first-seen order, not sorted
        if r['label'] not in seen:
            seen.add(r['label'])
            order.append(r['label'])
    log.info('   %-28s %7s %8s %10s %10s', 'offset', 'n', 'seated', 'seat x mm', 'outcomes')
    for lab in order:
        grp = [r for r in rows if r['label'] == lab]
        n_ok = sum(1 for r in grp if r['seated'])
        xs = [r['seat6'][0] for r in grp]
        tally = {}
        for r in grp:
            tally[r['outcome']] = tally.get(r['outcome'], 0) + 1
        log.info('   %-28s %7d %7.0f%% %+6.2f+-%-4.2f %s', lab, len(grp),
                 100.0 * n_ok / len(grp), float(np.mean(xs)), float(np.std(xs)),
                 ' '.join(f'{k}:{v}' for k, v in sorted(tally.items())))
    tot = sum(1 for r in rows if r['seated'])
    log.info('   OVERALL %d/%d seated (%.0f%%).', tot, len(rows), 100.0 * tot / len(rows))
    if tot == len(rows) and not dry_run:
        log.info('   Every offset seated -- the basin is WIDER than the range tested, so widen '
                 '`offsets` before concluding anything about the margin.')
    elif tot == 0 and not dry_run:
        log.warning('   NOTHING seated, including any nominal offset -- suspect the setup (frame, '
                    'fixture, seat tolerance) before the insertion strategy.')


def main():
    # with_gripper=False: the connector is fixtured between the closed fingers, exactly as in
    # calibration_check -- this script never opens or closes anything.
    run_app('Insertion tester: seat success vs injected offset, direct or wiggle',
            'insertion_tester', build_and_run, with_gripper=False)


if __name__ == '__main__':
    main()
