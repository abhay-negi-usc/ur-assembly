"""WIGGLE SAMPLING -- sustained-contact excitation for CONTACT-MODE discovery.

Data collection for analysis/engagement_modes/wiggle_mode_detection.md. Where uncertain_sampling
builds a POSE map, this builds a MODE dataset: it parks the connector at a series of labelled
stations, presses it against the socket, and holds it there for tens of seconds while a multisine
rocks it in x / z / pitch. The question it exists to answer is not "where is the part" but "how
many directions is it free to move in", which is a property of the local tangent cone and needs
CONTACT THAT LASTS.

WHY A SEPARATE APP, AND NOT A KNOB ON uncertain_sampling. That app's trial is a single approach:
the connector meets the socket once, at the very end, roughly along the socket normal, and the
trial is over within a settle. That is exactly right for mapping pose -> wrench and exactly wrong
for mode detection, for three reasons that no parameter can fix:

  * NO DWELL. A tangent-cone estimate is a covariance (or a compliance regression) over a WINDOW.
    A contact that lasts one settle gives one window, all of it in the post-impact transient.
  * NORMAL CONTACT ONLY. Driving straight down the axis loads the constraint the insertion already
    knows about. The lateral and rotational constraints -- the ones that actually distinguish a
    one-point graze from a seated bayonet -- are never excited, so the data cannot show them.
  * NO EXCITATION SPECTRUM. The commanded motion is a monotone advance, so any rank deficit in the
    observed motion is indistinguishable from "that direction was never driven". A multisine makes
    the COMMANDED motion provably full-rank, which is what turns an observed deficit into evidence.

THE PROTOCOL, per burst:

    approach   ramp axially to the station under admittance, under a LOOSE safety limit only.
               Not the tight seating guard uncertain_sampling arms: there a trip means "seated,
               stop advancing", while here the commanded depth is the independent variable, so a
               tight limit would hold every station at the limit instead of at its preload.
               Armed only when going deeper -- a withdrawal must never be blocked by a guard.
    press      preload_mm further along the connector's own +X. This is what makes the contact
               BILATERAL: without it the reachable set is a half-space and PCA on a half-disc
               reports near-full rank no matter how constrained the part is
               (wiggle_mode_detection.md 3.3).
    quiet      hold still. This segment is the NOISE FLOOR -- the eigengap threshold is meaningless
               without one, and it has to be measured in the same pose and grasp as the wiggle it
               will be compared against, not once at the start of the day.
    wiggle     multisine on the connector's own axes, superimposed on the preloaded reference.
    tug        TWO probes -- release along -X, and shove sideways at the preload -- each ~3 s, for
               a near-definitive label (3.2): the cheapest supervision in the experiment, and
               worth more than any amount of analysis cleverness. They are not redundant; for a
               bayonet before the collar is clocked they are expected to disagree (see the config).

and the run is those bursts over the product of

    stations x preloads x amplitude scales x passes(in, out) x offsets

with three station kinds that are labelled BY CONSTRUCTION (3.1):

    insert   in the socket at a stated depth -- the states under study, label unknown
    free     held well clear -- k = 0 by definition
    beside   the SAME DEPTH as an insert station but laterally displaced into free space.
             DO NOT SKIP THIS ONE. Depth correlates with engagement trivially, so without it a
             "mode detector" that has merely learned to read x cannot be told from one that has
             learned the geometry. If the statistic separates in-socket from beside-socket at
             identical depth, the signal is real.

ORDER IS PART OF THE EXPERIMENT. Within a pass the contact stations are visited in depth order and
the arm does NOT return to the standoff between them -- it ramps station to station along the axis,
so the contact history is continuous. The 'in' pass walks deeper, the 'out' pass walks shallower.
Comparing the same station between the two passes is the hysteresis test (4.1), which is the
strongest evidence in the document: a continuous function of pose cannot produce it, so a
difference at matched depth means a discrete state exists. Retracting between every station would
erase the history and quietly turn that test into noise.

SLIP IS THE CONFOUND THAT RUINS THIS (6). A grasp slip is discrete, hysteretic and sharply
transitioning -- it forges every diagnostic the analysis relies on (bimodality, matched-depth
hysteresis, window-length-independent sharp transitions) and it lives entirely in the jaws.
RETURN-TO-REFERENCE runs every N bursts: revisit one fixed free-space pose, re-read pose and
wrench, and log the drift. Growth there IS accumulated slip, and it is what lets affected bursts
be bracketed or dropped rather than quietly poisoning the result.

The document's other defence, DATUM RE-TOUCH (probe a rigid feature and record the pose at contact
onset), is NOT implemented here -- it needs a datum whose location is known independently of the
socket, which this cell does not have. Return-to-reference catches a slip that changes the grasp
pose; it will not catch one that leaves the part in the same place with less friction.

Output: data/wiggle_sampling/<cable>/run_<stamp>/{samples.csv, bursts.csv}
    samples.csv is a strict SUPERSET of uncertain_sampling's schema -- same column names for the
        tool0 pose, the connector-vs-target deviation and both wrenches -- so anything that reads
        that file reads this one. The added columns are the COMMANDED pose (which the compliance
        regression of 2.1 needs and uncertain_sampling never logged) and the segment labels.
    bursts.csv is one row per burst: the plan, the noise floor, and the tug verdict. It is the
        weak supervision -- join it to samples.csv on `burst`.

Run:  python -m urlab.apps.wiggle_sampling --config configs/wiggle_sampling.yaml
      python -m urlab.apps.wiggle_sampling --set wiggle.wiggle_s=30
"""

import csv as _csv
import math
import os
import time
from datetime import datetime, timedelta

import numpy as np

from .. import config as urconfig
from .. import log as urlog
from .. import tool_frames
from ..robot import AdmittanceController, ForceGuard
from ..skills import trajectory as traj
from ..transforms import (
    inverse, pose_error, slerp_matrix, translation_matrix, xyzrpy_to_matrix)
from ._runner import run_app
from .cable_pick_assemble import _guarded

log = urlog.get('wiggle-sampling')

# The six excitation axes, in the order they index every amplitude / frequency / phase vector.
# Same spelling as bnc_assembly and insertion_tester so a wiggle block can be copied between them.
DIMS = ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')

_POSE = ('x_mm', 'y_mm', 'z_mm', 'qx', 'qy', 'qz', 'qw', 'yaw_deg', 'pitch_deg', 'roll_deg')
_WRENCH = ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')

_HEADER = (
    # ---- uncertain_sampling's schema, verbatim, so its readers work unchanged ----------------
    ['trial', 'timestamp']
    + [f'tool0_base_{s}' for s in _POSE]           # raw tool0 wrt base
    + [f'connector_target_{s}' for s in _POSE]     # ACHIEVED connector deviation from the mate
    + [f'wrench_base_{s}' for s in _WRENCH]        # wrench as recorded (base_link)
    + [f'wrench_connector_{s}' for s in _WRENCH]   # wrench re-expressed in the connector frame
    # ---- what this app adds ------------------------------------------------------------------
    # The COMMANDED pose, same 10-field form. 2.1's compliance regression needs the input as well
    # as the output, and 3.3 asks for it outright: without it an unmoved axis cannot be told from
    # an undriven one, which is the entire distinction the multisine exists to make.
    + [f'commanded_target_{s}' for s in _POSE]
    + ['burst', 'segment', 'taper', 't_seg',
       'station', 'label', 'direction', 'depth_mm', 'lat_y_mm', 'lat_z_mm',
       'preload_mm', 'amp_scale'])

_BURST_HEADER = [
    'burst', 'station', 'label', 'direction', 'depth_mm', 'lat_y_mm', 'lat_z_mm',
    'offset_x_mm', 'offset_y_mm', 'offset_z_mm',
    'offset_roll_deg', 'offset_pitch_deg', 'offset_yaw_deg',
    'preload_mm', 'amp_scale',
    'amp_x_mm', 'amp_y_mm', 'amp_z_mm', 'amp_roll_deg', 'amp_pitch_deg', 'amp_yaw_deg',
    'f_x_hz', 'f_y_hz', 'f_z_hz', 'f_roll_hz', 'f_pitch_hz', 'f_yaw_hz',
    'n_quiet', 'n_wiggle', 'n_tug',
    # The noise floor, measured in THIS pose and grasp -- the eigengap threshold is a ratio
    # against it, so a single figure taken once at the start of the day will not do.
    'quiet_axial_mean_n', 'quiet_axial_std_n', 'quiet_fmag_std_n', 'quiet_tmag_std_nm',
    'wiggle_axial_mean_n', 'wiggle_axial_std_n', 'wiggle_fmag_mean_n', 'wiggle_flat_mean_n',
    # The tug oracle -- TWO probes, because they answer different questions and a bayonet before
    # clocking answers them differently. See the tug block in the config.
    'tug_axial_resist_n', 'tug_axial_min_n', 'tug_lat_resist_n', 'tug_verdict',
    'ref_drift_mm', 'ref_drift_deg', 'ref_force_n',
    't_start_s', 'duration_s']


# ===================================================================================================
# geometry
# ===================================================================================================

def _station_pose(depth_m, lat_m, bias, preload_m, wig6):
    """T_target_conn for one station: depth+lateral in the TARGET frame, then everything else in
    the CONNECTOR's own.

        T = Trans_target(depth, lat_y, lat_z) @ bias @ Trans(preload, 0, 0) @ Delta(wiggle)

    DEPTH IS A TARGET-FRAME TRANSLATION on purpose. It is the insertion depth relative to the mate,
    measured along the socket's axis, and that is what makes "matched depth" well defined -- both
    the beside-socket control (3.1) and the hysteresis comparison (4.1) are statements about two
    poses at the SAME depth, which is only meaningful if depth is measured against the socket
    rather than against the part's own tilted axis.

    EVERYTHING RIGHT OF THE BIAS ACTS IN THE PART'S FRAME, by right-multiplication -- the same
    convention as trajectory.perturb(frame='connector') and uncertain_sampling._axial_ref. A tilted
    connector therefore presses along its OWN axis and rocks about its OWN axes, which is what the
    part physically does; pressing along the target's axis instead would mean the preload had a
    lateral component that grows with the tilt.
    """
    depth = translation_matrix([float(depth_m), float(lat_m[0]), float(lat_m[1])])
    pre = translation_matrix([float(preload_m), 0.0, 0.0])
    w = np.asarray(wig6, dtype=float)
    wig = xyzrpy_to_matrix(w[:3] / 1000.0, np.radians(w[3:]))
    return depth @ bias @ pre @ wig


def _envelope(t, duration, taper_s):
    """Raised-cosine amplitude envelope, 0 -> 1 -> 0 over `taper_s` at each end.

    WITHOUT IT the multisine would step to its full value at t=0 (the per-axis phases are chosen
    for crest factor, so they are NOT all zero and sin(phi) != 0), and a step on the reference is
    an impulse into the contact -- which shows up in the wrench as a transient that has nothing to
    do with the constraint geometry. Tapered samples are FLAGGED rather than dropped, so the
    analysis can exclude them from a covariance window while still seeing them."""
    if taper_s <= 0.0:
        return 1.0
    if t < taper_s:
        return 0.5 * (1.0 - math.cos(math.pi * t / taper_s))
    if t > duration - taper_s:
        return 0.5 * (1.0 - math.cos(math.pi * max(0.0, duration - t) / taper_s))
    return 1.0


def _wig_at(t, amp, frq, pha, env):
    """The 6-vec excitation (mm / deg) at time t."""
    out = np.zeros(6)
    for i in range(6):
        if abs(amp[i]) > 0.0 and frq[i] > 0.0:
            out[i] = env * amp[i] * math.sin(2.0 * math.pi * frq[i] * t + pha[i])
    return out


# ===================================================================================================
# config parsing (everything that can be wrong is caught HERE, before the arm moves)
# ===================================================================================================

def _parse_stations(raw):
    """`stations:` -> a list of dicts, or None on a bad entry."""
    if not raw:
        log.error('wiggle.stations is empty -- there is nothing to sample.')
        return None
    out = []
    for i, e in enumerate(raw):
        if not isinstance(e, dict):
            log.error('stations[%d] must be a mapping, got %r.', i, e)
            return None
        name = str(e.get('name', f'st{i}'))
        label = str(e.get('label', 'insert'))
        try:
            depth_mm = float(e.get('depth_mm', 0.0))
            lat = [float(v) for v in (e.get('lateral_mm') or [0.0, 0.0])]
            off = [float(v) for v in (e.get('offset') or [0.0] * 6)]
        except (TypeError, ValueError):
            log.error('stations[%d] (%s): depth_mm / lateral_mm / offset must be numbers.', i, name)
            return None
        if len(lat) != 2:
            log.error('stations[%d] (%s): lateral_mm is [y, z] in the TARGET frame -- 2 entries, '
                      'got %d.', i, name, len(lat))
            return None
        if len(off) != 6:
            log.error('stations[%d] (%s): offset is [x, y, z (mm), roll, pitch, yaw (deg)] in the '
                      'CONNECTOR frame -- 6 entries, got %d.', i, name, len(off))
            return None
        out.append({
            'name': name,
            'label': label,
            'depth_mm': depth_mm,
            'depth_m': depth_mm / 1000.0,
            'lat_mm': lat,
            'lat_m': [v / 1000.0 for v in lat],
            'offset': off,
            'bias': xyzrpy_to_matrix(np.asarray(off[:3]) / 1000.0, np.radians(off[3:])),
            # A station the part is not expected to touch. Its preload sweep collapses to {0}:
            # pressing 4 mm into thin air records the same free-space rows three times over and
            # costs a minute each, and the number would be a lie in the burst table besides.
            'contact': bool(e.get('contact', True)),
        })
    names = [s['name'] for s in out]
    if len(set(names)) != len(names):
        log.error('station names must be unique (they key the CSV): %s', names)
        return None
    return out


def _parse_wave(w, key, default=0.0):
    """A per-DIM block (`amplitude:` / `frequency_hz:` / `phase_deg:`) -> a 6-vector."""
    blk = w.get(key) or {}
    if not isinstance(blk, dict):
        raise ValueError(f'wiggle.{key} must be a mapping keyed by {list(DIMS)}')
    unknown = [k for k in blk if k not in DIMS]
    if unknown:
        raise ValueError(f'wiggle.{key} has unknown axes {unknown}; expected {list(DIMS)}')
    return [float(blk.get(d, default)) for d in DIMS]


# ===================================================================================================
# the app
# ===================================================================================================

def build_and_run(cfg, robot, camera, args):
    w = cfg.section('wiggle')

    # ---- the held frame + its recorded mate, from the shared catalogue ----------------------
    held_name = cfg.get('held_frame')
    if not held_name:
        log.error('held_frame is required -- name the connector declared in %s.',
                  tool_frames.frames_path(cfg))
        return False
    frames = tool_frames.load_frames(cfg)
    targets = tool_frames.load_targets(cfg)
    if held_name not in frames or held_name not in targets:
        log.error('held_frame %r needs BOTH a frames: and a targets: entry in %s.',
                  held_name, tool_frames.frames_path(cfg))
        return False
    T_tool0_held = frames[held_name]
    T_base_tconn = targets[held_name]          # base_link <- target connector == the mate
    # The mate is the identity row of the connector-wrt-target frame, so the anchoring collapses:
    # T_base_targetobj IS the recorded mate and _station_pose's output places the part directly.
    T_base_targetobj = T_base_tconn
    log.info('Held frame %r + its recorded mate, from %s.', held_name, tool_frames.frames_path(cfg))

    # ---- stations --------------------------------------------------------------------------
    stations = _parse_stations(w.get('stations'))
    if stations is None:
        return False

    # ---- the excitation --------------------------------------------------------------------
    try:
        amp0 = _parse_wave(w, 'amplitude')
        frq = _parse_wave(w, 'frequency_hz')
        pha_deg = _parse_wave(w, 'phase_deg')
    except ValueError as exc:
        log.error('%s', exc)
        return False
    pha = [math.radians(v) for v in pha_deg]
    active = [i for i in range(6) if abs(amp0[i]) > 0.0]
    if not active:
        log.error('every wiggle.amplitude is 0 -- this app has nothing to excite. Set at least '
                  'x_mm / z_mm / pitch_deg.')
        return False
    for i in active:
        if frq[i] <= 0.0:
            log.error('wiggle.amplitude.%s is non-zero but its frequency is 0 -- that is a '
                      'constant OFFSET, not an excitation. Put a constant offset in the '
                      "station's `offset` instead.", DIMS[i])
            return False

    scales = [float(v) for v in (w.get('amplitude_scales') or [1.0])]
    if any(v <= 0.0 for v in scales):
        log.error('wiggle.amplitude_scales must all be > 0 (got %s).', scales)
        return False
    scales = sorted(scales)
    preloads = sorted(float(v) for v in (w.get('preloads_mm') or [0.0]))
    if any(v < 0.0 for v in preloads):
        log.error('wiggle.preloads_mm must all be >= 0 (got %s).', preloads)
        return False
    passes = [str(v).lower() for v in (w.get('passes') or ['in'])]
    if any(p not in ('in', 'out') for p in passes):
        log.error("wiggle.passes entries must be 'in' or 'out' (got %s).", passes)
        return False

    quiet_s = float(w.get('quiet_s', 2.0))
    wiggle_s = float(w.get('wiggle_s', 12.0))
    taper_s = float(w.get('taper_s', 1.0))
    if taper_s * 2.0 >= wiggle_s:
        log.error('wiggle.taper_s %.2f s x2 does not fit inside wiggle_s %.2f s -- the excitation '
                  'would never reach full amplitude.', taper_s, wiggle_s)
        return False

    rate = float(cfg.get_path('compliance.reference_rate_hz', 125.0))
    fmax = max(frq[i] for i in active)
    if rate < 4.0 * fmax:
        log.error('the reference is rebuilt at %.0f Hz but the fastest axis is %.2f Hz -- below '
                  '4x the sampled sine ALIASES into a slower one and the run looks correct while '
                  'exciting a frequency nobody chose.', rate, fmax)
        return False

    # NO TIME DILATION HERE, and this is a deliberate difference from bnc_assembly's engage.
    # That app caps the oscillation's peak speed by stretching the waveform clock, which is the
    # right trade when the oscillation is a means to an end. Here the frequencies ARE the
    # experiment: dilating them would make the effective spectrum a function of amplitude, so the
    # amplitude sweep (3.3) -- whose whole point is to vary the radius at a FIXED excitation
    # spectrum -- would confound the two. The cap is enforced by REFUSING TO RUN instead.
    cap_v = w.get('max_speed_mm_s')
    cap_w = w.get('max_rotation_deg_s')
    big = [amp0[i] * max(scales) for i in range(6)]
    peak_v = max([abs(big[i]) * math.tau * frq[i] for i in range(3) if frq[i] > 0], default=0.0)
    peak_w = max([abs(big[i]) * math.tau * frq[i] for i in range(3, 6) if frq[i] > 0], default=0.0)
    if cap_v and peak_v > float(cap_v):
        log.error('at the largest amplitude scale (%.2f) the wiggle peaks at %.1f mm/s, over the '
                  '%.1f mm/s cap. Lower the amplitude or the frequency -- this app will NOT dilate '
                  'the clock, because that would tie the spectrum to the amplitude and confound '
                  'the amplitude sweep.', max(scales), peak_v, float(cap_v))
        return False
    if cap_w and peak_w > float(cap_w):
        log.error('at the largest amplitude scale (%.2f) the wiggle peaks at %.1f deg/s, over the '
                  '%.1f deg/s cap.', max(scales), peak_w, float(cap_w))
        return False

    # THE ORBIT. Two axes at a simple frequency ratio retrace one closed Lissajous curve forever,
    # so the excitation would sweep a ONE-dimensional path through the rectangle and a rank
    # estimate could not tell that apart from a genuine constraint. Mutually-prime frequencies
    # close the orbit only at 1/gcd, which must be long enough to actually fill the box.
    live_mhz = [int(round(frq[i] * 1000.0)) for i in active]
    g = 0
    for n in live_mhz:
        g = math.gcd(g, n)
    orbit_s = (1000.0 / g) if g else 0.0
    slowest_s = 1.0 / min(frq[i] for i in active)
    if len(active) >= 2 and orbit_s < 3.0 * slowest_s:
        log.error('frequencies %s Hz close their orbit every %.1f s against a slowest single-axis '
                  'period of %.1f s -- the ratio is too simple, so the excitation traces a LINE '
                  'through the box instead of filling it. Pick mutually-prime frequencies.',
                  [frq[i] for i in active], orbit_s, slowest_s)
        return False
    if wiggle_s < orbit_s:
        log.warning('wiggle_s %.1f s is shorter than the %.1f s orbit -- each burst sees only '
                    '%.0f%% of the excitation pattern, so windows from different bursts are not '
                    'comparable. Raise wiggle_s to at least one orbit.',
                    wiggle_s, orbit_s, 100.0 * wiggle_s / orbit_s)

    # ---- the tug oracle --------------------------------------------------------------------
    # TWO probes, because "engaged" is not one question. See the config for why a bayonet before
    # clocking is expected to read FREE on the axial probe at every depth.
    tug = cfg.section('tug') or {}
    tug_on = bool(tug.get('enabled', True))
    tug_axial_mm = float(tug.get('axial_release_mm', 1.0))
    tug_lat_mm = float(tug.get('lateral_mm', 0.5))
    tug_lat_axis = str(tug.get('lateral_axis', 'z')).lower()
    if tug_lat_axis not in ('y', 'z'):
        log.error("tug.lateral_axis must be 'y' or 'z' (the connector's own lateral axes), got %r.",
                  tug_lat_axis)
        return False
    tug_s = float(tug.get('duration_s', 1.5))
    tug_thresh = float(tug.get('resist_n', 1.0))

    # ---- slip defences ---------------------------------------------------------------------
    ref_cfg = cfg.section('reference_check') or {}
    ref_on = bool(ref_cfg.get('enabled', True))
    ref_every = max(1, int(ref_cfg.get('every_n_bursts', 8)))
    ref_hold_s = float(ref_cfg.get('hold_s', 1.0))
    ref_depth_mm = float(ref_cfg.get('depth_mm', -60.0))
    ref_lat_mm = [float(v) for v in (ref_cfg.get('lateral_mm') or [0.0, 60.0])]

    # ---- speeds / compliance ---------------------------------------------------------------
    adm = AdmittanceController(robot.arm, cfg.section('compliance'))
    # TWO guards, with genuinely different jobs.
    #   guard  (force_guard, tight)  -- FREE-SPACE joint hops only. There contact is a surprise.
    #   safety (wiggle_guard, loose) -- every compliant phase: approach, press, hold, wiggle, tug.
    #
    # uncertain_sampling arms its tight guard on the compliant insert because a trip there MEANS
    # something: the connector seated, stop advancing. That reading does not transfer. Here the
    # commanded depth is the independent variable of the experiment, so a tight limit on the
    # approach would hold every contact station at the limit instead of at its configured
    # preload -- and it would do so invisibly, because the logged reference would still say the
    # station was reached. The loose limit is a collision backstop, not a contact detector.
    guard = ForceGuard(robot.arm, cfg.section('force_guard'))
    safety = ForceGuard(robot.arm, cfg.section('wiggle_guard'))
    tare = (lambda: robot.arm.zero_ft(settle=False)) \
        if bool(cfg.get_path('compliance.tare_before', True)) else None

    v_mm_s = float(cfg.get_path('speed.max_cartesian_translation_mm_s', 3.5))
    w_deg_s = float(cfg.get_path('speed.max_cartesian_rotation_deg_s', 5.0))
    rv_mm_s = float(cfg.get_path('speed.retract_translation_mm_s', v_mm_s))
    rw_deg_s = float(cfg.get_path('speed.retract_rotation_deg_s', w_deg_s))
    approach_mm_s = float(w.get('approach_speed_mm_s', v_mm_s))
    standoff_mm = float(w.get('standoff_mm', 20.0))
    min_seg_s = 1.0 / adm.rate
    dt = 1.0 / adm.rate
    decim = max(1, int(w.get('log_decimation', 1)))

    def seg_time(A, B, v=None, ang=None):
        v = v_mm_s if v is None else v
        ang = w_deg_s if ang is None else ang
        lin_m, ang_rad = pose_error(A, B)
        return max((lin_m * 1000.0 / v) if v > 0 else 0.0,
                   (math.degrees(ang_rad) / ang) if ang > 0 else 0.0,
                   min_seg_s)

    def ref_of(st, preload_mm=0.0, wig6=None, depth_mm=None):
        """The tool0 reference for a station, optionally at a different depth."""
        d_m = (st['depth_m'] if depth_mm is None else float(depth_mm) / 1000.0)
        T = _station_pose(d_m, st['lat_m'], st['bias'], preload_mm / 1000.0,
                          np.zeros(6) if wig6 is None else wig6)
        return traj.tool0_at(T_base_targetobj, T, T_tool0_held)

    # ---- build the burst plan --------------------------------------------------------------
    # A pass is: the non-contact anchors first (they are calibration, not a trajectory), then the
    # contact stations in depth order -- ascending on the way in, descending on the way out.
    anchors = [s for s in stations if not s['contact']]
    contacts = sorted([s for s in stations if s['contact']], key=lambda s: s['depth_mm'])
    offsets_note = sorted({tuple(s['offset']) for s in stations})

    plan = []                                   # (pass_dir, station, preload_mm, amp_scale)
    for direction in passes:
        ordered = anchors + (contacts if direction == 'in' else list(reversed(contacts)))
        for st in ordered:
            pres = preloads if st['contact'] else [0.0]
            for pre in pres:
                for sc in scales:
                    plan.append((direction, st, pre, sc))

    n_probe = (1 if tug_axial_mm > 0 else 0) + (1 if tug_lat_mm > 0 else 0)
    burst_s = quiet_s + wiggle_s + (2.0 * tug_s * n_probe if tug_on else 0.0)
    est_s = len(plan) * burst_s * 1.25          # +25% for the moves between stations
    log.info('PLAN: %d bursts = %d stations (%d contact, %d anchor) x %d preloads x %d amplitude '
             'scales x %d pass(es).', len(plan), len(stations), len(contacts), len(anchors),
             len(preloads), len(scales), len(passes))
    log.info('  excitation: %s; orbit closes every %.1f s; %.1f s of wiggle per burst '
             '(%.1f orbits).',
             ', '.join(f'{DIMS[i]} {amp0[i]:+.2f}@{frq[i]:.2f}Hz' for i in active),
             orbit_s, wiggle_s, wiggle_s / orbit_s if orbit_s else 0.0)
    log.info('  amplitude scales %s (x%.0f range), preloads %s mm, offsets %s.',
             scales, max(scales) / min(scales), preloads, [list(o) for o in offsets_note])
    log.info('  ~%.0f s per burst -> ESTIMATED RUN %s (done ~%s).',
             burst_s, _fmt_dur(est_s), _clock(est_s))

    # PRELOAD -> FORCE. This is the number that decides whether the contact is bilateral, and it
    # is set by the SPRING, not by the distance alone: the reference stops preload_mm past
    # wherever the part actually stopped, so the sustained force is (axial stiffness) x preload.
    # The axial direction is the CONNECTOR's +X, and the compliance runs in TOOL0, so the stiffness
    # facing the insertion is u' K u with u the connector's +X in tool0 -- not simply stiffness[0].
    K = np.asarray(cfg.get_path('compliance.stiffness') or [0.0] * 6, dtype=float)
    u = (T_tool0_held[:3, :3] @ np.array([1.0, 0.0, 0.0]))
    k_ax = float(u @ (K[:3] * u))
    log.info('  axial stiffness %.0f N/m -> sustained press %s N at preloads %s mm.',
             k_ax, [round(k_ax * p / 1000.0, 2) for p in preloads], preloads)
    max_amp_x = abs(amp0[0]) * max(scales)
    for p in preloads:
        if p > 0.0 and max_amp_x >= p:
            log.warning('preload %.2f mm is not larger than the largest x amplitude %.2f mm -- the '
                        'spring goes into TENSION at the bottom of each cycle, so contact breaks '
                        'and the reachable set is a half-space again. That is the exact failure '
                        'the preload exists to prevent (3.3).', p, max_amp_x)
    if 0.0 not in preloads:
        log.warning('no zero-preload burst in the sweep -- 3.3 asks for one as the UNILATERAL '
                    'control, and it is what shows the preload is doing anything.')

    # ---- reachability, before anything moves ------------------------------------------------
    q_home = robot.arm.q()
    seed_q = q_home
    for direction, st, pre, _sc in plan:
        for probe, what in ((ref_of(st, pre, depth_mm=st['depth_mm'] - standoff_mm), 'standoff'),
                            (ref_of(st, pre), 'station')):
            if robot.arm.ik(probe, seed_q) is None:
                log.error('IK fails for the %s of station %r at preload %.1f mm -- fix the station '
                          'list before running.', what, st['name'], pre)
                return False
    log.info('  every station and its standoff is reachable.')

    gates = bool(cfg.get('confirm_each_step', False))
    if gates and not robot.arm.dry_run:
        try:
            if input('\nStart the run? Enter to continue (q to abort): ').strip().lower() \
                    in ('q', 'quit', 'n', 'no'):
                log.warning('Aborted before any motion.')
                return False
        except EOFError:
            pass

    # ---- output ----------------------------------------------------------------------------
    out_dir = _run_dir(cfg, w.get('out_dir', 'data/wiggle_sampling'))
    os.makedirs(out_dir, exist_ok=True)
    fsam = open(os.path.join(out_dir, 'samples.csv'), 'w', newline='')
    fbur = open(os.path.join(out_dir, 'bursts.csv'), 'w', newline='')
    wsam, wbur = _csv.writer(fsam), _csv.writer(fbur)
    wsam.writerow(_HEADER)
    wbur.writerow(_BURST_HEADER)
    log.info('Logging to %s', out_dir)

    # ---- the sampling state the log callback reads ------------------------------------------
    # A mutable cell rather than arguments: AdmittanceController calls on_step() with no
    # arguments, once per servo cycle, so whatever the row needs has to be reachable from here.
    ctx = {'burst': 0, 'segment': '', 'taper': 0, 't0': 0.0, 'cmd': np.eye(4),
           'station': '', 'label': '', 'direction': '', 'depth_mm': 0.0,
           'lat': [0.0, 0.0], 'preload_mm': 0.0, 'amp_scale': 0.0}
    stats = {}                                  # segment -> list of (f_axial, |f|, |tau|)
    cnt = [0]

    def observe():
        T_base_tool0 = robot.tool0()
        T_base_conn = T_base_tool0 @ T_tool0_held
        w_base = np.asarray(robot.arm.wrench(), dtype=float)
        # wrench_in is given the flange pose explicitly so the moment is re-referenced off the
        # SAME sample the pose came from -- reading it again inside would pair a wrench with a
        # pose one cycle later.
        w_conn = np.asarray(robot.arm.wrench_in(T_base_conn, T_base_tool0), dtype=float)
        return T_base_tool0, T_base_conn, w_base, w_conn

    def log_cb():
        cnt[0] += 1
        T_base_tool0, T_base_conn, w_base, w_conn = observe()
        # Per-segment running stats, in the CONNECTOR frame: (signed axial, |f|, |tau|, |f_lat|).
        # Axial is signed on purpose -- the tug's whole verdict is a SIGN (see below) -- while the
        # lateral magnitude is what the lateral probe and the wiggle both report against.
        stats.setdefault(ctx['segment'], []).append(
            (float(w_conn[0]), float(np.linalg.norm(w_base[:3])),
             float(np.linalg.norm(w_base[3:])), float(np.linalg.norm(w_conn[1:3]))))
        if cnt[0] % decim:
            return
        achieved = inverse(T_base_tconn) @ T_base_conn          # identity at a perfect mate
        commanded = inverse(T_base_tconn) @ ctx['cmd'] @ T_tool0_held
        wsam.writerow(
            [ctx['burst'], time.time()]
            + _pose_fields_mm(T_base_tool0)
            + _pose_fields_mm(achieved)
            + list(w_base) + list(w_conn)
            + _pose_fields_mm(commanded)
            + [ctx['burst'], ctx['segment'], ctx['taper'],
               round(time.time() - ctx['t0'], 4),
               ctx['station'], ctx['label'], ctx['direction'], ctx['depth_mm'],
               ctx['lat'][0], ctx['lat'][1], ctx['preload_mm'], ctx['amp_scale']])

    def enter(segment, T_cmd, taper=0):
        # t_seg only restarts when the segment CHANGES, so the two halves of a tug (out and back,
        # two ramped() calls under one name) read as one continuous 0 -> 2*tug_s probe.
        if ctx['segment'] != segment:
            ctx['t0'] = time.time()
        ctx['segment'] = segment
        ctx['cmd'] = T_cmd
        ctx['taper'] = taper

    def ramped(A, B, duration, guard, segment):
        """Ramp the reference A -> B, keeping ctx['cmd'] on the INSTANTANEOUS reference.

        AdmittanceController.ramp slerps internally and does not expose where it currently is, so
        a segment driven by a single call would log its ENDPOINT as the commanded pose for every
        sample in it. That column is exactly what 2.1's compliance regression regresses against
        and what 3.3 asks to be logged alongside the achieved pose, and an endpoint is not an
        input -- it would say the reference teleported and then waited. Stepping the reference
        here costs one Python call per servo cycle, which is how the wiggle already runs.

        Returns (status, last_reference)."""
        enter(segment, A)
        n = max(1, int(round(duration * adm.rate)))
        prev = A
        for k in range(1, n + 1):
            cur = slerp_matrix(A, B, k / n)
            ctx['cmd'] = cur
            res = adm.ramp(prev, cur, dt, guard, on_step=log_cb)
            prev = cur
            if res == 'seated':
                return 'seated', cur
        return 'done', prev

    def col(segment, idx):
        rec = stats.get(segment) or []
        return np.asarray([r[idx] for r in rec], dtype=float) if rec else np.zeros(0)

    def summary(segment, idx):
        """(mean, std) of column `idx` over a segment, or (nan, nan) if it never ran."""
        c = col(segment, idx)
        return (float(c.mean()), float(c.std())) if c.size else (float('nan'), float('nan'))

    ok = True
    t_run = time.time()
    burst_no = 0
    cur_ref = None                              # the reference the arm is currently holding
    cur_key = None                              # (direction, station name, preload) it belongs to
    ref_pose0 = None                            # the return-to-reference datum, first visit
    try:
        for direction, st, pre, sc in plan:
            burst_no += 1
            t_burst = time.time()
            stats.clear()                        # the burst row summarises THIS burst only
            amp = [amp0[i] * sc for i in range(6)]
            ctx.update({'burst': burst_no, 'station': st['name'], 'label': st['label'],
                        'direction': direction, 'depth_mm': st['depth_mm'],
                        'lat': st['lat_mm'], 'preload_mm': pre, 'amp_scale': sc})
            log.info('--- burst %d/%d --- %s %s (%s, depth %+.1f mm, lat %s) '
                     'preload %.1f mm, amplitude x%.2f',
                     burst_no, len(plan), direction.upper(), st['name'], st['label'],
                     st['depth_mm'], st['lat_mm'], pre, sc)

            key = (direction, st['name'], pre)
            T_station = ref_of(st, pre)

            # ---- get there --------------------------------------------------------------
            if cur_key is not None and cur_key[0] == direction \
                    and cur_key[1] == st['name'] and cur_key[2] != pre:
                # SAME station, next preload: just change the press. Going out to the standoff
                # and coming back would reset the contact history for no reason.
                ramped(cur_ref, T_station, seg_time(cur_ref, T_station, approach_mm_s),
                       None, 'press')
            elif cur_key is not None and cur_key[0] == direction \
                    and _axially_connected(_by_name(stations, cur_key[1]), st):
                # AXIALLY CONNECTED to the previous station: ramp along the axis without letting
                # go. This is what preserves the contact history that the hysteresis test (4.1)
                # is a statement about; retracting between stations would erase it.
                #
                # Only the SAFETY limit is armed, and only when going DEEPER. Withdrawing from an
                # engaged part starts over the limit, so a guarded withdrawal returns 'seated' on
                # cycle one and never moves -- the trap ForceGuard.disable() documents.
                deeper = st['depth_mm'] > _by_name(stations, cur_key[1])['depth_mm']
                safety.reset()
                res, _ = ramped(cur_ref, T_station, seg_time(cur_ref, T_station, approach_mm_s),
                                safety if deeper else None, 'approach')
                if res == 'seated':
                    log.warning('  SAFETY guard tripped on the way to %s (%s) -- the station was '
                                'not reached, so this burst is at an unknown depth.',
                                st['name'], safety.tripped_by)
            else:
                # A NEW approach: retract from wherever we are, move in free space, come in
                # axially from the standoff.
                if cur_ref is not None:
                    prev_st = _by_name(stations, cur_key[1])
                    T_out = ref_of(prev_st, 0.0, depth_mm=prev_st['depth_mm'] - standoff_mm)
                    ramped(cur_ref, T_out, seg_time(cur_ref, T_out, rv_mm_s, rw_deg_s),
                           None, 'retract')
                    adm.stop()
                T_appr = ref_of(st, pre, depth_mm=st['depth_mm'] - standoff_mm)
                # The free-space hop IS guarded at the low limit -- that is the one move in the
                # burst where contact is a surprise rather than the objective.
                q = robot.arm.ik(T_appr, seed_q)
                if q is None or not _guarded(robot, guard,
                                             lambda _q=q: robot.arm.move_j(_q, label='standoff')):
                    log.warning('  could not reach the standoff for %s -- skipping this burst.',
                                st['name'])
                    cur_ref, cur_key = None, None
                    continue
                seed_q = q
                adm.reset()
                adm.warmup(T_appr, tare_fn=tare)
                # TOUCH, THEN PRESS -- two ramps, the same split uncertain_sampling makes, and for
                # the same reason inverted. There the seating guard STOPS the advance at first
                # contact and the preload is applied after, un-guarded, because at the seat the
                # guard has already tripped. Here reaching the commanded depth IS the measurement,
                # so a 5 N seating limit would end the approach at the first touch and every
                # contact station would be held at 5 N instead of at its configured preload --
                # silently, since the reference would still read as the station. Only the SAFETY
                # limit is armed, and it sits well above the intended press for exactly this
                # reason. Keeping the two ramps separate still buys the labelling: `approach` is
                # free travel and first contact, `press` is the spring loading up.
                T_touch = ref_of(st, 0.0)
                safety.reset()
                res, _ = ramped(T_appr, T_touch, seg_time(T_appr, T_touch, approach_mm_s),
                                safety, 'approach')
                if res == 'seated':
                    log.warning('  SAFETY guard tripped on the approach to %s (%s).',
                                st['name'], safety.tripped_by)
                elif pre > 0.0:
                    ramped(T_touch, T_station, seg_time(T_touch, T_station, approach_mm_s),
                           safety, 'press')
            cur_ref, cur_key = T_station, key

            # ---- quiet: the noise floor, in THIS pose and THIS grasp --------------------
            enter('quiet', T_station)
            safety.reset()
            adm.hold(T_station, quiet_s, safety, on_step=log_cb)

            # ---- wiggle -----------------------------------------------------------------
            enter('wiggle', T_station)
            safety.reset()
            nstep = max(1, int(round(wiggle_s * adm.rate)))
            prev = T_station
            for i in range(1, nstep + 1):
                t = i * dt
                env = _envelope(t, wiggle_s, taper_s)
                ctx['taper'] = 0 if env > 0.999 else 1
                T = _station_pose(st['depth_m'], st['lat_m'], st['bias'], pre / 1000.0,
                                  _wig_at(t, amp, frq, pha, env))
                cur = traj.tool0_at(T_base_targetobj, T, T_tool0_held)
                ctx['cmd'] = cur
                if adm.ramp(prev, cur, dt, safety, on_step=log_cb) == 'seated':
                    log.warning('  SAFETY guard tripped during the wiggle (%s) -- ending the '
                                'burst here.', safety.tripped_by)
                    break
                prev = cur
            ctx['taper'] = 0
            # Back to the un-excited station pose, so the tug starts from a known reference
            # rather than from wherever in the cycle the wiggle happened to end. Its OWN segment:
            # tagging it 'wiggle' would put un-excited rows into the window the rank estimate is
            # computed over, and inflate n_wiggle with samples that carry no excitation.
            ramped(prev, T_station, max(min_seg_s, 0.2), None, 'recentre')

            # ---- tug: the oracle --------------------------------------------------------
            # TWO probes. They are not redundant -- they measure different constraints, and for
            # a bayonet BEFORE the collar is clocked they are expected to disagree.
            ax_resist = ax_min = lat_resist = float('nan')
            verdict = 'skipped'
            if tug_on and st['contact']:
                held = []
                if tug_axial_mm > 0.0:
                    # AXIAL RELEASE. The reference goes tug_axial_mm SHORT of the station --
                    # NOT `preload - distance`, which would still be a press whenever the burst's
                    # preload is the larger of the two. Placing it short of the station guarantees
                    # the spring is in TENSION by exactly k x tug_axial_mm whatever the preload,
                    # so the number means the same thing across the whole preload sweep.
                    T_tug = ref_of(st, -tug_axial_mm)
                    safety.reset()
                    # OUT is un-guarded (a withdrawal must never be blocked by a guard);
                    # the return press back onto the station is guarded like any other.
                    ramped(T_station, T_tug, tug_s, None, 'tug_axial')
                    ramped(T_tug, T_station, tug_s, safety, 'tug_axial')
                    a = col('tug_axial', 0)
                    if a.size:
                        # SIGN IS THE VERDICT. wrench_in reports the external force ON the
                        # connector in its own frame: pressing home makes f_x NEGATIVE (the
                        # socket pushes back along -X), and a withdrawal the socket RESISTS makes
                        # it POSITIVE (the socket pulls the part back in). A part that is merely
                        # sitting in a hole cannot make it positive at all.
                        ax_resist, ax_min = float(a.max()), float(a.min())
                        if ax_resist >= tug_thresh:
                            held.append('axial')
                if tug_lat_mm > 0.0:
                    # LATERAL PROBE, at the burst's own preload -- the part stays pressed home and
                    # is pushed sideways. This is the one that discriminates in-socket from
                    # beside-socket, and it is the reason the axial probe alone is not enough.
                    d = np.zeros(6)
                    d[2 if tug_lat_axis == 'z' else 1] = tug_lat_mm
                    T_lat = ref_of(st, pre, wig6=d)
                    safety.reset()
                    # Both ways guarded: a lateral shove against a blocked direction is the
                    # one probe here that can genuinely jam, and neither leg frees a jam.
                    ramped(T_station, T_lat, tug_s, safety, 'tug_lateral')
                    ramped(T_lat, T_station, tug_s, safety, 'tug_lateral')
                    lat = col('tug_lateral', 3)
                    if lat.size:
                        lat_resist = float(lat.max())
                        if lat_resist >= tug_thresh:
                            held.append('lateral')
                verdict = '+'.join(held) if held else 'free'
                log.info('  tug: axial %+.2f N (min %+.2f), lateral %.2f N -> %s',
                         ax_resist, ax_min, lat_resist, verdict.upper())

            # ---- return-to-reference: the cheap slip check ------------------------------
            drift_mm = drift_deg = ref_f = float('nan')
            if ref_on and burst_no % ref_every == 0:
                T_out = ref_of(st, 0.0, depth_mm=st['depth_mm'] - standoff_mm)
                ramped(cur_ref, T_out, seg_time(cur_ref, T_out, rv_mm_s, rw_deg_s),
                       None, 'retract')
                adm.stop()
                T_ref_pose = traj.tool0_at(
                    T_base_targetobj,
                    _station_pose(ref_depth_mm / 1000.0, [v / 1000.0 for v in ref_lat_mm],
                                  np.eye(4), 0.0, np.zeros(6)),
                    T_tool0_held)
                q = robot.arm.ik(T_ref_pose, seed_q)
                if q is not None and _guarded(
                        robot, guard, lambda _q=q: robot.arm.move_j(_q, label='reference pose')):
                    seed_q = q
                    adm.reset()
                    adm.warmup(T_ref_pose, tare_fn=None)     # NO tare: the reading IS the check
                    enter('datum', T_ref_pose)
                    adm.hold(T_ref_pose, ref_hold_s, guard=None, on_step=log_cb)
                    adm.stop()
                    here = robot.tool0()
                    if ref_pose0 is None:
                        ref_pose0 = here
                        log.info('  reference pose recorded -- later visits are measured against '
                                 'it, and drift there IS accumulated grasp slip.')
                    d_lin, d_ang = pose_error(ref_pose0, here)
                    drift_mm, drift_deg = d_lin * 1000.0, math.degrees(d_ang)
                    ref_f, _ = summary('datum', 1)
                    lvl = log.warning if (drift_mm > 0.5 or drift_deg > 0.3 or ref_f > 2.0) \
                        else log.info
                    lvl('  reference check: drift %.2f mm / %.2f deg, |f| %.2f N at a pose that '
                        'should be free.', drift_mm, drift_deg, ref_f)
                cur_ref, cur_key = None, None

            # ---- the burst row ----------------------------------------------------------
            q_ax_m, q_ax_s = summary('quiet', 0)
            _, q_f_s = summary('quiet', 1)
            _, q_t_s = summary('quiet', 2)
            wg_ax_m, wg_ax_s = summary('wiggle', 0)
            wg_f_m, _ = summary('wiggle', 1)
            wg_lat_m, _ = summary('wiggle', 3)
            n_tug = len(stats.get('tug_axial') or []) + len(stats.get('tug_lateral') or [])
            wbur.writerow(
                [burst_no, st['name'], st['label'], direction, st['depth_mm'],
                 st['lat_mm'][0], st['lat_mm'][1]] + list(st['offset'])
                + [pre, sc] + [round(a, 6) for a in amp] + list(frq)
                + [len(stats.get('quiet') or []), len(stats.get('wiggle') or []), n_tug]
                + [q_ax_m, q_ax_s, q_f_s, q_t_s, wg_ax_m, wg_ax_s, wg_f_m, wg_lat_m]
                + [ax_resist, ax_min, lat_resist, verdict]
                + [drift_mm, drift_deg, ref_f]
                + [round(t_burst - t_run, 2), round(time.time() - t_burst, 2)])
            fsam.flush()
            fbur.flush()
            os.fsync(fsam.fileno())
            os.fsync(fbur.fileno())

            done = time.time() - t_run
            left = done / burst_no * (len(plan) - burst_no)
            log.info('  burst %.0f s | elapsed %s | %d left, ETA %s (done ~%s)',
                     time.time() - t_burst, _fmt_dur(done), len(plan) - burst_no,
                     _fmt_dur(left), _clock(left))
    except Exception:                            # noqa: BLE001
        ok = False
        log.exception('Sampling error:')
    finally:
        # Escape along the LAST station's axis before anything else: a part still in the socket
        # must come out the way it went in, not by a joint move.
        try:
            if cur_ref is not None and cur_key is not None:
                st = _by_name(stations, cur_key[1])
                T_out = ref_of(st, 0.0, depth_mm=st['depth_mm'] - standoff_mm)
                adm.ramp(cur_ref, T_out, seg_time(cur_ref, T_out, rv_mm_s, rw_deg_s), guard=None)
        except Exception:                        # noqa: BLE001
            log.exception('Retract failed -- the part may still be in the socket:')
        robot.arm.servo_stop()
        fsam.close()
        fbur.close()
    if ok:
        robot.arm.move_j(q_home, label='home')
        log.info('Done: %d bursts in %s -> %s', burst_no, _fmt_dur(time.time() - t_run), out_dir)
    return ok


# ===================================================================================================
# helpers
# ===================================================================================================

def _by_name(stations, name):
    for s in stations:
        if s['name'] == name:
            return s
    return None


def _axially_connected(a, b):
    """Can the arm go from station `a` to station `b` by sliding along the insertion axis?

    Only if they share a lateral offset AND a pose offset -- otherwise the move would sweep the
    part sideways through whatever is between them, which near a socket is the socket."""
    if a is None or b is None:
        return False
    return (np.allclose(a['lat_mm'], b['lat_mm'], atol=1e-9)
            and np.allclose(a['offset'], b['offset'], atol=1e-9))


def _pose_fields_mm(T):
    """traj.pose_fields with the translation in MILLIMETRES (matching the _mm column names)."""
    f = traj.pose_fields(T)
    return [f[0] * 1000.0, f[1] * 1000.0, f[2] * 1000.0] + f[3:]


def _fmt_dur(seconds):
    seconds = int(max(0.0, seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f'{h}:{m:02d}:{sec:02d}' if h else f'{m}:{sec:02d}'


def _clock(seconds_from_now):
    return (datetime.now() + timedelta(seconds=max(0.0, seconds_from_now))).strftime('%H:%M:%S')


def _run_dir(cfg, path):
    """One timestamped DIRECTORY per run -- samples.csv and bursts.csv have to stay together or
    the burst labels cannot be joined back onto the samples they describe."""
    path = urconfig.resolve(cfg, path)
    cable = cfg.get('cable')
    if cable:
        path = os.path.join(path, str(cable))
    return os.path.join(path, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')


def main():
    run_app('Wiggle sampling (contact-mode data collection)', 'wiggle_sampling', build_and_run,
            with_gripper=False)


if __name__ == '__main__':
    main()
