"""WIGGLE SAMPLING -- sustained-contact excitation for CONTACT-MODE discovery.

Data collection for analysis/engagement_modes/wiggle_mode_detection.md.  Where
uncertain_sampling builds a POSE map, this builds a MODE dataset: it parks the connector at a
series of labelled stations, presses it against the socket, and rocks it with a multisine for
tens of seconds.  The question is not "where is the part" but "how many directions is it free
to move in" -- a property of the local tangent cone, which needs contact that LASTS, lateral
and rotational excitation (not just the axial advance an insertion makes), and a commanded
motion that is provably full-rank.  None of those fit uncertain_sampling's single-approach
trial, which is why this is its own app.

THE PROTOCOL, per burst:
    approach   ramp axially to the station under admittance, under a LOOSE safety limit only
               (the commanded depth is the independent variable -- a tight seating guard would
               hold every station at the limit instead of at its preload)
    press      preload_mm further along the connector's own +X, making the contact BILATERAL
               (without it the reachable set is a half-space and PCA reports near-full rank no
               matter how constrained the part is; wiggle_mode_detection.md 3.3)
    quiet      hold still: the NOISE FLOOR, measured in the same pose and grasp as the wiggle
               it will be compared against
    wiggle     multisine on the connector's own axes, superimposed on the preloaded reference
    tug        TWO probes -- release along -X, and shove sideways at the preload -- the
               cheapest supervision in the experiment (3.2).  They are not redundant: for a
               bayonet before the collar is clocked they are expected to disagree.

The run sweeps stations x preloads x amplitude scales x passes(in, out) x offsets, with three
station kinds labelled BY CONSTRUCTION (3.1):
    insert   in the socket at a stated depth -- the states under study, label unknown
    free     held well clear -- k = 0 by definition
    beside   the SAME DEPTH as an insert station but laterally displaced into free space.
             Do not skip it: it is what separates a detector that learned the geometry from
             one that merely learned to read x.

ORDER IS PART OF THE EXPERIMENT: within a pass the contact stations are visited in depth order
WITHOUT returning to the standoff between them, so the contact history is continuous --
comparing the same station between the 'in' and 'out' passes is the hysteresis test (4.1), the
strongest evidence for a discrete mode.  Grasp SLIP is the confound that forges every
diagnostic (6); the RETURN-TO-REFERENCE check (revisit one fixed free-space pose every N
bursts and log the drift) is what lets affected bursts be bracketed or dropped.

Output: data/wiggle_sampling/<cable>/run_<stamp>/{samples.csv, bursts.csv}
    samples.csv is a strict SUPERSET of uncertain_sampling's schema (same column names for the
        tool0 pose, the connector-vs-target deviation and both wrenches), plus the COMMANDED
        pose (which the compliance regression of 2.1 needs) and the segment labels.
    bursts.csv is one row per burst: the plan, the noise floor, and the tug verdict -- join it
        to samples.csv on `burst`.

Run:  python -m urlab.apps.wiggle_sampling --config configs/wiggle_sampling.yaml
      python -m urlab.apps.wiggle_sampling --set wiggle.wiggle_s=30
"""

import csv as _csv
import itertools
import math
import os
import time

import numpy as np

from .. import behaviors as bt
from .. import log as urlog
from .. import tool_frames
from ..robot import AdmittanceController, ForceGuard
from ..skills import trajectory as traj
from ..skills import wiggle as wigmod
from ..transforms import inverse, pose_error, slerp_matrix, translation_matrix, xyzrpy_to_matrix
from ._common import ask, eta_clock, fmt_dur, guarded, pose_fields_mm, run_dir, tare_fn
from ._runner import run_app

log = urlog.get('wiggle-sampling')

# The six excitation axes, in the order they index every amplitude / frequency / phase vector.
# Same spelling as bnc_assembly and insertion_tester so a wiggle block can be copied between
# them.
DIMS = ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')

_POSE = ('x_mm', 'y_mm', 'z_mm', 'qx', 'qy', 'qz', 'qw', 'yaw_deg', 'pitch_deg', 'roll_deg')
_WRENCH = ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')

_HEADER = (
    # ---- uncertain_sampling's schema, verbatim, so its readers work unchanged --------------
    ['trial', 'timestamp']
    + [f'tool0_base_{s}' for s in _POSE]           # raw tool0 wrt base
    + [f'connector_target_{s}' for s in _POSE]     # ACHIEVED connector deviation from the mate
    + [f'wrench_base_{s}' for s in _WRENCH]        # wrench as recorded (base_link)
    + [f'wrench_connector_{s}' for s in _WRENCH]   # wrench re-expressed in the connector frame
    # ---- what this app adds: the COMMANDED pose (an unmoved axis cannot be told from an
    # undriven one without it) and the segment labels -----------------------------------------
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
    # The noise floor, measured in THIS pose and grasp (the eigengap threshold is a ratio
    # against it).
    'quiet_axial_mean_n', 'quiet_axial_std_n', 'quiet_fmag_std_n', 'quiet_tmag_std_nm',
    'wiggle_axial_mean_n', 'wiggle_axial_std_n', 'wiggle_fmag_mean_n', 'wiggle_flat_mean_n',
    # The tug oracle -- TWO probes; see the tug block in the config.
    'tug_axial_resist_n', 'tug_axial_min_n', 'tug_lat_resist_n', 'tug_verdict',
    'ref_drift_mm', 'ref_drift_deg', 'ref_force_n',
    't_start_s', 'duration_s']


# =================================================================================================
# geometry
# =================================================================================================

def _station_pose(depth_m, lat_m, bias, preload_m, wig6):
    """T_target_conn for one station:

        T = Trans_target(depth, lat_y, lat_z) @ bias @ Trans(preload, 0, 0) @ Delta(wiggle)

    DEPTH IS A TARGET-FRAME TRANSLATION on purpose: it is the insertion depth relative to the
    mate, measured along the socket's axis, which is what makes "matched depth" well defined
    for both the beside-socket control and the hysteresis comparison.  EVERYTHING RIGHT OF THE
    BIAS ACTS IN THE PART'S OWN FRAME (right-multiplication, the same convention as
    trajectory.perturb(frame='connector')): a tilted connector presses along its OWN axis and
    rocks about its OWN axes, as the part physically does."""
    depth = translation_matrix([float(depth_m), float(lat_m[0]), float(lat_m[1])])
    pre = translation_matrix([float(preload_m), 0.0, 0.0])
    w = np.asarray(wig6, dtype=float)
    wig = xyzrpy_to_matrix(w[:3] / 1000.0, np.radians(w[3:]))
    return depth @ bias @ pre @ wig


def _station_dict(name, label, depth_mm, lat_mm, off6, contact):
    return {
        'name': name,
        'label': label,
        'depth_mm': depth_mm,
        'depth_m': depth_mm / 1000.0,
        'lat_mm': lat_mm,
        'lat_m': [v / 1000.0 for v in lat_mm],
        'offset': off6,
        'bias': xyzrpy_to_matrix(np.asarray(off6[:3]) / 1000.0, np.radians(off6[3:])),
        'contact': contact,
    }


def _by_name(stations, name):
    for s in stations:
        if s['name'] == name:
            return s
    return None


def _axially_connected(a, b):
    """Can the arm slide from station `a` to `b` along the insertion axis?  Only if they share
    a lateral offset AND a pose offset -- otherwise the move would sweep the part sideways
    through whatever is between them, which near a socket is the socket."""
    if a is None or b is None:
        return False
    return (np.allclose(a['lat_mm'], b['lat_mm'], atol=1e-9)
            and np.allclose(a['offset'], b['offset'], atol=1e-9))


# =================================================================================================
# config parsing (everything that can be wrong is caught HERE, before the arm moves)
# =================================================================================================

def _parse_stations(raw, required=True):
    """`stations:` -> a list of dicts, or None on a bad entry. `required=False` (used when a
    station_grid will generate more) turns an empty list into [] instead of an error."""
    if not raw:
        if not required:
            return []
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
            log.error('stations[%d] (%s): depth_mm / lateral_mm / offset must be numbers.',
                      i, name)
            return None
        if len(lat) != 2:
            log.error('stations[%d] (%s): lateral_mm is [y, z] in the TARGET frame -- 2 '
                      'entries, got %d.', i, name, len(lat))
            return None
        if len(off) != 6:
            log.error('stations[%d] (%s): offset is [x, y, z (mm), roll, pitch, yaw (deg)] in '
                      'the CONNECTOR frame -- 6 entries, got %d.', i, name, len(off))
            return None
        # `contact: false` marks a station the part is not expected to touch; its preload
        # sweep collapses to {0} (pressing into thin air just repeats free-space rows).
        out.append(_station_dict(name, label, depth_mm, lat, off,
                                 bool(e.get('contact', True))))
    names = [s['name'] for s in out]
    if len(set(names)) != len(names):
        log.error('station names must be unique (they key the CSV): %s', names)
        return None
    return out


def _grid_stations(g):
    """`wiggle.station_grid:` -> generated stations, uncertain_sampling-style.

    The user gives RANGES and RESOLUTIONS and the app samples all of them:

        depth_mm / lateral_y_mm / lateral_z_mm:  {lower, upper, resolution}   (TARGET frame)
        offset: {lower: [6], upper: [6], resolution: [6]}                     (CONNECTOR frame)
        mode: grid | random     grid = the full Cartesian product, count DERIVED (a degenerate
                                axis, lower == upper, contributes one value and costs nothing);
                                random = `num_stations` uniform draws inside the box.

    Endpoints are hit exactly (traj._axis_values nudges the step).  Generated stations are
    named `<name_prefix><i>` and carry one shared label/contact flag; hand-written stations:
    entries (the free/beside anchors the analysis NEEDS) merge alongside them untouched."""
    def axis1(key):
        blk = g.get(key) or {}
        try:
            lo = float(blk.get('lower', 0.0))
            hi = float(blk.get('upper', 0.0))
            res = float(blk.get('resolution', 0.0))
        except (TypeError, ValueError):
            log.error('station_grid.%s: lower/upper/resolution must be numbers, got %r.',
                      key, blk)
            return None
        if hi < lo:
            log.error('station_grid.%s: upper %.3f < lower %.3f.', key, hi, lo)
            return None
        try:
            return traj._axis_values(lo, hi, res)
        except ValueError as exc:
            log.error('station_grid.%s: %s', key, exc)
            return None

    axes = [axis1(k) for k in ('depth_mm', 'lateral_y_mm', 'lateral_z_mm')]
    if any(a is None for a in axes):
        return None
    off = g.get('offset') or {}
    try:
        off_lo = [float(v) for v in (off.get('lower') or [0.0] * 6)]
        off_hi = [float(v) for v in (off.get('upper') or [0.0] * 6)]
        off_res = [float(v) for v in (off.get('resolution') or [0.0] * 6)]
    except (TypeError, ValueError):
        log.error('station_grid.offset: lower/upper/resolution must be numeric lists.')
        return None
    if not (len(off_lo) == len(off_hi) == len(off_res) == 6):
        log.error('station_grid.offset: lower/upper/resolution are 6-vectors '
                  '[x, y, z (mm), roll, pitch, yaw (deg)] in the CONNECTOR frame.')
        return None
    if any(h < l for l, h in zip(off_lo, off_hi)):
        log.error('station_grid.offset: upper < lower on DOF %s.',
                  [i for i, (l, h) in enumerate(zip(off_lo, off_hi)) if h < l])
        return None
    try:
        off_axes = [traj._axis_values(off_lo[i], off_hi[i], off_res[i]) for i in range(6)]
    except ValueError as exc:
        log.error('station_grid.offset: %s', exc)
        return None

    mode = str(g.get('mode', 'grid')).lower()
    if mode not in ('grid', 'random'):
        log.error("station_grid.mode %r must be 'grid' or 'random'.", g.get('mode'))
        return None
    if mode == 'grid':
        combos = list(itertools.product(*axes, *off_axes))
    else:
        n = int(g.get('num_stations', 20))
        if n <= 0:
            log.error('station_grid.num_stations must be > 0 in random mode (got %d).', n)
            return None
        seed = int(g.get('random_seed', 0))
        rng = np.random.default_rng(seed if seed > 0 else None)
        lows = [a[0] for a in axes] + off_lo
        highs = [a[-1] for a in axes] + off_hi
        combos = [tuple(rng.uniform(lo, hi) if hi > lo else lo
                        for lo, hi in zip(lows, highs)) for _ in range(n)]

    label = str(g.get('label', 'insert'))
    contact = bool(g.get('contact', True))
    prefix = str(g.get('name_prefix', 'g'))
    return [_station_dict(f'{prefix}{i:03d}', label, float(c[0]),
                          [float(c[1]), float(c[2])], [float(v) for v in c[3:9]], contact)
            for i, c in enumerate(combos, start=1)]


def _parse_wave(w, key, default=0.0):
    """A per-DIM block (`amplitude:` / `frequency_hz:` / `phase_deg:`) -> a 6-vector."""
    blk = w.get(key) or {}
    if not isinstance(blk, dict):
        raise ValueError(f'wiggle.{key} must be a mapping keyed by {list(DIMS)}')
    unknown = [k for k in blk if k not in DIMS]
    if unknown:
        raise ValueError(f'wiggle.{key} has unknown axes {unknown}; expected {list(DIMS)}')
    return [float(blk.get(d, default)) for d in DIMS]


def _merge_stations(w):
    """wiggle.stations + wiggle.station_grid -> one station list, or None on any error."""
    gridcfg = w.get('station_grid') or {}
    stations = _parse_stations(w.get('stations'), required=not gridcfg)
    if stations is None:
        return None
    if gridcfg:
        gen = _grid_stations(gridcfg)
        if gen is None:
            return None
        clash = {st['name'] for st in stations} & {st['name'] for st in gen}
        if clash:
            log.error('station_grid names collide with stations: %s -- set '
                      'station_grid.name_prefix.', sorted(clash))
            return None
        log.info('station_grid (%s): %d generated station(s) joined with %d hand-written.',
                 str(gridcfg.get('mode', 'grid')), len(gen), len(stations))
        stations = stations + gen
    if not stations:
        log.error('no stations at all -- give wiggle.stations and/or wiggle.station_grid.')
        return None
    return stations


def _parse_excitation(w):
    """amplitude / frequency / phase blocks -> (amp0, frq, pha_deg, active) or None."""
    try:
        amp0 = _parse_wave(w, 'amplitude')
        frq = _parse_wave(w, 'frequency_hz')
        pha_deg = _parse_wave(w, 'phase_deg')
    except ValueError as exc:
        log.error('%s', exc)
        return None
    active = [i for i in range(6) if abs(amp0[i]) > 0.0]
    if not active:
        log.error('every wiggle.amplitude is 0 -- this app has nothing to excite. Set at '
                  'least x_mm / z_mm / pitch_deg.')
        return None
    for i in active:
        if frq[i] <= 0.0:
            log.error('wiggle.amplitude.%s is non-zero but its frequency is 0 -- that is a '
                      'constant OFFSET, not an excitation. Put a constant offset in the '
                      "station's `offset` instead.", DIMS[i])
            return None
    return amp0, frq, pha_deg, active


def _parse_sweep(w):
    """amplitude_scales / preloads_mm / passes -> (scales, preloads, passes) or None."""
    scales = [float(v) for v in (w.get('amplitude_scales') or [1.0])]
    if any(v <= 0.0 for v in scales):
        log.error('wiggle.amplitude_scales must all be > 0 (got %s).', scales)
        return None
    preloads = sorted(float(v) for v in (w.get('preloads_mm') or [0.0]))
    if any(v < 0.0 for v in preloads):
        log.error('wiggle.preloads_mm must all be >= 0 (got %s).', preloads)
        return None
    passes = [str(v).lower() for v in (w.get('passes') or ['in'])]
    if any(p not in ('in', 'out') for p in passes):
        log.error("wiggle.passes entries must be 'in' or 'out' (got %s).", passes)
        return None
    return sorted(scales), preloads, passes


def _parse_tug(cfg):
    """`tug:` -> a dict of probe parameters, or None on a bad axis."""
    tug = cfg.section('tug') or {}
    lat_axis = str(tug.get('lateral_axis', 'z')).lower()
    if lat_axis not in ('y', 'z'):
        log.error("tug.lateral_axis must be 'y' or 'z' (the connector's own lateral axes), "
                  'got %r.', tug.get('lateral_axis'))
        return None
    return {'on': bool(tug.get('enabled', True)),
            'axial_mm': float(tug.get('axial_release_mm', 1.0)),
            'lat_mm': float(tug.get('lateral_mm', 0.5)),
            'lat_axis': lat_axis,
            'duration_s': float(tug.get('duration_s', 1.5)),
            'resist_n': float(tug.get('resist_n', 1.0))}


def _parse_reference_check(cfg):
    ref = cfg.section('reference_check') or {}
    return {'on': bool(ref.get('enabled', True)),
            'every': max(1, int(ref.get('every_n_bursts', 8))),
            'hold_s': float(ref.get('hold_s', 1.0)),
            'depth_mm': float(ref.get('depth_mm', -60.0)),
            'lat_mm': [float(v) for v in (ref.get('lateral_mm') or [0.0, 60.0])]}


def _build_plan(stations, preloads, scales, passes):
    """The burst plan: per pass, the non-contact anchors first (they are calibration, not a
    trajectory), then the contact stations in depth order -- ascending on the way in,
    descending on the way out.  Returns (plan, anchors, contacts)."""
    anchors = [s for s in stations if not s['contact']]
    contacts = sorted([s for s in stations if s['contact']], key=lambda s: s['depth_mm'])
    plan = []                                   # (pass_dir, station, preload_mm, amp_scale)
    for direction in passes:
        ordered = anchors + (contacts if direction == 'in' else list(reversed(contacts)))
        for st in ordered:
            for pre in (preloads if st['contact'] else [0.0]):
                for sc in scales:
                    plan.append((direction, st, pre, sc))
    return plan, anchors, contacts


# =================================================================================================
# the run
# =================================================================================================

class _BurstRunner:
    """All the state one sampling run threads through its bursts: the frames, the compliant
    controller and both guards, the CSV writers, the per-cycle logging context, and the motion
    primitives each burst's behavior tree is built from.

    TWO GUARDS, with genuinely different jobs: `guard` (force_guard, tight) covers FREE-SPACE
    joint hops only, where contact is a surprise; `safety` (wiggle_guard, loose) covers every
    compliant phase.  A tight limit on the approach would hold every contact station at the
    limit instead of at its configured preload -- invisibly, since the logged reference would
    still say the station was reached.  The loose limit is a collision backstop, not a contact
    detector.  Guards are only armed when moving DEEPER: a withdrawal from an engaged part
    starts over the limit, so a guarded withdrawal would return 'seated' on cycle one and
    never move."""

    def __init__(self, cfg, robot, w, frames_pack, wg, tug, refchk, stations, out_dir):
        self.robot = robot
        self.stations = stations
        self.T_tool0_held, self.T_base_tconn, self.T_base_targetobj = frames_pack
        self.wg = wg
        self.tug_cfg = tug
        self.ref_cfg = refchk

        self.adm = AdmittanceController(robot.arm, cfg.section('compliance'))
        self.guard = ForceGuard(robot.arm, cfg.section('force_guard'))
        self.safety = ForceGuard(robot.arm, cfg.section('wiggle_guard'))
        self.tare = tare_fn(robot, cfg.section('compliance'))

        self.v_mm_s = float(cfg.get_path('speed.max_cartesian_translation_mm_s', 3.5))
        self.w_deg_s = float(cfg.get_path('speed.max_cartesian_rotation_deg_s', 5.0))
        self.rv_mm_s = float(cfg.get_path('speed.retract_translation_mm_s', self.v_mm_s))
        self.rw_deg_s = float(cfg.get_path('speed.retract_rotation_deg_s', self.w_deg_s))
        self.approach_mm_s = float(w.get('approach_speed_mm_s', self.v_mm_s))
        self.standoff_mm = float(w.get('standoff_mm', 20.0))
        self.quiet_s = float(w.get('quiet_s', 2.0))
        self.wiggle_s = float(w.get('wiggle_s', 12.0))
        self.min_seg_s = 1.0 / self.adm.rate
        self.dt = 1.0 / self.adm.rate
        self.decim = max(1, int(w.get('log_decimation', 1)))

        self.out_dir = out_dir
        self._fsam = self._fbur = self._wsam = self._wbur = None

        # The sampling context the per-cycle log callback reads.  A mutable dict rather than
        # arguments: AdmittanceController calls on_step() with no arguments, once per servo
        # cycle, so whatever the row needs has to be reachable from here.
        self.ctx = {'burst': 0, 'segment': '', 'taper': 0, 't0': 0.0, 'cmd': np.eye(4),
                    'station': '', 'label': '', 'direction': '', 'depth_mm': 0.0,
                    'lat': [0.0, 0.0], 'preload_mm': 0.0, 'amp_scale': 0.0}
        self.stats = {}                         # segment -> list of (f_ax, |f|, |tau|, |f_lat|)
        self._cnt = 0

        self.seed_q = robot.arm.q()
        self.q_home = self.seed_q
        self.burst_no = 0
        self.t_run = time.time()
        self.plan_len = 0
        self.cur_ref = None                     # the reference the arm is currently holding
        self.cur_key = None                     # (direction, station name, preload) it belongs to
        self.ref_pose0 = None                   # the return-to-reference datum, first visit

        # per-burst fields, reset by begin_burst
        self.direction = self.st = None
        self.pre = self.sc = 0.0
        self.T_station = None
        self.t_burst = 0.0
        self.tug_out = (float('nan'), float('nan'), float('nan'), 'skipped')
        self.ref_out = (float('nan'), float('nan'), float('nan'))

    def open_logs(self):
        """Create the run directory + CSVs -- called only once the pre-run checks pass, so an
        aborted run leaves no empty directory behind."""
        os.makedirs(self.out_dir, exist_ok=True)
        self._fsam = open(os.path.join(self.out_dir, 'samples.csv'), 'w', newline='')
        self._fbur = open(os.path.join(self.out_dir, 'bursts.csv'), 'w', newline='')
        self._wsam, self._wbur = _csv.writer(self._fsam), _csv.writer(self._fbur)
        self._wsam.writerow(_HEADER)
        self._wbur.writerow(_BURST_HEADER)

    def close(self):
        for fh in (self._fsam, self._fbur):
            if fh is not None:
                fh.close()

    # ---- timing / geometry -------------------------------------------------------------------
    def seg_time(self, A, B, v=None, ang=None):
        v = self.v_mm_s if v is None else v
        ang = self.w_deg_s if ang is None else ang
        lin_m, ang_rad = pose_error(A, B)
        return max((lin_m * 1000.0 / v) if v > 0 else 0.0,
                   (math.degrees(ang_rad) / ang) if ang > 0 else 0.0,
                   self.min_seg_s)

    def ref_of(self, st, preload_mm=0.0, wig6=None, depth_mm=None):
        """The tool0 reference for a station, optionally at a different depth."""
        d_m = (st['depth_m'] if depth_mm is None else float(depth_mm) / 1000.0)
        T = _station_pose(d_m, st['lat_m'], st['bias'], preload_mm / 1000.0,
                          np.zeros(6) if wig6 is None else wig6)
        return traj.tool0_at(self.T_base_targetobj, T, self.T_tool0_held)

    # ---- per-cycle logging -------------------------------------------------------------------
    def _observe(self):
        T_base_tool0 = self.robot.tool0()
        T_base_conn = T_base_tool0 @ self.T_tool0_held
        w_base = np.asarray(self.robot.arm.wrench(), dtype=float)
        # The flange pose is handed over so the moment is re-referenced off the SAME sample
        # the pose came from.
        w_conn = np.asarray(self.robot.arm.wrench_in(T_base_conn, T_base_tool0), dtype=float)
        return T_base_tool0, T_base_conn, w_base, w_conn

    def log_cb(self):
        self._cnt += 1
        T_base_tool0, T_base_conn, w_base, w_conn = self._observe()
        # Per-segment running stats, in the CONNECTOR frame: (signed axial, |f|, |tau|,
        # |f_lat|).  Axial is SIGNED on purpose -- the tug's verdict is a sign.
        self.stats.setdefault(self.ctx['segment'], []).append(
            (float(w_conn[0]), float(np.linalg.norm(w_base[:3])),
             float(np.linalg.norm(w_base[3:])), float(np.linalg.norm(w_conn[1:3]))))
        if self._cnt % self.decim:
            return
        achieved = inverse(self.T_base_tconn) @ T_base_conn    # identity at a perfect mate
        commanded = inverse(self.T_base_tconn) @ self.ctx['cmd'] @ self.T_tool0_held
        self._wsam.writerow(
            [self.ctx['burst'], time.time()]
            + pose_fields_mm(T_base_tool0)
            + pose_fields_mm(achieved)
            + list(w_base) + list(w_conn)
            + pose_fields_mm(commanded)
            + [self.ctx['burst'], self.ctx['segment'], self.ctx['taper'],
               round(time.time() - self.ctx['t0'], 4),
               self.ctx['station'], self.ctx['label'], self.ctx['direction'],
               self.ctx['depth_mm'], self.ctx['lat'][0], self.ctx['lat'][1],
               self.ctx['preload_mm'], self.ctx['amp_scale']])

    def enter(self, segment, T_cmd, taper=0):
        # t_seg only restarts when the segment CHANGES, so the two halves of a tug (out and
        # back, two ramped() calls under one name) read as one continuous probe.
        if self.ctx['segment'] != segment:
            self.ctx['t0'] = time.time()
        self.ctx['segment'] = segment
        self.ctx['cmd'] = T_cmd
        self.ctx['taper'] = taper

    def ramped(self, A, B, duration, guard, segment):
        """Ramp the reference A -> B, keeping ctx['cmd'] on the INSTANTANEOUS reference.

        AdmittanceController.ramp slerps internally without exposing where it currently is, so
        a single call would log its ENDPOINT as the commanded pose for every sample -- and the
        commanded column is exactly what the compliance regression regresses against.
        Stepping the reference here costs one Python call per servo cycle, which is how the
        wiggle already runs.  Returns (status, last_reference)."""
        self.enter(segment, A)
        n = max(1, int(round(duration * self.adm.rate)))
        prev = A
        for k in range(1, n + 1):
            cur = slerp_matrix(A, B, k / n)
            self.ctx['cmd'] = cur
            res = self.adm.ramp(prev, cur, self.dt, guard, on_step=self.log_cb)
            prev = cur
            if res == 'seated':
                return 'seated', cur
        return 'done', prev

    def col(self, segment, idx):
        rec = self.stats.get(segment) or []
        return np.asarray([r[idx] for r in rec], dtype=float) if rec else np.zeros(0)

    def summary(self, segment, idx):
        """(mean, std) of column `idx` over a segment, or (nan, nan) if it never ran."""
        c = self.col(segment, idx)
        return (float(c.mean()), float(c.std())) if c.size else (float('nan'), float('nan'))

    # ---- burst phases (the behavior-tree leaves) ---------------------------------------------
    def begin_burst(self, direction, st, pre, sc):
        self.burst_no += 1
        self.t_burst = time.time()
        self.stats.clear()                       # the burst row summarises THIS burst only
        self.direction, self.st, self.pre, self.sc = direction, st, pre, sc
        self.T_station = self.ref_of(st, pre)
        self.tug_out = (float('nan'), float('nan'), float('nan'), 'skipped')
        self.ref_out = (float('nan'), float('nan'), float('nan'))
        self.ctx.update({'burst': self.burst_no, 'station': st['name'],
                         'label': st['label'], 'direction': direction,
                         'depth_mm': st['depth_mm'], 'lat': st['lat_mm'],
                         'preload_mm': pre, 'amp_scale': sc})
        log.info('--- burst %d/%d --- %s %s (%s, depth %+.1f mm, lat %s) '
                 'preload %.1f mm, amplitude x%.2f',
                 self.burst_no, self.plan_len, direction.upper(), st['name'], st['label'],
                 st['depth_mm'], st['lat_mm'], pre, sc)

    def goto_station(self):
        """Reach this burst's station, preserving contact history where the plan allows:
        same station at a new preload just changes the press; an axially-connected neighbour
        is ramped to along the axis WITHOUT letting go (this is what the hysteresis test is a
        statement about); anything else is a fresh standoff approach.  Returns False only when
        the standoff is unreachable (the burst is skipped)."""
        st, pre, direction = self.st, self.pre, self.direction
        key = (direction, st['name'], pre)
        T_station = self.T_station
        if self.cur_key is not None and self.cur_key[0] == direction \
                and self.cur_key[1] == st['name'] and self.cur_key[2] != pre:
            # SAME station, next preload: just change the press.
            self.ramped(self.cur_ref, T_station,
                        self.seg_time(self.cur_ref, T_station, self.approach_mm_s),
                        None, 'press')
        elif self.cur_key is not None and self.cur_key[0] == direction \
                and _axially_connected(_by_name(self.stations, self.cur_key[1]), st):
            # Ramp along the axis; only the SAFETY limit, and only when going DEEPER.
            deeper = st['depth_mm'] > _by_name(self.stations, self.cur_key[1])['depth_mm']
            self.safety.reset()
            res, _ = self.ramped(self.cur_ref, T_station,
                                 self.seg_time(self.cur_ref, T_station, self.approach_mm_s),
                                 self.safety if deeper else None, 'approach')
            if res == 'seated':
                log.warning('  SAFETY guard tripped on the way to %s (%s) -- the station was '
                            'not reached, so this burst is at an unknown depth.',
                            st['name'], self.safety.tripped_by)
        else:
            # A NEW approach: retract from wherever we are, hop in free space, come in
            # axially from the standoff.
            if self.cur_ref is not None:
                prev_st = _by_name(self.stations, self.cur_key[1])
                T_out = self.ref_of(prev_st, 0.0,
                                    depth_mm=prev_st['depth_mm'] - self.standoff_mm)
                self.ramped(self.cur_ref, T_out,
                            self.seg_time(self.cur_ref, T_out, self.rv_mm_s, self.rw_deg_s),
                            None, 'retract')
                self.adm.stop()
            T_appr = self.ref_of(st, pre, depth_mm=st['depth_mm'] - self.standoff_mm)
            # The free-space hop IS guarded at the tight limit -- the one move in the burst
            # where contact is a surprise rather than the objective.
            q = self.robot.arm.ik(T_appr, self.seed_q)
            if q is None or not guarded(self.robot, self.guard,
                                        lambda _q=q: self.robot.arm.move_j(
                                            _q, label='standoff')):
                log.warning('  could not reach the standoff for %s -- skipping this burst.',
                            st['name'])
                self.cur_ref, self.cur_key = None, None
                return False
            self.seed_q = q
            self.adm.reset()
            self.adm.warmup(T_appr, tare_fn=self.tare)
            # TOUCH, THEN PRESS -- two ramps, so `approach` labels free travel + first contact
            # and `press` labels the spring loading up.  Both under the loose safety limit
            # only: reaching the commanded depth IS the measurement here.
            T_touch = self.ref_of(st, 0.0)
            self.safety.reset()
            res, _ = self.ramped(T_appr, T_touch,
                                 self.seg_time(T_appr, T_touch, self.approach_mm_s),
                                 self.safety, 'approach')
            if res == 'seated':
                log.warning('  SAFETY guard tripped on the approach to %s (%s).',
                            st['name'], self.safety.tripped_by)
            elif pre > 0.0:
                self.ramped(T_touch, T_station,
                            self.seg_time(T_touch, T_station, self.approach_mm_s),
                            self.safety, 'press')
        self.cur_ref, self.cur_key = T_station, key
        return True

    def quiet_hold(self):
        """The noise floor, in THIS pose and THIS grasp."""
        self.enter('quiet', self.T_station)
        self.safety.reset()
        self.adm.hold(self.T_station, self.quiet_s, self.safety, on_step=self.log_cb)

    def do_wiggle(self):
        """The multisine, anchored at this station + preload, then a recentre back to the
        un-excited station pose so the tug starts from a known reference.  The recentre has
        its OWN segment: tagging it 'wiggle' would put un-excited rows into the rank window."""
        st, pre, sc = self.st, self.pre, self.sc
        self.enter('wiggle', self.T_station)
        self.safety.reset()

        def anchor(delta, _st=st, _pre=pre):
            T = _station_pose(_st['depth_m'], _st['lat_m'], _st['bias'], _pre / 1000.0,
                              np.zeros(6)) @ delta
            return traj.tool0_at(self.T_base_targetobj, T, self.T_tool0_held)

        def on_ref(cur, _t, tapered):
            self.ctx['cmd'] = cur
            self.ctx['taper'] = 1 if tapered else 0

        res_w, _ = wigmod.run(self.adm, self.wg, anchor, self.wiggle_s, self.dt,
                              guard=self.safety, on_step=self.log_cb, scale=sc, on_ref=on_ref)
        if res_w == 'seated':
            log.warning('  SAFETY guard tripped during the wiggle (%s) -- ending the burst '
                        'here.', self.safety.tripped_by)
        self.ctx['taper'] = 0
        prev = anchor(self.wg.delta(self.wiggle_s, self.wiggle_s, sc))
        self.ramped(prev, self.T_station, max(self.min_seg_s, 0.2), None, 'recentre')

    def do_tug(self):
        """The oracle: an axial release (does the socket PULL the part back in?) and a lateral
        shove at the preload (in-socket vs beside-socket).  wrench_in reports the external
        force ON the connector in its own frame, so pressing home makes f_x NEGATIVE and a
        resisted withdrawal makes it POSITIVE -- the SIGN is the verdict."""
        t = self.tug_cfg
        st, pre = self.st, self.pre
        ax_resist = ax_min = lat_resist = float('nan')
        held = []
        if not (t['on'] and st['contact']):
            self.tug_out = (ax_resist, ax_min, lat_resist, 'skipped')
            return
        if t['axial_mm'] > 0.0:
            # The reference goes axial_mm SHORT of the station (not `preload - distance`,
            # which would still be a press at high preloads), so the spring is in TENSION by
            # exactly k x axial_mm whatever the preload.
            T_tug = self.ref_of(st, -t['axial_mm'])
            self.safety.reset()
            # OUT is un-guarded (a withdrawal must never be blocked by a guard); the return
            # press back onto the station is guarded like any other.
            self.ramped(self.T_station, T_tug, t['duration_s'], None, 'tug_axial')
            self.ramped(T_tug, self.T_station, t['duration_s'], self.safety, 'tug_axial')
            a = self.col('tug_axial', 0)
            if a.size:
                ax_resist, ax_min = float(a.max()), float(a.min())
                if ax_resist >= t['resist_n']:
                    held.append('axial')
        if t['lat_mm'] > 0.0:
            d = np.zeros(6)
            d[2 if t['lat_axis'] == 'z' else 1] = t['lat_mm']
            T_lat = self.ref_of(st, pre, wig6=d)
            self.safety.reset()
            # Both ways guarded: a lateral shove against a blocked direction is the one probe
            # here that can genuinely jam, and neither leg frees a jam.
            self.ramped(self.T_station, T_lat, t['duration_s'], self.safety, 'tug_lateral')
            self.ramped(T_lat, self.T_station, t['duration_s'], self.safety, 'tug_lateral')
            lat = self.col('tug_lateral', 3)
            if lat.size:
                lat_resist = float(lat.max())
                if lat_resist >= t['resist_n']:
                    held.append('lateral')
        verdict = '+'.join(held) if held else 'free'
        log.info('  tug: axial %+.2f N (min %+.2f), lateral %.2f N -> %s',
                 ax_resist, ax_min, lat_resist, verdict.upper())
        self.tug_out = (ax_resist, ax_min, lat_resist, verdict)

    def reference_check(self):
        """The cheap slip defence: every N bursts, revisit one fixed free-space pose and log
        the drift against its first visit -- growth there IS accumulated grasp slip."""
        r = self.ref_cfg
        drift_mm = drift_deg = ref_f = float('nan')
        if not (r['on'] and self.burst_no % r['every'] == 0):
            self.ref_out = (drift_mm, drift_deg, ref_f)
            return
        st = self.st
        T_out = self.ref_of(st, 0.0, depth_mm=st['depth_mm'] - self.standoff_mm)
        self.ramped(self.cur_ref, T_out,
                    self.seg_time(self.cur_ref, T_out, self.rv_mm_s, self.rw_deg_s),
                    None, 'retract')
        self.adm.stop()
        T_ref_pose = traj.tool0_at(
            self.T_base_targetobj,
            _station_pose(r['depth_mm'] / 1000.0, [v / 1000.0 for v in r['lat_mm']],
                          np.eye(4), 0.0, np.zeros(6)),
            self.T_tool0_held)
        q = self.robot.arm.ik(T_ref_pose, self.seed_q)
        if q is not None and guarded(self.robot, self.guard,
                                     lambda _q=q: self.robot.arm.move_j(
                                         _q, label='reference pose')):
            self.seed_q = q
            self.adm.reset()
            self.adm.warmup(T_ref_pose, tare_fn=None)      # NO tare: the reading IS the check
            self.enter('datum', T_ref_pose)
            self.adm.hold(T_ref_pose, r['hold_s'], guard=None, on_step=self.log_cb)
            self.adm.stop()
            here = self.robot.tool0()
            if self.ref_pose0 is None:
                self.ref_pose0 = here
                log.info('  reference pose recorded -- later visits are measured against it, '
                         'and drift there IS accumulated grasp slip.')
            d_lin, d_ang = pose_error(self.ref_pose0, here)
            drift_mm, drift_deg = d_lin * 1000.0, math.degrees(d_ang)
            ref_f, _ = self.summary('datum', 1)
            lvl = log.warning if (drift_mm > 0.5 or drift_deg > 0.3 or ref_f > 2.0) \
                else log.info
            lvl('  reference check: drift %.2f mm / %.2f deg, |f| %.2f N at a pose that '
                'should be free.', drift_mm, drift_deg, ref_f)
        self.cur_ref, self.cur_key = None, None
        self.ref_out = (drift_mm, drift_deg, ref_f)

    def finish_burst(self, frq, amp0):
        """The burst row + flush/fsync (a crash must not lose completed bursts), then ETA."""
        st, pre, sc = self.st, self.pre, self.sc
        amp = [amp0[i] * sc for i in range(6)]
        q_ax_m, q_ax_s = self.summary('quiet', 0)
        _, q_f_s = self.summary('quiet', 1)
        _, q_t_s = self.summary('quiet', 2)
        wg_ax_m, wg_ax_s = self.summary('wiggle', 0)
        wg_f_m, _ = self.summary('wiggle', 1)
        wg_lat_m, _ = self.summary('wiggle', 3)
        n_tug = len(self.stats.get('tug_axial') or []) + len(self.stats.get('tug_lateral') or [])
        ax_resist, ax_min, lat_resist, verdict = self.tug_out
        drift_mm, drift_deg, ref_f = self.ref_out
        self._wbur.writerow(
            [self.burst_no, st['name'], st['label'], self.direction, st['depth_mm'],
             st['lat_mm'][0], st['lat_mm'][1]] + list(st['offset'])
            + [pre, sc] + [round(a, 6) for a in amp] + list(frq)
            + [len(self.stats.get('quiet') or []), len(self.stats.get('wiggle') or []), n_tug]
            + [q_ax_m, q_ax_s, q_f_s, q_t_s, wg_ax_m, wg_ax_s, wg_f_m, wg_lat_m]
            + [ax_resist, ax_min, lat_resist, verdict]
            + [drift_mm, drift_deg, ref_f]
            + [round(self.t_burst - self.t_run, 2), round(time.time() - self.t_burst, 2)])
        self._fsam.flush()
        self._fbur.flush()
        os.fsync(self._fsam.fileno())
        os.fsync(self._fbur.fileno())

        done = time.time() - self.t_run
        left = done / self.burst_no * (self.plan_len - self.burst_no)
        log.info('  burst %.0f s | elapsed %s | %d left, ETA %s (done ~%s)',
                 time.time() - self.t_burst, fmt_dur(done), self.plan_len - self.burst_no,
                 fmt_dur(left), eta_clock(left))

    def escape(self):
        """Back out along the LAST station's axis -- a part still in the socket must come out
        the way it went in, not by a joint move."""
        try:
            if self.cur_ref is not None and self.cur_key is not None:
                st = _by_name(self.stations, self.cur_key[1])
                T_out = self.ref_of(st, 0.0, depth_mm=st['depth_mm'] - self.standoff_mm)
                self.adm.ramp(self.cur_ref, T_out,
                              self.seg_time(self.cur_ref, T_out, self.rv_mm_s, self.rw_deg_s),
                              guard=None)
        except Exception:                        # noqa: BLE001
            log.exception('Retract failed -- the part may still be in the socket:')
        self.robot.arm.servo_stop()

    def burst_tree(self, frq, amp0):
        """One burst as a behavior sequence.  Only goto_station can FAIL (unreachable
        standoff), which skips the rest of the burst; every other phase reports through the
        CSVs and the run carries on."""
        return bt.sequence(
            f'burst {self.burst_no}',
            bt.Action('reach the station', self.goto_station),
            bt.Action('quiet (noise floor)', self.quiet_hold),
            bt.Action('wiggle + recentre', self.do_wiggle),
            bt.Action('tug oracle', self.do_tug),
            bt.Action('return-to-reference check', self.reference_check),
            bt.Action('write burst row', lambda: self.finish_burst(frq, amp0)))


# =================================================================================================
# pre-run reporting + checks
# =================================================================================================

def _announce_plan(runner, plan, stations, anchors, contacts, preloads, scales, amp0, frq,
                   active, orbit_s, tug, passes):
    n_probe = (1 if tug['axial_mm'] > 0 else 0) + (1 if tug['lat_mm'] > 0 else 0)
    burst_s = runner.quiet_s + runner.wiggle_s \
        + (2.0 * tug['duration_s'] * n_probe if tug['on'] else 0.0)
    est_s = len(plan) * burst_s * 1.25          # +25% for the moves between stations
    offsets_note = sorted({tuple(s['offset']) for s in stations})
    log.info('PLAN: %d bursts = %d stations (%d contact, %d anchor) x %d preloads x %d '
             'amplitude scales x %d pass(es).', len(plan), len(stations), len(contacts),
             len(anchors), len(preloads), len(scales), len(passes))
    log.info('  excitation: %s; orbit closes every %.1f s; %.1f s of wiggle per burst '
             '(%.1f orbits).',
             ', '.join(f'{DIMS[i]} {amp0[i]:+.2f}@{frq[i]:.2f}Hz' for i in active),
             orbit_s, runner.wiggle_s, runner.wiggle_s / orbit_s if orbit_s else 0.0)
    log.info('  amplitude scales %s (x%.0f range), preloads %s mm, offsets %s.',
             scales, max(scales) / min(scales), preloads, [list(o) for o in offsets_note])
    log.info('  ~%.0f s per burst -> ESTIMATED RUN %s (done ~%s).',
             burst_s, fmt_dur(est_s), eta_clock(est_s))


def _preload_notes(cfg, T_tool0_held, preloads, amp0, scales):
    """PRELOAD -> FORCE: the sustained press is (axial stiffness) x preload, and the axial
    direction is the CONNECTOR's +X while the compliance runs in TOOL0 -- so the stiffness
    facing the insertion is u'Ku with u the connector's +X in tool0, not simply stiffness[0]."""
    K = np.asarray(cfg.get_path('compliance.stiffness') or [0.0] * 6, dtype=float)
    u = (T_tool0_held[:3, :3] @ np.array([1.0, 0.0, 0.0]))
    k_ax = float(u @ (K[:3] * u))
    log.info('  axial stiffness %.0f N/m -> sustained press %s N at preloads %s mm.',
             k_ax, [round(k_ax * p / 1000.0, 2) for p in preloads], preloads)
    max_amp_x = abs(amp0[0]) * max(scales)
    for p in preloads:
        if p > 0.0 and max_amp_x >= p:
            log.warning('preload %.2f mm is not larger than the largest x amplitude %.2f mm '
                        '-- the spring goes into TENSION at the bottom of each cycle, so '
                        'contact breaks and the reachable set is a half-space again (3.3).',
                        p, max_amp_x)
    if 0.0 not in preloads:
        log.warning('no zero-preload burst in the sweep -- 3.3 asks for one as the '
                    'UNILATERAL control, and it is what shows the preload is doing anything.')


def _check_reachability(robot, runner, plan):
    """IK-probe every station and its standoff BEFORE anything moves."""
    seed_q = runner.q_home
    for direction, st, pre, _sc in plan:
        for probe, what in ((runner.ref_of(st, pre,
                                           depth_mm=st['depth_mm'] - runner.standoff_mm),
                             'standoff'),
                            (runner.ref_of(st, pre), 'station')):
            if robot.arm.ik(probe, seed_q) is None:
                log.error('IK fails for the %s of station %r at preload %.1f mm -- fix the '
                          'station list before running.', what, st['name'], pre)
                return False
    log.info('  every station and its standoff is reachable.')
    return True


# =================================================================================================
# the app
# =================================================================================================

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
    T_base_tconn = targets[held_name]            # base_link <- target connector == the mate
    # The mate is the identity row of the connector-wrt-target frame, so T_base_targetobj IS
    # the recorded mate and _station_pose's output places the part directly.
    T_base_targetobj = T_base_tconn
    log.info('Held frame %r + its recorded mate, from %s.', held_name,
             tool_frames.frames_path(cfg))

    # ---- stations, excitation, sweep, tug, slip defences ------------------------------------
    stations = _merge_stations(w)
    if stations is None:
        return False
    excitation = _parse_excitation(w)
    if excitation is None:
        return False
    amp0, frq, pha_deg, active = excitation
    sweep = _parse_sweep(w)
    if sweep is None:
        return False
    scales, preloads, passes = sweep
    tug = _parse_tug(cfg)
    if tug is None:
        return False
    refchk = _parse_reference_check(cfg)

    # THE SHARED WIGGLE (urlab/skills/wiggle.py): taper, phase, the connector-frame
    # right-multiply and the speed-cap refusal all live there, so bnc_assembly and
    # estimator_eval superimpose the identical excitation.
    try:
        wg = wigmod.Wiggle(amp0, frq, pha_deg, float(w.get('taper_s', 1.0) or 0.0), 'wiggle')
        # Aliasing, orbit co-primality and the speed caps are validated at the LARGEST
        # amplitude scale, so the sweep's worst case is the one that has to pass.
        wg.validate(rate_hz=float(cfg.get_path('compliance.reference_rate_hz', 125.0)),
                    cap_v=w.get('max_speed_mm_s'), cap_w=w.get('max_rotation_deg_s'),
                    scale=max(scales), duration_s=float(w.get('wiggle_s', 12.0)))
    except wigmod.WiggleError as exc:
        log.error('%s', exc)
        return False
    orbit_s, _slowest_s = wg.orbit_s()

    # ---- the runner + the plan --------------------------------------------------------------
    out_dir = run_dir(cfg, w.get('out_dir', 'data/wiggle_sampling'))
    runner = _BurstRunner(cfg, robot, w, (T_tool0_held, T_base_tconn, T_base_targetobj),
                          wg, tug, refchk, stations, out_dir)
    plan, anchors, contacts = _build_plan(stations, preloads, scales, passes)
    runner.plan_len = len(plan)

    _announce_plan(runner, plan, stations, anchors, contacts, preloads, scales, amp0, frq,
                   active, orbit_s, tug, passes)
    _preload_notes(cfg, T_tool0_held, preloads, amp0, scales)
    if not _check_reachability(robot, runner, plan):
        return False

    if bool(cfg.get('confirm_each_step', False)) and not robot.arm.dry_run:
        if not ask('\nStart the run? Enter to continue (q to abort): '):
            log.warning('Aborted before any motion.')
            return False
    runner.open_logs()
    log.info('Logging to %s', out_dir)

    # ---- the run ----------------------------------------------------------------------------
    ok = True
    try:
        for direction, st, pre, sc in plan:
            runner.begin_burst(direction, st, pre, sc)
            bt.run_tree(runner.burst_tree(frq, amp0), log)   # a skip is not a run failure
    except Exception:                            # noqa: BLE001
        ok = False
        log.exception('Sampling error:')
    finally:
        runner.escape()
        runner.close()
    if ok:
        robot.arm.move_j(runner.q_home, label='home')
        log.info('Done: %d bursts in %s -> %s', runner.burst_no,
                 fmt_dur(time.time() - runner.t_run), out_dir)
    return ok


def main():
    run_app('Wiggle sampling (contact-mode data collection)', 'wiggle_sampling', build_and_run,
            with_gripper=False)


if __name__ == '__main__':
    main()
