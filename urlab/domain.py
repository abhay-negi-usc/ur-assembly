"""The BNC assembly DOMAIN: the clocking state walk, the clocking window arithmetic, and
the per-behaviour compliance override -- pure, robot-free, promoted out of
apps/bnc_assembly.py so behaviours can import them instead of capturing them.
"""

import numpy as np


CLOCK_STATES = ('engaged', 'seated', 'locked')

UNCLOCK_STATES = ('locked', 'seated', 'engaged', 'removed')


def advance_state(state, expected):
    """The next state after `expected`, asserting that is where we actually are -- so an edit
    that advances twice, or skips a step, fails here instead of logging 'locked' for a connector
    that was never seated."""
    if state != expected:
        raise AssertionError(f'cannot advance from {state!r}: expected {expected!r}')
    return CLOCK_STATES[CLOCK_STATES.index(state) + 1]


def retreat_state(state, expected):
    """The state BELOW `expected`, asserting that is where we actually are -- the mirror of
    advance_state, so a disassembly step cannot claim a rung it never undid."""
    if state != expected:
        raise AssertionError(
            f'disassembly step expected the connector to be {expected!r}, but it is {state!r}')
    return UNCLOCK_STATES[UNCLOCK_STATES.index(expected) + 1]

def wrap_near(angle, centre):
    """`angle` (rad) shifted by whole turns so it lands within pi of `centre`.

    Every achieved clock angle here is recovered from a measured pose -- an Euler roll or
    `dot(rotvec, axis)` -- and both return a value in (-pi, pi]. Near a half turn that sign is a
    coin flip, and the sign IS the direction the unwind orbits. The achieved angle lies inside the
    swept band, so wrapping near the band centre picks the right branch."""
    return float(angle) + 2.0 * np.pi * np.round((float(centre) - float(angle)) / (2.0 * np.pi))


def clocking_plan(connector, collar):
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
            'collar_clocking.enabled is true but connector_clocking.enabled is '
            'false. Collar clocking only runs on a SEATED connector and takes the connector pose '
            'it places the collar against from connector clocking. Enable connector clocking, or disable '
            'collar clocking.')
    return connector_on, collar_on


def wrist3_window(arm, q_ref, span=2.0 * np.pi, tol=np.radians(0.25)):
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


def fit_turn(theta, lo, hi):
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


COMPLIANCE_KEYS = ('stiffness', 'mass', 'damping_ratio')


def compliance_override(shared, block):
    """The shared `compliance:` with a maneuver's own physics laid over it.

    A maneuver sets `stiffness` (and optionally `mass`/`damping_ratio`) in its OWN block to
    run at a different compliance from everything else -- STATIC for that whole behaviour.
    `null`, or the key omitted, inherits; it is not the same as writing zeros.

    Returns a NEW dict: the shared section is read by every other maneuver, so overriding one
    must not quietly retune the rest."""
    comp = dict(shared or {})
    for k in COMPLIANCE_KEYS:
        if (block or {}).get(k) is not None:
            comp[k] = [float(v) for v in block[k]]
    return comp


def path_time(lin_mm, ang_deg, v_mm_s, w_deg_s, min_s):
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


class Pacing:
    """The speed: block as one object -- global caps, phase scales, and the duration maths.

    Replaces three closures (phase/caps/seg_time) and seven captured locals. `phase()` sets the
    arm's speed scale for a named phase; `caps()` resolves a maneuver's null speed overrides to
    the global cap x the assemble scale; `seg_time()` is the shared duration arithmetic.
    """

    def __init__(self, cfg, arm, min_seg_s):
        spd = cfg.section('speed')
        self.arm = arm
        self.scales = spd.get('phase_scale', {}) or {}
        self.g_v = float(spd.get('max_cartesian_translation_mm_s', 3.5))
        self.g_w = float(spd.get('max_cartesian_rotation_deg_s', 5.0))
        self.s_asm = float(self.scales.get('assemble', 1.0))
        # ENGAGE HAS ITS OWN SCALE, falling back to `assemble`: the insertion is the slowest
        # thing the arm does and needs to be settable without slowing every assemble-phase move.
        self.s_eng = float(self.scales.get('engage', self.scales.get('assemble', 1.0)))
        self.s_ret = float(self.scales.get('retract', 1.0))
        self.s_std = float(self.scales.get('standoff', 1.0))
        self.min_seg_s = float(min_seg_s)

    def phase(self, name):
        self.arm.set_speed_scale(float(self.scales.get(name, 1.0)), name)

    def caps(self, v=None, w=None):
        """Resolve a maneuver's speed overrides; null = the global cap x the assemble scale."""
        return (self.g_v * self.s_asm if v is None else v),                (self.g_w * self.s_asm if w is None else w)

    def seg_time(self, A, B, v=None, w=None):
        from .transforms import pose_error
        lin_m, ang_rad = pose_error(A, B)
        cv, cw = self.caps(v, w)
        return path_time(lin_m * 1000.0, np.degrees(ang_rad), cv, cw, self.min_seg_s)


def clock_physics(arm, shared_compliance, shared_guard, compliance_over, guard_over,
                  axial_limit_n, name, log):
    """(AdmittanceController, ForceGuard) for one maneuver -- every compliance key inheriting
    the shared compliance and every guard key the shared force_guard when absent or null.

    Logs the AXIAL GIVE next to the limit that provokes it: an admittance loop holds force by
    deflecting w/S, so a soft stiffness against a modest limit can give back more than the whole
    insertion travels -- which reads on the bench as "pushed in, then came back out"."""
    from .robot.admittance import AdmittanceController
    from .robot.guard import ForceGuard
    comp = compliance_override(shared_compliance, compliance_over or {})
    over = {k: v for k, v in (guard_over or {}).items()
            if k in ('max_force_n', 'max_torque_nm', 'persistence_s') and v is not None}
    # NOT `enabled` -- that is the MANEUVER's own switch. force_guard_enabled names the guard's.
    if (guard_over or {}).get('force_guard_enabled') is not None:
        over['enabled'] = bool(guard_over['force_guard_enabled'])
    g = ForceGuard(arm, {**dict(shared_guard), **over})
    stiff = [float(v) for v in (comp.get('stiffness') or [2000.0] * 3 + [15.0] * 3)]
    lim = float(axial_limit_n) if axial_limit_n else float(g.max_force)
    give_mm = (lim / stiff[0] * 1000.0) if stiff[0] > 0 else float('inf')
    log.info('%s ON: stiffness %s, guard %.1f N / %.1f Nm.', name, comp.get('stiffness'),
             g.max_force, g.max_torque)
    log.info('    axial give %.1f mm at %.1f N (= force / stiffness %.0f N/m) -- the '
             'commanded pose backs off by this much to hold that force.',
             give_mm, lim, stiff[0])
    return AdmittanceController(arm, comp), g


class BncAssembly:
    """The run's shared state, ONE object the behaviour functions take instead of a closure
    scope. Grows an attribute per extraction step; `from_config` arrives once construction
    itself moves out of the script."""

    def __init__(self, robot, camera, cfg, spec):
        self.robot = robot
        self.camera = camera
        self.cfg = cfg
        self.spec = spec
        self.frames = None              # TargetFrames, set once the trajectory is loaded
        self.T_ftip_conn = None         # the MUTABLE in-hand belief (fingertip->connector)
        self.pace = None                # Pacing
        self.guard = None               # the shared ForceGuard
        self.adm = None                 # the shared AdmittanceController
        self.tare = None                # zero-ft callable (or None: tare_before off)
        self.adm_tug = None             # tug-verify controller (None: tug off)
        self.adm_cc = self.guard_cc = None    # connector-clocking pair (None: off)
        self.adm_cl = self.guard_cl = None    # collar-clocking pair (None: off)
        self.guard_push = None          # seat-push guard (trips = success)
        self.T_tool0_conn_engaged = None  # the belief AT engagement (clocking anchor)
        self.cc_geometry = None         # validated (sweep, legs, lo, hi, mid), radians
        self.adm_en = self.guard_en = None  # engage pair (None: estimate mode)
        self.engage_rig = None          # validated (wiggle, amp, frq, scale, pv, pw)
        self.dense = None               # resampled trajectory mats
        self.seed_q = None              # rolling IK seed for trajectory-frame moves
        self.estimator = None           # CheckedManifoldEstimator
        self.commit = None              # 'aggregator' | 'argmin'
        self.adm_final = self.guard_final = None  # the COMMIT's controller pair
        self.noise_rng = None           # trajectory-noise stream
        self.col_mode = None            # attempts | offset_sweep | peck
        self.sweep_offsets = None       # resolved offset_sweep 6-vectors
        self.live_path = None           # live-plot png (or None)
        self.mats = None                # raw trajectory mats (frames anchor these)
        self.debug_match = (None, False, None)  # (block, enabled, live_dir)
        self.clock_rows = None          # clocking.csv diagnostic rows
        self.settle_s = 0.5             # shared post-insertion settle
        self.hold_s = 0.0               # shared un-guarded dwell
        self.grasp = None               # GraspController (collision model source)
        self.scanner = self.geom = self.check = None      # the pick stack ...
        self.recovery = self.recorder = self.confirm = None  # ... (see build)
        self.targets = None             # name -> recorded target frame
        self.tname = None               # the target frame this run mates to
        self.vt_rig = None              # the target's marker rig (visual source)
        self.T_ftip_conn_catalogue = None  # nominal belief; the square-grasp anchor
        self.n_cycles = 1               # resolved cycle count for this run
        self.cc_on = self.cl_on = False # which clockings the run is configured to reach
        self.gates_on = True            # behaviour-boundary operator gates
        self.no_prompts = False         # --no-prompts: unattended run
        self.q_pick = None              # the joints the pick (and its recovery) start from
        self.out_dir = None             # this run's experiment folder
        self.ground_z = None            # bench plane, base-frame z (m)
        self.place_rng = None           # [np.random.Generator] cell for the place scatter


class TargetFrames:
    """The geometric spine of a run: every frame the planning derives from the socket pose.

    anchor() rebuilds ALL of them off a (re)measured socket -- rebuilding a subset would put the
    trajectory at one place and the clocking axes at another, which is the failure mode this
    class exists to prevent. It replaces a `nonlocal` web that forced the whole app into one
    3,700-line closure scope: every behaviour now reads `frames.X` and re-anchoring is one call.
    """

    def __init__(self, R_clock, mats, commit_preload_mm):
        from .transforms import inverse, translation_matrix
        self._inverse, self._translation = inverse, translation_matrix
        self.R_clock = R_clock              # engage_clock_deg as a roll about the socket +X
        self.mats = mats                    # the anchored assembly trajectory (last row = mate)
        self.commit_preload_mm = float(commit_preload_mm)
        self.T_base_socket = None
        self.T_base_tconn = None            # the TARGET connector frame (socket @ R_clock)
        self.T_clk = None                   # the clocking axis frame; 'believed' mode rebinds it
        self.T_base_targetobj = None        # where the trajectory's first row is expressed from
        self.T_base_commit = None           # targetobj pushed commit_preload_mm along +X

    def anchor(self, T_socket):
        """Re-derive every planning frame from `T_socket`."""
        self.T_base_socket = T_socket
        self.T_base_tconn = T_socket @ self.R_clock
        self.T_clk = self.T_base_tconn
        self.T_base_targetobj = self.T_base_tconn @ self._inverse(self.mats[-1])
        self.T_base_commit = self.T_base_targetobj @ self._translation(
            [self.commit_preload_mm / 1000.0, 0.0, 0.0])
        return self

# ---------------------------------------------------------------------------- config specs
# One frozen dataclass per behaviour block, mirroring the per-behaviour yaml schema
# (2026-08-28 swap; docs/bnc_config_migration.md maps the legacy assembly.* paths here).
# Field defaults ARE the defaults -- one place, not scattered .get() fallbacks.
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContactSpec:
    """The gentle find-the-socket phase run before engage."""
    enabled: bool = True
    force_n: float = 1.0
    persistence_s: float = 0.10


@dataclass(frozen=True)
class ConfirmSpec:
    """Post-engage seat confirmation: sustained RADIAL force while wiggling in place."""
    enabled: bool = True
    radial_force_n: float = 1.0
    radial_persistence_s: float = 0.10
    max_wiggle_mm: float = 1.0          # measured and reported; no longer a criterion
    cycles: int = 1
    fallback_s: float = 0.5


@dataclass(frozen=True)
class FailRetractSpec:
    """The failed-engage escape: long, because the arm travels to a place afterwards."""
    distance_mm: float = 100.0
    min_mm: float = 20.0                # measured clearance the recovery refuses to go without


@dataclass(frozen=True)
class EngageSpec:
    """The trajectory-following insertion. Wiggle parameters stay raw blocks here -- the Wiggle
    object is behaviour, built where the engage runs, from `wiggle`/`wiggle_from`."""
    travel_mm: float = 5.0
    timeout_s: float = 15.0
    align_s: float = 2.0
    max_axial_force_n: float = 0.0
    persistence_s: float | None = None
    preload_mm: float = 0.0
    max_misses: int = 2
    sample_rate_hz: float = 25.0
    amplitude: dict = field(default_factory=dict)
    frequency_hz: dict = field(default_factory=dict)
    wiggle: dict | None = None
    wiggle_from: str = 'wiggle_sampling.yaml'
    max_oscillation_speed_mm_s: float | None = None
    max_oscillation_rotation_deg_s: float | None = None
    speed_mm_s: float | None = None     # RETIRED key, carried so the loud failure still fires
    contact: ContactSpec = field(default_factory=ContactSpec)
    confirm: ConfirmSpec = field(default_factory=ConfirmSpec)
    fail_retract: FailRetractSpec = field(default_factory=FailRetractSpec)
    compliance: dict | None = None      # per-behaviour override of the shared compliance:
    force_guard: dict | None = None     # ... and of the shared force_guard:



@dataclass(frozen=True)
class SeatPushSpec:
    """The pre-collar seat push: press until force holds, bounded by travel."""
    enabled: bool = True
    force_n: float = 5.0
    persistence_s: float = 1.0
    max_travel_mm: float = 15.0


@dataclass(frozen=True)
class ConnectorClockingSpec:
    """The bayonet sweep. `sweep_deg` = ABSOLUTE roll positions wrt the target frame;
    `rotation_deg` is the legacy single relative stroke (becomes a one-entry sweep)."""
    enabled: bool | None = None         # interplay with the collar resolved by clocking_plan
    sweep_deg: list | None = None
    rotation_deg: float = 90.0
    push_mm: float = 5.0
    success_advance_mm: float = 5.0
    max_tries: int = 3
    settle_s: float | None = None       # None = inherit compliance.settle_s
    hold_after_s: float = 0.0
    open_gripper_after: bool = True
    speed_translation_mm_s: float | None = None
    speed_rotation_deg_s: float | None = None
    tare_before: bool = False
    compliance: dict | None = None
    force_guard: dict | None = None


@dataclass(frozen=True)
class CollarClockingSpec:
    """Grasp the locking collar AXIALLY and turn it with the wrist."""
    enabled: bool | None = None
    collar_offset_mm: float = 25.0
    rotation_deg: float = 90.0
    push_mm: float = 0.0
    grasp_clock_deg: float | None = None   # None = the engaged roll
    retract_mm: float = 300.0
    retreat_mm: float = 100.0
    wall_standoff_mm: float | None = None  # None = no wall check
    settle_s: float | None = None
    speed_translation_mm_s: float | None = None
    speed_rotation_deg_s: float | None = None
    tare_before: bool = False
    max_offaxis_tilt_deg: float = 5.0
    wrist3_margin_deg: float = 5.0
    axis_offset_mm: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    seat_push: SeatPushSpec = field(default_factory=SeatPushSpec)
    compliance: dict | None = None
    force_guard: dict | None = None


@dataclass(frozen=True)
class TugVerifySpec:
    """Pull along -X and watch displacement: a seated connector holds."""
    enabled: bool = True
    pull_force_n: float = 3.0
    pull_time_s: float = 3.0
    displacement_threshold_mm: float = 3.0
    extraction_distance_mm: float = 50.0
    compliance: dict | None = None


@dataclass(frozen=True)
class PlaceScatterSpec:
    """Uniform random x/y/yaw perturbation of the disassembly place (data collection)."""
    enabled: bool = False
    x_mm: float = 50.0
    y_mm: float = 50.0
    yaw_deg: float = 30.0
    seed: int | None = None


@dataclass(frozen=True)
class DisassemblyPlaceSpec:
    enabled: bool = True
    clearance_mm: float = 25.0
    retreat_mm: float = 100.0


@dataclass(frozen=True)
class ClockingRetractSpec:
    """The shared post-clocking escape: two guarded straight legs. Axes are unit vectors,
    distances mm; the behaviour converts."""
    gripper_axis: list = field(default_factory=lambda: [0.0, 0.0, -1.0])
    gripper_distance_mm: float = 100.0
    target_axis: list = field(default_factory=lambda: [-1.0, 0.0, 0.0])
    target_distance_mm: float = 100.0


@dataclass(frozen=True)
class ReorientRecoverySpec:
    """The reorient recovery: pick the badly-presented cable SQUARE, set it down on the socket
    heading, re-scan. place_offsets keeps the raw mapping (x_mm/y_mm/z_mm/yaw_deg) so the
    behaviour's defaults stay written at the read site, like the closure had them."""
    enabled: bool = True
    fingertip_in_connector: dict | None = None   # the SQUARE grasp override (xyz_mm/rpy_deg)
    release_clearance_mm: float = 10.0
    approach_mm: float = 120.0
    settle_s: float = 5.0
    place_offsets: dict = field(default_factory=dict)
    snap_to_ground: bool = True


@dataclass(frozen=True)
class VisualTargetSpec:
    """The marker sweep that re-anchors the run. max_shift_* of None disables that gate."""
    servo_refine: bool = True
    view_joints_deg: list | None = None
    return_home_after: bool = True
    max_shift_mm: float | None = 25.0
    max_shift_deg: float | None = 10.0


@dataclass(frozen=True)
class CelebrateSpec:
    """The post-verification flourish. Off by default; every gate lives in the behaviour."""
    enabled: bool = False
    when: str = 'last'                  # last | every | every_n
    every_n: int = 1
    rise_mm: float = 100.0
    nod_deg: float = 15.0
    spin_deg: float = 180.0
    repeats: int = 2
    gripper_flourish: bool = True


@dataclass(frozen=True)
class DisassemblySpec:
    enabled: bool = False
    unlock_collar: bool = True
    extract_mm: float = 60.0
    cycles: int = 1                     # whole localize->pick->assemble->disassemble passes
    place: DisassemblyPlaceSpec = field(default_factory=DisassemblyPlaceSpec)
    place_scatter: PlaceScatterSpec = field(default_factory=PlaceScatterSpec)


@dataclass(frozen=True)
class TrajectoryNoiseSpec:
    """Per-attempt jitter over the reference path (probing only)."""
    enabled: bool = False
    std: list = field(default_factory=lambda: [0.0] * 6)
    smooth_window: int = 10
    noise_decay_attempt: float = 0.0
    noise_decay_traj: float = 0.0


@dataclass(frozen=True)
class FinalInsertionSpec:
    """The COMMIT: the one insertion meant to seat. None inherits the shared compliance value."""
    enabled: bool = True
    settle_s: float | None = None
    hold_after_insertion_s: float | None = None
    speed_translation_mm_s: float | None = None
    speed_rotation_deg_s: float | None = None
    pause_s: float = 0.0
    preload_mm: float = 10.0
    trajectory_noise: TrajectoryNoiseSpec = field(default_factory=TrajectoryNoiseSpec)
    compliance: dict | None = None
    force_guard: dict | None = None


@dataclass(frozen=True)
class CollectionSpec:
    """How estimate-mode gathers observations along the path."""
    mode: str = 'attempts'              # attempts | offset_sweep | peck
    sweep_offsets: list | None = None   # None = the default pitch sweep
    peck_retract_mm: float = 5.0
    peck_timeout_s: float = 30.0


@dataclass(frozen=True)
class TrajectorySpec:
    """The recorded insertion path: where it lives, how it is resampled, how it is approached."""
    csv: str = 'assembly_trajectory.csv'
    angles_deg: bool = False
    translational_resolution_mm: float = 1.0
    rotational_resolution_deg: float = 1.0
    standoff: dict = field(default_factory=lambda: {'axis': [-1, 0, 0], 'distance_mm': 10.0})


@dataclass(frozen=True)
class RunSpec:
    """The estimate/engage loop: mode, budget, tolerances, logging cadence."""
    insertion_mode: str = 'estimate'    # estimate | engage
    target_source: str = 'kinematic'    # kinematic | visual
    post_engage_frame: str = 'target'   # target | believed
    engage_clock_deg: float = 0.0
    max_attempts: int = 5
    accumulate_observations: bool = True
    retract_distance_mm: float = 50.0
    log_decimation: int = 5
    success_pos_mm: float = 2.0
    success_rot_deg: float = 3.0
    target_frame: str | None = None     # targets: entry in frames.yaml
    gate_between_behaviors: bool = True
    live_plot: bool | str = True        # True | False | an explicit png path
    release_retract_distance_mm: float = 80.0
    debug_match: dict = field(default_factory=dict)

_UNIT_PAIRS = (('_m_s2', '_mm_s2'), ('_rad_s2', '_deg_s2'), ('_m_s', '_mm_s'),
               ('_rad_s', '_deg_s'), ('_m', '_mm'), ('_rad', '_deg'))


def strip_derived_units(block):
    """A loaded block minus the units-normaliser's derived SI siblings (X_mm -> X_m, X_deg ->
    X_rad, xyz_mm -> xyz, rpy_deg -> rpy) and minus explicit nulls, recursively. What reaches
    parse_block is exactly the keys the file MEANINGFULLY declares: an unknown key is a real
    typo, and `key: null` means "inherit / use the default", exactly as it always has."""
    out = {}
    for k, v in block.items():
        if v is None:
            continue
        if any(k.endswith(suf) and (k[:len(k) - len(suf)] + src) in block
               for suf, src in _UNIT_PAIRS):
            continue
        if (k == 'xyz' and 'xyz_mm' in block) or (k == 'rpy' and 'rpy_deg' in block):
            continue
        out[k] = strip_derived_units(v) if isinstance(v, dict) else v
    return out








@dataclass(frozen=True)
class BncSpec:
    """Every tuning block the bnc run reads, parsed once, unknown keys rejected. Grows a field
    per port slice until the whole config flows through it."""
    run: RunSpec = field(default_factory=RunSpec)
    trajectory: TrajectorySpec = field(default_factory=TrajectorySpec)
    engage: EngageSpec = field(default_factory=EngageSpec)
    final_insertion: FinalInsertionSpec = field(default_factory=FinalInsertionSpec)
    collection: CollectionSpec = field(default_factory=CollectionSpec)
    trajectory_noise: TrajectoryNoiseSpec = field(default_factory=TrajectoryNoiseSpec)
    connector_clocking: ConnectorClockingSpec = field(default_factory=ConnectorClockingSpec)
    collar_clocking: CollarClockingSpec = field(default_factory=CollarClockingSpec)
    tug_verify: TugVerifySpec = field(default_factory=TugVerifySpec)
    reorient_recovery: ReorientRecoverySpec = field(default_factory=ReorientRecoverySpec)
    visual_target: VisualTargetSpec = field(default_factory=VisualTargetSpec)
    clocking_retract: ClockingRetractSpec = field(default_factory=ClockingRetractSpec)
    celebrate: CelebrateSpec = field(default_factory=CelebrateSpec)
    disassembly: DisassemblySpec = field(default_factory=DisassemblySpec)

    @classmethod
    def from_config(cls, cfg):
        from dataclasses import fields as _fields

        from .config import parse_block
        blocks = {}
        for f in _fields(cls):
            blk = cfg.get(f.name)
            if blk:
                blocks[f.name] = strip_derived_units(blk)
        return parse_block(cls, blocks, 'bnc')

