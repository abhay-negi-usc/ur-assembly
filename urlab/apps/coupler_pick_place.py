"""COUPLER PICK AND PLACE -- find a catalogued object by its marker, mate, carry, set down.

    python -m urlab.apps.coupler_pick_place --set object_name=ORU

Everything about the object comes from configs/objects.yaml, keyed by that name: its marker id
and printed size, the MATING FEATURE's pose in that marker's frame, and the mass it adds to the
tool. Nothing about the object is spelled out in this app or in its config, so adding a second
object is a calibration run and not a code change.

THE CONCEPT OF OPERATIONS, in order:

    1. View the markers from several angles, then servo onto each one in turn at
       `marker_views.servo.distance_mm` (100 mm) for a close, square look, and fuse a pose for
       each from the sweep's views and the close ones together.
    2. Derive the coupling feature's pose from them -- every marker votes, and the votes are
       averaged (the catalogue entry).
    3. ALIGN the coupler with the feature at the `motion.mate_standoff` leg (250 mm back along
       the coupler's own -z by default).
    4. MATE under admittance, travelling the whole standoff compliantly, then PUSH along the
       mating axis until `mate_preload.force_n` is measured -- proof of contact rather than
       proof of arrival. The F/T is tared once, at the standoff, before any of it.
    5. ACTUATE the coupler to grasp the feature, and confirm it against the board's sensor.
    6. RETRACT with the object by `motion.pick_retract` (100 mm, coupler -z).
    7. CARRY from there to the `motion.place_standoff` leg off the place pose (100 mm, -z).
    8. PLACE under admittance, down to the place pose -- a delta from where the object was
       picked up -- then PUSH along the same axis until `place_preload.force_n` is measured,
       tared at the place standoff with the object hanging free.
    9. DEMATE, and RETRACT by `motion.final_retract` (100 mm, -z).

Each of the four legs is its own `distance_mm` / `axis` / `frame`, because they answer different
questions: the mate wants a long approach so the compliant descent has room to correct, while
the three short hops only have to clear the part.

THE WHOLE PICK IS ONE MULTIPLY. The camera measures T_base_marker and the catalogue holds
T_marker_grasp, so

    T_base_grasp = T_base_marker @ T_marker_grasp

is where configs/frames.yaml's `coupler_mate` frame has to end up. That is the entire geometry;
the rest of this file is how to get there safely and what to do afterwards.

WHERE IT GOES is a DELTA from where it was picked, not an absolute pose -- `place.xyz_mm` /
`place.rpy_deg`, applied either in base_link axes (`relative_to: base`, "300 mm to the left")
or in the object's own frame (`relative_to: object`, "rotated 90 degrees about its own mating
axis"). A delta is the right shape because the pick pose is not known until the camera has
looked: an absolute place pose would have to be re-taught whenever the object is set down
somewhere new, while a delta keeps meaning the same thing.

SOFT ADMITTANCE BY DEFAULT. Every motion that can touch something -- the mate, the retract
with the object, the set-down, the withdrawal -- runs under the software admittance law
(robot/admittance.py) with deliberately soft translational stiffness, so the coupler yields to
a misalignment instead of levering against it, and springs back to the reference when the
contact eases. Free-space transit between the two stations is a plain guarded moveL, because
compliance buys nothing where there is nothing to touch and costs a lot of time.

THE COMMANDED POSE IS THE CALIBRATED MATE. The catalogue stores the pose the coupler was
actually seated in during calibration, uncorrected, so this app drives to the seating that was
proven to work -- including its yaw about the mating axis, which the mechanism does not
constrain and therefore nothing else would pin. That is what makes the in-hand pose known: once
mated, the object's mating frame IS `coupler_mate`, so the set-down can be commanded in terms of
the object and not of the flange.

THE PAYLOAD IS UPDATED ON PICK AND ON RELEASE. A 3 kg object that the controller has not been
told about reads as 30 N of external force, which is most of a force-guard limit and a standing
bias in the admittance law. See combined_payload() for what is assumed about its CoG.
"""

import numpy as np

from .. import behaviors as bt
from .. import log as urlog
from .. import tool_frames
from ..apps._common import ask, prompts_off, seg_time
from ..robot.admittance import AdmittanceController
from ..robot.coupler import Coupler
from ..robot.guard import ForceGuard
from ..skills import marker_localize as mloc
from ..transforms import (fmt_pose, inverse, matrix_to_xyzrpy, pose_error, translation_matrix,
                          xyzrpy_to_matrix)
from ._runner import run_app

log = urlog.get('coupler-pnp')


# ---------------------------------------------------------------------------- pure geometry
LEGS = ('mate_standoff', 'pick_retract', 'place_standoff', 'final_retract')
DEFAULT_LEG_MM = {'mate_standoff': 250.0, 'pick_retract': 100.0,
                  'place_standoff': 100.0, 'final_retract': 100.0}


def parse_offset(block, where, default_mm=100.0):
    """A validated displacement leg: {distance_m, axis (unit), frame}.

    `axis` is a direction, `frame` says what it is a direction IN. The default everywhere is the
    coupler's own -z -- straight back out along the mating axis, which is the one direction that
    never levers the coupler against a feature it is still inside. `frame: base` instead reads
    the axis in base_link, for the case where "straight up" is meant literally regardless of how
    the object happens to be tilted.

    A ZERO AXIS IS REFUSED rather than normalised to something arbitrary: it would otherwise
    collapse the leg to no motion at all, and a standoff of zero means the compliant mate starts
    already inside the feature. Pure, so the schema is testable without a robot."""
    b = dict(block or {})
    frame = str(b.pop('frame', 'coupler'))
    if frame not in ('coupler', 'base'):
        raise ValueError(f'{where}.frame must be coupler or base, not {frame!r}')
    # BOTH SPELLINGS ARE EXPECTED HERE. config._normalise_units adds an SI `distance_m` sibling
    # to every `distance_mm` at load time, so a block written in mm arrives carrying both; a
    # strict reader that had not been told would reject the repo's own configs. Writing two
    # DIFFERENT values is already an error in the loader, so they cannot disagree by the time
    # this runs.
    if 'distance_m' in b:
        distance_m = float(b.pop('distance_m'))
        b.pop('distance_mm', None)
    else:
        distance_m = float(b.pop('distance_mm', default_mm)) / 1000.0
    if distance_m < 0.0:
        raise ValueError(f'{where}.distance_mm must not be negative -- reverse the axis instead')
    axis = np.asarray(b.pop('axis', [0.0, 0.0, -1.0]), dtype=float)
    if axis.shape != (3,):
        raise ValueError(f'{where}.axis must be three numbers, got {axis.tolist()}')
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError(f'{where}.axis is zero -- it is a DIRECTION, so there is no sensible '
                         'way to read it; write [0, 0, -1] for straight back out')
    if b:
        raise ValueError(f'{where} has unknown key(s) {sorted(b)} -- allowed: distance_mm, '
                         'axis, frame')
    return {'distance_m': distance_m, 'axis': axis / norm, 'frame': frame}


def offset_pose(T, spec):
    """`T` displaced by one parsed leg, orientation unchanged.

    `frame: coupler` post-multiplies, so the axis travels with the tool: -z is out along the
    mating axis wherever the coupler is pointing. `frame: base` adds the displacement in
    base_link instead. Only the origin moves either way -- a standoff that rotated the tool
    would not be a standoff."""
    step = spec['distance_m'] * spec['axis']
    if spec['frame'] == 'base':
        out = np.array(T, dtype=float)
        out[:3, 3] = out[:3, 3] + step
        return out
    return T @ translation_matrix(step)


def mating_direction(standoff_spec):
    """The direction the coupler advances to mate: the mate standoff leg, REVERSED.

    Derived rather than configured separately on purpose. The standoff is where the approach
    starts and the mate is the trip back along it, so they are one axis with two signs -- and two
    independent settings could disagree, which would have the arm approach along one line and
    preload along another. Same `frame`, so a standoff read in base preloads in base too."""
    spec = dict(standoff_spec)
    spec['axis'] = -np.asarray(spec['axis'], dtype=float)
    return spec


def parse_preload(block, where='mate_preload'):
    """A `*_preload:` block -- push along an axis until a contact force is reached.

    A MATE THAT MERELY REACHES THE TARGET POSE HAS NOT NECESSARILY SEATED. The pose came from a
    camera, so it is right to within the estimate's error; arriving there proves the arm got to
    where it believed the feature was, not that the two are touching. Pushing until a measured
    force appears proves contact, and leaves the mechanism loaded rather than resting."""
    b = dict(block or {})
    out = {'enabled': bool(b.pop('enabled', True)),
           'force_n': float(b.pop('force_n', 1.0)),
           'max_travel_m': float(b.pop('max_travel_mm', 8.0)) / 1000.0,
           'step_m': float(b.pop('step_mm', 0.5)) / 1000.0,
           'settle_s': float(b.pop('settle_s', 0.2))}
    for key in ('max_travel_m', 'step_m', 'force_n'):
        b.pop(key.replace('_m', '_mm') if key.endswith('_m') else key, None)
    b = {k: v for k, v in b.items() if not k.endswith(('_m', '_s2'))}
    if b:
        raise ValueError(f'{where} has unknown key(s) {sorted(b)} -- allowed: enabled, '
                         'force_n, max_travel_mm, step_mm, settle_s')
    if out['force_n'] <= 0.0:
        raise ValueError(f'{where}.force_n must be positive -- it is the contact force the '
                         'push stops at, and zero would stop before touching anything')
    if out['step_m'] <= 0.0 or out['max_travel_m'] <= 0.0:
        raise ValueError(f'{where}.step_mm and max_travel_mm must both be positive')
    return out


def law_bandwidth_hz(mass, stiffness):
    """The admittance law's natural frequency per axis, sqrt(S/M) / 2pi.

    Worth printing because it is the number the oscillation question is really about. The law is
    a virtual second-order system and its damping_ratio governs whether IT rings -- over-damped,
    it cannot. What rings is the LOOP: the law commands a reference, the arm tracks it with lag,
    the payload's inertia overshoots, the F/T reads the overshoot as force, and the law yields
    further. The closer the law's bandwidth is to what the loaded arm can actually track, the
    tighter that loop is wound."""
    return np.sqrt(np.asarray(stiffness, dtype=float)
                   / np.asarray(mass, dtype=float)) / (2.0 * np.pi)


def insertion_axis_is_stiff(spec, stiffness, ratio=2.0):
    """Does the stiffest translational axis line up with the direction the part is inserted?

    THE LAW RUNS IN THE TOOL0 FRAME (robot/admittance.py), and `coupler_mate` is a pure
    translation from tool0 -- no rotation -- so stiffness[0..2] are the coupler's own x, y, z. An
    insertion stiffness profile is therefore only meaningful relative to WHICH of those the part
    travels along: soft across it so a chamfer can guide the part in, stiffer along it so the
    push actually reaches the preload.

    Point the insertion down a different axis and that profile is exactly backwards -- compliant
    where the force is wanted and rigid where the give is. Nothing about the numbers says so,
    which is why it is worth checking out loud. Pure, so the rule is testable."""
    S = np.asarray(stiffness, dtype=float)[:3]
    if float(np.max(S)) <= ratio * float(np.min(S)):
        return True                  # near-isotropic: there is no profile to point the wrong way
    axis = np.abs(np.asarray(spec['axis'], dtype=float))
    axis = axis / (float(np.linalg.norm(axis)) + 1e-12)
    # For an axis-aligned diagonal stiffness the effective stiffness along a direction is the
    # quadratic form; what is left over, halved, is the average across the two perpendiculars.
    along = float((axis ** 2) @ S)
    across = (float(np.sum(S)) - along) / 2.0
    return along >= ratio * across if across > 0.0 else True


def compliance_blocks(cfg):
    """(free, loaded, insert) compliance sections -- the laws for the three states of the run.

    WHY MORE THAN ONE. The admittance law's `mass` is a VIRTUAL design parameter, not the
    payload; nothing
    in it changes when 3 kg is picked up. But the plant it drives changes enormously -- on this
    cell the mass at the tool goes from 0.3 kg to 3.6 kg, twelvefold -- and a law tuned to move
    a bare coupler briskly will command reference accelerations the loaded arm cannot follow. The
    tracking lag then feeds the F/T, which feeds the law, which yields further: the oscillation
    that shows up at pickup and not before.

    RAISING THE VIRTUAL MASS IS THE DIRECT FIX. It lowers the law's bandwidth (sqrt(S/M)), so the
    reference stops asking for accelerations the real arm cannot deliver. Lowering the stiffness
    would also work but costs preload resolution, and the preload happens while still unloaded.

    WHOLE-BLOCK, NOT MERGED, following the same rule as the machine layers in config.py: a
    per-key merge would mix a retuned mass with an inherited damping_ratio computed for the old
    one, and D = zeta*2*sqrt(M*S) means those two are not independent. Absent, the loaded law is
    simply the free one -- which is the old behaviour, and is fine for a light object.

    `insert` IS THE THIRD, for the leg that drives the part into its destination. An INSERTION
    wants a different SHAPE of compliance, not just a different speed: soft ACROSS the travel
    axis so a chamfer can guide the part in, stiffer ALONG it so the push reaches the preload
    without metres of reference travel. A carry wants neither -- it wants to be uniformly slow.
    Absent, it falls back to the loaded law, which is the old behaviour."""
    free = cfg.section('compliance')
    loaded = cfg.section('compliance_loaded') or free
    insert = cfg.section('compliance_insert') or loaded
    return free, loaded, insert


def step_gate(cfg, robot):
    """The per-step confirm, or None when it is off.

    `confirm_each_step: true` pauses before EVERY step that moves the arm or works the coupler,
    naming what is about to happen. bt.Action takes this as its `confirm` hook and treats a
    refusal as a failure, so declining stops the sequence where it stands rather than skipping
    a leg and carrying on -- which for a pick-and-place would mean releasing an object the arm
    never moved to the place pose.

    OFF for a dry run and under --no-prompts, so unattended and simulated runs never hang. NOT
    applied to the steps that move nothing (the payload updates, the report): a prompt there is
    noise, and noise is what trains people to hit Enter without reading."""
    if not bool(cfg.get('confirm_each_step', False)) or prompts_off(cfg) or robot.arm.dry_run:
        return None

    def confirm(label):
        return ask(f'  NEXT: {label}  --  Enter to run, q to stop: ')

    return confirm


def describe_leg(name, spec):
    """One readable line per leg, so a run's log says what geometry it was actually given."""
    return ('%s: %.0f mm along %s in the %s frame'
            % (name, spec['distance_m'] * 1000.0,
               np.round(spec['axis'], 3).tolist(), spec['frame']))


def place_pose(T_base_pick, offset, relative_to='base'):
    """Where to set the object down, as a DELTA from where it was picked up.

    `relative_to='base'` PRE-multiplies, so the delta is read in base_link axes: xyz is a
    displacement along the robot's own x/y/z and rpy is a rotation about them. This is what
    "put it 300 mm to the left" means, and it is the default.

    `relative_to='object'` POST-multiplies, so the delta is read in the object's own mating
    frame: +z is along its mating axis, and a yaw is a spin about that axis. This is what
    "rotate it a quarter turn and drop it back in" means.

    Both are one multiply; which one is meant is never inferable from the numbers, so it is
    named in the config rather than guessed. Pure, so the convention is testable."""
    T_delta = xyzrpy_to_matrix(np.asarray(offset.get('xyz', [0.0, 0.0, 0.0]), dtype=float),
                               np.asarray(offset.get('rpy', [0.0, 0.0, 0.0]), dtype=float))
    if relative_to == 'object':
        return T_base_pick @ T_delta
    if relative_to == 'base':
        return T_delta @ T_base_pick
    raise ValueError(f'place.relative_to must be base or object, not {relative_to!r}')


def combined_payload(tool_payload, held_mass_kg, held_cog_m):
    """{mass_kg, cog_m} for tool + object, as the controller needs it once the object is on.

    THE OBJECT'S CoG IS AN ASSUMPTION, and a load-bearing one: mass is easy to weigh and CoG is
    not, so objects.yaml records the first and not the second. The default put in by the config
    is the coupler's own engagement point -- the object hangs off there, so it is the least
    wrong single point available without a measurement -- and the combination is then the honest
    mass-weighted mean rather than "the tool's CoG with more mass at it", which would claim the
    object's weight acts 55 mm behind where it does.

    Getting this wrong does not fail loudly. It biases every wrench reading by the residual of
    the gravity compensation, which shows up as an admittance law that drifts in one direction
    and a force guard whose margin depends on arm orientation. Override `held_cog_mm` per run
    once it matters. Pure, so the arithmetic is testable without a robot."""
    m_tool = float((tool_payload or {}).get('mass_kg', 0.0) or 0.0)
    cog_tool = np.asarray((tool_payload or {}).get('cog_m', [0.0, 0.0, 0.0]), dtype=float)
    m_held = float(held_mass_kg or 0.0)
    if m_held <= 0.0:
        return {'mass_kg': m_tool, 'cog_m': cog_tool.tolist()}
    cog_held = np.asarray(held_cog_m, dtype=float)
    total = m_tool + m_held
    cog = (m_tool * cog_tool + m_held * cog_held) / total
    return {'mass_kg': total, 'cog_m': cog.tolist()}


def object_rig(obj, dictionary=None):
    """The catalogue entry, shaped as the `rig` that skills/marker_localize.locate() consumes.

    Reusing locate() rather than re-implementing the sweep gets the whole run-time pipeline for
    free -- multi-view fusion, the per-marker consistency gate, RANSAC over the votes, and the
    joint-PnP solve over every view's corners -- and keeps this app solving the SAME estimate the
    calibration was measured against.

    EVERY MARKER ON THE OBJECT BECOMES A VOTER. They all store the same grasp frame in their own
    coordinates, so `T_marker_target` is that frame for each, and locate() returns T_base_grasp
    directly. With three or more, a marker that has been knocked or re-stuck is outvoted instead
    of dragging the answer; with two, a disagreement fails the gate, which is the honest outcome
    because nothing can say which of them moved."""
    return {'dictionary': dictionary,
            'markers': {int(mid): {'size_m': float(m['size_m']),
                                   'T_marker_grasp': m['T_marker_grasp'],
                                   'T_marker_target': m['T_marker_grasp'],
                                   'meta': dict(m.get('meta') or {})}
                        for mid, m in obj['markers'].items()}}


# ---------------------------------------------------------------------------- the run
class CouplerCycle:
    """One find-mate-carry-set down cycle for one catalogued object.

    PUBLIC, AND SUBCLASSED. apps/coupler_pick_assemble is the same cycle with a different idea of
    where the object ends up -- a taught assembly pose instead of a delta from the pick -- and
    everything around that (locating, mating, preloading, locking, the payload, the two laws, the
    tare timing, the withdrawal) is identical. Duplicating it would mean fixing the next tare bug
    twice, so the target is a HOOK rather than a copy:

        _target_pose(T_pick)   where the object is going
        TARGET_STANDOFF        which leg stands off from it
        TARGET_PRELOAD         which preload block presses it home
    """

    LEGS = LEGS
    TARGET_STANDOFF = 'place_standoff'
    TARGET_PRELOAD = 'place_preload'
    TARGET_WORD = 'place'

    def __init__(self, cfg, robot, camera, detector, plan, name, obj, coupler):
        self.cfg = cfg
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.plan = plan
        self.name = name
        self.obj = obj
        self.coupler = coupler
        free, loaded, insert = compliance_blocks(cfg)
        # TWO LAWS, ONE PER PAYLOAD STATE. `_adm()` picks between them on self.holding, which
        # take_payload/drop_payload flip -- so the gains change at exactly the moment the mass
        # does, rather than one set being a compromise for both.
        self.adm = AdmittanceController(robot.arm, free)
        self.adm_loaded = AdmittanceController(robot.arm, loaded)
        self.adm_insert = AdmittanceController(robot.arm, insert)
        self.holding = False
        self._active_law = None       # the law the current leg is running, for the preload push
        self.guard = ForceGuard(robot.arm, cfg.section('force_guard'))
        self.T_tool0_coupler = tool_frames.coupler_mate(cfg)
        # FOUR INDEPENDENT LEGS, not one standoff reused four times: the mate wants a long
        # approach so the compliant descent has room to correct, while the three short hops
        # around it only have to clear the part. Tying them together would make either the mate
        # cramped or the cycle slow.
        motion = cfg.section('motion')
        self.legs = {name: parse_offset(motion.get(name), f'motion.{name}',
                                        DEFAULT_LEG_MM.get(name, 100.0))
                     for name in self.LEGS}
        self.preload_mate = parse_preload(cfg.section('mate_preload'), 'mate_preload')
        self.preload_place = parse_preload(cfg.section(self.TARGET_PRELOAD),
                                           self.TARGET_PRELOAD)
        self.tare_before = bool(cfg.get_path('compliance.tare_before', True))
        if (self.preload_mate['enabled'] or self.preload_place['enabled']) \
                and not self.tare_before:
            log.warning('A preload is on but compliance.tare_before is off. A preload is '
                        'measured against the tare, so without a fresh zero at the standoff it '
                        'is being compared to whatever offset the sensor was last left with.')
        self.settle_s = float(cfg.get('settle_s', 0.3))
        self.T_pick = None
        self.T_place = None

    # ---- locate ------------------------------------------------------------------------------
    def locate(self):
        """Where the coupler has to end up, straight out of the catalogue and the camera."""
        if self.plan.servo.enabled:
            log.info('Locating %r: sweep, then a close servo onto each of %d marker%s at '
                     '%.0f mm.', self.name, len(self.obj['markers']),
                     '' if len(self.obj['markers']) == 1 else 's',
                     self.plan.servo.distance_m * 1000.0)
        T = mloc.locate(self.robot, self.camera, self.detector, object_rig(
            self.obj, self.cfg.get_path('aruco.dictionary')), self.plan)
        if T is None:
            log.error('Could not locate %r -- marker(s) %s were not seen well enough, or '
                      'disagreed past the gate, to place the coupler. Nothing has moved.',
                      self.name, ', '.join(str(m) for m in sorted(self.obj['markers'])))
            return False
        self.T_pick = T
        log.info('PICK   %s', fmt_pose(T))
        try:
            self.T_place = self._target_pose(T)
        except (ValueError, KeyError) as exc:
            log.error('%s', exc)
            return False
        log.info('%-6s %s', self.TARGET_WORD.upper(), fmt_pose(self.T_place))
        lin, ang = pose_error(T, self.T_place)
        log.info('  the %s is %.1f mm / %.1f deg from the pick.',
                 self.TARGET_WORD, lin * 1000.0, np.degrees(ang))
        return self._sane()

    def _target_pose(self, T_pick):
        """Where the object is going: a DELTA from where it was picked up.

        The hook apps/coupler_pick_assemble replaces -- it reads a taught absolute pose instead.
        A delta is right for a set-down because the pick pose is not known until the camera has
        looked, so an absolute one would have to be re-taught every time the object starts
        somewhere new. It is WRONG for an assembly, where the fixture does not move with the
        part."""
        return place_pose(T_pick, self.cfg.section('place'),
                          self.cfg.get_path('place.relative_to', 'base'))

    def _sane(self):
        """Refuse a pick the geometry says is wrong BEFORE anything moves.

        A marker seen at a bad angle, or one whose size is mistyped, produces a pose that is
        perfectly well-formed and metres away; the arm would then drive at it confidently. The
        floor check is the cheap one that catches most of it -- a mating point below the bench
        is not a mating point."""
        floor_z = self.cfg.get('min_grasp_z_mm')
        if floor_z is None:
            return True
        for what, T in (('pick', self.T_pick), (self.TARGET_WORD, self.T_place)):
            z_mm = float(T[2, 3]) * 1000.0
            if z_mm < float(floor_z):
                log.error('The %s pose sits at z = %.1f mm, below min_grasp_z_mm (%.1f). Either '
                          'the marker was mis-solved or the catalogue entry is wrong -- '
                          'refusing to drive at it.', what, z_mm, float(floor_z))
                return False
        return True

    # ---- the pick ----------------------------------------------------------------------------
    def approach(self):
        """Align the coupler with the feature at the mate standoff.

        Free space, so a plain guarded moveL: compliance buys nothing where there is nothing to
        touch, and costs the whole transit in servo-rate ramping."""
        return self._move_to(offset_pose(self.T_pick, self.legs['mate_standoff']),
                             'align at the mate standoff')

    def descend_and_mate(self):
        """The whole standoff is travelled under admittance, not just the last few millimetres.

        The feature was found by a camera, so where contact begins is exactly what is uncertain;
        a compliant approach that only starts once something is already touching has missed the
        event it exists for.

        THE PRELOAD PUSH RIDES INSIDE THE SAME SERVO SESSION, as a continuation rather than a
        second leg, and that is not a tidiness choice. Every compliant leg tares the F/T at the
        pose it starts from; a preload run as its own leg would therefore re-zero the sensor
        AFTER contact, which subtracts exactly the force it is trying to measure. One session
        means one tare, taken at the standoff with nothing touching, and every newton read
        afterwards is contact."""
        return self._compliant(offset_pose(self.T_pick, self.legs['mate_standoff']), self.T_pick,
                               'mate with the coupling feature',
                               after=lambda T, w: self._push_to_preload(
                                   T, w, self.preload_mate, 'mate_standoff', 'mate'))

    def _push_to_preload(self, T_at, watch, preload, leg, what):
        """Advance the compliant reference along `leg` reversed until the preload is met.

        SHARED BY THE MATE AND THE PLACEMENT because they are the same manoeuvre with the sign of
        one axis and the state of the coupler swapped: arrive at a pose the camera computed, then
        push until a force proves contact. Splitting them would let the two drift apart.

        THE REFERENCE MOVES, NOT THE MEASURED POSE. Under admittance the contact force is the
        spring acting on the displacement between the two (F = S * delta), so walking the
        reference past the arrival pose is how force is commanded -- at 1000 N/m, 1 N is 1 mm of
        reference travel. Stepping it and re-reading is what makes this a measured preload rather
        than an open-loop shove.

        THE FORCE IS PROJECTED ONTO THE TRAVEL AXIS, not taken as a magnitude. A magnitude counts
        side loads from a misaligned entry just as eagerly as real contact, and would report the
        preload met while the coupler is jammed against a lip rather than resting on the bottom
        of the feature."""
        if not preload['enabled']:
            return True
        spec = mating_direction(self.legs[leg])
        dir_base = (T_at[:3, :3] @ spec['axis'] if spec['frame'] == 'coupler' else spec['axis'])
        ref = lambda T: T @ inverse(self.T_tool0_coupler)       # noqa: E731 -- one expression
        log.info('  %s: pushing along the travel axis to %.2f N (up to %.1f mm past the target, '
                 'in %.2f mm steps).', what, preload['force_n'],
                 preload['max_travel_m'] * 1000.0, preload['step_m'] * 1000.0)

        travelled, T_ref, reached = 0.0, T_at, 0.0
        while travelled < preload['max_travel_m']:
            travelled = min(travelled + preload['step_m'], preload['max_travel_m'])
            T_next = offset_pose(T_at, dict(spec, distance_m=travelled))
            (self._active_law or self._adm()).ramp(
                ref(T_ref), ref(T_next), max(preload['settle_s'], 0.1),
                self.guard, on_step=watch)
            T_ref = T_next
            # arm.wrench() is the EXTERNAL force ON the tool, so pushing INTO the surface is felt
            # as a push BACK along -travel_dir. Negating makes "pressing harder" positive.
            reached = -float(np.dot(self.robot.arm.wrench()[:3], dir_base))
            if reached >= preload['force_n']:
                log.info('  %s PRELOAD %.2f N reached after %.2f mm of reference travel.%s',
                         what, reached, travelled * 1000.0, self._weight_note(reached))
                return True
        log.error('The %s preload never reached %.2f N -- %.2f N after the full %.1f mm of '
                  'travel. Either nothing is being touched (the located pose is off, or the '
                  'object moved), or the compliance is soft enough that this much reference '
                  'travel cannot raise that force. Nothing is seated, so the run stops here.',
                  what, preload['force_n'], reached, preload['max_travel_m'] * 1000.0)
        return False

    def _weight_note(self, reached_n):
        """How much of the HELD object's weight the surface is taking, while placing.

        WORTH SAYING OUT LOUD because the same number means different things on the two ends of
        the cycle. Mating, 1 N is a firm seat against a fixture. PLACING, the arm is still
        carrying the object, so 1 N against a 32 N object means the bench has taken 3 per cent of
        it -- the object has kissed the surface, not settled onto it. That is a perfectly good
        touchdown detector and a poor 'safe to let go' test, and the difference is invisible
        unless it is printed."""
        held = float(self.obj.get('held_mass_kg') or 0.0)
        if not self.holding or held <= 0.0:
            return ''
        weight_n = held * 9.81
        return (' The surface is taking %.0f%% of the object\'s %.1f N weight.'
                % (100.0 * reached_n / weight_n, weight_n))

    def prepare_coupler(self):
        """Power the actuation and open the jaws, BEFORE the arm goes anywhere near the object.

        Done here rather than just before the hold because both failures it prevents are silent:
        an unpowered relay lets the servo swing with nothing behind it, and a board that booted
        latched answers a hold by re-reading the sensor and confirming without moving. Either
        way the run would mate, "grip", lift, and carry nothing -- and the first evidence would
        be the object still sitting on the bench."""
        if not self.coupler.prepare_to_mate():
            log.error('The coupler could not be powered and opened, so a mate would be a no-op. '
                      'Nothing has moved.')
            return False
        return True

    def park_coupler(self):
        """Switch the relay off once the object is down and the arm has withdrawn.

        ONLY ON THE WAY OUT, and only after the release: cutting the actuation while something
        is still held is how a part gets dropped."""
        self.coupler.set_motor(False)
        return True

    def lock(self):
        """Lock, then ask the board a SECOND time before anything lifts.

        hold() confirms once, at the instant the servo finished. A part that was merely resting
        where the probe could see it can settle back out in the moment after, and the next steps
        are a payload change and a lift -- both of which are wrong, silently, if the object is
        not actually on. The re-check costs one serial round trip."""
        if not self.coupler.hold():
            log.error('Not locked. The arm is left where it is, with the coupler still in the '
                      'feature -- back it out by hand once you know why.')
            return False
        again = self.coupler.verify()
        if again is False:
            log.error('The coupler confirmed the lock and then the sensor disagreed on a '
                      're-check. The object is not reliably held; refusing to lift it.')
            return False
        if again is None:
            log.warning('The coupler reports locked but NOTHING VERIFIED IT -- no board, or no '
                        'sensor. The lift below assumes the object is on.')
        return True

    def take_payload(self):
        """Tell the controller what it is now carrying, before the first move that carries it."""
        held = self.obj.get('held_mass_kg')
        if not held:
            log.warning('%r has no held_mass_kg in the catalogue, so the payload is unchanged. '
                        'Every wrench from here on reads the object\'s weight as external '
                        'force.', self.name)
            return True
        cog_mm = self.cfg.get('held_cog_mm')
        cog_m = (np.asarray(cog_mm, dtype=float) / 1000.0 if cog_mm is not None
                 else self.T_tool0_coupler[:3, 3])
        payload = combined_payload(self.cfg.section('robot').get('payload', {}), held, cog_m)
        log.info('Payload -> %.2f kg (tool + %s), CoG %s mm.', payload['mass_kg'], self.name,
                 np.round(np.asarray(payload['cog_m']) * 1000.0, 1).tolist())
        self.robot.arm.set_payload(payload)
        # THE LAW CHANGES WITH THE MASS. Everything after this point moves 12x what the mate
        # moved on this cell, and the compliance gains switch to `compliance_loaded:` at the same
        # instant so the reference stops asking for accelerations the loaded arm cannot track.
        self.holding = True
        log.info('Compliance -> LOADED law (%.1f Hz vs %.1f Hz free, translational).',
                 law_bandwidth_hz(self.adm_loaded.M, self.adm_loaded.S)[0],
                 law_bandwidth_hz(self.adm.M, self.adm.S)[0])
        # NO TARE HERE, DELIBERATELY. The object is locked on but STILL RESTING on whatever it
        # was picked from, so the bench is carrying its weight -- and with the payload now set
        # for the airborne case, that shows up as a genuine ~32 N of external force. Zeroing it
        # would declare that real force to be nothing; the instant the object leaves the surface
        # the true reading returns to zero and the law sees the whole weight as a phantom pulling
        # the other way. At the loaded stiffness that is 32 mm of compliant deflection, applied
        # in the tool frame while the phantom acts along gravity -- which is a lift that visibly
        # slews sideways. The tare belongs AFTER the object is clear; see settle_after_lift().
        return True

    def lift(self):
        """Compliant, not free-space: if the object is bolted down, fixtured, or simply did not
        release, this is where that is discovered -- and the guard trips instead of the arm
        pulling against it.

        TARE-FREE ON PURPOSE. This leg begins with the object locked on and still resting on the
        surface, so the sensor is reading the weight the surface is taking. That is a real force,
        not an offset, and zeroing it turns it into a phantom of equal size the moment the object
        lifts clear. See take_payload."""
        return self._compliant(self.T_pick, offset_pose(self.T_pick, self.legs['pick_retract']),
                               'retract with the object', tare=False)

    def settle_after_lift(self):
        """Tare NOW -- the one moment in the cycle when the object really is hanging free.

        Everything downstream reads force against this zero: the carry's guard, the set-down's
        contact detection, the placement preload. Taking it here also absorbs whatever the
        payload's assumed CoG got wrong, which is the other standing bias while carrying."""
        if self.tare_before and not self.robot.arm.dry_run:
            self.robot.arm.zero_ft()
            log.info('F/T tared with %r hanging free -- the first moment in the cycle it is.',
                     self.name)
        return True

    # ---- the place ---------------------------------------------------------------------------
    def carry(self):
        """Retracted pose -> the place standoff. Free space with the object in hand."""
        return self._move_to(offset_pose(self.T_place, self.legs[self.TARGET_STANDOFF]),
                             f'carry to the {self.TARGET_WORD} standoff')

    def set_down(self):
        """Lower to the place pose, then push until the surface pushes back.

        SAME REASONING AS THE MATE, same tare rule. The place pose is a computed pose -- a delta
        from a camera-found pick -- so arriving at it proves the arm got where it believed the
        surface was, not that the object is touching. The F/T is tared at the place standoff with
        the object hanging free, and NOT again, so every newton read on the way down is contact.

        THE THRESHOLD MEANS SOMETHING DIFFERENT HERE, though, and the run says so: the arm is
        still carrying the object, so the force is the share of its weight the surface has taken.
        See _weight_note."""
        return self._compliant(offset_pose(self.T_place, self.legs[self.TARGET_STANDOFF]),
                               self.T_place, f'{self.TARGET_WORD} the object',
                               after=lambda T, w: self._push_to_preload(
                                   T, w, self.preload_place, self.TARGET_STANDOFF,
                                   self.TARGET_WORD),
                               law=self.adm_insert)

    def unlock(self):
        """A refused release means the sensor still sees the object -- it has NOT let go, and
        withdrawing would drag it off the bench."""
        if not self.coupler.release():
            log.error('The coupler did not release. The object is still attached; do not '
                      'withdraw until it has let go.')
            return False
        return True

    def drop_payload(self):
        """Hand the object's mass back, and again DO NOT tare.

        The mirror of take_payload: the object has been released but the coupler is still down in
        it, carrying whatever the placement preload left. Zeroing against that turns it into a
        phantom the moment the withdrawal breaks contact -- smaller than the pick's 32 N, because
        it is only the preload, but the same mistake."""
        payload = dict(self.cfg.section('robot').get('payload', {}) or {})
        log.info('Payload -> %.2f kg (tool alone).', float(payload.get('mass_kg', 0.0)))
        self.robot.arm.set_payload(payload)
        self.holding = False
        return True

    def withdraw(self):
        """Compliant on the way out too. The coupler is still inside the feature until it has
        cleared it, and a stiff retreat against a part that has not let go drags the whole
        fixture rather than stopping."""
        return self._compliant(self.T_place,
                               offset_pose(self.T_place, self.legs['final_retract']),
                               'withdraw from the object', tare=False)

    # ---- motion primitives -------------------------------------------------------------------
    def _move_to(self, T_base_coupler, label):
        """Free-space move of the COUPLER frame, with the force guard armed as a canceller."""
        from ..apps._common import guarded
        ok = guarded(self.robot, self.guard,
                     lambda: self.robot.arm.move_frame_to(T_base_coupler, self.T_tool0_coupler,
                                                          label),
                     label=label)
        if not ok:
            log.error('%s did not finish -- check reach and the joint limits at that pose.',
                      label)
        return ok

    def _adm(self):
        """The law for the current payload state."""
        return self.adm_loaded if self.holding else self.adm

    def _compliant(self, T_from, T_to, what, after=None, tare=True, law=None):
        """Ramp the COUPLER frame T_from -> T_to under the admittance law.

        Mirrors skills/pick._compliant_move, which does the same for the gripper's fingertip;
        the difference is only which tool frame the reference is built from. Leaves the arm OUT
        of the servo loop whatever happens, because the next step is an ordinary move."""
        ref = lambda T: T @ inverse(self.T_tool0_coupler)          # noqa: E731 -- one expression
        duration = seg_time(T_from, T_to,
                            float(self.cfg.get('compliant_speed_mm_s', 20.0)),
                            float(self.cfg.get('compliant_rot_speed_deg_s', 15.0)),
                            min_s=0.5)
        # TARED HERE, AT T_from, AND NOWHERE ELSE IN THIS LEG. For the mate that pose is the
        # standoff, with nothing touching, which is the only place a zero means anything: every
        # newton read after it is contact. The tare happens MID-WARMUP, with the servo already
        # engaged and the arm static, because servoL engaging produces a joint-torque transient
        # that reads as tens of phantom newtons -- zeroing before it would bake that in.
        # `tare` is FALSE for any leg that starts with something already in contact. Zeroing
        # there does not remove an offset -- it declares a real external force to be zero, and
        # the moment that force goes away the law sees its negative as a phantom and yields to
        # it. See lift().
        tare_fn = ((lambda: self.robot.arm.zero_ft(settle=False))
                   if (tare and self.tare_before) else None)
        # `law` overrides the payload-state law for one leg -- the insertion uses its own
        # SHAPE of compliance, not just its own speed. Recorded on self so the preload
        # push that runs inside this session pushes against the same spring.
        adm = law or self._adm()
        self._active_law = adm
        adm.reset()
        adm.warmup(ref(T_from), tare_fn=tare_fn)
        if tare_fn is not None:
            log.info('  F/T tared at the start of this leg (nothing in contact yet).')
        elif self.tare_before:
            log.info('  NOT taring here -- something is already in contact, and a zero taken '
                     'against a real force becomes a phantom when that force goes away.')
        self.guard.reset()
        which = ('INSERTION' if adm is self.adm_insert
                 else 'LOADED' if self.holding else 'free')
        log.info('%s under the %s law (S=%.0f/%.0f/%.0f N/m trans, %.0f Nm/rad rot) over %.1f s, '
                 'guarded at %.0f N / %.1f Nm.', what, which,
                 adm.S[0], adm.S[1], adm.S[2], adm.S[3], duration,
                 self.guard.max_force, self.guard.max_torque)
        peak = {'f': 0.0, 'tau': 0.0}

        def watch():
            w = self.robot.arm.wrench()
            peak['f'] = max(peak['f'], float(np.linalg.norm(w[:3])))
            peak['tau'] = max(peak['tau'], float(np.linalg.norm(w[3:])))

        try:
            status = adm.ramp(ref(T_from), ref(T_to), duration, self.guard, on_step=watch)
            if status == 'seated':
                # A trip is a NORMAL outcome for a contact phase -- the coupler has met the
                # feature, or the object has met the bench -- and the compliant hold that
                # follows is what lets the mechanism settle into it.
                log.info('  contact reached -- holding here, compliant.')
            else:
                adm.hold(ref(T_to), self.settle_s, self.guard, on_step=watch)
            # The continuation runs INSIDE the servo session and after the tare, so it can read
            # contact force against a zero taken where nothing was touching.
            if after is not None and not after(T_to, watch):
                return False
            delta = adm.delta
            log.info('  %s: peak %.1f N / %.2f Nm; the law yielded %.1f mm / %.2f deg at the '
                     'end.', what, peak['f'], peak['tau'],
                     float(np.linalg.norm(delta[:3])) * 1000.0,
                     float(np.degrees(np.linalg.norm(delta[3:]))))
            return True
        finally:
            self._active_law = None
            self.robot.arm.servo_stop()

    # ---- reporting ---------------------------------------------------------------------------
    def report(self):
        """What actually happened to the object, in the object's own terms."""
        reached = self.robot.arm.tcp_pose() @ self.T_tool0_coupler
        lin, ang = pose_error(self.T_place, reached)
        xyz, rpy = matrix_to_xyzrpy(inverse(self.T_place) @ reached)
        log.info('PLACED %r: %.1f mm / %.2f deg from the commanded set-down '
                 '(dx %+.1f dy %+.1f dz %+.1f mm in the object frame).', self.name,
                 lin * 1000.0, np.degrees(ang), *[v * 1000.0 for v in xyz])
        log.info('  the object\'s mating frame was coincident with coupler_mate throughout, so '
                 'that offset is the object\'s, not the flange\'s.')
        return True


# ---------------------------------------------------------------------------- entry point
def build_and_run(cfg, robot, camera, args, cycle=None):
    """`cycle` is the CouplerCycle subclass to run -- apps/coupler_pick_assemble
    passes its own. Everything else about the run is identical."""
    cycle = cycle or CouplerCycle
    name = cfg.get('object_name')
    path = tool_frames.objects_path(cfg)
    try:
        catalogue = tool_frames.load_objects(cfg)
    except ValueError as exc:
        log.error('%s', exc)
        return False
    if not name:
        log.error('object_name is required. Catalogued objects in %s: %s.',
                  path, ', '.join(sorted(catalogue)) or '(none)')
        return False
    if name not in catalogue:
        log.error('%r is not in %s. Catalogued: %s. Calibrate it first with '
                  'urlab.apps.object_calibration.', name, path,
                  ', '.join(sorted(catalogue)) or '(none)')
        return False
    obj = catalogue[name]
    meta = obj.get('meta') or {}
    log.info('Object %r: %d marker%s, %s kg, calibrated %s from %s mate(s).', name,
             len(obj['markers']), '' if len(obj['markers']) == 1 else 's',
             obj.get('held_mass_kg'), meta.get('measured', '?'), meta.get('mates', '?'))
    for mid in sorted(obj['markers']):
        m = obj['markers'][mid]
        mm = m.get('meta') or {}
        log.info('  marker %d: %.2f mm, residual %s mm / %s deg over %s mate(s).', mid,
                 m['size_m'] * 1000.0, mm.get('residual_mm', '?'),
                 mm.get('residual_axis_deg', '?'), mm.get('mates', '?'))
    if len(obj['markers']) == 1:
        log.info('  ONE MARKER: nothing can outvote it, so a knock or a re-stick moves every '
                 'pick with it and there is no disagreement to notice. Three or more is what '
                 'makes the run-time RANSAC able to reject one.')
    if int(meta.get('mates', 0) or 0) < 2:
        log.warning('This entry came from a SINGLE mate, so its residuals are zero by '
                    'construction and say nothing about how repeatable the pick is. Re-run the '
                    'calibration with mates: >= 3 before trusting it.')

    try:
        plan = mloc.ViewPlan(cfg.section('marker_views'))
    except ValueError as exc:
        log.error('%s', exc)
        return False

    # Imported here, not at module load: perception pulls in cv2, and the geometry is worth
    # testing on a machine that has no OpenCV.
    from ..perception import ArucoDetector
    detector = ArucoDetector(cfg, sizes_m={int(mid): m['size_m']
                                           for mid, m in obj['markers'].items()})

    coupler = Coupler(cfg)
    step = step_gate(cfg, robot)
    if step is not None:
        log.info('confirm_each_step is ON -- every motion waits for Enter. --yes runs straight '
                 'through.')
    try:
        job = cycle(cfg, robot, camera, detector, plan, name, obj, coupler)
    except ValueError as exc:
        log.error('%s', exc)
        coupler.close()
        return False
    for leg in job.LEGS:
        log.info('  %s', describe_leg(leg, job.legs[leg]))
    free_hz = law_bandwidth_hz(job.adm.M, job.adm.S)[0]
    loaded_hz = law_bandwidth_hz(job.adm_loaded.M, job.adm_loaded.S)[0]
    log.info('  compliance: %.1f kg virtual / %.0f N/m -> %.2f Hz free; %.1f kg / %.0f N/m -> '
             '%.2f Hz holding %s (%.2f kg).', job.adm.M[0], job.adm.S[0], free_hz,
             job.adm_loaded.M[0], job.adm_loaded.S[0], loaded_hz, name,
             float(obj.get('held_mass_kg') or 0.0))
    ins = job.adm_insert
    log.info('  insertion law: M %.1f kg, S %s N/m -- %s.', ins.M[0],
             np.round(ins.S[:3], 0).tolist(),
             'its own shape' if ins is not job.adm_loaded else 'same as the loaded law')
    travel = mating_direction(job.legs[job.TARGET_STANDOFF])
    if not insertion_axis_is_stiff(travel, ins.S):
        log.warning('  THE INSERTION STIFFNESS PROFILE POINTS THE WRONG WAY. The part travels '
                    'along %s of the coupler, but that is not the stiff axis -- so the law is '
                    'compliant where the push is wanted and rigid where the give is. The law '
                    'runs in the tool0 frame, so reorder compliance_insert.stiffness to match '
                    'motion.%s.axis.', np.round(travel['axis'], 2).tolist(),
                    job.TARGET_STANDOFF)
    if job.adm_loaded.M[0] == job.adm.M[0] and (obj.get('held_mass_kg') or 0.0) > 1.0:
        log.warning('  the loaded law is the same as the free one while carrying %.1f kg. If the '
                    'arm oscillates after the pick, raise compliance_loaded.mass -- the virtual '
                    'mass is a design parameter and does not follow the payload on its own.',
                    float(obj['held_mass_kg']))
    # The one-off gate before the first motion toward the object is UNCONDITIONAL -- it is not
    # silenced by --yes, because "run the steps without asking" is not the same request as
    # "drive at the part without telling me". --no-prompts still skips it, and so does the
    # per-step gate, which has already asked by then.
    root = bt.sequence(
        'coupler-pick-place',
        bt.Action('locate the object', job.locate, confirm=step),
        bt.Action('power and open the coupler', job.prepare_coupler, confirm=step),
        bt.OperatorGate(robot, 'About to approach and MATE. Clear of the arm? (Enter / q): ',
                        label='before the mate',
                        skip=prompts_off(cfg) or step is not None),
        bt.Action('align at the mate standoff', job.approach, confirm=step),
        bt.Action('mate with the coupling feature', job.descend_and_mate, confirm=step),
        bt.Action('lock the coupler', job.lock, confirm=step),
        bt.Action('take the payload', job.take_payload),
        bt.Action('retract with the object', job.lift, confirm=step),
        bt.Action('tare with the object clear', job.settle_after_lift),
        bt.Action(f'carry to the {job.TARGET_WORD} standoff', job.carry,
                  confirm=step),
        bt.Action(f'{job.TARGET_WORD} the object', job.set_down, confirm=step),
        bt.Action('release the coupler', job.unlock, confirm=step),
        bt.Action('drop the payload', job.drop_payload),
        bt.Action('withdraw', job.withdraw, confirm=step),
        bt.Action('switch the coupler motor off', job.park_coupler),
        bt.Action('report', job.report))
    try:
        return bt.run_tree(root, log)
    finally:
        robot.arm.servo_stop()
        # NOT switching the motor off here. A run that failed part way may still be HOLDING the
        # object, and cutting the actuation under it is how a part gets dropped -- park_coupler
        # does it on the way out, after the release. Note that closing the port resets the board
        # and drops the relay anyway (the driver's dead-man switch), so a failed run still ends
        # unpowered; if that is the wrong trade for a heavy object, the driver's `latch` option
        # is what keeps it alive.
        coupler.close()


def main():
    # with_gripper=False: the coupler does the holding, and the 2F-85 may not even be fitted.
    run_app('Coupler pick and place: find a catalogued object, mate, carry, set it down',
            'coupler_pick_place', build_and_run, with_gripper=False, needs_camera=True)


if __name__ == '__main__':
    main()
