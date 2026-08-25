"""Grasp helpers shared by the cable demos -- pre-grasp geometry, the counts-based grasp check,
and the open/return/retry recovery.

Ported from PickPlace's grasp/lift/place geometry and CablePickPlace's grasp-check + recovery.
The grasp check is genuinely better here than in the ROS version, for a hardware reason: the
Robotiq talks to us in COUNTS directly (see robot/gripper.py), so the whole rad->counts
conversion through `full_close_rad` -- the config value that was permanently marked "TODO: verify
me" -- is gone. `closed_counts: 228` is now compared against a number the gripper reports.
"""

import numpy as np

from .. import log as urlog
from ..transforms import UR_JOINTS, fmt_delta, inverse, pose_error, translation_matrix

log = urlog.get('pick')

# --------------------------------------------------------------- the fingertip grasp target
# ONE TRANSFORM DECIDES THE WHOLE GRASP: `pickup.fingertip_in_connector` is the TARGET
# FINGERTIP FRAME EXPRESSED IN THE DETECTED CONNECTOR FRAME, so the arm is commanded to
#
#     T_base_fingertip = detected_connector @ fingertip_in_connector
#
# and there is nothing else in the chain. It replaces the pitch_deg / roll_deg / grip_offset_mm
# trio, which were three partial descriptions of that one pose composed in a fixed order -- a
# shape that made the approach angle depend on cables.yaml's junction_in_fingertip yaw and made
# "which rotation goes where" a standing question.
#
# THE FRAME IT IS WRITTEN IN is the detected connector (junction) frame, built by the scan
# (skills/scan -> transforms.frame_from_axis). Every axis of it is physical:
#
#     ORIGIN  the detected junction -- where the cable meets the connector -- AT THE GROUND
#             PLANE. The ground_plane scan puts it ON the plane, NOT on the barrel centreline,
#             so a connector resting on the bench needs a POSITIVE z or the pads aim at the
#             floor.
#     +x      ALONG THE CONNECTOR AXIS, from the cable toward the connector's FREE END.
#     +y      HORIZONTAL, ACROSS the cable (y = z cross x) -- also the direction the JAWS CLOSE.
#     +z      THE GROUND NORMAL: UP.


def fingertip_in_connector(cfg):
    """`pickup.fingertip_in_connector` as a 4x4 -- THE TARGET FINGERTIP FRAME WRT THE DETECTED
    CONNECTOR.

    The arm drives the fingertip to `detected_connector @ this`, so it alone says where the
    hand ends up: the translation places the bite point, the rotation aims the approach. See
    the frame definition above for what each axis means.

    THE ROTATION, in the common cases (the fingertip's own z is the approach direction):
        rpy_deg [0,   0, 0]   the fingers come straight DOWN onto the cable.
        rpy_deg [0, -60, 0]   tilted 60 deg over toward the connector's free end.
        rpy_deg [0, -90, 0]   AXIAL: tool0 +Z along the connector +X and tool0 -Y along its +Z,
                              the flange on the connector axis 183 mm behind the bite. Both
                              alignments are off parallel by exactly 90 - |pitch|, so the y of
                              this rpy is the ONE number that sets the whole approach.

    ACCEPTS MONITOR UNITS: xyz_mm / rpy_deg, paste-able off urlab.apps.monitor; xyz / rpy in
    m / rad also work. Both units for one triple is an error, not a silent preference.

    THE LEGACY FALLBACK, used when the block is absent, reproduces exactly what the old chain
    did for a config that never set an approach angle: slide `pickup.grip_offset_mm` along the
    connector axis and take the orientation from cables.yaml's junction_in_fingertip. That is
    what keeps the un-migrated pick apps bit-identical."""
    from ..config import _pose_si
    from ..transforms import from_cfg, translation_matrix
    block = cfg.get_path('pickup.fingertip_in_connector')
    if block is not None:
        return from_cfg(_pose_si(block))
    d = float(cfg.get_path('pickup.grip_offset_mm', 0.0) or 0.0) / 1000.0
    return (translation_matrix([d, 0.0, 0.0])
            @ inverse(from_cfg(cfg.section('junction_in_fingertip'))))


def connector_axis_height_m(cfg):
    """How far the connector's AXIS sits above the ground plane when the part lies flat on it:
    HALF ITS GREATEST DIAMETER.

    WHY IT BELONGS TO THE ESTIMATE, NOT THE GRASP. The ground_plane scan measures where the
    cable meets the connector and reports it ON THE PLANE -- that is what the scan can see. The
    connector is a solid resting on that plane, so its axis is one radius up, and it is the
    LARGEST radius that decides it: a stepped barrel rests on its fattest section and every
    other section is lifted clear along with it. Adding this to the DETECTED POSE fixes the
    measurement once, for everything downstream; folding it into the grasp offset instead would
    hide a property of the part inside a choice about the approach, and would silently be wrong
    the moment the part is picked up off a fixture rather than off the bench.

    WHERE THE DIAMETER COMES FROM, in order:
      1. `grasp_check.connector_diameter_mm` -- measured with calipers. Preferred.
      2. `grasp_check.connector_counts` -- the measured grasp-check band, converted through the
         calibrated gripper model. The LOW count is the FAT end of the band (more obstruction =
         less closed), so it is the one that gives the greatest diameter.
    Returns 0.0 when neither is available, and the caller says so rather than guessing."""
    d_conn = cfg.get_path('grasp_check.connector_diameter_mm')
    if d_conn:
        return max(float(v) for v in d_conn) / 1000.0 / 2.0
    counts = cfg.get_path('grasp_check.connector_counts')
    if counts:
        from ..robot.gripper_kinematics import width_from_counts
        groove = cfg.get_path('gripper.groove_depth_mm')
        kw = {} if groove is None else {'groove_depth_m': float(groove) / 1000.0}
        return width_from_counts(min(int(c) for c in counts), **kw) / 2.0
    return 0.0


def belief_offset(cfg):
    """The belief offset as a FULL POSE (4x4) in the CONNECTOR frame.

    `pickup.belief_offset` is a pose block in monitor units (`xyz_mm` + `rpy_deg`) or SI
    (`xyz` + `rpy`). The older translation-only `pickup.belief_offset_mm: [x, y, z]` still works
    and means the same thing with no rotation; setting both is an error rather than a silent
    preference.

    WHY ROTATION MATTERS AT LEAST AS MUCH AS TRANSLATION. The residual this carries is where the
    part sits in the jaws, and a part that is TILTED in the grooves is the common case, not the
    exotic one: the cable lies flat on the bench while the jaws come down at an angle, so the
    barrel is captured across its axis. A tilt of a degree at the grip becomes a lateral error of
    the connector's length times sin(theta) at the TIP -- which is the end that has to find the
    socket. Expressing the belief as a translation only meant that error had to be smeared into an
    xyz that was right at one depth and wrong at every other.

    THE MEASURED RESIDUAL, not a derived one -- and the reason this exists at all is that the
    two things it separates are genuinely different:

      * the GRASP COMMAND (grip_offset) says where the fingers GO. Change it and the arm moves
        somewhere else.
      * the BELIEF says where the connector then IS relative to those fingers. Change this and
        NOTHING moves at pickup -- only the assembly's idea of what it is carrying.

    held_belief() derives the belief from the command exactly, and its round trip is exact --
    but only under the assumption that the part seats in the jaws the same way at every
    approach angle. It does not. The cable lies on the ground and stays horizontal while the
    JAWS tilt, so tilted grooves capture the cylinder at a different depth than square ones.
    That difference is a contact fact: measurable, not derivable from any frame. This is where
    the measurement goes, so it cannot be confused with the geometry it corrects."""
    from ..config import _pose_si
    from ..transforms import from_cfg, translation_matrix
    block = cfg.get_path('pickup.belief_offset')
    legacy = cfg.get_path('pickup.belief_offset_mm')
    if block is not None and legacy is not None:
        raise ValueError(
            'pickup.belief_offset and pickup.belief_offset_mm are both set. They are the same '
            'quantity -- the full-pose form and the translation-only one -- and silently '
            'preferring one would apply a rotation nobody asked for, or drop one they did. '
            'Keep pickup.belief_offset.')
    if block is not None:
        return from_cfg(_pose_si(block))
    if legacy is None:
        return np.eye(4)
    v = [float(x) for x in legacy]
    if len(v) != 3:
        raise ValueError('pickup.belief_offset_mm must be [x, y, z] mm in the connector frame, '
                         f'got {len(v)} entries')
    return translation_matrix(np.asarray(v, dtype=float) / 1000.0)


def offset_belief(T_ftip_conn, T_offset):
    """Move the believed connector by a FULL POSE `T_offset`, read in ITS OWN frame.

    RIGHT-MULTIPLIED, so the delta is read in the CONNECTOR frame: +x along the connector axis,
    y and z its own transverse axes. "the part sits 10 mm further back along its own body",
    not "10 mm deeper into the hand".

    WHY THE PART'S FRAME AND NOT THE HAND'S. The residual it carries is a fact about where the
    connector ends up relative to the fingers, and either frame can express that -- but the
    part frame is the one the rest of this geometry is now written in. The grasp itself is
    `fingertip_in_connector`, and the socket, the insertion axis and the clocking rotations are
    all connector-frame quantities. A hand-frame delta was the odd one out, and its axes swing
    with the approach angle: the same physical error needed a different number at every
    pickup pitch. In the part's frame it does not.

    THE TRADE, stated plainly: a residual that really is a property of the JAWS -- the groove
    capturing the cylinder at a different depth along the approach -- is more naturally a
    hand-frame quantity, and in the part frame it will move as the pitch changes. Which one a
    given measurement belongs in depends on where the error comes from. Re-measure after a
    large change of approach angle either way."""
    T_offset = np.asarray(T_offset, dtype=float)
    if T_offset.shape != (4, 4):
        raise ValueError(
            'offset_belief takes a 4x4 pose, not a %s. The belief offset carries a ROTATION now; '
            'build it with belief_offset(cfg) rather than passing a translation vector, or a '
            'tilt in the config would be silently dropped.' % (T_offset.shape,))
    return T_ftip_conn @ T_offset


def grasp_pose(T_base_connector, T_conn_ftip):
    """The FINGERTIP grasp pose: the detected connector, with the grip offset applied.

    The entire grasp command, in one product. Note what is NOT here: cables.yaml's
    junction_in_fingertip no longer steers the arm at all. It survives as the descriptor of the
    NOMINAL grip that held_belief reads the part's own geometry out of -- see there."""
    return T_base_connector @ T_conn_ftip


def held_junction_in_fingertip(T_conn_ftip):
    """Where the detected connector frame sits in the FINGERTIP frame after the grasp.

    Just the inverse: the fingertip was commanded to `connector @ T_conn_ftip`, so from the
    fingertip's point of view the connector is back at `inverse(T_conn_ftip)`. Kept as a named
    function because that inversion is the step everything downstream actually needs, and
    writing it out at each call site is how the sign gets dropped."""
    return inverse(T_conn_ftip)


def held_belief(T_ftip_conn_nominal, T_ftip_junction_nominal, T_conn_ftip):
    """The in-hand connector belief this grasp produces.

    TWO INPUTS DESCRIBE THE PART, ONE DESCRIBES THE GRASP, and separating them is the point:

      * `T_ftip_conn_nominal` (estimation.initial_connector_in_fingertip / frames.yaml) and
        `T_ftip_junction_nominal` (cables.yaml junction_in_fingertip) are both written for the
        SAME nominal grip, so dividing one by the other cancels that grip out and leaves
        T_junction_connector -- a property of the PART, true however it is held.
      * `T_conn_ftip` is how this run actually took it.

    So the belief is the part's own geometry, seen from wherever the fingers ended up. Exact at
    every offset, because it is the same product the grasp was commanded from -- there is no
    second derivation to drift."""
    return held_junction_in_fingertip(T_conn_ftip) @ inverse(
        T_ftip_junction_nominal) @ T_ftip_conn_nominal



class GraspGeometry:
    """The grasp/pre-grasp/lift/place poses, all derived from one grasp target on demand -- so a
    refined estimate that updates the target automatically feeds every downstream pose."""

    def __init__(self, cfg):
        self.approach_distance = float(cfg.get('approach_distance_m', 0.10))
        self.approach_axis = np.asarray(cfg.get('approach_axis', [0.0, 0.0, -1.0]), dtype=float)
        # WHICH FRAME approach_axis IS READ IN -- see pre_grasp for why it matters.
        self.approach_frame = str(cfg.get('approach_frame', 'grasp')).lower()
        if self.approach_frame not in ('grasp', 'base'):
            raise ValueError("approach_frame must be 'grasp' or 'base', got "
                             f'{self.approach_frame!r}')
        self.lift_distance = float(cfg.get('lift_distance_m', 0.10))
        self.lift_axis = np.asarray(cfg.get('lift_axis', [0.0, 0.0, 1.0]), dtype=float)
        from ..transforms import xyzrpy_to_matrix
        self.place_offset = xyzrpy_to_matrix(cfg.get('place_offset_xyz', [0.0, 0.20, 0.0]),
                                             cfg.get('place_offset_rpy', [0.0, 0.0, 0.0]))
        self.T_base_grasp = np.eye(4)

    def pre_grasp(self):
        """Stand-off before the grasp -- and WHICH WAY it backs off is `approach_frame`.

        'grasp' (default) reads approach_axis in the GRASP frame, so the stand-off is straight
        back along the gripper's own approach and the descent runs down the tool axis. That is
        right for a square pickup, where the tool axis IS vertical.

        'base' reads it in BASE_LINK, so [0, 0, 1] stands off straight UP off the ground plane
        and the descent comes straight DOWN onto the part, whatever attitude the gripper is
        holding. This is the one to use once fingertip_in_connector tilts the approach: at a
        -60 deg pitch the grasp-frame stand-off sits 100 mm BACK ALONG THE CABLE and the
        descent drags the open jaw down the cable to reach the bite point -- it has to thread
        the cable into the jaw, and a cable that is not straight snags it. Coming from directly
        above, the pads drop past the barrel's two sides instead and nothing is threaded.

        The gripper ATTITUDE is identical either way -- only the direction it retreats along
        changes. It also puts the stand-off on the same axis the lift already uses, so the
        approach and the departure are mirror images."""
        step = translation_matrix(self.approach_axis * self.approach_distance)
        if self.approach_frame == 'base':
            return step @ self.T_base_grasp
        return self.T_base_grasp @ step

    def lift(self):
        return translation_matrix(self.lift_axis * self.lift_distance) @ self.T_base_grasp

    def place(self):
        return self.place_offset @ self.T_base_grasp

    def pre_place(self):
        return translation_matrix(self.lift_axis * self.lift_distance) @ self.place()


class GraspCheck:
    """Classify a completed close from the finger position, in counts.

    The current fingertips give THREE distinct closure levels (more obstruction = LESS closed):

        faces (<= faces_max_counts, ~220-223)  cable caught on the flat faces -> MISSED
        groove (~groove_counts, 225)           cable seated in the fingertip groove -> OK
        empty (>= empty_counts, 228)           fingers closed fully on nothing -> EMPTY

    So a seated cable STOPS the fingers short of full closure (unlike the previous fingertips, where
    a seated cable allowed full closure and EMPTY was indistinguishable). Empty is now reliably
    separable by POSITION (228 vs 225), which is why detect_empty defaults on."""

    def __init__(self, cfg):
        gc = cfg.section('grasp_check')
        self.enabled = bool(gc.get('enabled', True))
        self.groove_counts = int(gc.get('groove_counts', 225))     # SUCCESS: target in the groove
        self.empty_counts = int(gc.get('empty_counts', 228))       # full closure on nothing
        self.faces_max_counts = int(gc.get('faces_max_counts', 223))  # <= this: too thick (faces) -> miss
        gm = gc.get('groove_max_counts', None)                     # > this (but < empty): too THIN -> miss
        self.groove_max_counts = int(gm) if gm is not None else None  # (e.g. cable, not the connector)
        self.tolerance = int(gc.get('tolerance_counts', 1))
        self.detect_empty = bool(gc.get('detect_empty', True))
        self.max_retries = int(gc.get('max_retries', 2))
        self.settle_s = float(gc.get('settle_s', 1.0))
        # LIFT SLIP CHECK (grasp_check.lift_check): raise height_m, RE-CLOSE, re-read the counts.
        lc = gc.get('lift_check', {}) or {}
        self.lift_check_enabled = bool(lc.get('enabled', True))
        self.lift_check_height_m = float(lc.get('height_m', 0.02))
        # After a detected slip: rise this much straight up (no full reset) before rescanning.
        self.slip_raise_m = float(lc.get('slip_raise_m', 0.10))
        # Outer-retry grasp perturbation step along the junction x-axis (0 = off).
        self.retry_perturb_x_m = float(gc.get('retry_perturb_x_m', 0.0))

    def evaluate(self, gripper):
        """'ok' | 'missed' | 'empty' from the settled finger position."""
        if not self.enabled:
            return 'ok'
        import time
        time.sleep(self.settle_s)
        return gripper.grasp_result(self.groove_counts, self.empty_counts, self.faces_max_counts,
                                    self.tolerance, self.detect_empty, self.groove_max_counts)


class GraspRecovery:
    """Failed-grasp recovery for the CONNECTOR pick, classified purely by the stalled finger COUNT
    (more obstruction = LESS closed). The grasp target is the connector; the count after a close
    says what is between the fingers, and each state maps to a DIRECTED reseat in the junction frame
    (x = connector axis toward the connector's END; z = up, so -z = toward the ground):

      * CONNECTOR band  -> the connector is seated: SUCCESS.
      * CABLE (within tolerance of cable_counts) -> grabbed the thinner cable: open, shift
                           cable_shift_fraction * finger_width in +x (toward the connector end), retry.
      * EDGE PINCH (within edge_tolerance_counts of the model's FREE-CLOSURE counts, ~216) ->
                           separation ~0, held width at the 2*groove floor -- LESS than any
                           connector, so the grooves closed PAST the connector's fat section: the
                           grasp is too SHALLOW. Open, move edge_drop_m IN toward the connector
                           (-z), retry. (The trigger comes from the CALIBRATED gripper model, not
                           a hand-typed count.)
      * CLOSED (within tolerance of the closed/empty position) -> nothing grasped: open, drop
                           empty_drop_m in -z (toward the object on the ground), retry.
      * anything else    -> unexpected: open + blind retry (no move).

    The reseat ACCUMULATES into geom.T_base_grasp, so a successful reseat leaves the corrected pose
    for the lift/place. (The old faces/tips reseat is gone.)"""

    def __init__(self, cfg):
        gc = cfg.section('grasp_check')
        rc = (gc.get('recovery', {}) or {})
        self.enabled = bool(rc.get('enabled', True))
        self.max_tries = int(rc.get('max_tries', 5))
        fw = float(rc.get('finger_width_m', 0.02278))
        self.cable_shift = float(rc.get('cable_shift_fraction', 0.8)) * fw   # +x reseat for a cable grab
        self.empty_drop = float(rc.get('empty_drop_m', 0.003))              # -z reseat for an empty close
        # GROUND CONTACT, opt-in (null = off). An EARLY stall -- the fingers stopping while
        # still WIDE -- means something got in their way before they could reach the part, and
        # on a bench that something is the work surface: a finger tip touched down and the
        # closure jammed against it. It is a distinct reading from every other count the check
        # knows: too wide to be the connector band, nowhere near the free-closure point, and
        # the opposite end of the travel from an `empty` full closure. The correction is
        # therefore UP -- lift the fingers off the surface and try again -- where `empty`
        # (fingers shut on nothing, so they are ABOVE the part) drops onto it.
        #
        # Stated per cable rather than derived: what separation counts as 'blocked' depends on
        # the part, the pads and the approach angle.
        _gcnt = rc.get('ground_counts')
        self.ground_counts = None if _gcnt is None else int(_gcnt)
        self.ground_rise = float(rc.get('ground_rise_m', 0.0005))            # +z reseat, up off it
        # EDGE-PINCH band: a stall at ~the FREE-CLOSURE counts means separation ~0 (held width at
        # the 2*groove floor, thinner than any connector) -- the grooves closed PAST the
        # connector's fat section, i.e. the grasp is too SHALLOW. Resolution: -z, IN toward the
        # connector. The trigger counts come from the calibrated gripper model (~216).
        from ..robot.gripper_kinematics import COUNTS_CLOSED
        self.edge_counts = int(round(COUNTS_CLOSED))
        self.edge_tol = int(rc.get('edge_tolerance_counts', 1))
        self.ground_tol = int(rc.get('ground_tolerance_counts',
                                     rc.get('edge_tolerance_counts', 1)))
        self.edge_drop = float(rc.get('edge_drop_m', 0.003))                # -z reseat, deeper on

        # Count bands (from the grasp_check block, set per-cable by apply_cable_profile): the CONNECTOR
        # range is success, cable_counts / the closed position are the two miss states.
        conn = gc.get('connector_counts') or []
        self.connector_lo = int(min(conn)) if conn else int(gc.get('faces_max_counts', 223)) + 1
        self.connector_hi = (int(max(conn)) if conn
                             else int(gc.get('groove_max_counts', gc.get('groove_counts', 225))))
        cc = gc.get('cable_counts')
        self.cable_counts = int(cc) if cc is not None else None
        self.closed_counts = int(gc.get('empty_counts', 228))
        self.tol = int(gc.get('tolerance_counts', 1))
        self.settle_s = float(gc.get('settle_s', 1.0))

        # Optional: save the wrist-camera view at EACH grasp close, labelled with the gripper count
        # (in the filename and drawn on the image), for correlating the visual grasp state with the
        # count bands. Written to <data_dir>/<capture_subdir>/<timestamp>/.
        self.capture_images = bool(gc.get('capture_images', True))
        self._data_root = cfg.get('data_dir', 'data')
        self._capture_subdir = gc.get('capture_subdir', 'grasp_images')
        self._capture_dir = None
        self._capture_seq = 0

    def _capture_grasp(self, camera, count, tag, result):
        """Save the wrist-camera frame labelled with the gripper `count` (filename + on-image).
        Best-effort; a no-op if capture is off or there is no camera."""
        if not self.capture_images or camera is None or getattr(camera, 'dry_run', False):
            return
        try:
            import os
            import cv2
            from datetime import datetime
            if self._capture_dir is None:
                self._capture_dir = os.path.join(self._data_root, self._capture_subdir,
                                                 datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(self._capture_dir, exist_ok=True)
            img = camera.capture().color.copy()
            cv2.putText(img, f'{count} counts  {tag}  {result}', (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            path = os.path.join(self._capture_dir,
                                f'grasp_{self._capture_seq:03d}_{tag}_{count}counts_{result}.png')
            cv2.imwrite(path, img)
            self._capture_seq += 1
            log.info('  saved grasp image %s', path)
        except Exception as exc:                       # noqa: BLE001 -- capture is best-effort
            log.warning('  could not save grasp image: %s', exc)

    def grasp_with_recovery(self, robot, geom, check, camera=None):
        """Close, classify by the finger COUNT, and on a miss reseat in the DIRECTED way for that
        count and retry -- up to max_tries. Returns 'ok' | 'missed' | 'abort'. Leaves the (possibly
        corrected) grasp in geom.T_base_grasp. If `camera` is given and grasp_check.capture_images is
        on, the wrist view is saved (labelled with the count) at each close."""
        import time
        g = robot.gripper
        tries = self.max_tries if self.enabled else 0
        for attempt in range(tries + 1):
            if not g.close('grasp'):
                return 'abort'
            time.sleep(self.settle_s)
            pos = g.position()
            tag = 'grasp' if attempt == 0 else f'reseat{attempt}'
            delta = world_delta = None      # in the fingertip frame / in base_link

            if self.connector_lo <= pos <= self.connector_hi:      # connector seated -> success
                log.info('Grasp OK: %d counts in the connector band [%d, %d].',
                         pos, self.connector_lo, self.connector_hi)
                self._capture_grasp(camera, pos, tag, 'ok')
                return 'ok'
            if attempt >= tries:                                    # out of retries
                self._capture_grasp(camera, pos, tag, 'missed')
                break

            if self.cable_counts is not None and abs(pos - self.cable_counts) <= self.tol:
                log.warning('Grabbed the CABLE (%d ~ %d) -- open, shift %.1f mm +x toward the '
                            'connector end, retry.', pos, self.cable_counts, self.cable_shift * 1000)
                self._capture_grasp(camera, pos, tag, 'cable')
                delta = translation_matrix([self.cable_shift, 0.0, 0.0])   # +x = toward connector end
                reseat = 'reseat +x (toward connector)'
            elif abs(pos - self.edge_counts) <= self.edge_tol:
                log.warning('EDGE PINCH (%d ~ free closure %d): separation ~0 -- the grooves '
                            'closed past the connector, the grasp is too SHALLOW. Open, move '
                            '%.1f mm -z IN toward the connector, retry.',
                            pos, self.edge_counts, self.edge_drop * 1000)
                self._capture_grasp(camera, pos, tag, 'edge')
                delta = translation_matrix([0.0, 0.0, -self.edge_drop])    # -z = deeper onto it
                reseat = 'reseat -z (deeper onto the connector)'
            elif (self.ground_counts is not None
                  and abs(pos - self.ground_counts) <= self.ground_tol):
                # The pads reached full closure because they went past the part and bottomed on
                # the work surface -- so the correction is UP, not the drop an `empty` reading
                # would ask for. Small on purpose: the miss is a fraction of a diameter, and a
                # large rise would clear the part altogether on the next try.
                log.warning('GROUND CONTACT (%d ~ %d): the fingers stalled while still '
                            'WIDE, so something stopped them before they reached the part -- '
                            'a finger is down on the work surface. Open, rise %.1f mm +z, '
                            'retry.', pos, self.ground_counts, self.ground_rise * 1000)
                self._capture_grasp(camera, pos, tag, 'ground')
                # IN THE WORLD, not in the hand. Every other reseat is a nudge along the
                # APPROACH and is right to follow the fingertip; this one is the only reseat
                # aimed at the GROUND PLANE, which does not tilt when the approach does. The
                # two coincide at pitch 0 (fingertip +z IS world up there), so this changes
                # nothing for a square pickup -- but at pickup.pitch_deg -75 the fingertip's
                # own +z is 97% backwards along the cable and 26% up, i.e. the 'rise' would
                # retreat instead of lift.
                world_delta = translation_matrix([0.0, 0.0, self.ground_rise])
                reseat = 'reseat +z world (up off the ground)'
            elif abs(pos - self.closed_counts) <= self.tol:
                log.warning('EMPTY close (%d ~ closed %d) -- open, drop %.1f mm -z toward the '
                            'object, retry.', pos, self.closed_counts, self.empty_drop * 1000)
                self._capture_grasp(camera, pos, tag, 'empty')
                delta = translation_matrix([0.0, 0.0, -self.empty_drop])   # -z = toward the ground
                reseat = 'reseat -z (toward ground)'
            else:                                                   # not connector/cable/closed
                log.warning('Grasp count %d is not connector/cable/closed -- open + blind retry.', pos)
                self._capture_grasp(camera, pos, tag, 'other')
                delta = None
                reseat = 'blind retry'

            if not g.open('reposition'):
                return 'abort'
            if delta is not None:
                geom.T_base_grasp = geom.T_base_grasp @ delta       # ACCUMULATE, in the hand
            if world_delta is not None:
                geom.T_base_grasp = world_delta @ geom.T_base_grasp  # ACCUMULATE, in base_link
            if delta is not None or world_delta is not None:
                if not robot.move_fingertip(geom.T_base_grasp, reseat):
                    return 'abort'

        log.error('Grasp not seated in the connector band after %d tries.', self.max_tries)
        return 'missed'


def _compliant_move(robot, adm, guard, T_start_ftip, T_target_ftip, duration,
                    tare_before=True, settle_s=0.5, what='move'):
    """Move the fingertip from its CURRENT pose to T_target under SOFTWARE ADMITTANCE
    (robot/admittance.py) instead of a stiff moveL -- the SAME law the assembly insert uses. The
    fingertip yields to contact (a misplaced cable, the work surface, a cable that resists the lift)
    and springs back toward the reference. Returns True on completion (or an early guard trip =
    contact); leaves the arm OUT of the servo loop.

    CARTESIAN reference: the tool0 pose is slerped start -> target. This is the straight-line tool
    path, but near an IK branch boundary / SINGULARITY the servoL Jacobian inverse spikes the joint
    velocity and the shoulder jerks into the controller's COLLISION-DETECTION protective stop (no
    real contact). Keep the working region away from singularities (mid-range elbow/wrist)."""
    T_t0_ft = robot.T_tool0_fingertip

    def ref(T_ft):
        return T_ft @ inverse(T_t0_ft)                  # fingertip pose -> tool0 servoL reference

    # Tare MID-WARMUP (servo engaged, static) so the guard baseline matches the servo-active reading.
    tare = (lambda: robot.arm.zero_ft(settle=False)) if tare_before else None
    adm.reset()
    adm.warmup(ref(T_start_ftip), tare_fn=tare)         # settle the servo (+ tare) before the guard
    if guard is not None:
        guard.reset()
    log.info('%s under ADMITTANCE (S=%.0f N/m trans, %.0f Nm/rad rot) over %.1fs%s.',
             what, adm.S[0], adm.S[3], duration,
             '' if guard is None
             else f', guarded at {guard.max_force:.0f} N / {guard.max_torque:.1f} Nm')
    # PEAK WRENCH over the whole move. Under admittance the arm YIELDS, so the commanded
    # pose says nothing about how hard the part (or the table) was actually pushed -- the only
    # record of that is the force seen while it happened. Worth a line at every pickup: a
    # descent that peaks near the guard limit is one nudge away from tripping, and one that
    # peaks at nothing never touched the part at all.
    peak = {'f': 0.0, 'tau': 0.0}

    def _watch():
        w = robot.arm.wrench()
        peak['f'] = max(peak['f'], float(np.linalg.norm(w[:3])))
        peak['tau'] = max(peak['tau'], float(np.linalg.norm(w[3:])))

    try:
        result = adm.ramp(ref(T_start_ftip), ref(T_target_ftip), duration, guard,
                          on_step=_watch)
        if result == 'seated':
            log.info('  contact reached -- holding here (compliant).')
        else:
            adm.hold(ref(T_target_ftip), settle_s, guard, on_step=_watch)
        log.info('  %s peak wrench: %.1f N / %.2f Nm%s.', what, peak['f'], peak['tau'],
                 '' if guard is None
                 else ' (guard %.0f N / %.1f Nm)' % (guard.max_force, guard.max_torque))
        return True
    finally:
        robot.arm.servo_stop()                          # leave the servo loop before the next step


def verify_cable_held(robot, check, where=''):
    """Cable-in-gripper check: RE-CLOSE the gripper and re-run the counts check -- is the
    connector still between the fingers? Same principle as the lift slip check (the stalled
    fingers HOLD position when a part vanishes, so only a re-close can reveal the loss), without
    any arm motion. True = still held. Skipped (True) when the grasp check is disabled or in a
    dry run (a dry-run gripper closes to 'empty' by construction)."""
    if not check.enabled or getattr(getattr(robot, 'arm', None), 'dry_run', False):
        return True
    if not robot.gripper.close(f're-close ({where or "held check"})'):
        return False
    result = check.evaluate(robot.gripper)
    if result != 'ok':
        log.error('Cable-in-gripper check%s: %s -- the connector is no longer held.',
                  f' at {where}' if where else '', result)
        return False
    log.info('Cable-in-gripper check%s passed -- still holding the connector.',
             f' at {where}' if where else '')
    return True


def retry_offset_x(attempt, step_m):
    """The grasp perturbation (m, along the junction x-axis) for OUTER retry `attempt` (0 = the
    first try). Pattern: 0, +d, -d, +2d, -2d, ... -- a scan->grasp loop that fails
    DETERMINISTICALLY is a FIXED POINT (the fresh scan reproduces the same junction estimate, so
    the retry reproduces the same wrong grasp); stepping alternately outward along the connector
    axis is what breaks it. step_m = 0 disables (every retry at the nominal pose)."""
    if attempt <= 0 or step_m == 0.0:
        return 0.0
    k = (attempt + 1) // 2
    return k * float(step_m) * (1.0 if attempt % 2 == 1 else -1.0)


class GraspController:
    """Moves the fingertip in the configured pickup mode -- used for BOTH the grasp DESCENT
    (grasp-align -> grasp) and the LIFT (grasp -> lift), so they always match:

      * 'position'   -- a stiff moveL (default; unchanged behaviour).
      * 'compliance' -- SOFTWARE ADMITTANCE (the same spring-mass-damper law as the assembly
                        insert): the fingertip yields to contact and springs back toward the
                        reference. Parameters mirror assembly.compliance, plus an optional
                        force_guard backstop and the ramp duration.

    Keeping the lift in the SAME mode matters: a compliant grasp leaves the arm holding the cable
    under the admittance law, and a stiff guarded lift then fights the cable's resistance and trips
    the force guard. A compliant lift yields instead.

    Reads the `pickup:` config section. The admittance controller (and optional guard) are built
    lazily on first compliant move, so a position-mode run never touches the servo layer."""

    def __init__(self, cfg):
        p = cfg.section('pickup')
        self.mode = str(p.get('mode', 'position')).lower()
        self.compliance = p.get('compliance', {}) or {}
        self.compliance_enabled = bool(self.compliance.get('enabled', True))
        self.tare_before = bool(self.compliance.get('tare_before', True))
        # NEVER tare at the LIFT by default: the lift starts IN CONTACT (part on the ground), so a
        # tare there bakes the ground reaction into the baseline -- the moment the part lifts off,
        # that baseline reads as a PHANTOM DOWNWARD force and the admittance drives the tool back
        # into the ground (jerky scraping, force climbing to the pendant's safety stop). The
        # descent's FREE-SPACE tare (at grasp-align) stays valid for the lift.
        self.tare_before_lift = bool(p.get('tare_before_lift', False))
        self.descent_time_s = float(p.get('descent_time_s', 2.0))
        # SPEED-based pacing (preferred): when either limit is set, the ramp duration is derived
        # from the ACTUAL distance -- so the touchdown speed no longer changes silently when
        # approach_distance_m does. The pickup keys override; absent, the GLOBAL cartesian limits
        # (speed.max_cartesian_translation_mm_s / _rotation_deg_s) pace the descent and lift like
        # every other pre-assembly motion. descent_time_s is the legacy fallback when neither the
        # pickup nor the global limits are set.
        spd = cfg.section('speed')
        self.descent_v_mm_s = p.get('descent_translation_mm_s',
                                    spd.get('max_cartesian_translation_mm_s'))
        self.descent_w_deg_s = p.get('descent_rotation_deg_s',
                                     spd.get('max_cartesian_rotation_deg_s'))
        # Per-phase scaling of the global limits (speed.phase_scale): the descent runs at the
        # 'pickup' scale, the lift at the 'lift' scale (typically slower -- the cable is in hand).
        scales = spd.get('phase_scale', {}) or {}
        self.pickup_scale = float(scales.get('pickup', 1.0))
        self.lift_scale = float(scales.get('lift', 1.0))
        # The FREE-SPACE move onto the pre-grasp (grasp-align). It used to inherit whatever
        # phase the caller was in -- 'scan' -- which paced a long reposition next to the work
        # surface at scanning speed. It gets its own scale so it can be slowed without also
        # slowing the multi-view scan, and falls back to 'scan' so nothing changes for a config
        # that does not set it.
        self.align_scale = float(scales.get('grasp_align', scales.get('scan', 1.0)))
        # THE IK BRANCH FOR THE GRASP. A tool0 pose has up to eight joint solutions, and the
        # controller returns the one NEAREST the seed -- which by default is wherever the scan
        # left the arm. Two of those solutions differ by the WRIST FLIP (wrist_1 and wrist_3
        # turned half a revolution, wrist_2 negated): the same fingertip pose with the wrist
        # body on the opposite side. At a square pickup either is fine. At a steeply tilted
        # fingertip_in_connector one of them swings wrist_2 down toward the work surface.
        # Naming a seed here makes that a choice instead of an accident.
        seed = p.get('approach_seed_joints_deg')
        self.approach_seed = (None if seed is None
                              else [float(np.radians(float(v))) for v in seed])
        self.approach_seed_tol = float(np.radians(
            float(p.get('approach_seed_tolerance_deg', 45.0))))
        # A WAYPOINT TO ROUTE AROUND THE BENCH. One moveJ from wherever the scan ended to the
        # pre-grasp is a single long arc, and a long arc near the work surface is exactly what
        # dips through it -- both ends can be clear while the middle is not. Naming a high,
        # known-good configuration here gives the move somewhere to go via. Only used when the
        # DIRECT path is refused, so a clean approach costs nothing.
        via = p.get('approach_via_joints_deg')
        self.approach_via = (None if via is None
                             else [float(np.radians(float(v))) for v in via])
        # GROUND COLLISION over the grasp-align move. Built LAZILY -- importing pybullet at
        # construction would make it a hard dependency of every app that touches a gripper.
        self._collision_cfg = dict(p.get('collision', {}) or {})
        self._collision_ground_z = cfg.get_path('ground_plane.z_m')
        self._collision_ftip_z = float(
            (cfg.get_path('fingertip_grasp.xyz') or [0.0, 0.0, 0.183])[2])
        self._collision = False        # False = not built yet, None = unavailable
        self.settle_s = float(p.get('settle_s', 0.5))
        self._guard_cfg = p.get('force_guard', {}) or {}
        self._adm = None
        self._guard = None
        # WHY the last approach was refused: 'unreachable' (no IK, wrong branch, or a path that
        # goes through something) or None. A caller that can DO something about it needs to
        # tell that apart from a comms failure or an operator abort, which is all a bare False
        # conveys -- see the reorient recovery in bnc_assembly.
        self.last_refusal = None

    @property
    def compliant(self):
        return self.mode == 'compliance' and self.compliance_enabled

    def _lazy_build(self, robot):
        if self._adm is None:
            from ..robot import AdmittanceController, ForceGuard
            self._adm = AdmittanceController(robot.arm, self.compliance)
            if self._guard_cfg.get('enabled', False):
                self._guard = ForceGuard(robot.arm, self._guard_cfg)

    def _duration(self, T_from, T_to, scale=1.0):
        """Ramp seconds for a compliant move: distance / the configured SPEED limits (whichever of
        translation/rotation needs longer), or the legacy fixed descent_time_s when no speed is
        configured. `scale` is the phase's speed.phase_scale factor (0.5 = half speed = double
        time). Floored at 0.1 s so a zero-length move still ramps sanely."""
        scale = max(float(scale), 1e-6)
        if self.descent_v_mm_s is None and self.descent_w_deg_s is None:
            return self.descent_time_s / scale
        lin_m, ang_rad = pose_error(T_from, T_to)
        v = float(self.descent_v_mm_s or 0.0) * scale
        w = float(self.descent_w_deg_s or 0.0) * scale
        t_lin = (lin_m * 1000.0 / v) if v > 0 else 0.0
        t_ang = (np.degrees(ang_rad) / w) if w > 0 else 0.0
        return max(t_lin, t_ang, 0.1)

    def _to(self, robot, T_target, label, what, position_guard=None, tare=None, scale=1.0):
        """Move the fingertip to T_target in the configured mode. In POSITION mode, `position_guard`
        (a callable taking the move thunk) runs it force-guarded; in COMPLIANCE mode the guard is
        ignored -- the admittance bounds the contact force itself (a stiff guarded move is exactly
        what trips on the cable's resistance). `tare` overrides whether to re-tare mid-warmup
        (default: the descent's tare_before) -- a move that STARTS IN CONTACT must not tare.
        `scale` is the phase's speed.phase_scale factor."""
        if self.compliant:
            self._lazy_build(robot)
            T_start = robot.fingertip()
            return _compliant_move(robot, self._adm, self._guard, T_start, T_target,
                                   self._duration(T_start, T_target, scale),
                                   self.tare_before if tare is None else tare,
                                   self.settle_s, what)
        if position_guard is not None:
            return position_guard(lambda: robot.move_fingertip(T_target, label))
        return robot.move_fingertip(T_target, label)

    def collision_model(self):
        """The ground model, or None if it is switched off or pybullet is missing.

        A MISSING pybullet DISABLES THE CHECK RATHER THAN THE RUN: this is a guard bolted onto
        an app that worked without it, and refusing to start because an optional package is
        absent would be the wrong trade. It says so loudly -- once -- so an unguarded run is
        never silent."""
        if self._collision is not False:
            return self._collision
        self._collision = None
        if not bool(self._collision_cfg.get('enabled', True)):
            log.info('Ground-collision checking is OFF (pickup.collision.enabled).')
            return None
        if self._collision_ground_z is None:
            log.warning('Ground-collision checking needs ground_plane.z_m -- not set, so the '
                        'pick path is UNCHECKED.')
            return None
        try:
            from ..robot.collision import GroundCollisionModel
            self._collision = GroundCollisionModel(
                self._collision_cfg, ground_z_m=float(self._collision_ground_z),
                fingertip_z_m=self._collision_ftip_z)
            log.info('Ground-collision model: %s', self._collision.describe())
        except ImportError:
            log.warning('pybullet is not installed, so the pick path is UNCHECKED against the '
                        'ground plane. `pip install pybullet` to turn the guard on.')
        except Exception as exc:                               # noqa: BLE001
            # A GUARD THAT CANNOT BE BUILT MUST NOT BE SKIPPED QUIETLY. This used to warn and
            # return None, so ANY fault in the collision code -- including a SyntaxError, which is
            # an Exception -- downgraded the run to no checking at all with one WARNING line in a
            # busy log. That happened: a corrupted collision.py made the module unimportable, the
            # exception was swallowed here, and the arm drove a path into the floor. A missing
            # OPTIONAL DEPENDENCY is a legitimate reason to run unguarded (above, and the operator
            # can see it); our own code being broken is not.
            log.error('Ground-collision model could not be built: %s: %s. REFUSING to run '
                      'unguarded -- fix the model, or set pickup.collision.enabled: false to '
                      'proceed deliberately without it.', type(exc).__name__, exc)
            raise
        return self._collision

    def verify_collision_model(self, robot):
        """Check our DH chain against the controller's FK. Call once, at start-up."""
        model = self.collision_model()
        return None if model is None else model.verify_against_controller(robot.arm)

    def align(self, robot, geom, label='grasp-align'):
        """Move the fingertip onto the PRE-GRASP, at the 'grasp_align' phase scale and on the
        SEEDED IK branch.

        A plain position move -- nothing is in front of the pads yet -- but it is the longest
        motion that happens near the work surface, and at a steeply tilted
        fingertip_in_connector it carries the gripper BODY down toward the ground plane rather
        than just the fingers. Slow, and on a configuration you picked.

        THE BRANCH IS CHECKED BEFORE THE ARM MOVES. A seed is a hint -- the controller returns
        the solution nearest it, which is not always the one you meant if the seed is far from
        the target. So the solution is compared against the seed joint by joint first, and a
        run that would land on a different branch is REFUSED while the arm is still parked,
        naming the joint. A wrist flip shows up here as ~180 deg on wrist_1/wrist_3 or a
        sign change on wrist_2; an elbow or shoulder flip shows up on those joints."""
        robot.arm.set_speed_scale(self.align_scale, 'grasp_align')
        self.last_refusal = None
        T = geom.pre_grasp()
        # SOLVE FIRST, MOVE SECOND -- both the branch check and the ground check need the
        # configuration in hand while the arm is still parked.
        q = robot.arm.ik(T @ inverse(robot.T_tool0_fingertip), self.approach_seed)
        if q is None:
            self.last_refusal = 'unreachable'
            log.error('%s: no IK solution for the STAND-OFF%s.', label,
                      '' if self.approach_seed is None else
                      ' near pickup.approach_seed_joints_deg %s deg'
                      % np.round(np.degrees(self.approach_seed), 1).tolist())
            self.diagnose(robot, geom, 'the stand-off has no IK solution', label)
            return False
        # THE GRASP ITSELF, BEFORE COMMITTING TO THE PRE-GRASP. Checking only the
        # stand-off let the arm drive all the way there and discover at the descent that the
        # pose it was standing off FROM could not be reached -- by which point it has moved,
        # and the reorient recovery (which re-picks from where the run started) has lost that.
        # The grasp is 100 mm away and its configuration is knowable right now. Checked here,
        # before the branch split, so BOTH the seeded and unseeded paths get it.
        if not self._grasp_is_reachable(robot, geom, q, label):
            return False
        if self.approach_seed is None:
            log.info('  grasp-align configuration: %s deg (no approach_seed_joints_deg set, so '
                     'this is whatever branch the scan left the arm nearest).',
                     np.round(np.degrees(q), 1).tolist())
            return self._go(robot, geom, T, q, label)
        # WRAPPED ONTO [-pi, pi], because a joint at +179 and one at -179 are 2 deg apart,
        # not 358. Done with a modulo rather than min(d, 2pi - d): the latter returns a
        # NEGATIVE distance once a joint differs by more than a full turn -- and a negative
        # sorts below every real distance, so the worst joint would be missed and the check
        # would pass exactly when it most needed to fail. UR joints run to +/-360, so
        # differences past 2pi are reachable, not hypothetical.
        raw = np.asarray(q, dtype=float) - np.asarray(self.approach_seed, dtype=float)
        d = np.abs((raw + np.pi) % (2.0 * np.pi) - np.pi)
        worst = int(np.argmax(d))
        if d[worst] > self.approach_seed_tol:
            self.last_refusal = 'unreachable'
            log.error('%s: IK landed on a DIFFERENT BRANCH from the seed -- %s is %.1f deg away '
                      '(limit %.1f). Solution %s deg vs seed %s deg. The wrist would sit on the '
                      'other side; refusing while the arm is still parked. Either re-seed from a '
                      'configuration that can actually reach this grasp, or raise '
                      'pickup.approach_seed_tolerance_deg if the flip is acceptable.',
                      label, UR_JOINTS[worst], np.degrees(d[worst]),
                      np.degrees(self.approach_seed_tol),
                      np.round(np.degrees(q), 1).tolist(),
                      np.round(np.degrees(self.approach_seed), 1).tolist())
            self.diagnose(robot, geom, 'the IK landed on a different branch from the seed',
                          label)
            return False
        log.info('  grasp-align on the seeded branch: %s deg (worst joint %s, %.1f deg from the '
                 'seed).', np.round(np.degrees(q), 1).tolist(), UR_JOINTS[worst],
                 np.degrees(d[worst]))
        return self._go(robot, geom, T, q, label)

    def diagnose(self, robot, geom, why, label='grasp'):
        """Print everything known about why this grasp could not be taken.

        THE ONE-LINE REFUSALS NAME THE CHECK BUT NOT THE CAUSE. "no IK solution" is true of a
        pose 50 mm into the bench, a pose past the reach, and a pose the wrist cannot twist to,
        and they want completely different fixes. So on any refusal this dumps the whole
        approach at once: where the grasp actually is, whether each of the two poses solves,
        and what every body's clearance is.

        MOST OF IT SURVIVES AN IK FAILURE. The tool bodies depend only on tool0's pose, so
        "the gripper body is 20 mm into the bench" is still answerable when the arm is not --
        and that is usually the real answer behind an unreachable coaxial grasp."""
        model = self.collision_model()
        from ..transforms import inverse, matrix_to_xyzrpy
        T_inv = inverse(robot.T_tool0_fingertip)
        T_g_f, T_p_f = geom.T_base_grasp, geom.pre_grasp()
        T_g_0, T_p_0 = T_g_f @ T_inv, T_p_f @ T_inv
        gz = model.ground_z if model is not None else None

        def line(tag, T):
            xyz, rpy = matrix_to_xyzrpy(T)
            h = '' if gz is None else '  | %+.0f mm above the bench' % ((T[2, 3] - gz) * 1000.0)
            return '%-18s xyz %-26s rpy %s deg%s' % (
                tag, np.round(xyz * 1000.0, 1).tolist(),
                np.round(np.degrees(rpy), 1).tolist(), h)

        log.error('--- WHY THE GRASP WAS REFUSED: %s ---', why)
        log.error('   %s', line('grasp  fingertip', T_g_f))
        log.error('   %s', line('grasp  tool0', T_g_0))
        log.error('   %s', line('stand-off tool0', T_p_0))
        if getattr(geom, 'T_base_detection', None) is not None:
            log.error('   %s', line('detected junction', geom.T_base_detection))

        # ---- the two solves, reported separately: which one fails is the whole diagnosis ----
        q_now = robot.arm.q()
        log.error('   arm is now at    %s deg', np.round(np.degrees(q_now), 1).tolist())
        q_p = robot.arm.ik(T_p_0, self.approach_seed if self.approach_seed is not None else q_now)
        log.error('   IK stand-off     %s', 'UNREACHABLE' if q_p is None
                  else str(np.round(np.degrees(q_p), 1).tolist()))
        q_g = robot.arm.ik(T_g_0, q_p if q_p is not None else q_now)
        log.error('   IK grasp         %s', 'UNREACHABLE' if q_g is None
                  else str(np.round(np.degrees(q_g), 1).tolist()))
        if q_p is not None and q_g is not None:
            d = np.abs((np.asarray(q_g) - np.asarray(q_p) + np.pi) % (2 * np.pi) - np.pi)
            log.error('   stand-off -> grasp joint change %s deg (worst %s)',
                      np.round(np.degrees(d), 1).tolist(), UR_JOINTS[int(np.argmax(d))])
        if self.approach_seed is not None and q_p is not None:
            d = np.abs((np.asarray(q_p) - np.asarray(self.approach_seed) + np.pi)
                       % (2 * np.pi) - np.pi)
            log.error('   stand-off is %.1f deg from the seed at %s (tolerance %.0f)',
                      np.degrees(d.max()), UR_JOINTS[int(np.argmax(d))],
                      np.degrees(self.approach_seed_tol))
        if model is None:
            log.error('   (no collision model -- clearances unavailable)')
            return

        # ---- clearances that need NO joint solution -----------------------------------------
        for tag, T0 in (('AT THE GRASP', T_g_0), ('at the stand-off', T_p_0)):
            cl = model.tool_clearances(T0)
            worst = sorted(cl, key=cl.get)[:4]
            log.error('   tool vs ground %s:', tag)
            for b in worst:
                allow = model.allowance(b)
                log.error('       %-15s %+8.1f mm   (allowed %+.1f)%s', b, cl[b] * 1000.0,
                          allow * 1000.0, '   <-- OVER' if cl[b] < allow else '')
        # ---- and the ones that do -----------------------------------------------------------
        if q_g is None:
            log.error('   arm/self clearances need an IK solution for the grasp, and there is '
                      'none -- the tool numbers above are what there is.')
            return
        cl = model.clearances(q_g)
        arm = {k: v for k, v in cl.items() if k not in model.tool.body_names()}
        for b in sorted(arm, key=arm.get)[:3]:
            log.error('       %-15s %+8.1f mm   (allowed %+.1f)%s', b, arm[b] * 1000.0,
                      model.margin * 1000.0, '   <-- OVER' if arm[b] < model.margin else '')
        sc = model.self_clearances(q_g)
        for b in sorted(sc, key=sc.get)[:3]:
            log.error('       %-28s %+8.1f mm   (allowed %+.1f)%s', b, sc[b] * 1000.0,
                      model.self_margin * 1000.0,
                      '   <-- OVER' if sc[b] < model.self_margin else '')

    def _grasp_is_reachable(self, robot, geom, q_pregrasp, label):
        """Is the GRASP pose -- not the stand-off -- reachable and clear?

        WHAT THIS CATCHES THAT THE OTHER CHECKS DO NOT. `align` checks the path to the
        PRE-GRASP; `descend` checks the TOOL along the cartesian descent. Neither ever solves
        for the arm at the grasp, so an approach whose stand-off is fine but whose grasp puts
        the FOREARM through the bench, or folds the arm into itself, was only discovered by
        driving there.

        SOLVED ON THE PRE-GRASP'S BRANCH. Seeding from `q_pregrasp` is what makes the answer
        mean anything: a descent is a small move, so the grasp must be reachable from the same
        configuration. A solution found on some other branch would be a pose the arm cannot
        actually get to from the stand-off.

        ONLY THE ENDPOINT, deliberately. The real descent is a straight cartesian line, and a
        joint interpolation between these two is NOT that line -- checking it would invent
        refusals for a path the arm never takes. The tool along the true line is covered
        exactly by descend's cartesian check; this covers the arm at the end of it."""
        model = self.collision_model()
        if model is None:
            return True
        from ..transforms import inverse
        T_grasp_tool0 = geom.T_base_grasp @ inverse(robot.T_tool0_fingertip)
        q_grasp = robot.arm.ik(T_grasp_tool0, q_pregrasp)
        if q_grasp is None:
            self.last_refusal = 'unreachable'
            log.error('%s: the STAND-OFF is reachable but the GRASP pose below it is not -- '
                      'no IK solution on that branch. Refusing before the arm moves.', label)
            self.diagnose(robot, geom, 'the grasp has no IK solution on the stand-off branch',
                          label)
            return False
        ok, body, over = model.check_q(q_grasp)
        if not ok:
            self.last_refusal = 'unreachable'
            log.error('%s: the GRASP pose is not collision-free -- %s is %.1f mm past its '
                      'allowance when the arm is AT the grasp. The stand-off above it is fine, '
                      'which is why this has to be checked separately. Refusing before the arm '
                      'moves.', label, body, over * 1000.0)
            self.diagnose(robot, geom, '%s is %.1f mm into something at the grasp'
                          % (body, over * 1000.0), label)
            return False
        return True

    def _go(self, robot, geom, T_target, q_goal, label):
        """Drive to the pre-grasp: straight there if that arc is clear, otherwise via the
        waypoint. Every leg is checked, and a refusal happens with the arm still parked."""
        if self._path_is_clear(robot, q_goal, label, quiet=self.approach_via is not None):
            return robot.move_fingertip(T_target, label, qnear=self.approach_seed)
        if self.approach_via is None:
            return False
        # THE DIRECT ARC DIPS, so try the two legs through the waypoint. Both are checked
        # before anything moves -- a route that only half works is no better than none.
        via = self.approach_via
        if not self._path_is_clear(robot, via, label + ' (leg 1: to the waypoint)'):
            return False
        ok, body, over, frac = self._clear_between(via, q_goal)
        if not ok:
            self.last_refusal = 'unreachable'
            log.error('%s: routing via pickup.approach_via_joints_deg does not help -- leg 2 '
                      'still puts %s %.1f mm past its allowance at %.0f%% along. The waypoint '
                      'needs to be higher, or the grasp itself is too low.',
                      label, body, over * 1000.0, frac * 100.0)
            self.diagnose(robot, geom, 'no route to the stand-off clears the bench', label)
            return False
        log.info('%s: the direct arc dips into the bench, so routing via the waypoint %s deg.',
                 label, np.round(np.degrees(via), 1).tolist())
        if not robot.arm.move_j(via, label=label + ' (waypoint)'):
            return False
        return robot.move_fingertip(T_target, label, qnear=self.approach_seed)

    def _clear_between(self, q_from, q_to):
        """(ok, body, over_m, frac) for the ground/self check between two configurations."""
        model = self.collision_model()
        if model is None:
            return True, None, 0.0, 0.0
        return model.check_path(q_from, q_to)

    def plan_is_valid(self, q_from, q_to):
        """(ok, reason) -- collision AND joint limits for a candidate joint move.

        The single call a planner should ask. `reason` is a finished sentence, because the two
        checks measure their violations in different units (mm of clearance, degrees past a stop)
        and a caller should not have to know which one it got back."""
        model = self.collision_model()
        if model is None:
            return True, None
        return model.check_plan(q_from, q_to)

    def _path_is_clear(self, robot, q_goal, label, quiet=False):
        """Refuse a move whose JOINT PATH puts the arm through the ground plane OR outside the
        joint limits.

        THE ENDPOINTS ARE NOT THE PATH. A moveJ interpolates in joint space, so the tool swings
        through an arc: both ends can be comfortably clear while the middle is not. That is the
        failure this exists for, and it is why the whole interpolation is sampled rather than
        just the target.

        JOINT LIMITS ARE CHECKED TOO, through the same call. A pose can be perfectly clear of the
        bench and still be one the arm cannot hold -- a wrist wound past its stop -- and finding
        that out from the controller mid-move is strictly worse than refusing while parked."""
        model = self.collision_model()
        if model is None:
            return True
        q_now = robot.arm.q()
        plan_ok, why = model.check_plan(q_now, q_goal)
        if not plan_ok and 'outside its limit' in (why or ''):
            if not quiet:
                self.last_refusal = 'unreachable'
            (log.info if quiet else log.error)(
                '%s: REFUSED before moving -- %s. Nothing has moved.', label, why)
            return False
        ok, body, over, frac = model.check_path(q_now, q_goal)
        if ok:
            return True
        # The fingertips carry their own allowance (they are MEANT to reach the work surface);
        # everything else, the gripper wrist included, is held to the strict margin.
        # NOT a speed problem -- a moveJ traces the same arc at any speed. The three things
        # that actually move the arc are the BRANCH it ends on, the ROUTE it takes, and how low
        # the target is.
        if not quiet:
            self.last_refusal = 'unreachable'
        (log.info if quiet else log.error)(
            '%s: the joint path goes THROUGH THE GROUND PLANE -- %s is %.1f mm past its '
            'allowance at %.0f%% along the move.%s The arc is the same at any speed; what '
            'moves it is (a) pickup.approach_via_joints_deg, a high waypoint to route through, '
            '(b) pickup.approach_seed_joints_deg, to land on a branch on the near side, or '
            '(c) a less extreme fingertip_in_connector rpy, which raises the target.',
            label, body, over * 1000.0, frac * 100.0,
            '' if quiet else ' Refusing before the arm moves.')
        return False

    def descend(self, robot, geom, label='grasp'):
        """Move to the grasp pose (from wherever the arm is -- the grasp-align pose). Tares in
        FREE SPACE (at grasp-align), which is the baseline the lift keeps. Runs at the 'pickup'
        phase scale.

        CHECKED AGAINST THE GROUND FIRST. This is the move that actually approaches the bench,
        so leaving it unguarded made the guard on grasp-align close to useless: the arc that
        needed watching was never the one being watched. The descent is CARTESIAN, so it is the
        tool bodies that are checked (they depend only on tool0's pose, no IK required) -- and
        the tool is what arrives at the bench first anyway."""
        self.last_refusal = None
        if not self._descent_is_clear(robot, geom, label):
            self.last_refusal = 'unreachable'
            return False
        return self._to(robot, geom.T_base_grasp, label, 'Grasp descent', scale=self.pickup_scale)

    def _descent_is_clear(self, robot, geom, label):
        """Refuse a descent that would put the gripper through the bench."""
        model = self.collision_model()
        if model is None:
            return True
        from ..transforms import inverse
        T_inv = inverse(robot.T_tool0_fingertip)
        ok, body, over, frac = model.check_tool_path(geom.pre_grasp() @ T_inv,
                                                     geom.T_base_grasp @ T_inv)
        if ok:
            return True
        log.error('%s: the descent puts %s %.1f mm past its ground allowance at %.0f%% of the '
                  'way down. Refusing. The fingertips may intersect by %.0f mm -- the gripper '
                  'body and wrist may not -- so this is either too steep an approach angle '
                  '(pickup.fingertip_in_connector rpy y) or a grasp target set too low.',
                  label, body, over * 1000.0, frac * 100.0,
                  model.fingertip_margin * 1000.0)
        self.diagnose(robot, geom, 'the descent puts %s %.1f mm into the bench'
                      % (body, over * 1000.0), label)
        return False

    def lift(self, robot, geom, label='lift', position_guard=None):
        """Lift to geom.lift() in the SAME mode as the descent, at the 'lift' phase scale.
        `position_guard` guards the position-mode lift only. Does NOT re-tare (tare_before_lift,
        default off): the lift starts in contact, and taring there turns the ground reaction into
        a phantom downward force at liftoff -- the arm would chase it back into the ground."""
        return self._to(robot, geom.lift(), label, 'Lift', position_guard=position_guard,
                        tare=self.tare_before_lift, scale=self.lift_scale)

    def lift_verified(self, robot, geom, check, label='lift', position_guard=None):
        """Lift with SLIP DETECTION -- for the failure mode where the grasp check PASSES but the
        cable slips out of the fingers during the lift. Raise a SMALL amount first
        (grasp_check.lift_check.height_m), RE-CLOSE the gripper, and re-run the counts check:
        a held connector stalls the fingers in the same band ('ok' -> finish the lift); a slipped
        cable lets them run on to the cable/empty counts -> 'slipped' (the caller opens and
        retries the whole scan->grasp). The RE-CLOSE is what makes a slip visible at all: after
        the original close stalled, the fingers HOLD POSITION even if the part vanishes, so
        reading the position without closing again would still show the old, healthy band.

        Returns 'ok' | 'slipped' | 'abort' (a move failed)."""
        if not check.lift_check_enabled or getattr(getattr(robot, 'arm', None), 'dry_run', False):
            return 'ok' if self.lift(robot, geom, label, position_guard) else 'abort'
        T_partial = (translation_matrix(geom.lift_axis * check.lift_check_height_m)
                     @ geom.T_base_grasp)
        if not self._to(robot, T_partial, f'{label} (slip check)', 'Partial lift',
                        position_guard=position_guard, tare=self.tare_before_lift,
                        scale=self.lift_scale):
            return 'abort'
        if not robot.gripper.close('re-close (slip check)'):
            return 'abort'
        result = check.evaluate(robot.gripper)
        if result != 'ok':
            log.warning('Slip check after the %.0f mm partial lift: %s -- the cable is no longer '
                        'held.', check.lift_check_height_m * 1000, result)
            return 'slipped'
        log.info('Slip check passed -- still holding the connector; completing the lift.')
        return 'ok' if self.lift(robot, geom, label, position_guard) else 'abort'


class GraspImageRecorder:
    """Save wrist-camera images at a fixed RATE for the DURATION of a grasp, in a background thread,
    so the whole approach -> close -> reseat is captured (not just the close events that
    GraspRecovery labels with the count). Enabled by grasp_check.capture_images (default TRUE) at
    grasp_check.capture_rate_hz (default 1 Hz); written to <data_dir>/<capture_subdir>/<timestamp>/.

        with recorder.recording(camera):
            ... grasp motions ...
    """

    def __init__(self, cfg):
        gc = cfg.section('grasp_check')
        self.enabled = bool(gc.get('capture_images', True))
        self.rate_hz = max(0.1, float(gc.get('capture_rate_hz', 1.0)))
        self._data_root = cfg.get('data_dir', 'data')
        self._subdir = gc.get('capture_subdir', 'grasp_images')
        self._dir = None
        self._seq = 0

    def recording(self, camera):
        """Context manager: record at self.rate_hz for the length of the `with` block."""
        return _GraspRecording(self, camera)

    def _capture(self, camera):
        """Save one wrist frame; best-effort (a failure never interrupts the grasp)."""
        try:
            import os
            import cv2
            from datetime import datetime
            if self._dir is None:
                self._dir = os.path.join(self._data_root, self._subdir,
                                         datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(self._dir, exist_ok=True)
            # reconnect=False: this runs in a BACKGROUND thread at ~1 Hz while the grasp
            # happens. If the camera vanished, waiting for it here would hold the recording
            # open long after the grasp it was recording had finished -- and the run's own
            # captures already do the waiting, in the foreground, where a pause is visible.
            img = camera.capture(reconnect=False).color.copy()
            cv2.imwrite(os.path.join(self._dir, f'grasp_{self._seq:04d}.png'), img)
            self._seq += 1
        except Exception as exc:                       # noqa: BLE001 -- capture is best-effort
            log.warning('  grasp image capture failed: %s', exc)


class _GraspRecording:
    """Runs GraspImageRecorder._capture at the recorder's rate in a daemon thread for the length of
    the `with` block. No-op if capture is off / no camera / dry-run."""

    def __init__(self, rec, camera):
        self._rec = rec
        self._camera = camera
        self._stop = None
        self._thread = None

    def __enter__(self):
        r = self._rec
        if not r.enabled or self._camera is None or getattr(self._camera, 'dry_run', False):
            return self
        import threading
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info('  recording grasp images at %.1f Hz.', r.rate_hz)
        return self

    def _loop(self):
        period = 1.0 / self._rec.rate_hz
        while not self._stop.is_set():
            self._rec._capture(self._camera)
            self._stop.wait(period)

    def __exit__(self, *exc):
        if self._stop is not None:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        return False


def log_grasp_delta(robot, T_base_grasp, label):
    """Report the fingertip's pose vs the grasp target -- the end-to-end error of the whole
    perception -> IK -> motion chain, in the units that matter (mm at the fingers).

    A report, not a gate: it prints before the grasp's confirm prompt so a bad perception estimate
    can be vetoed before the arm commits, but it never blocks on its own."""
    T_cur = robot.fingertip()
    log.info('[%s] fingertip vs grasp target: %s', label, fmt_delta(T_cur, T_base_grasp))
    return True
