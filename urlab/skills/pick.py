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
from ..transforms import fmt_delta, inverse, pose_error, translation_matrix

log = urlog.get('pick')

# ---------------------------------------------------------------------------- pickup pitch
# The detected junction frame (skills/scan -> transforms.frame_from_axis) has
#   x = the CABLE AXIS,  z = the ground normal (as close to `up` as orthogonality allows),
#   y = z x x -- horizontal, perpendicular to the cable.
# So a rotation about the junction frame's OWN y is exactly "pitch the gripper about the axis
# orthogonal to the ground normal and to the connector axis", and because it acts through the
# junction origin, the bite point on the cable does not move: only the approach tilts.


def pickup_pitch_rad(cfg):
    """The configured pickup pitch in radians (`pickup.pitch_deg` in the app config).
    0 = the fingers come straight down.

    A PER-APP setting, not a per-cable one: it lives beside the other pickup geometry
    (pickup.height_from_model, pickup.compliance) rather than in cables.yaml, because it is a
    choice about how this run approaches -- not a property of the cable. It must NOT go in
    cables.yaml: apply_cable_profile runs after the config is read, so a cable entry would
    silently overwrite whatever the app config set."""
    return float(np.radians(float(cfg.get_path('pickup.pitch_deg', 0.0) or 0.0)))


def grip_offset_matrix(cfg):
    """`pickup.grip_offset` as a 4x4 -- WHERE THE BITE POINT SITS RELATIVE TO THE DETECTION.

    THE FRAME IS THE DETECTED JUNCTION FRAME, and every axis of it is physical:

        +x   ALONG THE CONNECTOR AXIS, pointing from the cable toward the connector's free
             END. Positive slides the bite further onto the connector body.
        +y   HORIZONTAL AND ACROSS the cable (z cross x). This is also the axis the jaws
             close along, so positive shifts the bite sideways off the barrel's centreline.
        +z   THE GROUND NORMAL, i.e. UP. Positive lifts the bite point off the work surface.
        origin  the detected junction -- where the cable meets the connector, AT THE GROUND
             PLANE (the ground_plane scan puts it there; it does NOT sit at the barrel's
             centreline, so a resting connector needs a positive z here or from
             pickup.height_from_model, or the fingers aim at the floor).

    The ROTATION is applied about the translated point, and it tilts the APPROACH -- roll
    about +x, pitch about +y, yaw about +z, extrinsic XYZ like every other rpy in the repo.

    IT DOES NOT SUPERSEDE pickup.pitch_deg / roll_deg -- all three COMPOSE, in the order
    grip_offset, then pitch, then roll (grip_delta). What that means in practice:

      * the TRANSLATION is fully independent. It only moves the pivot, so sliding the bite
        point never changes the approach angle and changing the angle never changes where
        along the connector the fingers close. This is the part that is safe to tune alone.
      * the ROTATION is NOT independent -- it multiplies with the other two. An rpy y here IS
        pitch_deg (they are the same axis, so they simply ADD: y +10 with pitch_deg -75 is
        exactly pitch_deg -65). An rpy x or z here TILTS THE AXIS the pitch then turns about,
        so the result is not any pitch_deg value at all.

    So: use the translation for bite-point corrections, and keep the approach angle in
    pitch_deg / roll_deg where it is one number people can sweep. Reach for the rotation here
    only for a correction that genuinely is not a pitch or a roll -- and expect it to compose,
    not replace.

    ACCEPTS MONITOR UNITS: xyz_mm / rpy_deg, so a reading can be pasted straight off the
    monitor; xyz / rpy (m/rad) also work. Setting both units for one triple is an error.

    THE OLD SCALAR still works. `pickup.grip_offset_mm: 10` means exactly
    `grip_offset: {xyz_mm: [10, 0, 0]}` and is read when the block is absent."""
    from ..config import _pose_si
    from ..transforms import from_cfg, translation_matrix
    block = cfg.get_path('pickup.grip_offset')
    if block is not None:
        return from_cfg(_pose_si(block))
    return translation_matrix(
        [float(cfg.get_path('pickup.grip_offset_mm', 0.0) or 0.0) / 1000.0, 0.0, 0.0])


def pickup_roll_rad(cfg):
    """The configured pickup ROLL in radians (`pickup.roll_deg`). 0 = the wrist sits where
    junction_in_fingertip's yaw puts it.

    A ROTATION ABOUT THE APPROACH AXIS -- the fingertip's own z, i.e. the direction the gripper
    advances along -- applied AFTER the pitch, so it turns about the pitched approach and not
    about a vertical the pitch has already left behind.

    WHY 180 IS THE VALUE THAT MATTERS. A parallel jaw is symmetric under half a turn about its
    approach axis: the pads land on the same two sides of the barrel either way, so 180 is the
    SAME PHYSICAL BITE with the wrist on the other side. That makes it the one roll that is
    free at every pitch, and it is what flips tool0 -Y from pointing along the connector -Z to
    along its +Z. (At pitch +/-90 the approach runs down the connector axis and a cylinder has
    no preferred diameter, so there every roll is free -- but only 180 stays valid as the pitch
    backs off.)

    IT IS THE SAME FLIP AS junction_in_fingertip's YAW, and deliberately not done there.
    cables.yaml's yaw is baked into the NOMINAL grip, so changing it silently invalidates the
    in-hand belief in frames.yaml (bnc_connector_in_fingerpads) and estimation.
    initial_connector_in_fingertip -- flip one and the belief lands 180 deg and ~91 mm out.
    Rolling here instead keeps those describing the nominal grip and lets pitched_belief carry
    the flip through automatically, exactly as it already does for the pitch."""
    return float(np.radians(float(cfg.get_path('pickup.roll_deg', 0.0) or 0.0)))


def belief_offset_m(cfg):
    """`pickup.belief_offset_mm` as a 3-vector in metres, in the FINGERTIP frame.

    THE MEASURED RESIDUAL, not a derived one -- and the reason this exists at all is that the
    two things it separates are genuinely different:

      * the GRASP COMMAND (pitch_deg, grip_offset_mm, junction_in_fingertip) says where the
        fingers GO. Change one and the arm moves somewhere else.
      * the BELIEF says where the connector then IS relative to those fingers. Change this and
        NOTHING moves at pickup -- only the assembly's idea of what it is carrying.

    pitched_belief() derives the belief from the command exactly, and its round trip is exact
    -- but only under the assumption that the part seats in the jaws the same way at every
    approach angle. It does not. The cable lies on the ground and stays horizontal while the
    JAWS tilt, so tilted grooves capture the cylinder at a different depth than square ones.
    That difference is a contact fact: measurable, not derivable from any frame. This is where
    the measurement goes, so it cannot be confused with the geometry it corrects."""
    v = cfg.get_path('pickup.belief_offset_mm') or [0.0, 0.0, 0.0]
    v = [float(x) for x in v]
    if len(v) != 3:
        raise ValueError('pickup.belief_offset_mm must be [x, y, z] mm in the fingertip frame, '
                         f'got {len(v)} entries')
    return np.asarray(v, dtype=float) / 1000.0


def offset_belief(T_ftip_conn, offset_m):
    """Shift the believed connector in the FINGERTIP frame by `offset_m`, leaving its
    orientation alone. Left-multiplied because the offset is expressed in the fingertip's own
    axes -- 'the part sits this much lower in the hand', not 'this much along its own body'."""
    from ..transforms import translation_matrix
    return translation_matrix(np.asarray(offset_m, dtype=float)) @ T_ftip_conn


def pitch_delta(pitch_rad):
    """The pitch as a transform in the JUNCTION frame: a rotation about its own y."""
    from ..transforms import xyzrpy_to_matrix
    return xyzrpy_to_matrix([0.0, 0.0, 0.0], [0.0, float(pitch_rad), 0.0])


def _offset_matrix(offset):
    """The grip offset as a 4x4, from either form: a full transform (grip_offset_matrix) or the
    legacy scalar in METRES along the connector axis (pickup.grip_offset_mm / 1000)."""
    from ..transforms import translation_matrix
    if np.ndim(offset) == 2:
        return np.asarray(offset, dtype=float)
    return translation_matrix([float(offset), 0.0, 0.0])


def roll_delta(roll_rad):
    """The roll as a transform: a rotation about the APPROACH axis. Written in the frame the
    pitch leaves behind, so composing it after pitch_delta turns about the pitched approach."""
    from ..transforms import xyzrpy_to_matrix
    return xyzrpy_to_matrix([0.0, 0.0, 0.0], [0.0, 0.0, float(roll_rad)])


def grip_delta(pitch_rad, offset_m=0.0, roll_rad=0.0):
    """The full bite-point transform in the JUNCTION frame:

        grip_delta = grip_offset @ Ry(pitch) @ Rz(roll)

    THE THREE COMPOSE -- none of them replaces another. `offset_m` is either the 6DOF
    grip_offset pose or the legacy scalar along the connector axis.

    Order matters and all three positions are deliberate:

      * grip_offset FIRST, so its TRANSLATION becomes the pivot for what follows. That is what
        keeps the bite point and the approach angle independent of each other. Its ROTATION,
        if any, is not independent -- it premultiplies, so an rpy y adds to `pitch_rad` and an
        rpy x/z tilts the axis the pitch then turns about.
      * roll LAST, so it turns about the approach direction the pitch actually produced. Roll
        first and it would turn about the junction's own z, which after a large pitch is
        nowhere near the direction the gripper advances along."""
    return _offset_matrix(offset_m) @ pitch_delta(pitch_rad) @ roll_delta(roll_rad)


def pitched_grasp(T_base_junction, T_ftip_junction, pitch_rad, offset_m=0.0, roll_rad=0.0):
    """The FINGERTIP grasp pose for a pitched / offset / rolled pickup.

    Nominally the fingertip goes to `detected_junction @ inverse(junction_in_fingertip)`. All
    three knobs are inserted in the JUNCTION frame: `offset_m` slides the bite point along the
    connector axis, the pitch tilts the approach about that point, and the roll spins the wrist
    about the approach."""
    return T_base_junction @ grip_delta(pitch_rad, offset_m, roll_rad) @ inverse(T_ftip_junction)


def held_junction_in_fingertip(T_ftip_junction, pitch_rad, offset_m=0.0, roll_rad=0.0):
    """Where the junction ACTUALLY sits in the fingertip frame after a pitched / offset /
    rolled pickup.

    The nominal `junction_in_fingertip` describes a square grip AT the junction. Gripping
    `offset_m` further along the axis leaves the junction that much further back in the hand,
    pitching by phi leaves the part rotated by -phi, and rolling by psi leaves it rotated by
    -psi about the approach -- which is what every downstream user of the grasp geometry has to
    be told about.

    LITERALLY INVERTS what pitched_grasp inserted -- it calls the same grip_delta rather than
    re-deriving the inverse by hand, so the two cannot drift when a knob is added (they did
    have to be kept in step by hand, and that is exactly the bug this shape removes)."""
    return T_ftip_junction @ inverse(grip_delta(pitch_rad, offset_m, roll_rad))


def pitched_belief(T_ftip_conn, T_ftip_junction, pitch_rad, offset_m=0.0, roll_rad=0.0):
    """The in-hand connector belief a pitched / offset / rolled pickup actually produces.

    The connector is rigid with the junction, so T_junction_connector is a property of the PART
    and does not change; only the fingertip-to-junction relation does. Substituting
    T_ftip_conn = T_ftip_junction @ T_junction_conn and replacing the latter's left factor with
    held_junction_in_fingertip gives a conjugation of the nominal belief -- a rotation of -phi
    about the junction's y, through the junction origin, expressed in the fingertip frame."""
    return (held_junction_in_fingertip(T_ftip_junction, pitch_rad, offset_m, roll_rad)
            @ inverse(T_ftip_junction) @ T_ftip_conn)


class GraspGeometry:
    """The grasp/pre-grasp/lift/place poses, all derived from one grasp target on demand -- so a
    refined estimate that updates the target automatically feeds every downstream pose."""

    def __init__(self, cfg):
        self.approach_distance = float(cfg.get('approach_distance_m', 0.10))
        self.approach_axis = np.asarray(cfg.get('approach_axis', [0.0, 0.0, -1.0]), dtype=float)
        self.lift_distance = float(cfg.get('lift_distance_m', 0.10))
        self.lift_axis = np.asarray(cfg.get('lift_axis', [0.0, 0.0, 1.0]), dtype=float)
        from ..transforms import xyzrpy_to_matrix
        self.place_offset = xyzrpy_to_matrix(cfg.get('place_offset_xyz', [0.0, 0.20, 0.0]),
                                             cfg.get('place_offset_rpy', [0.0, 0.0, 0.0]))
        self.T_base_grasp = np.eye(4)

    def pre_grasp(self):
        """Stand-off before the grasp, along approach_axis IN THE GRASP FRAME."""
        return self.T_base_grasp @ translation_matrix(self.approach_axis * self.approach_distance)

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
        self.settle_s = float(p.get('settle_s', 0.5))
        self._guard_cfg = p.get('force_guard', {}) or {}
        self._adm = None
        self._guard = None

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

    def align(self, robot, geom, label='grasp-align'):
        """Move the fingertip onto the PRE-GRASP, at the 'grasp_align' phase scale.

        A plain position move -- nothing is in front of the pads yet -- but it is the longest
        motion that happens near the work surface, and at a large pickup.pitch_deg it carries
        the gripper BODY down toward the ground plane rather than just the fingers. Slow."""
        robot.arm.set_speed_scale(self.align_scale, 'grasp_align')
        return robot.move_fingertip(geom.pre_grasp(), label)

    def descend(self, robot, geom, label='grasp'):
        """Move to the grasp pose (from wherever the arm is -- the grasp-align pose). Tares in
        FREE SPACE (at grasp-align), which is the baseline the lift keeps. Runs at the 'pickup'
        phase scale."""
        return self._to(robot, geom.T_base_grasp, label, 'Grasp descent', scale=self.pickup_scale)

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
            img = camera.capture().color.copy()
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
