"""GROUND-COLLISION CHECKING for the UR10e + Robotiq 2F-85, in pybullet.

WHAT THIS IS FOR. A moveJ interpolates in JOINT space, so the tool sweeps an arc between its
endpoints -- both ends can be well clear of the bench while the middle of the move is not. That
is the failure this catches: it samples the joint path and refuses BEFORE the arm moves, rather
than discovering the bench with the gripper.

TWO MODELS OF THE ARM, and it prefers the accurate one.

    'urdf'    -- Universal Robots' own collision meshes, via the generated
                 description/ur10e/ur10e.urdf (see fetch_ur10e.py for provenance). This is the
                 real shell, so clearances are the real clearances.
    'capsule' -- a conservative capsule per link, built from the published UR10e DH parameters.
                 The fallback for when the description has not been fetched. Capsules CONTAIN
                 the real shell, so a PASS stays trustworthy and a FAIL can only be pessimistic
                 -- the right direction for a guard to err in, but it will refuse some poses
                 the real arm can reach.

THE KINEMATICS ARE CROSS-CHECKED, and by two independent things. Offline, the URDF is built
from UR's published joint origins while fk_links() is built from the published DH table -- two
descriptions from different upstream files -- and the tests assert their tool0 agrees to
nanometres. On hardware, `verify_against_controller()` compares both against the controller's
own FK. That matters because a DH typo would otherwise produce a confident, wrong answer.

THE FINGERTIP EXCEPTION -- the one deliberate hole in the guard.
    Picking a connector that LIES ON the bench means closing the pads around something whose
    centreline is one barrel-radius up. The pads have to reach beside and slightly below that
    centreline, and the fingertips are the part of the gripper that is SUPPOSED to arrive at
    the work surface. Holding them to a hard zero would refuse every legitimate pick.
    So the fingertips alone are allowed `fingertip_margin_mm` (default 5 mm) of intersection
    with the ground plane.
    IT APPLIES TO THE FINGERTIPS AND NOTHING ELSE. The gripper WRIST, its body, the spacer and
    every arm link are held to `margin_mm` -- normally zero or positive clearance. A wrist that
    touches the bench is a crash; a fingertip that grazes it is the job. The two are separate
    bodies in this model precisely so the exception cannot leak from one to the other.
"""

import os

import numpy as np

from .. import log as urlog
from ..transforms import BASE_LINK_FROM_UR_BASE, UR_JOINTS

log = urlog.get('collision')

# ---- UR10e, the published Denavit-Hartenberg parameters (metres, radians) -------------------
# Standard DH: T_i = Rz(theta_i) @ Tz(d_i) @ Tx(a_i) @ Rx(alpha_i). These are UR's own published
# UR10e figures; the frame they produce is the UR `base`, NOT ROS `base_link` -- see
# BASE_LINK_FROM_UR_BASE, applied at the end of fk_links().
UR10E_A = (0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0)
UR10E_D = (0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655)
UR10E_ALPHA = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)

# Conservative capsule radii per link (m). NOT the real shell -- an envelope that contains it.
# Shrinking one only makes the guard more permissive, so treat these as a floor.
UR10E_RADII = (0.090, 0.085, 0.075, 0.060, 0.060, 0.058)

# The six link lengths, which are a property of the CHAIN and not of the configuration -- the
# distance between consecutive joint origins never changes. Derived here rather than typed, so
# they cannot drift from the DH table above, and asserted in the tests.
UR10E_LINK_LENGTHS = (UR10E_D[0], abs(UR10E_A[1]), abs(UR10E_A[2]),
                      UR10E_D[3], UR10E_D[4], UR10E_D[5])


def dh_matrix(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def fk_links(q):
    """Every joint frame plus tool0, in ROS `base_link`. Returns a (7, 4, 4) array.

    Index i is the frame at the OUTBOARD end of link i; index 6 is tool0. The capsule for link i
    spans origin[i] -> origin[i+1], which is what makes the geometry fall out of the chain
    rather than being a second set of numbers to keep in step."""
    T = np.eye(4)
    out = []
    for i in range(6):
        T = T @ dh_matrix(float(q[i]), UR10E_D[i], UR10E_A[i], UR10E_ALPHA[i])
        out.append(T.copy())
    origins = [np.eye(4)] + out
    return np.array([BASE_LINK_FROM_UR_BASE @ T_i for T_i in origins])


class ToolModel:
    """The 80 mm spacer + Robotiq 2F-85, as z-ranges along tool0.

    ANCHORED ON THE REPO'S OWN CALIBRATION. `fingertip_grasp` says the pads sit
    `fingertip_z_m` along tool0 (183 mm as measured), and that number is treated as the truth:
    the spacer occupies [0, spacer], the gripper BODY the span above it, and the FINGERS the
    remainder up to the pads. Deriving the split that way keeps the model and the calibrated
    grasp frame from disagreeing -- if they ever do, the arm goes where the calibration says
    and the check would be guarding a robot that does not exist."""

    def __init__(self, cfg=None, fingertip_z_m=0.183):
        c = dict(cfg or {})
        self.fingertip_z = float(fingertip_z_m)
        self.spacer_len = float(c.get('spacer_length_mm', 80.0)) / 1000.0
        self.spacer_r = float(c.get('spacer_radius_mm', 37.5)) / 1000.0
        self.body_len = float(c.get('gripper_body_length_mm', 63.0)) / 1000.0
        self.body_r = float(c.get('gripper_body_radius_mm', 45.0)) / 1000.0
        self.finger_r = float(c.get('finger_radius_mm', 12.0)) / 1000.0
        # The jaws close along tool0 +/-X (fingertip y = tool0 -X), so the two fingers are
        # offset along X. Half the OPEN separation is the worst case for a ground check.
        self.finger_half_gap = float(c.get('finger_half_gap_mm', 42.0)) / 1000.0
        # BOXES BOLTED TO tool0 -- brackets, camera mounts, cable guides. Written as tool0
        # extents in mm, which is how you measure one: put a rule on the flange and read off
        # where the thing starts and stops on each axis. A slab is NOT a capsule -- wrapping a
        # 50 x 110 x 35 mm bracket in a capsule would give it a 60 mm radius and refuse most of
        # the workspace -- so these are real boxes.
        self.boxes = []
        for i, b in enumerate(c.get('boxes', []) or []):
            lo = np.asarray([float(v) for v in b['min_mm']], dtype=float) / 1000.0
            hi = np.asarray([float(v) for v in b['max_mm']], dtype=float) / 1000.0
            if np.any(hi <= lo):
                raise ValueError('collision tool box %r: max_mm must exceed min_mm on every '
                                 'axis, got %s .. %s' % (b.get('name', i), b['min_mm'],
                                                         b['max_mm']))
            self.boxes.append((str(b.get('name', 'box_%d' % i)),
                               (lo + hi) / 2.0, (hi - lo) / 2.0))

        used = self.spacer_len + self.body_len
        self.finger_len = self.fingertip_z - used
        # A SANITY CHECK ON THE CHAIN, because this split is derived, not measured. Robotiq's
        # own URDF puts the stock 2F-85 at 98.3 mm from its mounting face to the pads (base ->
        # knuckle 54.9, knuckle -> finger -3.8, finger -> tip 47.2), of which ~43 mm is finger.
        # A derived finger far longer than that means either genuinely custom fingertips (this
        # cell does have them) or a stale fingertip_grasp -- and the second would be wrong for
        # far more than this guard, since fingertip_grasp is the frame every grasp is written
        # against. Say so rather than silently modelling a gripper nobody owns.
        if self.finger_len > 2.0 * 0.0434:
            log.warning('TOOL MODEL: the spacer (%.0f mm) and gripper body (%.0f mm) leave '
                        '%.0f mm of FINGER to reach the calibrated fingertip at %.0f mm. A '
                        'stock 2F-85 finger is ~43 mm. Either these are custom fingertips, or '
                        'fingertip_grasp is stale -- measure it, because that frame is what '
                        'every grasp and the in-hand belief are written against.',
                        self.spacer_len * 1000, self.body_len * 1000,
                        self.finger_len * 1000, self.fingertip_z * 1000)
        if self.finger_len <= 0.0:
            raise ValueError(
                'the tool model does not fit: spacer %.0f mm + body %.0f mm is already past the '
                'calibrated fingertip at %.0f mm. Shorten one, or re-measure fingertip_grasp.'
                % (self.spacer_len * 1000, self.body_len * 1000, self.fingertip_z * 1000))

    def segments(self):
        """[(name, z_from, z_to, radius, x_offset)] along tool0, outboard-positive."""
        s, b = self.spacer_len, self.spacer_len + self.body_len
        return [
            # THE WRIST SIDE -- held to the strict margin. A touch here is a crash.
            ('spacer', 0.0, s, self.spacer_r, 0.0),
            ('gripper_body', s, b, self.body_r, 0.0),
            # THE FINGERTIPS -- and ONLY these carry the intersection allowance. See the module
            # docstring: the pads are supposed to reach the work surface; the wrist is not.
            ('fingertip_a', b, self.fingertip_z, self.finger_r, +self.finger_half_gap),
            ('fingertip_b', b, self.fingertip_z, self.finger_r, -self.finger_half_gap),
        ]


    def body_names(self):
        """Every tool body, capsules and boxes alike."""
        return [s[0] for s in self.segments()] + [b[0] for b in self.boxes]


FINGERTIP_BODIES = ('fingertip_a', 'fingertip_b')

# ---- self-collision ------------------------------------------------------------------------
# The kinematic chain, in order. Only meaningful in 'urdf' mode: the capsule envelope is
# deliberately fatter than the real shell, so capsules overlap each other at every joint and
# would report a permanent self-collision.
CHAIN = ('base_link_inertia', 'shoulder_link', 'upper_arm_link', 'forearm_link',
         'wrist_1_link', 'wrist_2_link', 'wrist_3_link')

# ADJACENT LINKS OVERLAP BY DESIGN. Their housings interpenetrate at the joint so the arm looks
# continuous -- measured at -2 to -5 mm on every working pose in this cell, at every
# configuration, because it is how the meshes are drawn and not a function of the joint angles.
# Checking them would fire constantly and mean nothing. Pairs two or more apart in the chain
# are the ones that can actually come together: on the same working poses the closest of those
# sits at +18 mm, so there is real signal to read.
SELF_PAIRS = tuple((a, b) for i, a in enumerate(CHAIN) for b in CHAIN[i + 2:])

# TOOL vs ARM, split by whether the pair can MOVE relative to each other.
#
# The tool is bolted to wrist_3 and every tool body is either on the tool axis or symmetric
# about it, so wrist_3's rotation does not change where the tool sits relative to wrist_3 OR
# wrist_2. Measured over 150 random configurations, every tool/wrist_2 distance has a range of
# exactly 0.0 mm: spacer 10.1, gripper_body 10.6, fingertips 98.6. They are RIGID offsets.
#
# A per-pose check on a rigid offset is meaningless -- it is either always fine or always
# broken -- and it is actively harmful here, because spacer/wrist_2 at 10.1 mm is the tightest
# pair in the whole set and would cap the usable self-collision margin at 10 mm for a pair that
# cannot collide. So they are checked ONCE at construction instead (see _check_fixed_pairs),
# which still catches a tool model that would foul the wrist -- a longer spacer, say -- but
# catches it as the design error it is rather than as a runtime refusal.
# wrist_3 is the MOUNTING FACE. The spacer is bolted to it and the link's mesh includes the
# flange, so the two overlap by ~38 mm at every configuration BY CONSTRUCTION. Excluded from
# both checks; there is nothing it could tell us.
TOOL_MOUNT_LINK = 'wrist_3_link'
# wrist_2 is rigid relative to the tool but SHOULD be clear -- so it is the one worth checking
# once, at build, where a tool that fouls it is reported as the design error it is.
TOOL_FIXED_LINKS = ('wrist_2_link',)
TOOL_SELF_SKIP = (TOOL_MOUNT_LINK,) + TOOL_FIXED_LINKS


class GroundCollisionModel:
    """UR10e + spacer + 2F-85 against a horizontal ground plane, in pybullet DIRECT mode.

    One body per link, posed from fk_links() at every query, and one static plane. pybullet is
    doing the distance work (so an obstacle can be added later without changing the callers)
    while the kinematics stay here, where they can be checked against the controller."""

    URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'description', 'ur10e', 'ur10e.urdf')

    def __init__(self, cfg, ground_z_m=None, fingertip_z_m=0.183):
        import pybullet as pb
        self.pb = pb
        c = dict(cfg or {})
        self.enabled = bool(c.get('enabled', True))
        self.ground_z = (float(ground_z_m) if ground_z_m is not None
                         else float(c.get('ground_z_m', 0.0)))
        self.margin = float(c.get('margin_mm', 0.0)) / 1000.0
        # THE EXCEPTION, and it is deliberately a separate number from `margin` so that widening
        # one can never widen the other.
        self.fingertip_margin = float(c.get('fingertip_margin_mm', 5.0)) / 1000.0
        self.path_samples = int(c.get('path_samples', 25))
        # SELF-COLLISION. Its own margin, and deliberately NOT the fingertip ground allowance:
        # a fingertip is meant to reach the work surface, it is never meant to reach the
        # forearm. Nothing about the tool gets an allowance against the arm.
        sc = dict(c.get('self_collision', {}) or {})
        self.self_enabled = bool(sc.get('enabled', True))
        self.self_margin = float(sc.get('margin_mm', 5.0)) / 1000.0
        self.tool = ToolModel(c.get('tool'), fingertip_z_m=fingertip_z_m)
        self.radii = [float(r) for r in c.get('link_radii_m', UR10E_RADII)]

        self.client = pb.connect(pb.DIRECT)
        self.plane = pb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=pb.createCollisionShape(
                pb.GEOM_BOX, halfExtents=[5.0, 5.0, 0.5], physicsClientId=self.client),
            basePosition=[0.0, 0.0, self.ground_z - 0.5],
            physicsClientId=self.client)
        # THE REAL MESHES IF WE HAVE THEM. pybullet does the arm's forward kinematics itself
        # once the URDF is loaded, so in this mode fk_links() is only used for the tool chain
        # and for the cross-check -- the clearances come off UR's own collision shells.
        self.robot = None
        self.mode = 'capsule'
        self._link_index = {}
        if bool(c.get('use_urdf', True)) and os.path.isfile(self.URDF):
            self.robot = pb.loadURDF(self.URDF, useFixedBase=True,
                                     flags=pb.URDF_USE_SELF_COLLISION,
                                     physicsClientId=self.client)
            for j in range(pb.getNumJoints(self.robot, physicsClientId=self.client)):
                info = pb.getJointInfo(self.robot, j, physicsClientId=self.client)
                self._link_index[info[12].decode()] = j
                if info[2] != pb.JOINT_FIXED:
                    self._joint_index = getattr(self, '_joint_index', [])
                    self._joint_index.append(j)
            self.mode = 'urdf'

        self._bodies = {}
        # EACH CAPSULE IS ITS OWN LINK'S LENGTH. Building them all the same length was a real
        # bug: a 1 m capsule on the 120 mm wrist_2 link hung 44 cm past both joint origins and
        # reported the floor as a collision from half a metre up. The lengths are constant, so
        # they belong here at construction -- and the tests assert capsule length == segment
        # length at random configurations, which is what would catch it happening again.
        if self.mode == 'capsule':
            for i in range(6):
                self._bodies[UR_JOINTS[i]] = self._capsule(self.radii[i],
                                                           UR10E_LINK_LENGTHS[i])
        for name, z0, z1, r, _x in self.tool.segments():
            self._bodies[name] = self._capsule(r, max(z1 - z0, 1e-4))
        for name, _centre, half in self.tool.boxes:
            self._bodies[name] = self._box(half)
        # The tool must clear the wrist it is bolted to, at every configuration. Fixed offsets,
        # so this is a build-time question, not a per-move one.
        self._pose_all(np.zeros(6))
        self._check_fixed_pairs()

    def _check_fixed_pairs(self):
        """The tool/arm pairs whose relative pose is FIXED, verified once.

        These cannot be checked per-pose usefully (see TOOL_SELF_SKIP), but they still have to
        be clear or the tool fouls the wrist at every configuration. Checking at build turns
        that from a runtime refusal nobody can act on into a startup error naming the part."""
        if self.mode != 'urdf' or not self.self_enabled:
            return True
        ok = True
        for tname in self.tool.body_names():
            for link in TOOL_FIXED_LINKS:
                if link not in self._link_index:
                    continue
                pts = self.pb.getClosestPoints(self._bodies[tname], self.robot, distance=0.5,
                                               linkIndexB=self._link_index[link],
                                               physicsClientId=self.client)
                d = min((c[8] for c in pts), default=0.5)
                if d < 0.0:
                    ok = False
                    log.error('TOOL MODEL FOULS THE WRIST: %s overlaps %s by %.1f mm, and that '
                              'offset is RIGID -- it is wrong at every configuration, not just '
                              'this one. Check pickup.collision.tool against the hardware.',
                              tname, link, -d * 1000.0)
        return ok

    # ------------------------------------------------------------------ construction
    def _box(self, half_extents):
        pb = self.pb
        shape = pb.createCollisionShape(pb.GEOM_BOX,
                                        halfExtents=[float(v) for v in half_extents],
                                        physicsClientId=self.client)
        return pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                  basePosition=[0, 0, 50.0], physicsClientId=self.client)

    def _capsule(self, radius, length):
        pb = self.pb
        shape = pb.createCollisionShape(pb.GEOM_CAPSULE, radius=radius, height=length,
                                        physicsClientId=self.client)
        return pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                  basePosition=[0, 0, 50.0], physicsClientId=self.client)

    def close(self):
        try:
            self.pb.disconnect(self.client)
        except Exception:                                      # noqa: BLE001 -- teardown only
            pass

    # ------------------------------------------------------------------ posing
    @staticmethod
    def _segment_pose(p0, p1):
        """Place a capsule (its own axis is +z, centred) on the segment p0 -> p1."""
        from scipy.spatial.transform import Rotation
        v = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
        n = float(np.linalg.norm(v))
        mid = (np.asarray(p0, dtype=float) + np.asarray(p1, dtype=float)) / 2.0
        if n < 1e-9:
            return mid, [0.0, 0.0, 0.0, 1.0], n
        z = v / n
        # any rotation taking +z onto the segment direction
        axis = np.cross([0.0, 0.0, 1.0], z)
        s = float(np.linalg.norm(axis))
        if s < 1e-12:
            quat = ([0.0, 0.0, 0.0, 1.0] if z[2] > 0 else [1.0, 0.0, 0.0, 0.0])
        else:
            ang = float(np.arctan2(s, float(np.dot([0.0, 0.0, 1.0], z))))
            quat = Rotation.from_rotvec(axis / s * ang).as_quat().tolist()
        return mid, quat, n

    def _pose_all(self, q):
        """Move every body onto the configuration `q`. Returns the tool0 pose."""
        pb = self.pb
        frames = fk_links(q)
        if self.mode == 'urdf':
            for j, v in zip(self._joint_index, q):
                pb.resetJointState(self.robot, j, float(v), physicsClientId=self.client)
        for i in range(6) if self.mode == 'capsule' else ():
            p0, p1 = frames[i][:3, 3], frames[i + 1][:3, 3]
            mid, quat, _n = self._segment_pose(p0, p1)
            body = self._bodies[UR_JOINTS[i]]
            # A capsule's length is baked into its shape, so a zero-length link (wrist offsets
            # that coincide) is simply parked at the joint -- the spherical caps still cover it.
            pb.resetBasePositionAndOrientation(body, mid.tolist(), quat,
                                               physicsClientId=self.client)
        T = frames[6]
        self._pose_tool(T)
        return T

    # ------------------------------------------------------------------ queries
    def clearances(self, q):
        """{body: signed clearance to the ground plane, m}. Negative = intersecting.

        In 'urdf' mode the ARM links are queried per-link off the loaded robot (so each one is
        named and reported separately, which is what makes the error message useful), and the
        tool chain is queried off its own bodies. In 'capsule' mode every body is standalone."""
        self._pose_all(q)
        out = {}
        if self.mode == 'urdf':
            for name, idx in self._link_index.items():
                pts = self.pb.getClosestPoints(self.robot, self.plane, distance=1.0,
                                               linkIndexA=idx, physicsClientId=self.client)
                out[name] = min((c[8] for c in pts), default=1.0)
        for name, body in self._bodies.items():
            pts = self.pb.getClosestPoints(body, self.plane, distance=1.0,
                                           physicsClientId=self.client)
            out[name] = min((c[8] for c in pts), default=1.0)
        return out

    def self_clearances(self, q):
        """{pair: signed clearance, m} for the arm against itself and the tool against the arm.

        ONLY IN 'urdf' MODE. The capsule envelope is intentionally fatter than the real shell,
        so capsules overlap at every joint and would report a permanent, meaningless
        self-collision -- returning nothing is honest, and the caller says so once.

        Adjacent chain links are excluded because their meshes interpenetrate by design; see
        SELF_PAIRS. The tool is excluded against wrist_3 for the same reason -- it is bolted
        there.

        WHAT IS DELIBERATELY NOT CHECKED: TOOL BODY vs TOOL BODY. The spacer, the gripper and
        any bracket are all bolted to tool0, so their relative poses are RIGID -- a per-pose
        check on them can only ever return the same answer. Worse, several overlap BY
        CONSTRUCTION: the camera bracket starts at the tool0 origin and so does the spacer, so
        they share space and a check would fire on every pose. The camera is checked against
        the ARM and the GROUND -- the things it can actually move relative to."""
        if self.mode != 'urdf' or not self.self_enabled:
            return {}
        self._pose_all(q)
        out = {}
        for a, b in SELF_PAIRS:
            pts = self.pb.getClosestPoints(self.robot, self.robot, distance=0.5,
                                           linkIndexA=self._link_index[a],
                                           linkIndexB=self._link_index[b],
                                           physicsClientId=self.client)
            out['%s~%s' % (a, b)] = min((c[8] for c in pts), default=0.5)
        for tname in self.tool.body_names():
            for link in CHAIN:
                if link in TOOL_SELF_SKIP:
                    continue
                pts = self.pb.getClosestPoints(self._bodies[tname], self.robot, distance=0.5,
                                               linkIndexB=self._link_index[link],
                                               physicsClientId=self.client)
                out['%s~%s' % (tname, link)] = min((c[8] for c in pts), default=0.5)
        return out

    def check_q(self, q):
        """(ok, worst_body, violation_m) for one configuration -- GROUND and SELF together.

        `violation_m` is how far past its OWN allowance the worst offender is, so the fingertip
        ground exception is already accounted for and the number stays comparable across very
        different checks. A self-collision name reads 'link_a~link_b'."""
        worst_name, worst = None, 0.0
        for name, clear in self.clearances(q).items():
            allow = (-self.fingertip_margin if name in FINGERTIP_BODIES else self.margin)
            over = allow - clear            # > 0 means it broke its own allowance
            if over > worst:
                worst_name, worst = name, over
        # THE FINGERTIP ALLOWANCE DOES NOT REACH HERE. It exists because the pads must arrive
        # at the work surface; nothing on the tool is ever meant to arrive at the forearm.
        for name, clear in self.self_clearances(q).items():
            over = self.self_margin - clear
            if over > worst:
                worst_name, worst = name, over
        return (worst_name is None), worst_name, worst

    def _pose_tool(self, T_base_tool0):
        """Place ONLY the tool bodies, from a tool0 pose. No joint angles needed."""
        from scipy.spatial.transform import Rotation
        for name, z0, z1, _r, xoff in self.tool.segments():
            p0 = (T_base_tool0 @ np.array([xoff, 0.0, z0, 1.0]))[:3]
            p1 = (T_base_tool0 @ np.array([xoff, 0.0, z1, 1.0]))[:3]
            mid, quat, _n = self._segment_pose(p0, p1)
            self.pb.resetBasePositionAndOrientation(self._bodies[name], mid.tolist(), quat,
                                                    physicsClientId=self.client)
        # A BOX TAKES tool0's ORIENTATION, not just its position -- it is bolted to the flange
        # and turns with it. (A capsule only needs its two endpoints, which is why the two are
        # posed differently.)
        if self.tool.boxes:
            quat_t = Rotation.from_matrix(T_base_tool0[:3, :3]).as_quat().tolist()
            for name, centre, _half in self.tool.boxes:
                p = (T_base_tool0 @ np.append(centre, 1.0))[:3]
                self.pb.resetBasePositionAndOrientation(self._bodies[name], p.tolist(), quat_t,
                                                        physicsClientId=self.client)

    def tool_clearances(self, T_base_tool0):
        """{tool body: clearance to the ground, m} from a tool0 pose alone.

        NO IK REQUIRED, which is the point: when the arm cannot be solved for a grasp this is
        still answerable, and "the gripper would be 20 mm into the bench" explains an
        unreachable pose that a bare "no IK solution" does not."""
        self._pose_tool(T_base_tool0)
        out = {}
        for name in self.tool.body_names():
            pts = self.pb.getClosestPoints(self._bodies[name], self.plane, distance=1.0,
                                           physicsClientId=self.client)
            out[name] = min((c[8] for c in pts), default=1.0)
        return out

    def allowance(self, name):
        """The clearance this body must keep: the fingertips may intersect, nothing else may."""
        return -self.fingertip_margin if name in FINGERTIP_BODIES else self.margin

    def check_tool_pose(self, T_base_tool0):
        """(ok, worst_body, violation_m) for the SPACER, GRIPPER BODY and FINGERTIPS only.

        WHY A TOOL-ONLY CHECK EXISTS. The descent is a straight CARTESIAN line, so checking it
        the way a moveJ is checked would need an IK solution per sample -- and arm.ik is a
        controller call that does not work offline. The tool bodies, though, depend on nothing
        but tool0's pose, so they can be checked from the pose alone, on any machine, with no
        robot connected. And they are the bodies that matter here: during a descent onto the
        bench it is the gripper that arrives first, not the elbow.

        The fingertip allowance applies exactly as it does everywhere else -- see the module
        docstring. The gripper WRIST and body get none of it."""
        self._pose_tool(T_base_tool0)
        worst_name, worst = None, 0.0
        for name in self.tool.body_names():
            pts = self.pb.getClosestPoints(self._bodies[name], self.plane, distance=1.0,
                                           physicsClientId=self.client)
            clear = min((c[8] for c in pts), default=1.0)
            allow = (-self.fingertip_margin if name in FINGERTIP_BODIES else self.margin)
            over = allow - clear
            if over > worst:
                worst_name, worst = name, over
        return (worst_name is None), worst_name, worst

    def check_tool_path(self, T_from, T_to, samples=None):
        """Sample a straight CARTESIAN tool0 path, tool bodies only. Rotation is not
        interpolated -- a descent holds its attitude, and taking the START attitude for every
        sample is the conservative reading when it does not."""
        n = int(samples or self.path_samples)
        a, b = np.asarray(T_from, dtype=float), np.asarray(T_to, dtype=float)
        worst_name, worst, worst_f = None, 0.0, 0.0
        for k in range(n + 1):
            f = k / float(n)
            T = np.array(a, dtype=float)
            T[:3, 3] = a[:3, 3] + f * (b[:3, 3] - a[:3, 3])
            ok, name, over = self.check_tool_pose(T)
            if not ok and over > worst:
                worst_name, worst, worst_f = name, over, f
        return (worst_name is None), worst_name, worst, worst_f

    def check_path(self, q_from, q_to, samples=None):
        """Sample the STRAIGHT JOINT PATH -- which is what a moveJ actually drives.

        The endpoints are not enough and that is the whole point: interpolating in joint space
        swings the tool through an arc, so a move between two clear poses can still put the
        gripper through the bench halfway along."""
        a = np.asarray(q_from, dtype=float)
        b = np.asarray(q_to, dtype=float)
        n = int(samples or self.path_samples)
        worst_name, worst, worst_f = None, 0.0, 0.0
        for k in range(n + 1):
            f = k / float(n)
            ok, name, over = self.check_q(a + f * (b - a))
            if not ok and over > worst:
                worst_name, worst, worst_f = name, over, f
        return (worst_name is None), worst_name, worst, worst_f

    # ------------------------------------------------------------------ trust
    def verify_against_controller(self, arm, q=None, tol_mm=1.0):
        """Compare our DH forward kinematics against the CONTROLLER's for one configuration.

        The whole model hangs off fk_links, and a DH typo would make it confidently wrong in a
        direction nobody would notice until the gripper hit something. Offline `arm.fk` is a
        fixed stand-in, so this can only run on real hardware -- it returns None there rather
        than pretending to have checked."""
        if getattr(arm, 'dry_run', False):
            log.warning('COLLISION MODEL UNVERIFIED: dry run, so arm.fk is a stand-in. The '
                        'ground check is running on our own DH chain with nothing to compare '
                        'it against.')
            return None
        q = arm.q() if q is None else q
        mine, theirs = fk_links(q)[6], arm.fk(q)
        err = float(np.linalg.norm(mine[:3, 3] - theirs[:3, 3])) * 1000.0
        if err > float(tol_mm):
            log.error('COLLISION MODEL FK MISMATCH: our tool0 is %.2f mm from the controller\'s '
                      '(limit %.2f). The DH chain in urlab/robot/collision.py does not describe '
                      'this robot -- every clearance it reports is suspect.', err, tol_mm)
            return False
        log.info('Collision model FK agrees with the controller to %.3f mm.', err)
        return True

    def describe(self):
        seg = ', '.join('%s %.0f-%.0f mm r%.0f' % (n, z0 * 1000, z1 * 1000, r * 1000)
                        for n, z0, z1, r, _x in self.tool.segments())
        for n, centre, half in self.tool.boxes:
            seg += ', %s box %s mm at %s' % (n, np.round(half * 2000, 0).astype(int).tolist(),
                                             np.round(centre * 1000, 0).astype(int).tolist())
        arm = ("UR10e collision MESHES (UR's own description)" if self.mode == 'urdf'
               else 'UR10e capsule envelope (conservative -- fetch the description for meshes)')
        if self.mode != 'urdf':
            self_txt = 'self-collision UNCHECKED (needs the meshes -- capsules overlap at every '
            'joint)'
        elif not self.self_enabled:
            self_txt = 'self-collision OFF'
        else:
            self_txt = ('self-collision %.1f mm over %d link pairs + %d tool/arm pairs'
                        % (self.self_margin * 1000, len(SELF_PAIRS),
                           len(self.tool.body_names()) * (len(CHAIN) - len(TOOL_SELF_SKIP))))
        return ('arm: %s | ground z %.3f m | margin %.1f mm (fingertips %.1f mm INTERSECTION '
                'allowed) | %s | tool: %s'
                % (arm, self.ground_z, self.margin * 1000, self.fingertip_margin * 1000,
                   self_txt, seg))
