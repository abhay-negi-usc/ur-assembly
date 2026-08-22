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


FINGERTIP_BODIES = ('fingertip_a', 'fingertip_b')


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

    # ------------------------------------------------------------------ construction
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
        for name, z0, z1, _r, xoff in self.tool.segments():
            p0 = (T @ np.array([xoff, 0.0, z0, 1.0]))[:3]
            p1 = (T @ np.array([xoff, 0.0, z1, 1.0]))[:3]
            mid, quat, _n = self._segment_pose(p0, p1)
            pb.resetBasePositionAndOrientation(self._bodies[name], mid.tolist(), quat,
                                               physicsClientId=self.client)
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

    def check_q(self, q):
        """(ok, worst_body, violation_m) for one configuration.

        `violation_m` is how far past its OWN allowance the worst body is, so the fingertip
        exception is already accounted for and the number is comparable across bodies."""
        worst_name, worst = None, 0.0
        for name, clear in self.clearances(q).items():
            allow = (-self.fingertip_margin if name in FINGERTIP_BODIES else self.margin)
            over = allow - clear            # > 0 means it broke its own allowance
            if over > worst:
                worst_name, worst = name, over
        return (worst_name is None), worst_name, worst

    def _pose_tool(self, T_base_tool0):
        """Place ONLY the tool bodies, from a tool0 pose. No joint angles needed."""
        for name, z0, z1, _r, xoff in self.tool.segments():
            p0 = (T_base_tool0 @ np.array([xoff, 0.0, z0, 1.0]))[:3]
            p1 = (T_base_tool0 @ np.array([xoff, 0.0, z1, 1.0]))[:3]
            mid, quat, _n = self._segment_pose(p0, p1)
            self.pb.resetBasePositionAndOrientation(self._bodies[name], mid.tolist(), quat,
                                                    physicsClientId=self.client)

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
        for name in self._bodies:
            if name not in [s[0] for s in self.tool.segments()]:
                continue
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
        arm = ("UR10e collision MESHES (UR's own description)" if self.mode == 'urdf'
               else 'UR10e capsule envelope (conservative -- fetch the description for meshes)')
        return ('arm: %s | ground z %.3f m | margin %.1f mm (fingertips %.1f mm INTERSECTION '
                'allowed) | tool: %s' % (arm, self.ground_z, self.margin * 1000,
                                         self.fingertip_margin * 1000, seg))
