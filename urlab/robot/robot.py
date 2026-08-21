"""Robot -- the one class a script talks to: arm + registered gripper/camera + a frame
registry + motion primitives.

Composition, not inheritance.  A demo holds a Robot and calls skills/behaviors as free
functions; sharing happens because the skills are shared, not because demos are relatives.

FRAME SYSTEM (mirrors the sunrise-wrapper KukaRobot).  Frames are named transforms with a
declared parent.  Two frames exist without registration:

    base_link   the world/robot-base frame (identity)
    tool0       the flange, read live from RTDE on every resolve

Everything else is registered:

    robot.register_frame('probe_tip', T, parent='tool0')     # rides on the arm
    robot.register_frame('fixture', T, parent='base_link')   # bolted to the world
    robot.register_frame('slot_3', T, parent='fixture')      # chains resolve recursively
    robot.load_frame_catalogue()                             # everything in configs/frames.yaml

and every registered frame then works uniformly in observation and motion:

    robot.pose('probe_tip', reference='fixture')             # any frame in any frame
    robot.move_cartesian(T, frame='probe_tip', reference='fixture')
    robot.move_relative(delta, expressed_in='probe_tip')

MOTION PRIMITIVES are the easy functions everything builds on -- behaviors
(urlab/behaviors) wrap exactly these, so a new script is a chain of behaviors and a
behavior is a thin shell over one primitive:

    move_joints      joint move, optionally force-guarded
    move_cartesian   put a robot-attached frame ON a target pose expressed in any frame
                     ('ptp' = IK + joint move, 'lin' = straight line), back-solving the
                     flange pose;  a mutable `seed` dict keeps consecutive IK solutions on
                     the same branch
    move_relative    jog by a delta expressed in any frame (tool frames post-multiply)
"""

import numpy as np

from .. import log as urlog
from ..frames import FrameGraph
from ..transforms import from_cfg, inverse, xyzrpy_to_matrix
from .arm import URArm
from .gripper import Robotiq2F85

log = urlog.get('robot')


def _as_matrix(pose):
    """Accept a 4x4 matrix or a 6-sequence [x, y, z (m), roll, pitch, yaw (rad)]."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (6,):
        return xyzrpy_to_matrix(pose[:3], pose[3:])
    raise ValueError(f'pose must be a 4x4 matrix or a 6-vector [xyz m, rpy rad]; '
                     f'got shape {pose.shape}')


class Robot:
    """Arm + registered gripper/camera + frame registry + motion primitives.  Owns the
    connections; close it or use it as a context manager."""

    def __init__(self, cfg, with_gripper=True):
        self.cfg = cfg
        self.frames = FrameGraph()
        self.arm = URArm(cfg, frames=self.frames)
        self.gripper = None
        self.camera_device = None              # registered with register_camera()

        self.base_frame = cfg.get('base_frame', 'base_link')
        self.tip_frame = cfg.get('tip_frame', 'tool0')
        self.camera_frame = cfg.get('camera_frame', 'camera1_color_optical_frame')
        self._parents = {}                     # name -> parent, for tool-attachment checks
        self._targets = {}                     # name -> recorded base_link pose

        if with_gripper:
            self.register_gripper(Robotiq2F85(cfg))

        # Hand-eye: tool0 -> camera. Static, so it never expires -- this single edge replaces
        # the old static_transform_publisher.
        self.T_tool0_cam = from_cfg(cfg.section('hand_eye'))
        self.register_frame(self.camera_frame, self.T_tool0_cam)

        # Frames rigidly attached to tool0. `grasp` is the TCP a grasp pose is expressed
        # for; `fingertip` is where the fingers actually meet.
        self.T_tool0_grasp = from_cfg(cfg.section('grasp_tcp_offset'))
        self.T_tool0_fingertip = from_cfg(cfg.section('fingertip_grasp'))
        self.register_frame('grasp', self.T_tool0_grasp)
        self.register_frame('fingertip', self.T_tool0_fingertip)

    # ------------------------------------------------------------------ registration
    def register_gripper(self, gripper):
        """Attach a gripper to this robot (constructed by default; call directly to swap)."""
        self.gripper = gripper
        return gripper

    def register_camera(self, camera):
        """Attach a camera DEVICE (available as robot.camera_device).  Its pose is already a
        registered frame (`camera_frame`, via the hand_eye edge); `robot.camera()` reads it."""
        self.camera_device = camera
        return camera

    def register_frame(self, name, T, parent=None):
        """Register a named frame at pose `T` (4x4 or [xyz m, rpy rad]) relative to `parent`.

        parent defaults to tool0 (a robot-attached frame).  `base_link` makes a world-fixed
        frame; any registered frame chains recursively -- the graph resolves the whole chain,
        reading live FK wherever tool0 is on the path."""
        parent = parent or self.tip_frame
        if name in (self.base_frame, self.tip_frame):
            raise ValueError(f'{name!r} is a built-in frame and cannot be re-registered')
        if parent not in (self.base_frame, self.tip_frame) and parent not in self._parents:
            raise KeyError(f'unknown parent frame {parent!r}; register it first')
        self._parents[name] = parent
        self.frames.set_static(parent, name, _as_matrix(T))

    def register_target(self, name, T_base):
        """Record where a frame IS in base_link (a measured mate, a taught pose).  Kept apart
        from register_frame: the same name can be both a tool-attached frame (the part in the
        fingers) and a recorded target (where it must end up)."""
        self._targets[name] = _as_matrix(T_base)

    def load_frame_catalogue(self, cfg=None):
        """Register everything configs/frames.yaml declares: `frames:` as tool0-attached
        frames, `targets:` as recorded base_link poses."""
        from .. import tool_frames
        for name, T in tool_frames.load_frames(cfg or self.cfg).items():
            if name not in (self.base_frame, self.tip_frame):   # built-ins exist already
                self.register_frame(name, T)
        for name, T in tool_frames.load_targets(cfg or self.cfg).items():
            self.register_target(name, T)

    def target(self, name):
        """The recorded base_link pose registered under `name`."""
        try:
            return self._targets[name]
        except KeyError:
            raise KeyError(f'no target registered as {name!r}; '
                           f'known: {sorted(self._targets)}') from None

    def is_tool_attached(self, frame):
        """True if `frame` rides on the arm (its parent chain reaches tool0)."""
        seen = set()
        while frame in self._parents and frame not in seen:
            seen.add(frame)
            frame = self._parents[frame]
        return frame == self.tip_frame

    # ------------------------------------------------------------------ observation
    def pose(self, frame, reference=None):
        """T_reference_frame: the current pose of any frame expressed in any other frame
        (default base_link), reading live FK wherever the chain crosses the arm."""
        T = self.frames.lookup(reference or self.base_frame, frame)
        if T is None:
            raise KeyError(f'no path from {reference or self.base_frame!r} to {frame!r} in '
                           f'the frame graph; register the frame first')
        return T

    def joints(self):
        return self.arm.q()

    def tool0(self):
        return self.arm.tcp_pose()

    def camera(self):
        """Camera pose in base_link (tool0 @ hand-eye)."""
        return self.arm.tcp_pose() @ self.T_tool0_cam

    def fingertip(self):
        return self.arm.tcp_pose() @ self.T_tool0_fingertip

    def grasp_tcp(self):
        return self.arm.tcp_pose() @ self.T_tool0_grasp

    # ------------------------------------------------------------------ motion primitives
    def _tool_offset(self, frame):
        """T_tool0_frame for a robot-attached frame (identity for tool0 itself)."""
        if frame in (None, self.tip_frame):
            return np.eye(4)
        if not self.is_tool_attached(frame):
            raise ValueError(f'{frame!r} is not attached to the robot -- only tool-attached '
                             'frames can be moved')
        return self.frames.lookup(self.tip_frame, frame)

    def _flange_target(self, target, frame, reference):
        """Back-solve the tool0 pose that puts `frame` on `target` (expressed in `reference`):
        T_base_tool0 = T_base_ref @ target @ inv(T_tool0_frame)."""
        T = _as_matrix(target)
        if reference not in (None, self.base_frame):
            T = self.pose(reference) @ T
        return T @ inverse(self._tool_offset(frame))

    def move_joints(self, q, label='move joints', guard=None):
        """Joint move, optionally with the force guard armed as a canceller."""
        if guard is None:
            return self.arm.move_j(list(q), label=label)
        from .guard import guarded_move
        return guarded_move(self, guard, lambda: self.arm.move_j(list(q), label=label))

    def move_cartesian(self, target, frame=None, reference=None, interpolation='ptp',
                      label='move', seed=None, guard=None):
        """Put a robot-attached `frame` ON `target` (a pose expressed in `reference`).

        interpolation 'ptp' = IK + joint move; 'lin' = Cartesian straight line.  `seed` is a
        mutable dict holding {'q': ...}: the IK solution seeds from it and is written back,
        keeping consecutive moves on the same IK branch.  `guard` arms the force guard as a
        canceller (a trip = hit something unexpected)."""
        T_base_tool0 = self._flange_target(target, frame, reference)
        if interpolation == 'lin':
            def move():
                return self.arm.move_l(T_base_tool0, label=label)
        elif interpolation == 'ptp':
            q = self.arm.ik(T_base_tool0, (seed or {}).get('q'))
            if q is None:
                log.error('[%s] IK unreachable.', label)
                return False

            def move(_q=q):
                return self.arm.move_j(_q, label=label)
        else:
            raise ValueError(f"interpolation must be 'ptp' or 'lin', got {interpolation!r}")
        if guard is not None:
            from .guard import guarded_move
            ok = guarded_move(self, guard, move)
        else:
            ok = move()
        if ok and seed is not None and interpolation == 'ptp':
            seed['q'] = q
        return ok

    def move_relative(self, delta, expressed_in=None, interpolation='lin', label='jog',
                      guard=None):
        """Displace the arm by `delta` expressed in any frame.

        A robot-attached `expressed_in` (default tool0) post-multiplies -- a jog along the
        tool's own axes wherever it points; a world frame conjugates the delta into base."""
        expressed_in = expressed_in or self.tip_frame
        D = _as_matrix(delta)
        T_now = self.tool0()
        if expressed_in == self.tip_frame or self.is_tool_attached(expressed_in):
            T_off = self._tool_offset(expressed_in)
            T_target = T_now @ T_off @ D @ inverse(T_off)
        else:
            # Conjugate: the delta's axes are the REFERENCE frame's, the pivot is the arm's
            # current pose.
            T_ref = self.pose(expressed_in)
            T_target = T_ref @ D @ inverse(T_ref) @ T_now
        return self.move_cartesian(T_target, interpolation=interpolation, label=label,
                                   guard=guard)

    # ------------------------------------------------------------------ legacy move aliases
    def move_tool0(self, T, label='move'):
        return self.arm.move_to(T, label)

    def move_camera(self, T, label='move camera'):
        """Move so the CAMERA lands on T (used by every scan and visual-servo step)."""
        return self.arm.move_frame_to(T, self.T_tool0_cam, label)

    def move_fingertip(self, T, label='move fingertip'):
        return self.arm.move_frame_to(T, self.T_tool0_fingertip, label)

    def move_grasp_tcp(self, T, label='move grasp'):
        return self.arm.move_frame_to(T, self.T_tool0_grasp, label)

    def move_frame(self, T_base_target, frame, label='move'):
        """Move so the named tool0-attached frame lands on the target. `frame` is one of
        'tool0' | 'grasp' | 'fingertip', matching the target.frame key in the assembly
        configs."""
        offsets = {'tool0': None, 'grasp': self.T_tool0_grasp,
                   'fingertip': self.T_tool0_fingertip}
        if frame not in offsets:
            raise ValueError(f'Unknown frame {frame!r}; expected one of {sorted(offsets)}')
        T_tool0_frame = offsets[frame]
        if T_tool0_frame is None:
            return self.arm.move_to(T_base_target, label)
        return self.arm.move_frame_to(T_base_target, T_tool0_frame, label)

    def home(self, q_home, label='return home'):
        return self.arm.move_j(q_home, label=label)

    # ------------------------------------------------------------------ lifecycle
    def close(self):
        self.arm.disconnect()
        if self.gripper is not None:
            self.gripper.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
