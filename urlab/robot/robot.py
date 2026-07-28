"""Robot -- the facade the skills and apps talk to.

Composition, not inheritance. The ROS stack grew a four-deep chain

    PickPlace -> CablePickPlace -> CableTouchPickPlace
                               \\-> CablePickAssemble

where a subclass changed behaviour by overriding a hook in a grandparent, and where the touch
demo inherited a marker-tracking visual servo it never used. Here a demo is a short script that
holds a Robot and calls skills as free functions. Sharing happens because the skills are shared,
not because the demos are relatives -- which is what makes it possible to combine "scan" and
"insert" without one of them being an ancestor of the other.
"""

from .. import log as urlog
from ..frames import FrameGraph
from ..transforms import from_cfg, inverse
from .arm import URArm
from .gripper import Robotiq2F85

log = urlog.get('robot')


class Robot:
    """Arm + gripper + frame graph. Owns the connections; close it or use it as a context
    manager."""

    def __init__(self, cfg, with_gripper=True):
        self.cfg = cfg
        self.frames = FrameGraph()
        self.arm = URArm(cfg, frames=self.frames)
        self.gripper = Robotiq2F85(cfg) if with_gripper else None

        self.base_frame = cfg.get('base_frame', 'base_link')
        self.tip_frame = cfg.get('tip_frame', 'tool0')
        self.camera_frame = cfg.get('camera_frame', 'camera1_color_optical_frame')

        # Hand-eye: tool0 -> camera. Static, so it never expires -- this single edge replaces
        # ur_tf_demo's static_transform_publisher, its launch file, and its whole package.
        self.T_tool0_cam = from_cfg(cfg.section('hand_eye'))
        self.frames.set_static(self.tip_frame, self.camera_frame, self.T_tool0_cam)

        # Frames rigidly attached to tool0. `grasp` is the TCP a grasp pose is expressed for;
        # `fingertip` is where the fingers actually meet. Both are commanded via
        # arm.move_frame_to, so a demo says where the FINGERTIP should go and the flange pose
        # that puts it there falls out.
        self.T_tool0_grasp = from_cfg(cfg.section('grasp_tcp_offset'))
        self.T_tool0_fingertip = from_cfg(cfg.section('fingertip_grasp'))
        self.frames.set_static(self.tip_frame, 'grasp', self.T_tool0_grasp)
        self.frames.set_static(self.tip_frame, 'fingertip', self.T_tool0_fingertip)

        # Connector holder: another tool0-attached frame (the holder that presents the connector).
        # +X aligns with tool0 -Y, +Z with tool0 -Z (see connector_holder in the config).
        self.T_tool0_connector_holder = from_cfg(cfg.section('connector_holder'))
        self.frames.set_static(self.tip_frame, 'connector_holder', self.T_tool0_connector_holder)

        # Connector held IN the holder (connector_holder -> connector); identity until measured. This
        # is the physically-held connector used by the assembly / uncertain-sampling (cf. insert.py's
        # 'connector' target frame), distinct from the perception's detected 'connector'.
        self.T_connector_holder_connector = from_cfg(cfg.section('connector_in_holder'))
        self.T_tool0_connector = self.T_tool0_connector_holder @ self.T_connector_holder_connector
        self.frames.set_static('connector_holder', 'connector', self.T_connector_holder_connector)

    # ------------------------------------------------------------------ poses
    def tool0(self):
        return self.arm.tcp_pose()

    def camera(self):
        """Camera pose in base_link -- the composition tf2 used to do for us."""
        return self.arm.tcp_pose() @ self.T_tool0_cam

    def fingertip(self):
        return self.arm.tcp_pose() @ self.T_tool0_fingertip

    def grasp_tcp(self):
        return self.arm.tcp_pose() @ self.T_tool0_grasp

    def connector_holder(self):
        return self.arm.tcp_pose() @ self.T_tool0_connector_holder

    def connector(self):
        """The held connector's pose in base_link (tool0 -> connector_holder -> connector)."""
        return self.arm.tcp_pose() @ self.T_tool0_connector

    # ------------------------------------------------------------------ moves
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
        'tool0' | 'grasp' | 'fingertip', matching the target.frame key in the assembly configs."""
        offsets = {'tool0': None, 'grasp': self.T_tool0_grasp, 'fingertip': self.T_tool0_fingertip}
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
