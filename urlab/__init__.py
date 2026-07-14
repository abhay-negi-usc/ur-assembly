"""urlab -- a ROS-free control + perception library for a UR10e + Robotiq 2F-85 + RealSense D405.

A refactor of the ur-assembly ROS2 workspace into plain Python. Where the ROS version needed a
running graph -- a controller_manager, move_group, tf2, realsense2_camera, two SAM3 nodes -- this
talks to the robot over RTDE and the camera over pyrealsense2, in one process, with no launch
files.

Layout:
    transforms.py         pose math (the only place rpy/quat/rotvec conventions live)
    frames.py             FrameGraph -- the tf2 replacement, with explicit staleness
    config.py, log.py     config loading + logging/step-runner
    robot/                URArm (RTDE), Robotiq2F85 (Modbus), Robot facade, ForceGuard
    perception/           RealSenseCamera, ArUco, SAM3 adapter, ConnectorEstimator
    skills/               reusable behaviours: scan, servo, insert, pick, touch
    apps/                 thin demo scripts, one per old *_demo package
"""

__version__ = '0.1.0'
