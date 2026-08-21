"""Shared app entry point -- parse args, build the config, set up logging, run, tear down cleanly.

Replaces the identical `main()` + `rclpy.init/spin/shutdown` boilerplate that every ROS node
carried. An app is `run(cfg, robot, args)`; this wraps it so Ctrl-C, force-mode teardown and the
non-zero exit code are handled once.
"""

import logging
import sys

from .. import config as urconfig
from .. import log as urlog

log = urlog.get('app')


def run_app(description, default_config, build_and_run, with_gripper=True, needs_camera=False):
    """Standardised CLI + lifecycle. `build_and_run(cfg, robot, camera, args)` returns truthy on
    success. `robot` and `camera` are constructed here so teardown is guaranteed."""
    parser = urconfig.arg_parser(description, default_config)
    args = parser.parse_args()
    cfg = urconfig.from_args(args)
    urlog.setup(logging.DEBUG if cfg.get('debug') else logging.INFO)
    log.info('Config: %s', cfg.get('_config_path'))

    # Imported here, not at module load, so `--help` and dry runs work without ur_rtde /
    # pyrealsense2 installed.
    from ..robot import Robot

    ok = False
    robot = camera = None
    try:
        robot = Robot(cfg, with_gripper=with_gripper)
        if needs_camera:
            from ..perception import RealSenseCamera
            camera = robot.register_camera(RealSenseCamera(cfg, pose_fn=robot.camera))
        ok = bool(build_and_run(cfg, robot, camera, args))
    except KeyboardInterrupt:
        log.warning('Interrupted.')
    except Exception:                          # noqa: BLE001 -- log the traceback, still tear down
        log.exception('Unhandled error:')
    finally:
        if camera is not None:
            camera.close()
        if robot is not None:
            robot.close()
    sys.exit(0 if ok else 1)
