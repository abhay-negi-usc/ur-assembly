"""BEHAVIOR DEMO -- the worked example of writing a new script.

A script is three things:

    1. the Robot, which registers the gripper/camera and the frames (here: the shared
       frames.yaml catalogue plus one ad-hoc probe frame riding on the fingertip);
    2. a CHAIN of behaviors from the library (urlab/behaviors) -- each a thin shell over
       one Robot motion primitive;
    3. `bt.app(...)`, which supplies the CLI / config / lifecycle boilerplate.

Every motion here is a small relative jog, so it is safe to run anywhere the arm has a few
centimetres of clearance.  Start with a dry run to see the whole tree without moving:

    python -m urlab.apps.behavior_demo --dry-run --debug
"""

from .. import behaviors as bt
from ..transforms import translation_matrix as trans


def build(cfg, robot, camera, args):
    # Frames: the shared catalogue, plus one ad-hoc frame 20 mm past the fingertip.
    # Chains resolve recursively (probe -> fingertip -> tool0 -> live FK).
    robot.load_frame_catalogue()
    robot.register_frame('probe', trans([0.0, 0.0, 0.020]), parent='fingertip')

    return bt.chain(
        'behavior-demo',
        ('say', 'a new script = registered frames + a chain of behaviors'),
        ('reset', robot, cfg),
        ('open_gripper', robot),
        ('move_relative', robot, trans([0.0, 0.0, -0.05]), 'descend 50 mm (world z)',
         {'expressed_in': 'base_link'}),
        ('move_relative', robot, trans([0.0, 0.0, 0.03]), 'advance 30 mm along the probe',
         {'expressed_in': 'probe'}),
        ('move_relative', robot, trans([0.0, 0.0, -0.03]), 'back off along the probe',
         {'expressed_in': 'probe'}),
        ('close_gripper', robot),
        ('move_relative', robot, trans([0.0, 0.0, 0.05]), 'rise 50 mm (world z)',
         {'expressed_in': 'base_link'}),
        ('open_gripper', robot),
        ('reset', robot, cfg, 'end reset'),
    )


main = bt.app('Behavior-chain demo (small relative jogs only)', 'cartesian', build)

if __name__ == '__main__':
    main()
