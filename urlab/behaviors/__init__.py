"""Behavior trees for the app scripts (py_trees).

`core` is the glue (blocking Action/Check leaves, sequence/selector/retry, run_tree);
`library` is the dictionary of common robot behaviors the apps compose their trees from.
Import the package and everything useful is at the top level:

    from urlab import behaviors as bt
    root = bt.sequence('probe', bt.make('open_gripper', robot), ...)
    ok = bt.run_tree(root)
"""

from .core import Action, Check, retry, run_tree, selector, sequence
from .library import (LIBRARY, AdmittanceRamp, CloseGripper, Hold, MoveFrame, MoveJoints,
                      MoveLinear, MoveRelative, MoveToPose, OpenGripper, OperatorGate,
                      ResetRobot, Say, ServoStop, VerifyHeld, Warmup, make)
from .script import app, chain

__all__ = [
    'Action', 'Check', 'sequence', 'selector', 'retry', 'run_tree', 'chain', 'app',
    'LIBRARY', 'make', 'MoveJoints', 'MoveToPose', 'MoveLinear', 'MoveFrame',
    'MoveRelative', 'OpenGripper', 'CloseGripper', 'ResetRobot', 'OperatorGate',
    'VerifyHeld', 'Warmup', 'AdmittanceRamp', 'Hold', 'ServoStop', 'Say',
]
