"""Write a new app by stringing behaviors together.

A script is three things: a Robot (which registers the gripper, the camera and the frames),
a CHAIN of behaviors from the library, and the shared runner for the CLI/lifecycle
boilerplate.  The whole of a new app:

    from urlab import behaviors as bt
    from urlab.transforms import translation_matrix as trans

    def build(cfg, robot, camera, args):
        robot.load_frame_catalogue()                       # everything in frames.yaml
        return bt.chain(
            'touch-demo',
            ('reset', robot, cfg),
            ('open_gripper', robot),
            ('move_frame', robot, robot.target('socket'), 'approach',
             {'frame': 'fingertip'}),
            ('move_relative', robot, trans([0.02, 0, 0]), 'advance 20 mm',
             {'expressed_in': 'fingertip'}),
            ('close_gripper', robot),
        )

    main = bt.app('Touch demo', 'touch_demo', build)

Steps are either ready py_trees nodes (so hand-built Actions and nested sequences mix in
freely) or `(library_name, *args, {kwargs})` tuples resolved through the behavior
dictionary.  `bt.app(...)` wraps the tree builder in the standard run_app lifecycle
(argparse, config, robot/camera construction, teardown, exit code).
"""

import py_trees

from .core import run_tree, sequence
from .library import make


def chain(name, *steps):
    """A Sequence built from behavior specs.

    Each step is a py_trees Behaviour, or a tuple `(library_name, *args)` optionally ending
    in a dict of keyword arguments: `('move_frame', robot, T, 'approach', {'frame': 'tip'})`.
    """
    nodes = []
    for step in steps:
        if isinstance(step, py_trees.behaviour.Behaviour):
            nodes.append(step)
        elif isinstance(step, (tuple, list)) and step and isinstance(step[0], str):
            spec = list(step)
            # Keyword arguments ride in a trailing PLAIN dict -- exact type on purpose, so a
            # dict subclass passed as a real argument (e.g. the Config) is never mistaken
            # for one.
            kwargs = spec.pop() if len(spec) > 1 and type(spec[-1]) is dict else {}
            nodes.append(make(spec[0], *spec[1:], **kwargs))
        else:
            raise TypeError(f'chain step must be a Behaviour or a (name, *args) tuple, '
                            f'got {step!r}')
    return sequence(name, *nodes)


def app(description, default_config, build_tree, **run_app_kwargs):
    """A ready `main()` for a behavior-tree app.

    `build_tree(cfg, robot, camera, args)` returns the root behavior (build it with chain()
    or sequence()); everything around it -- CLI, config, robot/camera lifecycle, exit code --
    is the shared runner's.  Extra keyword arguments (with_gripper, needs_camera) pass
    through to run_app."""
    def build_and_run(cfg, robot, camera, args):
        return run_tree(build_tree(cfg, robot, camera, args))

    def main():
        from ..apps._runner import run_app
        run_app(description, default_config, build_and_run, **run_app_kwargs)
    return main
