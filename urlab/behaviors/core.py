"""py_trees glue for the app scripts.

The apps are sequential robot procedures, so their leaves are BLOCKING actions: each leaf does
its whole job inside update() and returns SUCCESS or FAILURE, never RUNNING.  A tree of such
leaves executes in order under a Sequence, retries under a Retry decorator, and falls back
under a Selector -- which is exactly the control flow the scripts used to hand-roll with
`if not step(): return False` chains, now stated as structure instead.

run_tree() ticks the tree to completion and reports the outcome; the tree layout itself is
logged once at DEBUG so a new user can see the whole procedure before the arm moves.
"""

import py_trees
from py_trees.common import Status

from .. import log as urlog

log = urlog.get('behavior')


class Action(py_trees.behaviour.Behaviour):
    """A named blocking action as a leaf.

    `fn` runs once per traversal; returning False means FAILURE, anything else (including
    None) means SUCCESS.  An optional `confirm` callable is asked first -- the interactive
    step gate -- and a refusal is a FAILURE without running the action."""

    def __init__(self, name, fn, confirm=None):
        super().__init__(name=name)
        self._fn = fn
        self._confirm = confirm

    def update(self):
        if self._confirm is not None and not self._confirm(self.name):
            log.warning('step %r declined by the user.', self.name)
            return Status.FAILURE
        log.info('--> %s', self.name)
        return Status.FAILURE if self._fn() is False else Status.SUCCESS


class Check(py_trees.behaviour.Behaviour):
    """A named predicate as a leaf: SUCCESS iff `fn()` is truthy. Use for guards/conditions
    that must not log as steps."""

    def __init__(self, name, fn):
        super().__init__(name=name)
        self._fn = fn

    def update(self):
        return Status.SUCCESS if self._fn() else Status.FAILURE


def sequence(name, *children):
    """A memory Sequence: children run in order, resuming (not restarting) across ticks."""
    node = py_trees.composites.Sequence(name=name, memory=True)
    node.add_children(list(children))
    return node


def selector(name, *children):
    """A memory Selector: the first child to SUCCEED wins; a child's FAILURE falls through."""
    node = py_trees.composites.Selector(name=name, memory=True)
    node.add_children(list(children))
    return node


def retry(child, attempts, name=None):
    """Retry `child` up to `attempts` times before reporting FAILURE."""
    return py_trees.decorators.Retry(name=name or f'retry x{attempts}',
                                     child=child, num_failures=attempts)


def run_tree(root, logger=None):
    """Tick a tree of blocking actions to completion. Returns True on SUCCESS.

    A single tick normally completes the whole tree (no leaf returns RUNNING); decorators
    such as Retry surface RUNNING between attempts, so tick until the root settles."""
    (logger or log).debug('behavior tree:\n%s', py_trees.display.unicode_tree(root))
    tree = py_trees.trees.BehaviourTree(root)
    tree.setup()
    while True:
        tree.tick()
        if root.status != Status.RUNNING:
            return root.status == Status.SUCCESS
