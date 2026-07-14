"""Logging + the interactive step gate -- replaces get_logger() and the _confirm/_do pattern.

`step()` is the one worth reading. Every ROS demo wrapped each phase in

    if not self._do('close gripper', lambda: self.gripper_to(...)):
        return False

which meant every sequence was a chain of `and`s that silently stopped at the first False and
left you to work out which link broke. Here a sequence is a list of steps and the runner reports
exactly where it stopped and why.
"""

import logging
import sys
import time

_FORMAT = '[%(asctime)s] %(levelname)-7s %(name)s: %(message)s'


def setup(level=logging.INFO):
    logging.basicConfig(level=level, format=_FORMAT, datefmt='%H:%M:%S', stream=sys.stdout)


def get(name):
    return logging.getLogger(name)


class Aborted(Exception):
    """The user declined a confirmation prompt, or a guard tripped. Carries the step's name so
    the runner can say WHICH step stopped the sequence instead of just returning False."""


class StepRunner:
    """Runs named steps in order, with optional per-step confirmation.

    A step returning False (or raising Aborted) stops the sequence. Anything else -- including
    None, so a step that just does its job and returns nothing counts as success -- continues."""

    def __init__(self, log, confirm=True):
        self.log = log
        self.confirm = confirm
        self.completed = []

    def ask(self, label):
        """Blocking confirmation. Returns False if the user aborts."""
        if not self.confirm:
            return True
        try:
            answer = input(f'\n[{label}] Enter to proceed (q to abort): ')
        except EOFError:                     # piped stdin: treat as an unattended run
            return True
        return answer.strip().lower() not in ('q', 'quit', 'n', 'no')

    def step(self, label, fn, *args, **kwargs):
        """Confirm, run, report. Raises Aborted on refusal or failure."""
        if not self.ask(label):
            raise Aborted(f'{label} (declined by the user)')
        self.log.info('--> %s', label)
        t0 = time.monotonic()
        result = fn(*args, **kwargs)
        if result is False:
            raise Aborted(label)
        self.completed.append(label)
        self.log.debug('    %s done in %.1fs', label, time.monotonic() - t0)
        return result

    def run(self, steps):
        """Run a list of (label, callable). Returns True if all completed.

        The point of collecting them into a list first: on failure we can say `stopped at step 4
        of 9 (close gripper); completed: open gripper, scan, approach` -- which the chained-`and`
        version could never do."""
        for i, (label, fn) in enumerate(steps, start=1):
            try:
                self.step(label, fn)
            except Aborted as exc:
                self.log.error('STOPPED at step %d/%d: %s', i, len(steps), exc)
                if self.completed:
                    self.log.error('Completed: %s', ', '.join(self.completed))
                return False
        self.log.info('All %d steps complete.', len(steps))
        return True
