"""ForceGuard -- the contact watchdog.

In the ROS stack the force guard was three different things in three files: a `_wrench_cb` that
cached magnitudes, a `_contact_exceeded()` that compared them, and an `_abort_move()` override
that the base class polled between executor spins. Two of the three demos armed it only during
the phases that expected contact, which is precisely backwards -- the phases that DON'T expect
contact are the ones where a collision is a surprise worth stopping for.

Here it is one object, armed once, and `arm.add_guard()` polls it inside every move.

PERSISTENCE (optional, `persistence_s`): the limit must be exceeded CONTINUOUSLY for this long
before the guard trips. A raw F/T stream spikes over the limit for a few samples on every hard
contact transient; with persistence 0 (the default, the original behaviour) one such sample
ends the attempt, while e.g. 0.5 s requires a SUSTAINED press -- the difference between "the
rim was clipped for 30 ms" and "the part is genuinely jammed". The clock resets the moment the
wrench drops back under the limit.
"""

import time

import numpy as np

from .. import log as urlog

log = urlog.get('guard')


class ForceGuard:
    """Trips when the tared contact wrench exceeds a limit. Callable, so it plugs straight into
    `arm.add_guard(guard)`."""

    def __init__(self, arm, cfg_section):
        c = cfg_section or {}
        self.arm = arm
        self.max_force = float(c.get('max_force_n', 0.0))
        self.max_torque = float(c.get('max_torque_nm', 0.0))
        self.persistence_s = float(c.get('persistence_s', 0.0))
        self.enabled = bool(c.get('enabled', True))
        self.tripped_by = None
        self._over_since = None

        if self.enabled and self.max_force <= 0.0 and self.max_torque <= 0.0:
            log.warning('Force guard is enabled but both limits are 0 -- it will never trip.')
        if self.persistence_s > 0.0:
            log.info('Force guard persistence: the limit must hold for %.2f s to trip.',
                     self.persistence_s)

    def __call__(self):
        if not self.enabled:
            return False
        w = self.arm.wrench()
        force = float(np.linalg.norm(w[:3]))
        torque = float(np.linalg.norm(w[3:]))

        over = None
        if self.max_force > 0.0 and force >= self.max_force:
            over = f'force {force:.1f} N >= {self.max_force:.1f} N'
        elif self.max_torque > 0.0 and torque >= self.max_torque:
            over = f'torque {torque:.2f} Nm >= {self.max_torque:.2f} Nm'
        if over is None:
            self._over_since = None                # dropped under the limit: clock resets
            return False
        if self.persistence_s > 0.0:
            now = time.time()
            if self._over_since is None:
                self._over_since = now
            if now - self._over_since < self.persistence_s:
                return False                       # transient so far -- keep watching
            over += f' for {now - self._over_since:.2f} s'
        self.tripped_by = over
        return True

    def reset(self):
        self.tripped_by = None
        self._over_since = None

    def check(self):
        """Non-latching read -- for loops that want to test contact without arming a move."""
        return self()

    def disable(self):
        """Turn the guard off. Needed for extraction: a jammed part is ALREADY over the limit, so
        a guard checked before each step blocks the very motion that would free it. The ROS
        sampling node had exactly this bug -- disassembly reported success while the arm never
        moved, because every ramp returned at step 1."""
        self.enabled = False

    def enable(self):
        self.enabled = True
        self.tripped_by = None
        self._over_since = None
