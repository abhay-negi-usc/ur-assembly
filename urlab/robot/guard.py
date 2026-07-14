"""ForceGuard -- the contact watchdog.

In the ROS stack the force guard was three different things in three files: a `_wrench_cb` that
cached magnitudes, a `_contact_exceeded()` that compared them, and an `_abort_move()` override
that the base class polled between executor spins. Two of the three demos armed it only during
the phases that expected contact, which is precisely backwards -- the phases that DON'T expect
contact are the ones where a collision is a surprise worth stopping for.

Here it is one object, armed once, and `arm.add_guard()` polls it inside every move.
"""

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
        self.enabled = bool(c.get('enabled', True))
        self.tripped_by = None

        if self.enabled and self.max_force <= 0.0 and self.max_torque <= 0.0:
            log.warning('Force guard is enabled but both limits are 0 -- it will never trip.')

    def __call__(self):
        if not self.enabled:
            return False
        w = self.arm.wrench()
        force = float(np.linalg.norm(w[:3]))
        torque = float(np.linalg.norm(w[3:]))

        if self.max_force > 0.0 and force >= self.max_force:
            self.tripped_by = f'force {force:.1f} N >= {self.max_force:.1f} N'
            return True
        if self.max_torque > 0.0 and torque >= self.max_torque:
            self.tripped_by = f'torque {torque:.2f} Nm >= {self.max_torque:.2f} Nm'
            return True
        return False

    def reset(self):
        self.tripped_by = None

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
