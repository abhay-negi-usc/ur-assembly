"""Progress/termination DETECTORS for guarded compliant motion.

All are shaped like ForceGuard -- check() -> bool, reset(), tripped_by -- so
AdmittanceController.ramp can terminate on any of them, and AnyGuard ORs a set while
remembering which member fired. Promoted from apps/bnc_assembly.py: these are conditions
any assembly behaviour can compose, not app internals.
"""

import time as _t

import numpy as np

from ..transforms import inverse


class ScrewAdvance:
    """Progress detector for a clocking screw, shaped like ForceGuard so that
    AdmittanceController.ramp can terminate on it (ramp returns 'seated' the cycle check() first
    returns True).

    That early termination is the point: once the cams have pulled the connector far enough there
    is nothing to gain by finishing the rotation.

    Advance is measured from the MEASURED arm pose, never the commanded reference. Under
    admittance the two differ by the compliant deflection, and here that deflection IS the signal
    -- the reference is deliberately a virtual target the connector cannot reach."""

    def __init__(self, robot, T_tool0_conn, T_base_conn_engaged, threshold_m):
        self.robot = robot
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)
        self._inv_engaged = inverse(np.asarray(T_base_conn_engaged, dtype=float))
        self.threshold_m = float(threshold_m)
        self.peak_m = 0.0                  # best advance seen across ALL tries (never reset)
        self.tripped_by = None

    def rebase(self, T_tool0_conn):
        """Adopt a new connector-in-gripper relationship, keeping the engaged reference frame.

        Needed after any REGRASP: the connector stays put in the socket while the gripper
        travels, so the two afterwards differ by exactly the progress made. Rebasing keeps advance
        measured from the ORIGINAL engaged pose instead of re-zeroing.

        NO CALLER IN THE CURRENT SWEEP -- the oscillating screw never lets go. Kept as the
        detector's contract for anything that does."""
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)

    def advance_m(self):
        """Connector translation along the ENGAGED connector frame's +X, in metres."""
        rel = self._inv_engaged @ (self.robot.tool0() @ self.T_tool0_conn)
        return float(rel[0, 3])

    def check(self):
        d = self.advance_m()
        self.peak_m = max(self.peak_m, d)
        if self.threshold_m > 0.0 and d >= self.threshold_m:
            self.tripped_by = (f'advance {d * 1000.0:.2f} mm >= '
                               f'{self.threshold_m * 1000.0:.2f} mm')
            return True
        return False

    def reset(self):
        self.tripped_by = None


class AxialForce:
    """Trips on the contact force ALONG THE CONNECTOR'S OWN +X -- the insertion reaction.

    ForceGuard watches |f|, which a lateral graze raises without opposing the push at all -- so a
    magnitude limit tight enough to catch real resistance also stops on every glancing touch.
    Projecting onto the insertion axis separates the two, which lets the ENGAGE limit be set low
    without becoming a hair-trigger.

    Shaped like ForceGuard (check/reset/tripped_by) so AnyGuard can OR it with the others.
    Persistence means the limit must hold CONTINUOUSLY for that long."""

    def __init__(self, robot, T_tool0_conn, max_force_n, persistence_s=0.0):
        self.robot = robot
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)
        self.max_force_n = float(max_force_n)
        self.persistence_s = float(persistence_s)
        self.peak_n = 0.0                  # best axial force seen; never reset, for the log
        self.tripped_by = None
        self._over_since = None

    def axial_n(self):
        """|force along the connector +X|, in newtons. Magnitude, not signed -- an inverted sign
        convention would turn this guard off rather than make it noisy."""
        T_base_tool0 = self.robot.tool0()
        T_base_conn = T_base_tool0 @ self.T_tool0_conn
        w = self.robot.arm.wrench_in(T_base_conn, T_base_tool0)
        return abs(float(w[0]))

    def check(self):
        f = self.axial_n()
        self.peak_n = max(self.peak_n, f)
        if self.max_force_n <= 0.0 or f < self.max_force_n:
            self._over_since = None
            return False
        now = _t.time()
        if self.persistence_s > 0.0:
            if self._over_since is None:
                self._over_since = now
                return False
            if now - self._over_since < self.persistence_s:
                return False
        self.tripped_by = (f'axial force {f:.1f} N >= {self.max_force_n:.1f} N'
                           + (f' for {now - self._over_since:.2f} s'
                              if self.persistence_s > 0.0 and self._over_since else ''))
        return True

    def reset(self):
        self.tripped_by = None
        self._over_since = None


class TravelReached:
    """Satisfied once the connector has ADVANCED far enough along its own +X since contact.

    THE POSITIVE SUCCESS SIGNAL. A force limit says something is resisting, which is also what a
    connector jammed on a rim does. Travel says the part actually WENT somewhere -- it is the one
    condition a jam cannot fake, so a mate that slides home under low force is recognised as the
    success it is instead of running until the force criterion eventually trips.

    Measured from the MEASURED arm pose, never the commanded reference: under admittance the two
    differ by the compliant deflection, and here it is the real motion that matters."""

    def __init__(self, robot, T_tool0_conn, T_base_conn_contact, threshold_mm):
        self.robot = robot
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)
        self._inv_contact = inverse(np.asarray(T_base_conn_contact, dtype=float))
        self.threshold_mm = float(threshold_mm)
        self.peak_mm = 0.0
        self.tripped_by = None

    def travel_mm(self):
        """Advance along the connector's own +X since contact, in mm. Signed, then clipped at 0 --
        backing OFF is not progress and must never be counted as any."""
        T_now = self.robot.tool0() @ self.T_tool0_conn
        return max(0.0, float((self._inv_contact @ T_now)[0, 3]) * 1000.0)

    def check(self):
        d = self.travel_mm()
        self.peak_mm = max(self.peak_mm, d)
        if self.threshold_mm <= 0.0 or d < self.threshold_mm:
            return False
        self.tripped_by = f'travel {d:.2f} mm >= {self.threshold_mm:.2f} mm since contact'
        return True

    def reset(self):
        self.tripped_by = None


class RadialConfirm:
    """Was there sustained force ACROSS the connector axis -- the signature of being IN a socket?

    THE AXIAL LIMIT ALONE CANNOT TELL A SEAT FROM A FACE. Pushing a connector flat against any
    surface -- the fixture body, the bench, the rim of the wrong hole -- develops axial reaction
    exactly like a real insertion does, and stops the engage in the same way. What a SOCKET adds is
    LATERAL constraint: once the barrel is inside, the walls resist sideways motion, so a connector
    being wiggled inside a socket pushes back across its own axis. A connector resting on a flat
    face does not; it just slides.

    So this measures |force| in the connector's OWN Y-Z plane -- the two axes perpendicular to the
    insertion direction -- and asks that it hold above a threshold CONTINUOUSLY. It is a
    satisfaction detector, not a canceller: nothing is stopped when it trips, it simply records
    that the condition was met."""

    def __init__(self, robot, T_tool0_conn, min_force_n, persistence_s=0.0):
        self.robot = robot
        self.T_tool0_conn = np.asarray(T_tool0_conn, dtype=float)
        self.min_force_n = float(min_force_n)
        self.persistence_s = float(persistence_s)
        self.peak_n = 0.0
        self.held_s = 0.0                  # longest continuous stretch above the threshold
        self.satisfied = False
        self._over_since = None

    def radial_n(self):
        """|force across the connector +X|, in newtons -- the Y-Z magnitude in the CONNECTOR
        frame, so it means the same thing whatever attitude the part is held at."""
        T_base_tool0 = self.robot.tool0()
        T_base_conn = T_base_tool0 @ self.T_tool0_conn
        w = self.robot.arm.wrench_in(T_base_conn, T_base_tool0)
        return float(np.linalg.norm(w[1:3]))

    def check(self):
        f = self.radial_n()
        self.peak_n = max(self.peak_n, f)
        if self.min_force_n <= 0.0:
            self.satisfied = True                      # no bar set -> nothing to fail
            return True
        now = _t.time()
        if f < self.min_force_n:
            self._over_since = None
            return False
        if self._over_since is None:
            self._over_since = now
        self.held_s = max(self.held_s, now - self._over_since)
        if self.held_s >= self.persistence_s:
            self.satisfied = True
        return self.satisfied

    def reset(self):
        self.satisfied = False
        self._over_since = None
        self.held_s = 0.0


class AnyGuard:
    """ORs several ForceGuard-shaped watchdogs onto one ramp, remembering WHICH one tripped.

    The screw needs a success detector and a force limit watching the same motion, and the two
    mean opposite things -- satisfied vs jammed -- so which fired has to survive the call.
    `ramp` only reports 'seated'."""

    def __init__(self, *guards):
        self.guards = [g for g in guards if g is not None]
        self.tripped = None
        self.tripped_by = None

    def check(self):
        for g in self.guards:
            if g.check():
                self.tripped = g
                self.tripped_by = getattr(g, 'tripped_by', None)
                return True
        return False

    def reset(self):
        self.tripped = None
        self.tripped_by = None
        for g in self.guards:
            g.reset()



