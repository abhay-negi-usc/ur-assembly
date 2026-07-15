"""AdmittanceController -- the admittance control law, in software, over servoL.

Reimplements the ros2_control admittance_controller the dev branch used, so there is a real,
finite RESTORING STIFFNESS again (forceMode has none -- it is pure force control and yields
freely). No controller to install, load, switch, or parameterise: this is a plain servo loop.

THE LAW, per axis, in the BASE frame:

    F_ext = M x'' + D x' + S (x - x_d)      =>     x'' = M^-1 ( F_ext - D x' - S (x - x_d) )

integrated each cycle (v += x'' dt ; x += v dt). We track the DISPLACEMENT delta = x - x_d from
the reference rather than x itself: contact force grows delta, the spring S pulls it back to zero,
so when contact eases the arm RETURNS to the reference (the ideal insertion trajectory). A static
force F holds a steady deflection delta = F / S -- with S = 2000 N/m, 20 N deflects 10 mm.

    D = damping_ratio * 2 sqrt(M S)         (damping_ratio = 1 -> critically damped)

FRAME. The law runs in the TOOL0 frame, matching the dev controller. getActualTCPForce() reports
the wrench at the TCP in BASE axes, so each cycle it is rotated into tool0 (by R^T, R = the tool0
orientation in base -- a pure rotation, no lever arm, since the wrench is already referenced to the
TCP = tool0 origin). delta is then a TOOL0-frame displacement -- delta[:3] a translation and
delta[3:] a rotation vector about the TOOL axes -- and it is applied to the reference by
POST-multiplying: T_cmd = T_ref @ Delta. So "z compliant" means yielding along the tool's own z
wherever the tool points, which is what an insertion wants.

SAFETY. delta is clamped (max_delta_m / max_delta_rad): a bad tare or an F/T fault would otherwise
integrate into a runaway. The caller still arms a force guard on top for the hard limit.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .. import log as urlog

log = urlog.get('admittance')


class AdmittanceController:
    """Software admittance over servoL. Holds the integrator state so a chunked insertion is one
    continuous, compliant motion."""

    def __init__(self, arm, cfg_section):
        c = cfg_section or {}
        self.arm = arm
        self.M = np.asarray(c.get('mass', [5.0, 5.0, 5.0, 0.5, 0.5, 0.5]), dtype=float)
        self.S = np.asarray(c.get('stiffness', [2000.0] * 3 + [15.0] * 3), dtype=float)
        zeta = np.asarray(c.get('damping_ratio', [1.0] * 6), dtype=float)
        self.D = zeta * 2.0 * np.sqrt(self.M * self.S)          # critically damped at zeta = 1
        self.selected = np.asarray([1.0 if v else 0.0
                                    for v in c.get('selected_axes', [1] * 6)], dtype=float)
        self.rate = float(c.get('reference_rate_hz', 125.0))
        self.lookahead = float(c.get('lookahead_time_s', 0.1))
        self.gain = float(c.get('servo_gain', 300.0))
        self.max_delta = float(c.get('max_delta_m', 0.05))      # runaway clamp (translation)
        self.max_delta_rot = float(c.get('max_delta_rad', 0.5))
        self._delta = np.zeros(6)
        self._vel = np.zeros(6)

    def reset(self):
        """Zero the integrator -- call before an insertion so it starts on the reference."""
        self._delta = np.zeros(6)
        self._vel = np.zeros(6)

    def _command_pose(self, T_ref):
        """The reference tool0 pose displaced by the compliant delta IN THE TOOL0 FRAME
        (post-multiply): T_cmd = T_ref @ Delta."""
        Delta = np.eye(4)
        Delta[:3, :3] = Rotation.from_rotvec(self._delta[3:]).as_matrix()
        Delta[:3, 3] = self._delta[:3]
        return T_ref @ Delta

    def _step(self, T_ref, dt):
        """Advance the ODE one cycle (in the tool0 frame) and command the compliant pose."""
        R = T_ref[:3, :3]                                   # tool0 orientation in base
        wb = self.arm.wrench()                              # at the TCP, base axes (tared)
        w = np.concatenate([R.T @ wb[:3], R.T @ wb[3:]]) * self.selected   # -> tool0 axes
        accel = (w - self.D * self._vel - self.S * self._delta) / self.M
        self._vel += accel * dt
        self._delta += self._vel * dt
        self._delta[:3] = np.clip(self._delta[:3], -self.max_delta, self.max_delta)
        self._delta[3:] = np.clip(self._delta[3:], -self.max_delta_rot, self.max_delta_rot)
        self.arm.servo_l(self._command_pose(T_ref), dt, self.lookahead, self.gain)

    def ramp(self, T_ref_start, T_ref_end, duration, guard=None):
        """Ramp the tool0 reference start -> end over `duration` s under admittance. Returns
        'seated' if the guard trips (contact), else 'done'. Integrator state persists across calls,
        so consecutive ramps form one continuous compliant motion."""
        from ..transforms import slerp_matrix
        dt = 1.0 / self.rate
        steps = max(1, int(duration * self.rate))
        for k in range(1, steps + 1):
            self._step(slerp_matrix(T_ref_start, T_ref_end, k / steps), dt)
            if guard is not None and guard.check():
                return 'seated'
        return 'done'

    def hold(self, T_ref, seconds, guard=None):
        """Hold the reference for `seconds` under admittance (let the mate settle / keep yielding)."""
        return self.ramp(T_ref, T_ref, seconds, guard)

    def stop(self):
        self.arm.servo_stop()
