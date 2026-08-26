"""The engage trace exists to answer one question: WHAT MOVED -- reference, Delta, or the arm.

Its value is entirely in the columns lining up with what the report claims they are. An off-by-one
here would not crash; it would confidently blame the wrong subsystem, which is worse than no trace.
"""

import numpy as np

from urlab.apps.bnc_assembly import ENGAGE_TRACE_COLS, _report_engage_trace
from urlab.robot.admittance import AdmittanceController


def _row(t, stage, ref, cmd, meas, dx, flange_x, fx):
    return [t, stage, ref, cmd, meas, dx, 0.0, 0.0, 0.0, flange_x, 0.0, 0.0, fx, 0.0, 0.0]


def test_the_header_matches_the_row_the_engage_builds():
    assert len(ENGAGE_TRACE_COLS) == len(_row(0.0, 'engage', 0, 0, 0, 0, 0, 0))
    # The report reads these by INDEX; pin the names to those positions.
    for i, name in ((2, 'ref_x_mm'), (3, 'cmd_x_mm'), (4, 'meas_x_mm'),
                    (5, 'delta_x_mm'), (9, 'flange_x_mm')):
        assert ENGAGE_TRACE_COLS[i] == name, f'column {i} moved: the report would mislabel it'


def test_the_report_survives_each_fault_signature(caplog):
    """Three shapes, one per subsystem. Exercised mostly to prove the report cannot raise on them
    -- it runs at the end of a real insertion, where a traceback would cost the run."""
    caplog.set_level('INFO')
    controller_yields = [_row(i * 0.1, 'engage', 10.0, 10.0 - i, 10.0 - i, -float(i), 100.0 - i, 5.0)
                         for i in range(6)]
    trajectory_retreats = [_row(i * 0.1, 'engage', 10.0 - i, 10.0 - i, 10.0 - i, 0.0, 100.0 - i, 0.0)
                           for i in range(6)]
    arm_lags = [_row(i * 0.1, 'engage', 10.0, 10.0, 10.0 - i, 0.0, 100.0 - i, 0.0)
                for i in range(6)]
    for trace in (controller_yields, trajectory_retreats, arm_lags, []):
        _report_engage_trace(trace)
    assert 'BACKED OUT' in caplog.text


def test_the_controller_reports_the_reference_and_command_it_used():
    """The trace reads ref/delta/cmd off the controller, so they must actually be recorded."""
    class Arm:
        dry_run = False

        def wrench(self):
            return np.array([-5.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        def servo_l(self, T, dt, lookahead, gain):
            pass

        def servo_stop(self):
            pass

    adm = AdmittanceController(Arm(), {'stiffness': [500.0] * 3 + [10.0] * 3,
                                       'mass': [1.0] * 3 + [0.1] * 3,
                                       'reference_rate_hz': 125.0})
    ref = np.eye(4)
    ref[0, 3] = 0.100
    assert adm.last_ref is None and adm.last_cmd is None
    adm.reset()
    adm.hold(ref, 2.0)
    assert np.allclose(adm.last_ref, ref), 'the reference must be recorded as given'
    # Delta is the compliant give; the command is the reference displaced by it.
    assert adm.delta[0] < 0, 'a -x contact force must push the command back out'
    assert adm.last_cmd[0, 3] == float(np.round(ref[0, 3] + adm.delta[0], 12)) or \
        abs(adm.last_cmd[0, 3] - (ref[0, 3] + adm.delta[0])) < 1e-9
