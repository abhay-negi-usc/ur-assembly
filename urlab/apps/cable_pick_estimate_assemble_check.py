"""Cable pick + estimate-while-assemble + SOLUTION CHECK -- the trusted-estimate variant.

Identical to cable_pick_estimate_assemble (same phases, same trajectory execution, same
run figure) except that the estimator is wrapped in the runtime trust stack from the
2026-08 offline uncertainty campaign (urlab/skills/solution_check.py):

  * after every estimate, ~128 candidate corrections are re-scored against the attempt's
    own observations; the resulting signals (posterior width, ensemble spread, temporal-
    split disagreement, residual, depth, optional config-disagreement) are combined into
    BOTH trust scores -- rankavg2 (calibration-ECDF mean) and the Cauchy combination --
    every attempt, and logged side by side with the applied correction;
  * the check is PURELY OBSERVATIONAL: every correction is applied and every re-attempt
    runs exactly as in the base app, and the OPERATOR remains the decision maker at the
    y/Enter/q prompt -- `estimation.check.method` only chooses which score carries the
    logged LOW/ok trust label;
  * the intended estimation config is the campaign winner: the AUGMENTED manifold
    (configs/data/banana_manifold_augmented.csv -- clean map + 30k truth-corrected
    production-process rows) with the rawcap wrench representation (force direction x
    magnitude/10 N saturated at 30 N, torque dropped).

Run:  python -m urlab.apps.cable_pick_estimate_assemble_check --config configs/cable_pick_estimate_assemble_check.yaml

The wrapper swaps the estimator CLASS seen by the base app for the checked subclass; all
motion/grasp/plotting behaviour is the base app's, byte for byte.
"""

from ..skills.solution_check import CheckedManifoldEstimator
from . import cable_pick_estimate_assemble as _base
from ._runner import run_app


def build_and_run(cfg, robot, camera, args):
    """The base app with its ManifoldEstimator swapped for CheckedManifoldEstimator."""
    prev = _base.ManifoldEstimator
    _base.ManifoldEstimator = CheckedManifoldEstimator
    try:
        return _base.build_and_run(cfg, robot, camera, args)
    finally:
        _base.ManifoldEstimator = prev


def main():
    run_app('Cable pick + estimate-while-assemble + solution check',
            'cable_pick_estimate_assemble_check', build_and_run, needs_camera=True)


if __name__ == '__main__':
    main()
