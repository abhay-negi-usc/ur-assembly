"""Cable pick + estimate-while-assemble + SOLUTION CHECK -- the trusted-estimate variant.

Identical to cable_pick_estimate_assemble (same phases, same trajectory execution, same
run figure) except that the estimator is wrapped in the runtime trust stack from the
2026-08 offline uncertainty campaign (urlab/skills/solution_check.py):

  * after every estimate, ~128 candidate corrections are re-scored against the attempt's
    own observations; the resulting signals (posterior width, ensemble spread, temporal-
    split disagreement, residual, depth, optional config-disagreement) are combined into
    BOTH trust scores -- rankavg2 (calibration-ECDF mean) and the Cauchy combination --
    every attempt, and logged side by side;
  * `estimation.check.method` selects which score gates the decision; a flagged estimate
    is SKIPPED (belief kept, attempt retried) under on_flag: skip_correction, or applied
    with a warning under on_flag: warn;
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
