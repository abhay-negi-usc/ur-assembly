# estimator_eval — the manifold estimator scored against a known ground truth

`cable_pick_estimate_assemble` tests the estimator end-to-end, but after a real pick the true
in-hand pose is unknown, so the estimate can never be scored. This app isolates the estimator:
the connector is **rigidly fixtured between the closed gripper fingers**, so the true tool0 →
connector pose **is** the shared catalogue's `held_frame:` entry (`configs/frames.yaml`, the same
frame + recorded mate `uncertain_sampling` uses). Each trial then injects a **known** belief
error and lets the estimator try to remove it — the remaining ground-truth error is measured
exactly after every update.

```
for each of eval.num_trials trials:
    perturb    the BELIEF only: T_believed = T_true @ delta   (delta ~ eval.perturbation)
    LOOP (max eval.max_attempts):
        assemble   admittance-follow the trajectory planned from the BELIEF, collecting
                   observations (believed connector-wrt-target + wrench in the believed
                   connector frame) — same law/logging as cable_pick_estimate_assemble
        retract    straight back along the believed connector's own -X (peg-in-hole)
        estimate   manifold ICP -> T_corr;  T_believed <- T_believed @ T_corr
        score      ground-truth error = inverse(T_true) @ T_believed  ->  trials.csv;
                   within eval.success_tolerance -> converged (early stop)
    final      OPTIONAL (eval.final_insertion): one more guarded insertion from the FINAL
               corrected belief under a different stiffness — seats-or-not, no estimation
    disassemble: return to the stand-off (free space) before the next trial
```

```bash
python -m urlab.apps.estimator_eval
```

**The gripper is never opened or closed** — the app runs `with_gripper=False`, so the gripper
object is never even constructed. Fixture the connector between the closed fingers before
starting (close the gripper from the pendant or a separate script).

## Why perturbing the belief is the same experiment as a bad grasp
The robot plans every reference from `T_believed`, so the *true* part physically rides the
trajectory offset by exactly the injected error — identical contact physics to a mis-grasped
part — while the observations are logged against the believed frame, exactly as the pick app
logs them. The only difference from production is that here `inverse(T_true) @ T_believed` is
computable, so every correction can be scored instead of eyeballed.

## Prerequisites
1. The connector **fixtured** in the closed fingers such that tool0 → connector matches the
   catalogue frame (`held_frame:` in `configs/frames.yaml` `frames:`).
2. That frame's recorded mate in `frames.yaml` `targets:` (hand-guide to a good mate, paste
   `base_link <- <frame>` off the monitor).
3. A **contact manifold** for this connector at `estimation.manifold_csv`.
4. `estimation:` values **identical to the ones `cable_pick_estimate_assemble` runs with** —
   this app exists to score exactly that configuration.

## Config highlights (`configs/estimator_eval.yaml`)
| key | meaning |
|---|---|
| `held_frame` | catalogue frame = the ground-truth in-hand pose AND (via `targets:`) the mate |
| `eval.num_trials` / `eval.max_attempts` | N trials (default 50) × M estimate updates each (default 5) |
| `eval.mode` | `random` draws inside the bounds; `grid` sweeps them (count derived) |
| `eval.perturbation.lower/upper` | injected belief-error bounds, `[x,y,z (m), r,p,y (deg)]` in the part's own frame — default x/z ±5 mm, pitch ±5 deg |
| `eval.success_tolerance` | convergence gate on the **ground-truth** error (not the believed check) |
| `eval.stop_when_converged` | skip a trial's remaining attempts once converged |
| `eval.save_observations` / `save_plots` | per-attempt raw CSVs / the per-trial error figure (both default on) |
| `eval.live_plot` | mirror the current trial's figure to ONE fixed path outside the experiment folder (atomic overwrite — keep it open in an image viewer); `true` = `data/experiments/estimator_eval_live.png`, a string = explicit path |
| `eval.accumulate_observations` | (default **on**) each estimate uses ALL of the trial's observations so far, prior attempts re-projected into the current belief (`rel_new = rel_old @ T_corr`, wrench re-based likewise); recency weighting decays the older attempts. `false` = current attempt only |
| `eval.trajectory_noise` | optional smoothed per-waypoint Gaussian noise in the connector's own frame, redrawn per attempt; its own random stream, so the injected-error draws are unchanged |
| `eval.final_insertion` | optional extra guarded assemble per trial from the final corrected belief with a `stiffness` override — seat-check only, logged as `attempt=final_insertion`, excluded from `summary.csv` |
| `estimation.*` | the estimator under test — same schema as `cable_pick_estimate_assemble` |
| `estimation.aggregator` | `ransac` (residual-gated consensus vote, default) or `softmax` (residual-softmax weighted mean over all starts, `softmax_temp` relative to the best residual — the offline ablation's winner) |
| `compliance` / `force_guard` / `speed` | mirror the pick app's assembly values so observations are production-like |
| `retract_distance_m` | per-attempt escape along the believed connector's own −X; must exceed the insertion depth |

## Output — `data/experiments/estimator_eval_<timestamp>/`
- `trials.csv` — one row per attempt: injected error, ground-truth error before/after the update
  (per-dim mm/deg + norms), the applied correction, ICP inliers/residual, `n_observations`,
  `seated`, `converged`. Written incrementally (flushed per row), so an aborted run keeps its data.
- `trial_TTT_attempt_AA_observations.csv` — manifold-compatible raw observations per attempt.
- `trial_TTT_errors.png` — ONE figure per trial, **re-saved after every attempt** so it can be
  watched live. Left column (shared attempt axis): one panel per estimated dim (signed,
  symmetric ylim about the zero line), then the combined L2 error in the estimator's
  mm-equivalent metric (rotation × `scaling_constant_deg_to_mm`). Right column: the ICP
  residuals per attempt on a log y axis — **every guess's final residual** as a faint column
  (the population the aggregator votes over, so consensus spread and outlier guesses are
  visible) with the aggregated residual bold on top — then a scatter of residual vs the L2 error left
  **after** that attempt's update (points labelled by attempt) — the residual is only
  trustworthy if that scatter trends up-right.
- `estimator_eval_live.png` (outside the experiment folder, with `eval.live_plot`) — the current
  trial's figure, atomically overwritten after every attempt.
- `trial_TTT_final_insertion_observations.csv` (with `eval.final_insertion`) — the seat-check
  insertion's observations; its `trials.csv` row has `attempt=final_insertion` and only the
  before-error / `seated` / kinematic-check fields.
- `summary.csv` — per-attempt-index aggregate: mean/median |error| per estimated dim +
  convergence fraction, plus a `final` row (each trial's last attempt) — the headline numbers.
- `eval_config.json` — the eval/estimation/compliance sections as run.

Post-analysis pairs naturally with `analysis/manifold_interp_ablation.py`-style plots: `trials.csv`
has everything needed for error-vs-attempt convergence curves and per-dim error distributions.

## Cautions
- Linear (peg-in-hole) assembly assumed, like uncertain_sampling / the pick app.
- The injected error and the estimator's correction live in the **same right-multiplied
  connector frame**, so per-dim columns compare directly (`inj_x_mm` ↔ `corr_x_mm` ↔
  `err_after_x_mm`).
- A belief error outside `estimate_dims` (y/roll/yaw injections) cannot be corrected by
  construction — keep those bounds at 0 unless that residual floor is itself the question.
- The catalogue frame must match the *physical* fixture pose; any fixturing error becomes a
  systematic bias in every trial's score.
