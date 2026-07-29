# cable_pick_estimate_assemble — pick, then estimate-while-assembling

`cable_pick_assemble` with the contact manifold in the loop. The **pick** is identical (scan →
grasp → check → recovery). The **assembly** no longer trusts the grasp: the robot knows the target
connector pose and holds an *estimate* of the connector-in-hand (fingertip → connector), which
carries in-hand error — and it corrects that estimate from touch, between attempts.

```
[pick] -> lift -> stand-off ->
    LOOP (max assembly.max_attempts):
        assemble   admittance-follow the trajectory (same law as uncertain_sampling),
                   collecting observations (believed connector-wrt-target pose + wrench
                   in the believed connector frame)
        check      believed connector pose vs target, within success_tolerance? -> done
        retract    straight back along the connector's own -X (peg-in-hole assumption)
        estimate   ICP of the observations against the CONTACT MANIFOLD
                   (urlab/skills/manifold.py -- the same algorithm validated offline by
                   analysis/manifold_icp_validation.py)
        update     T_fingertip_connector <- T_fingertip_connector @ T_corr
        realign    recompute the trajectory references from the new estimate
-> release -> retract steps -> end reset
```

```bash
python -m urlab.apps.cable_pick_estimate_assemble
```

## Why the check can be trusted
The check compares the **believed** connector pose to the target — but under admittance a wrong
belief cannot fake success. If the part jams short of the mate, the arm deflects off its reference,
the actual tool0 (and therefore the believed connector pose) lags the target, and the check fails.
The failed attempt's observations are exactly the data the estimator needs.

## Why the estimate works
An in-hand belief error is a **rigid right-multiplied offset** on every believed pose, while the
recorded **wrench is the true contact signature** (physics doesn't care what the robot believes).
The contact manifold pairs true poses with their wrenches, so aligning the observations back onto
it — multi-start ICP across the 12-D pose+wrench space, correction restricted to
`estimation.estimate_dims`, residual-gated RANSAC consensus — recovers the offset:
`believed @ T_corr ≈ true`, hence the estimate update `E ← E @ T_corr`.

## Prerequisites
1. A **contact manifold** for this connector type (`analysis/contact_manifold.py` over
   uncertain_sampling runs) at `estimation.manifold_csv`.
2. The **target connector pose** measured (`assembly.target_connector`, base_link, m/rad).
3. The **trajectory CSV** (connector w.r.t. target connector, last row identity, −X → 0).
4. The scaling constants under `estimation:` should be the ones that **validated offline** in
   `analysis/manifold_icp_validation.py` — that script is the rehearsal for this app.

## Config highlights (`configs/cable_pick_estimate_assemble.yaml`)
| key | meaning |
|---|---|
| `assembly.target_connector` | the known mate pose for the CONNECTOR (base_link, m/rad) |
| `assembly.max_attempts` | assemble→estimate loop budget |
| `assembly.success_tolerance.pos_mm / rot_deg` | believed-pose-vs-target check after each attempt |
| `assembly.retract_distance_m` | between-attempt escape along the connector's own −X |
| `assembly.log_decimation` | observation every Nth servo cycle |
| `estimation.manifold_csv` | the contact manifold for this connector |
| `estimation.estimate_dims` | which belief dims the grasp can be wrong in (only these are corrected) |
| `estimation.initial_connector_in_fingertip` | optional override of the initial in-hand estimate (default: grasp geometry) |
| `estimation.min_force_n` / `min_observations` | contact filter; too few contact samples ⇒ estimation is **skipped**, belief kept |
| `estimation.*` (scalings, ICP, RANSAC, `residual_gate`) | same meanings as `analysis/manifold_icp_validation.py` |

## Output
A run folder `data/cable_pick_estimate_assemble/<timestamp>/` with per-attempt observation CSVs
(manifold-compatible columns: `connector_target_*` in mm/deg + `wrench_connector_*`) and
`estimates.csv` — per attempt: the check errors, the estimated correction per dim, ICP inliers and
residual, and the success flag.

## Cautions
- **Linear (peg-in-hole) assembly assumed** — same as uncertain_sampling; no curved or twist mates.
- The correction is restricted to `estimate_dims`; a belief error outside those dims (or one whose
  exact inverse needs a coupled component, e.g. the x-term of a z+pitch offset) leaves a small
  residual by construction.
- Observations come from the **latest attempt only** — earlier attempts were collected under a
  different belief, so they cannot be pooled naively.
