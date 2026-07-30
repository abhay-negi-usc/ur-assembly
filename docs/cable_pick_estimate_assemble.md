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
        check      the believed-vs-target numbers are logged; the OPERATOR decides success
                   at a prompt (y = done / Enter = retry / q = abort); --dry-run falls back
                   to success_tolerance
        retract    straight back along the connector's own -X (peg-in-hole assumption)
        estimate   ICP of the observations against the CONTACT MANIFOLD
                   (urlab/skills/manifold.py -- the same algorithm validated offline by
                   analysis/manifold_icp_validation.py)
        update     T_fingertip_connector <- T_fingertip_connector @ T_corr
        realign    recompute the trajectory references from the new estimate
-> release -> retract along the connector's own -X -> end reset
```

```bash
python -m urlab.apps.cable_pick_estimate_assemble
```

## The check
After each attempt the believed-vs-target error is computed and logged, but **success is the
operator's call** at the prompt (y / Enter = retry / q) — the operator can see the physical mate,
the numbers only see the belief; `success_tolerance` is the printed reference and the `--dry-run`
fallback. There is also an **unconditional pause at the stand-off** before the first contact,
regardless of `confirm_each_step`.

The kinematic numbers are still trustworthy context: the check compares the **believed** connector pose to the target — but under admittance a wrong
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
| `assembly.success_tolerance.pos_mm / rot_deg` | reference numbers at the check prompt; automatic decision only in `--dry-run` |
| `assembly.release_retract_distance_m` | post-release escape along the **connector's own −X** (never a base-frame axis) |
| `assembly.retract_distance_m` | between-attempt escape along the connector's own −X |
| `assembly.log_decimation` | observation every Nth servo cycle |
| `estimation.manifold_csv` | the contact manifold for this connector |
| `estimation.estimate_dims` | which belief dims the grasp can be wrong in (only these are corrected) |
| `estimation.initial_connector_in_fingertip` | optional override of the initial in-hand estimate (default: grasp geometry) |
| `estimation.min_force_n` / `min_observations` | contact filter; too few contact samples ⇒ estimation is **skipped**, belief kept |
| `estimation.*` (scalings, ICP, RANSAC, `residual_gate`) | same meanings as `analysis/manifold_icp_validation.py` |

## Output
An experiment folder `data/experiments/cable_pick_estimate_assemble_<timestamp>/` with, per attempt:
- `attempt_NN_observations.csv` — manifold-compatible columns (`connector_target_*` mm/deg +
  `wrench_connector_*`),
- `attempt_NN_estimate.png` — the ICP convergence figure (same layout as
  `analysis/manifold_icp_validation`: per-dim correction, guesses faint, RANSAC consensus bold,
  dashed zero, log-scale residual). Plotting is best-effort — a missing seaborn is logged, never fatal,
- `estimates.csv` — check errors, per-dim corrections, ICP inliers/residual, success flag.

## Speeds — no time-based motion
Every motion in this app is paced by an explicit config limit, never a fixed duration, and there
are exactly **two limit sets of the same four keys**:

| key | bounds |
|---|---|
| `max_joint_velocity_deg_s` | every joint, directly (moveJ speed) |
| `max_joint_acceleration_deg_s2` | the moveJ accel/decel ramp |
| `max_cartesian_translation_mm_s` | the tool: moveL speed, compliant-reference ramp pacing, and an equivalent joint bound inside moveJ |
| `max_cartesian_rotation_deg_s` | the tool's rotation: compliant-reference pacing + the moveJ equivalent bound |

There is **one global `speed:` block**, and every phase of the app applies a **scale factor** to
all four limits via `speed.phase_scale` (1.0 = the global limit itself; an absent phase = 1.0):

| phase | scales |
|---|---|
| `reset` | start/end home moves, and the between-retry homes |
| `scan` | the multi-view scan + grasp-align + reseat moves |
| `pickup` | the compliant grasp descent |
| `lift` | the compliant lift, including the slip-check partial lift |
| `standoff` | the move to stand-off + the per-attempt realign `moveJ`s |
| `assemble` | the insertion reference ramp |
| `retract` | the between-attempt retract + the release escape |

Free-space moves inherit the current phase's scale through `arm.set_speed_scale` (the app marks
each boundary); compliant ramps are paced directly from the scaled limits. All four limits are
enforced simultaneously — whichever binds. (Legacy keys `max_joint_velocity_rad_s`,
`joint_acceleration_rad_s2`, `max_cartesian_velocity_m_s` still parse for the older configs, and
`move_j`/`move_l` accept `caps=` as either a four-key mapping or a bare scale factor.)

## Grasp robustness
Two failure modes are handled around the pick:
- **Slip during the lift** (`grasp_check.lift_check`): the lift first raises `height_m`, then
  **re-closes** the gripper and re-runs the counts check — the fingers hold their stalled position
  when a part vanishes, so only a re-close can reveal the loss. Still held → finish the lift;
  slipped → open and retry the whole scan→grasp.
- **Deterministic grasp failure** (`grasp_check.retry_perturb_x_m`): a scan→grasp→fail loop is a
  fixed point (the fresh scan reproduces the same junction estimate). Full retries perturb the
  grasp along the junction x by 0, +d, −d, +2d, … to break it. `0` disables.

## Pickup height (`pickup.height_from_model`)
With the connector **on the ground plane**, the grasp target height is derived from physics
instead of a hand-tuned z-trim: plane + `connector_diameter_mm` max/2 (the centerline) + the
fingertip **advance** between the separation `fingertip_grasp` was calibrated at
(`fingertip_ref_separation_mm`: 0 = gripper closed, 83.56 = full open) and the expected stall
separation — the physical pad travels ~12.8 mm along the approach axis over the stroke, so the
static tool0→fingertip transform is exact at one separation only. All separations are
zero-compression values; real grasps squeeze the pads (desired — grip pressure), which the
calibrated groove depth absorbs on average (`pad_compression_mm` exists for completeness but the
advance is insensitive to it near closure).

## Cautions
- **Linear (peg-in-hole) assembly assumed** — same as uncertain_sampling; no curved or twist mates.
- The correction is restricted to `estimate_dims`; a belief error outside those dims (or one whose
  exact inverse needs a coupled component, e.g. the x-term of a z+pitch offset) leaves a small
  residual by construction.
- Observations come from the **latest attempt only** — earlier attempts were collected under a
  different belief, so they cannot be pooled naively.
