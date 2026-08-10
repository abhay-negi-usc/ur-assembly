# uncertain_sampling — perturbed-insertion data collection (held connector)

A **data-collection** run: repeatedly drive a PERTURBED **connector** into the mate under
compliance, logging each sample, then retract and repeat. Produces a CSV for studying how a
misaligned connector behaves during insertion.

```
for each trial: perturb (connector frame) -> move to the perturbed start (stiff, free space) ->
follow the path under ADMITTANCE (logging at the servo rate, guarded) -> settle ->
retract straight back along the connector's -X -> CSV
```

```bash
python -m urlab.apps.uncertain_sampling [--dry-run]
```

> **Assumption: linear (peg-in-hole) assembly.** The mate is taken to be a single-axis insertion
> along the connector's **+X** — the trajectory is a straight −X → 0 approach and the retract is the
> reverse translation along that same axis. A curved, multi-axis, or twist-to-lock mate is **not**
> supported and would need a true reverse-path retract.

## Control — compliance (admittance), NOT force control
The insertion runs under **software admittance** (`urlab/robot/admittance.py`): the arm **follows the
assembly trajectory** as a position reference *and* **yields to contact** through a virtual
spring-mass-damper with **finite restoring stiffness**, springing back toward the reference when
contact eases. This is deliberately **not** UR `forceMode` — that is pure force control with **no
stiffness**, so it floats freely off the trajectory.

The free-space moves (approach the stand-off, move to each trial's perturbed start) stay **stiff
position control**; only the contact phase is compliant. A `force_guard` trip during insertion means
the connector **seated**; the retract runs **un-guarded** (a seated part is already over the limit,
so a guarded retract would block the motion that frees it).

## Which cable/connector is under test
Set both keys in `configs/uncertain_sampling.yaml`:
```yaml
cable: banana                                  # grasp params from configs/cables.yaml
held_frame: banana_connector_finger_holder     # held pose + mate from configs/frames.yaml
```
`cable:` pulls the grasp calibration from `configs/cables.yaml`; `held_frame:` (REQUIRED) names a
`configs/frames.yaml` frame — the held connector is declared there **directly wrt tool0** (the old
`connector_holder` chain is retired).

## Frames & the recorded target
`held_frame` selects **both** poses from the shared catalogue, `configs/frames.yaml`:
- `frames:` — the held connector's pose wrt tool0 (one entry, monitor units welcome);
- `targets:` — the recorded mate (`base_link <- <frame>`): hand-guide to a good mate, read the
  frame's line off the monitor, and paste it — **in the monitor's own units** (`xyz_mm`/`rpy_deg`),
  so there is nothing to convert:

```yaml
targets:
  banana_connector_finger_holder:
    xyz_mm:  [82.69, 1071.80, -185.15]     # straight off the monitor line
    rpy_deg: [-0.72, 0.69, 88.73]
```

(`xyz`/`rpy` in m/rad still work. The unit is in the **key name**, so the two can't be confused;
setting both for one triple raises rather than silently picking one.) The assembled tool0 pose +
the anchoring fall out automatically (last trajectory row ↦ the assembled pose).

The **assembly trajectory** (`assembly_trajectory.csv`) is the **connector w.r.t. the target
connector** (`T_targetconn_conn`); its **last row must be identity** (connector == target connector
at the mate). It is a direct **−X insertion**: the connector backs off along the target's −X and
drives to 0 (`standoff_axis: [-1, 0, 0]`).

## How the perturbation is applied — `sampling.perturb_frame`
Which frame the bias acts in selects **which error source you are studying**. Both are legitimate;
pick the one you mean.

| `perturb_frame` | formula | models | effect of a 30° pitch bias |
|---|---|---|---|
| **`connector`** (default, also `held`) | `ideal[i] · bias` | **In-hand pose error** — the part is held wrong in the gripper; bias is along the **part's** axes. The robot doesn't know, so it runs its **nominal** motion. | Part is tilted 30°; **travel stays along the target's axis** and the part rides in crooked. |
| **`target`** | `bias · ideal[i]` | **Socket/target pose error** — the robot's belief about the mate is wrong; bias is along the **target's** axes. The whole approach is rigidly misaimed. | Part is tilted 30° **and travel rotates 30°** onto the part's own axis. |

Both tilt the part identically — **only the travel direction distinguishes them**, which is why the
regression test asserts travel direction rather than the endpoint.

## How the uncertainty is sampled
The range is **absolute lower/upper bounds per DOF** — not half-widths — so it need not be centred
on zero (a connector that always sags can be given `[-4, -1]` mm):

```yaml
uncertainty:
  lower: [0.0, 0.0, -0.005, 0.0, -15.0, 0.0]    # x,y,z in m; roll,pitch,yaw in deg
  upper: [0.0, 0.0,  0.005, 0.0,  15.0, 0.0]
```

`sampling.mode` picks how points inside that box are chosen:

| `mode` | behaviour |
|---|---|
| **`random`** | uniform draw inside the bounds, independently per DOF, once per trial, `num_trials` times. **Hard limits** — never exceeded, no tail (these are *not* σ). |
| **`grid`** | **ordered, exhaustive sweep** of every combination of the per-DOF grid values at `grid_resolution`. Deterministic, each point visited exactly once, **and the trial count is derived from the grid** (`num_trials` is ignored). |

Grid order is the Cartesian product with the **last axis varying fastest**; DOFs whose `lower ==
upper` contribute a single value, so all-zero DOFs cost nothing. The step is nudged so **both
endpoints are hit exactly** — a range that isn't a whole multiple of the step won't silently drop its
upper end. A DOF that spans a range with `grid_resolution: 0` is a loud error, not a silent single
point. For example, z ±5 mm step 5 mm × pitch ±15° step 15° gives **9 trials**:

```
(-5mm,-15°) (-5mm,0°) (-5mm,+15°) (0,-15°) (0,0°) (0,+15°) (+5mm,-15°) (+5mm,0°) (+5mm,+15°)
```

`sampling.noise` uses the same lower/upper form, drawn freshly per waypoint. Rotations are applied as
**extrinsic XYZ** rpy (the repo-wide convention). `sampling.random_seed > 0` seeds a dedicated RNG so
a `random` run replays exactly; `grid` is reproducible by construction.

## Progress reporting
Before each trial moves, its offsets are printed; after it finishes, its duration, the running **mean
cycle time**, and the **ETA** plus expected wall-clock finish:

```
--- trial 4/9 ---
offsets (connector frame): xyz=[  +0.00,   +0.00,   -5.00] mm  rpy=[ +0.00, -15.00,  +0.00] deg
trial 4/9 took 31.2 s | mean cycle 30.8 s | 5 left, ETA 2:34 (done ~14:21:07)
```

## Config `configs/uncertain_sampling.yaml`

### Part, frames, path
| key | meaning |
|---|---|
| `cable` | the connector under test; pulls its grasp calibration from `cables.yaml` |
| `held_frame` | REQUIRED: the `configs/frames.yaml` frame naming the held connector (its `frames:` pose wrt tool0 + its `targets:` mate) |
| `trajectory_csv` / `trajectory_angles_deg` | the ideal path (connector wrt target connector; last row identity) |
| `standoff_axis` | approach/insertion axis in the target-connector frame (`[-1,0,0]` = direct −X → 0) |
| `standoff_distance_m` | how far **beyond the trajectory's FIRST row** the one-time stand-off sits (measured from the path start, not the mate — so it is always clear of the path) |
| `retract_distance_m` | how far to back out after each trial, straight along the **connector's own −X** from wherever the insert stopped (magnitude; compliant but un-guarded) |

### Sampling
| key | meaning |
|---|---|
| `sampling.mode` | `random` (uniform draws, `num_trials` of them) or `grid` (ordered exhaustive sweep; **derives** the trial count) |
| `sampling.num_trials` | how many perturbed insertions — **ignored in `grid` mode** |
| `sampling.grid_resolution` | per-DOF grid step, same order/units as the bounds (`grid` mode only) |
| `sampling.perturb_frame` | `connector` (in-hand pose error, default) or `target` (socket pose error) |
| `sampling.uncertainty.lower` / `.upper` | per-DOF **absolute bounds** `[x,y,z (m), r,p,y (deg)]` in the frame above. Hard limits, not σ; need not be symmetric |
| `sampling.noise.lower` / `.upper` | extra jitter, same form but drawn **per waypoint** (usually all 0) |
| `sampling.chunk_fraction` | fraction of the (resampled) trajectory to execute per trial |
| `sampling.translational_resolution_m` / `rotational_resolution_deg` | densify the CSV to this spacing |
| `sampling.log_decimation` | log every Nth servo cycle (125 Hz / N ≈ rows/s); `5` ≈ 25 Hz |
| `sampling.random_seed` | `0` = nondeterministic; `>0` seeds a dedicated RNG for replayable trials |
| `sampling.csv_path` | output stem; a `<cable>/` subfolder is inserted and a `_YYYYmmdd_HHMMSS` appended, so runs are grouped by cable and unique |

### Compliance & contact
| key | meaning |
|---|---|
| `compliance.stiffness` / `mass` / `damping_ratio` | the admittance spring-mass-damper (per TOOL0 axis); `stiffness` sets deflection-per-force (`500 N/m` → 10 N ≈ 20 mm). `damping_ratio` 1 = critically damped |
| `compliance.selected_axes` | which TOOL0 axes yield to contact (`[1,1,1,1,1,1]` = all) |
| `compliance.settle_s` / `warmup_s` | settle-at-target after each insert / servo warm-up before the guard arms |
| `compliance.tare_before` | zero the F/T mid-warmup (servo-active) so the guard baseline is correct |
| `force_guard.max_force_n` / `max_torque_nm` | contact limit; a trip during insertion = the connector **seated** |

### Speed
| key | meaning |
|---|---|
| `speed.max_cartesian_translation_mm_s` / `max_cartesian_rotation_deg_s` | cartesian limits for the compliant **INSERT** (the contact phase). Each ramp segment gets the time its own geometry needs, so neither limit is exceeded; `0` disables that limit |
| `speed.retract_translation_mm_s` / `retract_rotation_deg_s` | same, for the **RETRACT** — free-space escape, so it need not crawl at insertion speed. Omit to fall back to the insert limits |
| `speed.max_joint_velocity_deg_s` / `max_cartesian_velocity_m_s` | the **free-space** `moveJ` caps (stand-off approach, per-trial reorient). `_rad_s` is still accepted for the joint cap |
| `speed.joint_acceleration_rad_s2` | free-space accel. **For short moves this binds, not the velocity cap** — a hop shorter than the accel/decel distance never reaches top speed, so raise this (not the cap) to speed up the per-trial reorient |

## Output columns (per sample)
`trial`, `timestamp`, then:
- **`tool0_base_*`** — raw tool0 pose in base_link (`x_mm, y_mm, z_mm`, quat, yaw/pitch/roll deg),
- **`connector_target_*`** — the connector's DEVIATION from the ideal mate, same field set (identity
  at a perfect mate),
- **`wrench_base_*`** — the contact wrench in **ROS `base_link`** (`getActualTCPForce`, bridged out of
  the UR `base` frame by `arm.wrench()` — see the README conventions),
- **`wrench_connector_*`** — that wrench re-expressed in the connector frame.

**Units are in the column names.** Translations are logged in **millimetres** (`_mm`), rotations in
**degrees** (`_deg`); quaternion components are dimensionless and wrenches are **N / Nm**. The
library itself works in metres throughout — the conversion happens only at the logging boundary
(`_pose_fields_mm`), so nothing upstream is affected.

`trial` increments each time the assembly trajectory is performed. The perturbation RNG is separate
from everything else, so a seeded run is reproducible.

> **Data notes — older logs.**
> - Logs written **before translations moved to mm** have un-suffixed `x`/`y`/`z` **in metres**.
>   `contact_manifold.py` skips them by name rather than silently mixing metres with millimetres; to
>   reuse one, scale x/y/z by 1000 and rename to `*_mm`.
> - Logs recorded before the `base_link` wrench bridge landed have their `wrench_*` **x and y
>   components negated** (z is unaffected). Re-collect, or flip those two columns.

## Building the contact manifold
`analysis/contact_manifold.py` concatenates a set of runs into one CSV of just the **frame-invariant**
pair — `connector_target_*` (the misalignment) and `wrench_connector_*` (the response along the
part's own axes). Inputs are files, directories, or globs; the output path **and filename are
yours** via the required `--out`:

```bash
# every sampling run for the banana connector
python analysis/contact_manifold.py data/uncertain_assembly_sampling/banana \
    --out data/uncertain_assembly_sampling/banana/banana_connector_contact_manifold.csv

# sampling runs + estimator_eval campaigns, contact samples only (the estimator's own threshold)
python analysis/contact_manifold.py data/uncertain_assembly_sampling/banana \
    "data/experiments/estimator_eval_2026*" --min-force 3.0 \
    --out configs/data/banana_manifold_augmented.csv
```

Two source formats are understood, freely mixed in one build:

- **uncertain_sampling runs** — the manifold columns copy out directly: the sampler logs the
  *true* relative pose (the part is fixtured, the belief is never wrong).
- **estimator_eval experiment folders** (`data/experiments/estimator_eval_*/`) — each
  `trial_TTT_attempt_AA_observations.csv` is logged in the **believed** frame, wrong by exactly
  that attempt's belief error, which is known because the part is fixtured. The builder **rebases**
  every file into the true frame using the `err_before_*` columns of the run's own `trials.csv`
  (`rel_true = rel_logged @ inverse(E)`; the wrench changes frames the same way) — the recipe that
  built the augmented manifold. `trials.csv`/`summary.csv` are read as metadata, never as samples;
  a run folder missing its `trials.csv` is skipped (believed-frame rows can't be rebased).

Because both column groups are relative to the connector and its target, runs recorded on different
days at different socket positions concatenate directly. The cell-specific `tool0_base_*` and
`wrench_base_*` columns are deliberately dropped.

| flag | effect |
|---|---|
| `--out PATH` | **required** — the output CSV, path and filename included |
| `--min-force N` | keep only samples with `|f| >= N` in the connector frame — drops the free-space approach, leaving actual contact |
| `--with-source` | prepend `source_file` and `trial` for provenance (eval rows: `trial/attempt`) |

Rebuilds are **idempotent** — an existing `*_contact_manifold.csv` is skipped when sweeping a
directory, so re-running can't read its own output back in and double every sample (keep that
suffix on `--out` when the output lands beside its inputs). Logs in an older column format are
skipped with a warning rather than failing the batch.

## Validating the manifold — ICP offset recovery
`analysis/manifold_icp_validation.py` checks whether the manifold carries enough information to
localise a misaligned connector. Both CSVs are reduced to a common mm-equivalent 12-D space
(translation as-is; rotation × `scaling_constant_deg_to_mm`; **unit-normalized** force/torque ×
their scaling constants). Per validation trial, the chosen `perturb_dims` of the initial pose are
zeroed (a hidden rigid offset applied to every observation, wrench left as recorded), then
multi-start **ICP** (matching across all 12 dims, correcting only the perturbed dims) with
**residual-gated RANSAC** aggregation recovers the offset. `trials_per_observation` groups
consecutive trials into one observation with one shared offset (use when they share a physical
offset — e.g. one grasp, several insertions). Edit the `CONFIG` dict at the top and run:

```bash
python analysis/manifold_icp_validation.py
```

Output: a timestamped folder under `analysis/` with per-trial figures (per-dim error + NN residual
vs iteration; every guess faint, consensus bold, dashed zero), a results CSV (per-trial true
offsets, estimates, signed + absolute errors), and `config.json` for provenance.
