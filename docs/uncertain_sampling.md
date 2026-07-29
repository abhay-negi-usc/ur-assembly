# uncertain_sampling — perturbed-insertion data collection (held connector)

A **data-collection** run: repeatedly drive a PERTURBED **connector** into the mate under
compliance, logging each sample, then retract and repeat. Produces a CSV for studying how a
misaligned connector behaves during insertion.

```
for each trial: perturb (connector frame) -> move to the perturbed start (stiff, free space) ->
follow the path under ADMITTANCE (logging at the servo rate, guarded) -> settle -> retract -> CSV
```

## Control — compliance (admittance), NOT force control
The insertion runs under **software admittance** (`urlab/robot/admittance.py`): the arm **follows the
assembly trajectory** as a position reference *and* **yields to contact** through a virtual
spring-mass-damper with **finite restoring stiffness**, springing back toward the reference when
contact eases. This is deliberately **not** UR `forceMode` — that is pure force control with **no
stiffness**, so it floats freely off the trajectory (the drift this design replaces). The free-space
moves (approach the stand-off, move to each trial's perturbed start) stay **stiff position control**;
only the contact phase is compliant. A `force_guard` trip during insertion means the connector
**seated**; the retract runs **un-guarded** (a seated part is already over the limit, so a guarded
retract would block the motion that frees it).

```bash
python -m urlab.apps.uncertain_sampling [--dry-run]
```

## Which cable/connector is under test
Set it in `configs/uncertain_sampling.yaml`:
```yaml
cable: banana        # or: bnc | c13   (or on the CLI: --set cable=bnc)
```
That one key pulls this connector's **calibration** from `configs/cables.yaml` — most importantly
`connector_in_holder` (the connector's pose in the holder). Add/measure a connector by editing its
entry under `cables:` in `cables.yaml`; **`banana` is the default and set up there.**

## Which cable/connector is under test
Set it in `configs/uncertain_sampling.yaml`:
```yaml
cable: banana        # or: bnc | c13   (or on the CLI: --set cable=bnc)
```
That one key pulls this connector's **calibration** from `configs/cables.yaml` — most importantly
`connector_in_holder` (the connector's pose in the holder). Add/measure a connector by editing its
entry under `cables:` in `cables.yaml`; **`banana` is the default and set up there.**

## Frames & the recorded target
The connector is held via the holder: **tool0 → `connector_holder` → `connector`**. `connector_holder`
(the shared holder mount) is in the config; **`connector_in_holder` (per-connector) comes from the
`cable:` profile above**. Both frames are shown live in the `monitor`. The **assembly target is
recorded for the `connector_holder`** and is **per-cable, in `cables.yaml`**: hand-guide to a good
mate, read `base_link <- connector_holder` off the monitor, and paste it into that cable's
`connector_holder_target` in `cables.yaml` — **in the monitor's own units**, using the `xyz_mm` /
`rpy_deg` keys, so there is nothing to convert by hand:

```yaml
connector_holder_target:
  xyz_mm:  [90.71, 1073.48, -184.20]     # straight off the monitor line
  rpy_deg: [-0.79, -0.25, 88.92]
```

(`xyz`/`rpy` in m/rad still work. The unit is in the **key name**, so the two can't be confused;
setting both for one triple raises rather than silently picking one.) The connector target is then
`connector_holder_target · (holder→connector)`, and the assembled tool0 pose + the anchoring
fall out automatically (last trajectory row ↦ the assembled pose).

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
**Uniform, independent per DOF, with hard limits.** Each value in `sampling.uncertainty` is a
**half-width**, not a standard deviation: DOF *i* is drawn `U(−half_widthᵢ, +half_widthᵢ)`, once per
trial. Draws are zero-centred, fill the range, and **never exceed it** — a `0` entry means that DOF
never moves. `sampling.noise` is drawn the same way, but freshly per waypoint.

> Reading these as σ would be badly wrong: ~32% of Gaussian draws fall outside ±σ, and a 30° σ would
> produce occasional 90°+ misalignments. There is no tail here — `±30°` means exactly that.

Rotations are applied as **extrinsic XYZ** rpy (the repo-wide convention). `sampling.random_seed > 0`
seeds a dedicated RNG so a run replays exactly.

**To calibrate `connector_in_holder` for a connector:** hold it in the holder, hand-guide, and read
`base_link <- connector` vs `base_link <- connector_holder` off the monitor; paste the relative pose
into that cable's `connector_in_holder` in `cables.yaml`. Identity = unmeasured (connector coincides
with the holder).

## Config `configs/uncertain_sampling.yaml`
| key | meaning |
|---|---|
| `cable` | the connector under test; pulls its calibration from `cables.yaml` (`connector_in_holder`, …) |
| `connector_holder` | the shared holder mount (tool0 → connector_holder) |
| `connector_holder_target` | recorded base→connector_holder at the mate (from the monitor); **per-cable, in `cables.yaml`** |
| `standoff_axis` | approach/insertion axis in the target-connector frame (`[-1,0,0]` = direct −X → 0) |
| `sampling.num_trials` | how many perturbed insertions |
| `sampling.chunk_fraction` | fraction of the (resampled) trajectory to execute per trial |
| `sampling.translational_resolution_m` / `rotational_resolution_deg` | densify the CSV to this spacing |
| `sampling.perturb_frame` | `connector` (in-hand pose error, default) or `target` (socket pose error) — see above |
| `sampling.uncertainty` | per-DOF **uniform half-widths** `[x,y,z (m), r,p,y (deg)]`, drawn once per trial in the frame above. Hard limits, not σ |
| `sampling.noise` | extra jitter, drawn the same way but **per waypoint** (usually 0) |
| `sampling.log_decimation` | log every Nth servo cycle (125 Hz / N ≈ rows/s); `5` ≈ 25 Hz |
| `sampling.random_seed` | `0` = nondeterministic; `>0` seeds a dedicated RNG for replayable trials |
| `sampling.csv_path` | output stem; a `<cable>/` subfolder is inserted and a `_YYYYmmdd_HHMMSS` appended, so runs are grouped by cable and unique |
| `compliance.stiffness` / `mass` / `damping_ratio` | the admittance spring-mass-damper (per TOOL0 axis); `stiffness` sets deflection-per-force (`2000 N/m` → 20 N ≈ 10 mm) |
| `compliance.selected_axes` | which TOOL0 axes yield to contact (`[1,1,1,1,1,1]` = all) |
| `compliance.settle_s` / `warmup_s` | settle-at-target after each insert / servo warm-up before the guard arms |
| `speed.max_cartesian_translation_mm_s` / `max_cartesian_rotation_deg_s` | **cartesian speed limits for the compliant INSERT** (the contact phase), mm/s and deg/s. Each ramp segment gets the time its own geometry needs, so neither limit is exceeded; `0` disables that limit |
| `speed.retract_translation_mm_s` / `retract_rotation_deg_s` | same, for the **RETRACT** — free-space escape, so it need not crawl at insertion speed. Omit to fall back to the insert limits |
| `speed.max_joint_velocity_rad_s` / `max_cartesian_velocity_m_s` | the **free-space** `moveJ` caps (stand-off approach, per-trial reorient). The joint cap usually binds on a reorient |
| `compliance.tare_before` | zero the F/T mid-warmup (servo-active) so the guard baseline is correct |
| `force_guard.max_force_n` / `max_torque_nm` | contact limit; a trip during insertion = the connector **seated** |

## Output columns (per sample)
`trial`, `timestamp`, then:
- **`tool0_base_*`** — raw tool0 pose in base_link (x, y, z, quat, yaw/pitch/roll deg),
- **`connector_target_*`** — the connector's DEVIATION from the ideal mate (identity at a perfect mate),
- **`wrench_base_*`** — the contact wrench in **ROS `base_link`** (`getActualTCPForce`, bridged out of
  the UR `base` frame by `arm.wrench()` — see the README conventions),
- **`wrench_connector_*`** — that wrench re-expressed in the connector frame.

> **Data note.** Logs recorded before the `base_link` wrench bridge landed have their `wrench_*`
> **x and y components negated** (z is unaffected). Re-collect them, or flip those two columns.

`trial` increments each time the assembly trajectory is performed. The perturbation RNG is separate
from everything else, so a seeded run is reproducible.
