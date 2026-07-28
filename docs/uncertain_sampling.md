# uncertain_sampling — perturbed-insertion data collection (held connector)

A **data-collection** run: repeatedly drive a PERTURBED **connector** into the mate under
compliance, logging each sample, then disassemble along the ideal path and repeat. Produces a CSV
for studying how a misaligned connector behaves during insertion.

```
resample the ideal trajectory -> for each trial: tare, perturb (connector frame), drive in
(compliant, guarded, LOGGING), snap to the closest ideal pose, disassemble -> write the CSV
```

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

## Frames & the recorded target
The connector is held via the holder: **tool0 → `connector_holder` → `connector`**. `connector_holder`
(the shared holder mount) is in the config; **`connector_in_holder` (per-connector) comes from the
`cable:` profile above**. Both frames are shown live in the `monitor`. The **assembly target is
recorded for the `connector_holder`** and is **per-cable, in `cables.yaml`**: hand-guide to a good
mate, read `base_link <- connector_holder` off the monitor, and paste it into that cable's
`connector_holder_target` in `cables.yaml`. The connector target is then
`connector_holder_target · (holder→connector)`, and the assembled tool0 pose + the anchoring
fall out automatically (last trajectory row ↦ the assembled pose).

The **assembly trajectory** (`assembly_trajectory.csv`) is the **connector w.r.t. the target
connector** (`T_targetconn_conn`); its **last row must be identity** (connector == target connector
at the mate). It is a direct **−X insertion**: the connector backs off along the target's −X and
drives to 0 (`standoff_axis: [-1, 0, 0]`).

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
| `sampling.uncertainty` | per-DOF uncertainty RANGE — half-widths `[x,y,z (m), r,p,y (deg)]`, drawn once per trial, **in the connector frame** |
| `sampling.noise` | extra per-waypoint jitter (usually 0), in the connector frame |
| `sampling.random_seed` | `0` = nondeterministic; `>0` seeds a dedicated RNG for replayable trials |
| `sampling.csv_path` | output stem; a `<cable>/` subfolder is inserted and a `_YYYYmmdd_HHMMSS` appended, so runs are grouped by cable and unique |

## Output columns (per sample)
`trial`, `timestamp`, then:
- **`tool0_base_*`** — raw tool0 pose in base_link (x, y, z, quat, yaw/pitch/roll deg),
- **`connector_target_*`** — the connector's DEVIATION from the ideal mate (identity at a perfect mate),
- **`wrench_base_*`** — the contact wrench as recorded (`getActualTCPForce`, base_link),
- **`wrench_connector_*`** — that wrench re-expressed in the connector frame.

`trial` increments each time the assembly trajectory is performed. The perturbation RNG is separate
from everything else, so a seeded run is reproducible.
