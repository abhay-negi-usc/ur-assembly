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

## Frames & the recorded target
The connector is held via the holder: **tool0 → `connector_holder` → `connector`** (`connector_holder`
+ `connector_in_holder` in the config; both shown live in the `monitor`). The **assembly target is
recorded for the `connector_holder`**: hand-guide to a good mate, read `base_link <- connector_holder`
off the monitor, and paste it into `connector_holder_target`. The connector target is then
`connector_holder_target · (holder→connector)`, and the assembled tool0 pose + the [anchoring trick]
fall out automatically (last trajectory row ↦ the assembled pose).

## Config `configs/uncertain_sampling.yaml`
| key | meaning |
|---|---|
| `connector_holder` / `connector_in_holder` | tool0→holder→connector frames (the held connector) |
| `connector_holder_target` | recorded base→connector_holder at the mate (from the monitor) |
| `sampling.num_trials` | how many perturbed insertions |
| `sampling.chunk_fraction` | fraction of the (resampled) trajectory to execute per trial |
| `sampling.translational_resolution_m` / `rotational_resolution_deg` | densify the CSV to this spacing |
| `sampling.bias` | ONE draw per trial — `[x,y,z (m), r,p,y (deg)]` half-widths, **in the connector frame** |
| `sampling.noise` | a fresh draw per waypoint, in the connector frame |
| `sampling.random_seed` | `0` = nondeterministic; `>0` seeds a dedicated RNG for replayable trials |
| `sampling.csv_path` | output stem; a `_YYYYmmdd_HHMMSS` is appended so each run is unique |

## Output columns (per sample)
`trial`, `timestamp`, then:
- **`tool0_base_*`** — raw tool0 pose in base_link (x, y, z, quat, yaw/pitch/roll deg),
- **`connector_target_*`** — the connector's DEVIATION from the ideal mate (identity at a perfect mate),
- **`wrench_base_*`** — the contact wrench as recorded (`getActualTCPForce`, base_link),
- **`wrench_connector_*`** — that wrench re-expressed in the connector frame.

`trial` increments each time the assembly trajectory is performed. The perturbation RNG is separate
from everything else, so a seeded run is reproducible.
