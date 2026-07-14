# uncertain_sampling — perturbed-insertion data collection

Port of `ur_uncertain_assembly_sampling`. A **data-collection** run: repeatedly drive a PERTURBED
held part into the mate under compliance, logging commanded/actual poses and the contact wrench,
then disassemble along the ideal path and repeat. Produces a CSV for studying how a misaligned
part behaves during insertion.

```
resample the ideal trajectory -> for each trial: tare, perturb, drive in (compliant, guarded,
LOGGING), snap to the closest ideal pose, disassemble along the ideal path -> write the CSV
```

```bash
python -m urlab.apps.uncertain_sampling [--dry-run]
```

Config: `configs/uncertain_sampling.yaml` (shares the anchoring trick + CSV with kinematic_assembly)

| key | meaning |
|---|---|
| `sampling.num_trials` | how many perturbed insertions |
| `sampling.chunk_fraction` | fraction of the (resampled) trajectory to execute per trial |
| `sampling.translational_resolution_m` / `rotational_resolution_deg` | densify the CSV to this spacing |
| `sampling.bias` | ONE perturbation draw per trial — `[x,y,z (m), r,p,y (deg)]` half-widths |
| `sampling.noise` | a fresh perturbation draw per waypoint |
| `sampling.random_seed` | `0` = nondeterministic; `>0` seeds a dedicated RNG for replayable trials |
| `sampling.csv_path` | output stem; a `_YYYYmmdd_HHMMSS` is appended so each run is unique |

Output columns: trial, timestamp, then commanded/actual tool0-in-base and held-in-target poses
(x,y,z,quat,yaw,pitch,roll) and the six-axis wrench. The perturbation RNG is separate from
everything else, so a seeded run is reproducible.
