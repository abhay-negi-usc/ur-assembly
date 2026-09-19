# configs/bnc_assembly.yaml — legacy `assembly:` schema → per-behaviour schema (2026-08-28)

The `assembly:` block was dissolved into top-level per-behaviour blocks that mirror the spec
dataclasses in `urlab/domain.py` one-to-one. The shipped file was converted mechanically and the
parsed `BncSpec` was verified EQUAL before and after the swap. A config still carrying an
`assembly:` block is refused at startup with a pointer to this file.

Everything OUTSIDE the old `assembly:` block is unchanged (`compliance:`, `force_guard:`,
`speed:`, `estimation:`, `pickup:`, `scan:`, `sam3:`, `marker_views:`, `robot_config:`, …).

## Block moves (`assembly.X…` → `X…`)

| old | new |
|---|---|
| `assembly.engage` | `engage` |
| `assembly.final_insertion` | `final_insertion` |
| `assembly.collection` | `collection` |
| `assembly.trajectory_noise` | `trajectory_noise` |
| `assembly.connector_clocking` | `connector_clocking` |
| `assembly.collar_clocking` | `collar_clocking` |
| `assembly.clocking_retract` | `clocking_retract` |
| `assembly.tug_verify` | `tug_verify` |
| `assembly.disassembly` (incl. `place`, `place_scatter`) | `disassembly` |
| `assembly.reorient_recovery` | `reorient_recovery` |
| `assembly.visual_target` | `visual_target` |
| `assembly.celebrate` | `celebrate` |

## Run-level flat keys → `run:`

`insertion_mode`, `target_source`, `target_frame`, `post_engage_frame`, `engage_clock_deg`,
`max_attempts`, `accumulate_observations`, `retract_distance_mm`, `log_decimation`,
`gate_between_behaviors`, `live_plot`, `release_retract_distance_mm`, `debug_match` — all move
verbatim under `run:`. Two renames:

| old | new |
|---|---|
| `assembly.success_tolerance.pos_mm` | `run.success_pos_mm` |
| `assembly.success_tolerance.rot_deg` | `run.success_rot_deg` |

## Trajectory keys → `trajectory:`

| old | new |
|---|---|
| `assembly.trajectory_csv` | `trajectory.csv` |
| `assembly.trajectory_angles_deg` | `trajectory.angles_deg` |
| `assembly.translational_resolution_mm` | `trajectory.translational_resolution_mm` |
| `assembly.rotational_resolution_deg` | `trajectory.rotational_resolution_deg` |
| `assembly.standoff` | `trajectory.standoff` (`distance_m` → `distance_mm`) |

## Flat override keys → nested `compliance:` / `force_guard:`

Inside `engage`, `final_insertion`, `connector_clocking`, `collar_clocking` and `tug_verify`,
the per-behaviour controller overrides used to sit flat on the block. They now nest:

- `stiffness`, `mass`, `damping_ratio` → `<block>.compliance.{…}`
- `max_force_n`, `max_torque_nm`, `force_guard_enabled` — and, for
  final_insertion/connector_clocking/collar_clocking, `persistence_s` — → `<block>.force_guard.{…}`

`engage.persistence_s` stays flat: it is the engage's own axial-force persistence (it still
doubles as its guard's persistence when the override does not set one). `tug_verify` has no
guard override — it runs under the global guard by design.

## Semantics that did NOT change

- Units convention: `_mm`/`_deg` keys in the file; the loader derives SI siblings.
- `key: null` still means "inherit the shared value / use the default".
- Absent keys read the dataclass defaults in `urlab/domain.py` — the single source of truth.
- Unknown keys in any spec block are rejected at load with their dotted path (this is now
  STRICTER than before: the old translator silently ignored stray keys).
