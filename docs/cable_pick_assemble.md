# cable_pick_assemble — pick a cable, then assemble it

Port of `ur_cable_pick_assemble_demo`. The **pick** is the exact cable-pick-place pipeline. What
replaces "place" is a pluggable assembly: stand-off, compliant chunked insertion, release,
multi-step retract. Only `kinematic` is implemented (the target pose is given outright); `vision`
fails loudly.

```
[PICK] -> lift -> stand-off -> enter compliance -> insert (chunked, force-guarded)
  -> open (release) -> end compliance -> retract (multi-step) -> home
```

**Compliance is UR `forceMode`**, not the ros2_control admittance controller — so there is no
controller to load, no controller switch, and no joint-0 velocity fault to prevent. It is a mode
of the running controller and starts from where the arm is.

**Chunked insertion** makes the mate inspectable: contact is checked *between* chunks, so a jam is
caught after a fraction of the travel, and each fraction can be vetoed. During insertion a force
trip means the part **seated** (success); the same trip during any free-space move means an
unexpected collision (failure) — the guard distinguishes them by phase.

```bash
python -m urlab.apps.cable_pick_assemble [--dry-run] [--yes]
```

Config: `configs/cable_pick_assemble.yaml` (pick half identical to cable_pick_place)

| key | meaning |
|---|---|
| `assembly.method` | `kinematic` (implemented) or `vision` (fails loudly) |
| `assembly.target.frame` / `xyz` / `rpy` | where the assembly ends; `frame` ∈ `fingertip`/`tool0`/`connector` — **measure this** |
| `assembly.standoff.axis` / `distance_m` | pre-insertion back-off, in the target frame |
| `assembly.compliance.selected_axes` / `force_limits` / `target_wrench` | forceMode: which axes yield and how |
| `assembly.force_guard.max_force_n` / `max_torque_nm` | collision / seat limits (30 N / 5 Nm) |
| `assembly.insertion.chunk_fraction` / `chunk_time_s` | insertion granularity |
| `assembly.retract.frame` / `steps` | multi-leg escape path (base/target = fixed; tool0/fingertip = move with the arm) |

`assembly.target` is placeholder geometry — measure it by jogging to a good mate and reading the
chosen frame off the robot. The retract steps are **large free-space moves**; clear the path.
