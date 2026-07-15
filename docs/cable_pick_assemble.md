# cable_pick_assemble — pick a cable, then assemble it

Port of `ur_cable_pick_assemble_demo`. The **pick** is the exact cable-pick-place pipeline. What
replaces "place" is a pluggable assembly: stand-off, compliant chunked insertion, release,
multi-step retract. Only `kinematic` is implemented (the target pose is given outright); `vision`
fails loudly.

```
[PICK] -> lift -> stand-off -> insert (admittance, force-guarded) -> open (release)
  -> retract (multi-step) -> home
```

**Compliance is a software ADMITTANCE law** (`robot/admittance.py`): a virtual spring-mass-damper,
`F = M·ẍ + D·ẋ + S·(x − x_d)`, integrated each cycle and streamed over `servoL`. It reimplements
the dev branch's ros2_control admittance controller in Python, so there's a real **restoring
stiffness** (default `S = 2000 N/m`): a contact force deflects the arm by `F/S` and it springs back
toward the reference (the ideal insertion pose) when contact eases. No controller to load or
switch. (This replaces an earlier forceMode attempt, which had *zero* restoring stiffness and
couldn't push the part into its mate.)

**Chunked insertion** is inspectable: the contact wrench is checked between chunks, so a jam is
caught after a fraction of the travel. During insertion a force‑guard trip means the part
**seated** (success); the same trip during any free‑space move means an unexpected collision
(failure) — the guard distinguishes them by phase.

> **⚠️ `servoL` needs steady timing.** The admittance loop streams `servoL` at
> `assembly.compliance.reference_rate_hz` (default **125 Hz**), pacing itself with
> `initPeriod`/`waitPeriod`. If Python + RTDE can't hold that rate on your host, the mate turns
> **jerky** — drop `reference_rate_hz` to **62.5** (the ODE integrates the same, just coarser).
> One more consequence of streaming `servoL`: the per‑chunk confirm prompts are gone (you can't
> pause `servoL` without dropping servo control), so the only veto is the single confirm **before**
> the insertion starts. The law runs in the **`tool0` frame** (matching dev) — the base‑frame F/T
> wrench is rotated into `tool0` each cycle, and `stiffness`/`selected_axes` are about the tool
> axes. **Keep the e‑stop in hand** on the first mate.

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
