# bnc_assembly — architecture after the 2026-08 refactor

Run it exactly as before:

```bash
python -m urlab.apps.bnc_assembly            # add --dry-run --yes to rehearse without a robot
```

The behaviour of the app is unchanged (pinned by `tests/characterize_bnc.py` and the unit
suite); what changed is WHERE the code lives. The monolith (4,327 lines, one closure web) is
now three layers, chosen so that each kind of edit has one obvious home.

## The three layers

```
urlab/apps/bnc_assembly.py      THE ALGORITHM  (~800 lines)
    build_and_run: construct the bag -> setup(asm) -> read the tuning once ->
    the reset/pick/estimate/assemble/clock/verify/disassemble loop, written as plain
    loops and if/else over behaviour calls. No behaviour bodies live here.

urlab/skills/bnc.py             THE BEHAVIOURS (~3,300 lines)
    setup(asm)                       construct + validate everything, pre-motion
    locate_target_visually(asm, ...) marker sweep; re-anchors the run
    engage_insertion(asm)            trajectory + oscillation, stop on axial force
    run_insertion(asm, ...)          one admittance insertion collecting observations
    connector_clocking(asm)          the bayonet screw sweep
    collar_clocking(asm, ...)        axial collar grasp + wrist twist (also unlocks)
    tug_verify_in_place(asm)         does the lock actually hold?
    clocking_retract / retract_from / retract_along_target   the escapes
    place_after_failed_engage(asm, ...) / disassembly(asm, ...) / reorient_recovery(asm)
    phase_gate / aligned_place_pose / place_scatter / celebrate / ... helpers
    Every behaviour is `f(asm, ...) -> status`; the algorithm branches on the status.

urlab/domain.py                 THE DOMAIN     (~680 lines, pure & robot-free)
    BncSpec + one frozen dataclass per behaviour   -> ALL tuning, typed, unknown-key-rejecting
    BncAssembly                                    -> the run's shared state ("the bag")
    TargetFrames / Pacing / clock_physics          -> frames, speed policy, controller pairs
    the clocking state walk + clocking math        -> CLOCK_STATES, wrap_near, fit_turn, ...
urlab/robot/detectors.py        ScrewAdvance, AxialForce, TravelReached, RadialConfirm, AnyGuard
```

## The two objects every behaviour sees

- **`asm` (domain.BncAssembly)** — the run's shared state: robot, camera, cfg, spec, the
  anchored `TargetFrames`, `Pacing`, every controller/guard pair, the mutable in-hand belief
  `asm.T_ftip_conn`, the pick stack, the experiment folder, RNG streams. It replaces the old
  closure scope: what used to be captured is now a named attribute you can grep.
- **`asm.spec` (domain.BncSpec)** — every tuning number, parsed once from the per-behaviour
  yaml blocks (`strip_derived_units` + `parse_block`). Unknown keys are rejected with their
  dotted path. Behaviours read `asm.spec.<behaviour>.<field>`; none of them touch the yaml.

## Where to make a change

| you want to… | edit |
|---|---|
| change the order/logic of the run | `apps/bnc_assembly.py` (build_and_run — it reads top to bottom) |
| change how a behaviour moves the arm | that behaviour in `skills/bnc.py` |
| add/rename a tuning knob | its Spec in `domain.py` and the matching yaml block |
| change frames/pacing/state arithmetic | `domain.py` (pure functions — unit-test them directly) |
| change a termination condition | `robot/detectors.py` |

## Verification harness (keep using it)

- `./.venv/Scripts/python -m pytest tests/` — the FAILED set must stay exactly the
  14-test baseline listed in `docs/bnc_port_log.md` (they are environment/tuning pins that
  predate the refactor; the venv interpreter is the reference — Anaconda reads one fewer).
- `./.venv/Scripts/python tests/characterize_bnc.py` — byte-compares the dry-run trace to
  `tests/goldens/bnc_dryrun_trace.txt`. `--update` regenerates it; explain why in the commit.

The step-by-step port record, including every test that was re-anchored and why, is
`docs/bnc_port_log.md`.

## Config schema

`configs/bnc_assembly.yaml` uses the per-behaviour schema (2026-08-28): one top-level block
per spec dataclass (`run:`, `trajectory:`, `engage:`, `connector_clocking:`, …), controller
overrides nested under `compliance:`/`force_guard:` inside their behaviour. The exact
old→new mapping is `docs/bnc_config_migration.md`; a file still carrying the legacy
`assembly:` block is refused at startup. Unknown keys in any block fail the load with their
dotted path.

## Still open

- Bench checkpoints: the port moved servo-critical code (engage, insertions, clockings,
  retracts). Dry-run and the suite cannot exercise force control — run the standard bench
  validation before trusting a hardware run.
