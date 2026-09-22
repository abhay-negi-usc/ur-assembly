# bnc_assembly port log

Working log for the OO + behaviour-tree port (plan: the approved refactor plan; steps there).

## Step 0 — baseline (2026-08-27)

**FAILED set** (stable across 3 consecutive runs, `.venv` Python 3.10, numpy 1.26.4; the
invariant for every step is this SET of ids, not the count):

```
tests/test_behaviors.py::test_commanded_speeds_are_clamped_to_the_controller_ceiling
tests/test_behaviors.py::test_joint_pnp_recovers_the_target_from_synthetic_corners
tests/test_behaviors.py::test_multiview_refine_recovers_the_marker_from_synthetic_corners
tests/test_behaviors.py::test_the_aligned_place_pose_is_flat_and_on_the_socket_heading
tests/test_smoke.py::test_a_clocking_stroke_follows_the_arc_and_not_the_chord
tests/test_smoke.py::test_bnc_assembly_shares_the_tuned_estimator
tests/test_smoke.py::test_bnc_clocking_geometry
tests/test_smoke.py::test_payload_and_joint_acceleration_are_fleet_wide_constants
tests/test_smoke.py::test_preload_force_is_a_spike_not_a_press
tests/test_smoke.py::test_the_collar_is_grasped_axially_and_turned_by_a_wrist_twist
tests/test_smoke.py::test_the_connector_sweep_rocks_between_absolute_roll_positions
tests/test_smoke.py::test_the_target_and_the_belief_describe_the_same_connector_datum
tests/test_smoke.py::test_tool_frames_shared_yaml_source
tests/test_smoke.py::test_wiggle_station_grid_and_the_retuned_excitation_stay_runnable
```

These are pre-existing tuning-pin failures (values drifted from what tests pin) plus the two
marker-PnP failures that only run where cv2 is installed. Do not chase them during the port;
do not let the set grow.

**Characterization golden**: `tests/goldens/bnc_dryrun_trace.txt`, produced by
`python tests/characterize_bnc.py --update` and checked by `python tests/characterize_bnc.py`
after every step. Invocation and its limits are documented in the harness docstring: the
dry-run black frame means the scanner prompts and `q` takes the abort path, so the golden pins
construction + entry sequencing; deep phases are pinned by unit tests co-migrated with each
extraction.

**Preconditions on this machine**: `estimation.manifold_csv` points at the local
`./data/test data/banana_manifold_20260807_clean.csv` (the robot host's `bnc_manifold_v6.csv`
is not in the repo); the visual target needs real markers, hence the kinematic override in the
harness.

## Step 0.5 — machine-config split (2026-08-27, user-requested mid-plan)

`_common.yaml` replaced by EXPLICIT machine files: `configs/robot.yaml` (robot:, speed:,
base/tip frames, compute:) and `configs/camera.yaml` (camera:, camera_frame), referenced from
every config via `robot_config:` / `camera_config:` (loader: `_apply_machine_layers` in
`urlab/config.py`; dangling reference = load error). `hand_eye` now comes from
`configs/frames.yaml`'s `camera` frame entry (`tool_frames.hand_eye()`), and the `aruco`
defaults derive from `marker_rigs` (`tool_frames.aruco_defaults()`); config blocks override
both wholesale — the four older-calibration configs (cable_pick_place, cable_touch_pick_place,
pick_place, visual_servo: hand_eye x = 0 vs the current −9 mm) keep their own values on
purpose. Tests: `tests/test_machine_config.py`.

> **SUPERSEDED 2026-09-22** for hand-eye only. The per-config override was the mechanism by
> which one calibration was spelled two ways (x = 0 vs −9 mm) across the fleet, which is a way
> for a cell to be wrong silently. `hand_eye:` blocks are now GONE from every config and
> `tool_frames.hand_eye()` has no override path: `configs/frames.yaml`'s `camera` entry is the
> only source, and it reads `[0, −120, 25] mm`. The `aruco` override described above is
> unchanged.

Also restored after an afternoon working-tree rollback reverted uncommitted session work:
robot.ip 192.168.10.106, gripper port COM3, the compute.device knob
(`sam3_backend.resolve_device`), and the local manifold path.

## Step 0.6 — estimation simplification (2026-08-28, user-requested)

Two amputations, both "one validated path instead of options nobody switched":

1. **SAM3 detection**: junction method + graph tracing + slope selection only, prompts fixed to
   "cable"/"connector". Removed as configuration: `sam3.mode` (neck|junction|tip), `sam3.trace`,
   `sam3.junction_select`, `cable_prompt`/`connector_prompt`, and the neck-only
   `adaptive`/`confidence_floor` (explained in each config's sam3 comment). The geometry keeps
   `trace=`/`select=` as function arguments so regression tests still compare strategies.
   Remaining knobs (`min_contrast`, `work_dim`, `overlay_opacity`, `connector_peak_min`)
   documented in the configs.
2. **Multi-view connector estimation removed**: `skills/scan.py` (CableScanner orbit),
   `perception/connector.py` (ConnectorEstimator triangulation fusion), and
   `perception/cable_recon.py` (reconstruction mode) deleted — every shipped config ran the
   single-image ground-plane estimate, which is now THE scan method (`build_scanner` always
   builds GroundPlaneScanner; its `.estimator` is a null stub kept for the one
   `scanner.estimator.reset()` call site). `cable_touch_pick_place` (app + config + doc) was
   deleted with it — its flow was built on the multi-view tip estimate and could not run
   without it. Seven tests pinning the deleted behaviour were removed with the behaviour
   (6× fusion/RANSAC/reconstruction in test_smoke, 1× fusion weights in test_behaviors);
   suite total returns to the Step-0 285 with the FAILED set unchanged.

## Steps 1-3 (2026-08-28)

**Step 1 — packaging**: `urlab.behaviors` added to pyproject packages (it was missing; a
non-editable install shipped without the behaviour layer).

**Step 2 — app-private imports → skills**: `corr_to_m`/`observe`/`save_observations` moved to
`urlab/skills/estimate.py` (public names; donor re-exports under the old `_names`);
`axial_ref`/`retract_ref` moved to `urlab/skills/insert.py` (uncertain_sampling re-exports);
bnc's `_guarded` now comes straight from `apps/_common.guarded` (the app-to-app alias chain is
gone); `_plot_run` imported from its real home `_estimate_plots`. `_argmin_estimate`/`_landscape`
stay in estimator_eval for now — `_landscape` is entangled with that app's private plotting
helpers and moves later with the estimate loop. Source-pinning tests repointed in the same step
(shared-not-duplicated intent preserved; `wrench_in` flange-pairing assert follows `observe`).

**Step 3 — detectors + pure domain out of the monolith**:
- `urlab/robot/detectors.py` (new): ScrewAdvance, AxialForce, TravelReached, RadialConfirm,
  AnyGuard — public names, verbatim behaviour, next to guard.py whose protocol they share.
  (TerminationSet.snapshot arrives with the engage move, Step 9.)
- `urlab/domain.py` (new, single MODULE not a package — fewer files by request): the clocking
  state walk (CLOCK_STATES/UNCLOCK_STATES, advance_state/retreat_state), the clocking window
  arithmetic (wrap_near, fit_turn, wrist3_window, clocking_plan), and compliance_override.
- bnc imports these under its old `_names` and re-exports CLOCK_STATES/UNCLOCK_STATES, so every
  existing test import stays valid until the cleanup step. Slice tests over the moved classes
  repointed at detectors.py.

bnc_assembly.py: 4327 → 3994 lines. Gates after every step: FAILED set = the stable 14,
characterization golden unchanged, pyflakes clean.

**Next**: Step 4 (config spine — per-behaviour Spec dataclasses + `parse_block` unknown-key
rejection + `legacy_translate`; the domain layer consolidates into `urlab/domain.py`, not a
package). Budgeted as a full session in the plan.

## Direction change (2026-08-28): NO behaviour trees

The composition layer is now PLAIN PYTHON, by user decision: trees are harder to follow. The
target `build_and_run` is an explicit algorithm — behaviour functions `f(asm) -> status` chained
by ordinary control flow, with named condition-checker functions (`engaged()`, `seated()`,
`tug_passed()`, ...) deciding every switch between behaviours. Steps 13-14 of the plan (subtree
factories + tree cutover) are replaced by "write the algorithmic build_and_run"; everything else
(config spine, BncAssembly, per-behaviour extraction, YAML swap) is unchanged and was always
composition-agnostic. urlab/behaviors stays as-is for the four apps already using it.

## Step 4 in progress (2026-08-28): the config spine, slices 1-2

`config.parse_block` (unknown keys rejected with the dotted path) + `urlab/domain.py` gains the
Spec dataclasses and `legacy_translate()` (the executable old->new schema table). Defaults live
ON the dataclasses -- one source of truth. Slice 1: EngageSpec (+Contact/Confirm/FailRetract).
Slice 2: ConnectorClocking/CollarClocking(+SeatPush)/TugVerify/Disassembly(+Place/PlaceScatter).
bnc's en_*/cc_*/cl_*/sp_*/tv_*/dis_* locals are now ASSIGNED FROM the spec under their old names
(zero downstream churn); the raw `en`/`cc`/`cl` block variables survive only for
`_clock_physics` until the compliance bank lands. Equality pinned by tests/test_bnc_spec.py
(field-by-field vs raw old-schema reads of the shipped config). Two source-text pins repointed
with their targets. Gotcha worth remembering: the units normaliser plants derived `_m` siblings
in every loaded block -- legacy_translate must copy EXPLICIT key lists, never dict(blk).

ALSO this session: the composition layer is now PLAIN PYTHON (no behaviour trees) by user
decision -- see the direction-change entry above and memory/no-behavior-trees.md.

Remaining Step-4 slices: final_insertion (fi_*), collection (col_*), trajectory noise (tn_*),
target/trajectory/run-level keys. Then Step 5 (BncAssembly object dissolving _anchor_target).

## Step 4 complete + Step 5a (2026-08-28)

**Step 4 slices 3**: FinalInsertionSpec (+TrajectoryNoiseSpec, shared shape), CollectionSpec,
RunSpec (insertion_mode/target_source/post_engage_frame/engage_clock_deg/max_attempts/
tolerances/retract/log_decimation). All of bnc's fi_*/col_*/tn_* and run-level locals now come
from `spec`; equality pinned (tests/test_bnc_spec.py, 6 tests). The whole tuning surface of the
run flows through BncSpec.

**Step 5a — TargetFrames**: the `_anchor_target` closure and its `nonlocal` web are GONE.
`domain.TargetFrames` holds the five planning frames (socket/tconn/T_clk/targetobj/commit),
`anchor()` rebuilds them all from a (re)measured socket; the trajectory load was hoisted above
the catalogue lookup so construction happens in one place; all ~70 frame reads in bnc are now
`frames_t.X`; the `post_engage_frame: believed` switch is a plain attribute write. THE reason
build_and_run could not be split into functions is dissolved.

Test co-migrations this step: the fold-once/rolled-anchor/socket-write-once pins now read
domain.py; the re-anchor test is BEHAVIOURAL (anchor twice, every frame must move, formulas
checked) instead of slicing closure source; ordering pins repointed at first-motion. Lesson
that cost several rounds: source pins with backslash continuations never exact-match through
heredocs -- go straight to line surgery.

bnc_assembly.py 3,994 -> 3,964; domain.py 488. Gates green after every slice.

**Next**: Step 5b -- Pacing (phase/caps/seg_time), ComplianceBank (the adm_*/guard_* farm via
compliance_override), RunSession (out_dir/RNGs/rows), and the BncAssembly aggregate; then the
behaviour extraction (engage first) toward the algorithmic script.

## Step 6 — first behaviour out: celebrate → `urlab/skills/bnc.py` (2026-08-28)

The extraction pattern the rest of the port repeats:

- **`urlab/skills/bnc.py`** (new, ONE module for all bnc behaviours): `celebrate(asm, state,
  cycle, tug_res, ret_ok)` — body verbatim from the closure, WHY-docstring intact. Captured
  names became explicit: config dict → `asm.spec.celebrate`, `robot` → `asm.robot`,
  `guard_shared` → `asm.guard`, `phase` → `asm.pace.phase`, `grasp` → `asm.grasp`,
  `n_cycles` → `asm.n_cycles`, `cc_on/cl_on` → `asm.*`; `UR_JOINTS`/`translation_matrix`
  imported directly.
- **`urlab/domain.py`**: `CelebrateSpec` (all eight keys, shipped defaults) wired into
  `legacy_translate` + `BncSpec`; `BncAssembly` attribute bag introduced — the run's shared
  state, one attribute per closure-web strand, grown as behaviours move out.
- **`bnc_assembly.py`**: `spec`/`asm` construction hoisted to the top of `build_and_run`
  (BncSpec.from_config is pure — no trace change); the bag is fed where each piece is built
  (`asm.pace`, `asm.frames`, `asm.guard`, `asm.grasp`, `asm.n_cycles` after the one-cycle
  clamp, `asm.cc_on/cl_on`); call site is `bnc_skills.celebrate(asm, ...)` inside the same
  try/except ('CELEBRATE raised' stays app-side). 3793 lines.
- **Tests co-migrated**: test_cable_tag celebrate pins now read
  `inspect.getsource(bnc_skills.celebrate)` (same assertions, incl. the `asm.`-form want/cadence
  lines); the neighbouring place_after_failed_engage slice re-anchored on `def disassembly(` —
  its old end anchor was `def celebrate(`, which left the file. test_bnc_spec: celebrate
  field-equality added to the slice-2 test.

Gates: stable-14, golden 61 lines byte-identical, pyflakes clean (touched files), ast clean.

**Environment note**: gates MUST run under `./.venv/Scripts/python`. A bare `python` here can
resolve to Anaconda (numpy 1.26.4, 203-warning run) where `test_bnc_clocking_geometry` PASSES —
its baseline failure is the numpy-2.x float boundary in `_ScrewAdvance.check` — so the FAILED
set reads 13 there and the diff lies in both directions.

## Step 7 — retract family → `urlab/skills/bnc.py` (2026-08-28)

- Moved `retract_from`, `retract_along_target`, `clocking_retract` (leg() inner intact) as
  `f(asm, ...)`; `ClockingRetractSpec` added (axes + *_mm distances; behaviour converts, so the
  spec carries the SHIPPED schema, not the derived `_m` siblings the closure read);
  `asm.adm` feeds the bag; the dead `retract_m` local dropped from build_and_run (its value now
  read from `spec.run.retract_distance_mm` inside retract_from). App: 3726 lines.
- Tests co-migrated: seat-hold test's `def retract_from` pin + retry-retract index →
  skills source / `bnc_skills.retract_from(asm, ...)`; T_clk-frame invariant for
  clocking_retract re-checked against skills source (app-side tuple keeps the three
  still-unmoved maneuvers); axis_offset_base slice re-anchored on `def tug_verify_in_place(`;
  ESCAPE-tail pin → `clocking_retract(asm)`; screw_ramp slice re-anchored on `\n    decim =`
  (the deleted `retract_m` local was its end anchor). Spec equality: clocking_retract fields
  added to slice-2.
- **Root cause found for the recurring exact-match failures**: this Bash tool's heredoc
  HALVES double backslashes — `\n` inside a quoted heredoc reaches Python as the `\n` escape.
  Any patch text containing literal backslashes must be built via `chr(92)` or written to a
  script file with the Write tool instead.

Gates: stable-14 (venv), golden 61 lines byte-identical, pyflakes clean, ast clean.

## Step 8 — screw_ramp → `urlab/skills/bnc.py` (2026-08-28)

- `screw_ramp(asm, adm, ref_at, guard, v, w, ang_deg, label='')`: `caps`/`min_seg_s` read from
  `asm.pace`, `_path_time` → `domain.path_time` imported directly. Three call sites prefixed
  (`bnc_skills.screw_ramp(asm, ...)`). The now-dead `caps` and `min_seg_s` aliases dropped from
  build_and_run (`phase, seg_time = pace.phase, pace.seg_time` remains). App: 3684 lines.
- Pins: null-speed-override test reads screw_ramp from skills source (app `src` read dropped
  there — unused); arc-vs-chord test's `def screw_ramp` pin → skills; collar session's
  `res, f_done = screw_ramp(` → `bnc_skills.screw_ramp(`; the `screw_ramp(` substring pins in
  the still-app-side clocking bodies survive the rename unchanged.

Gates: stable-14 (venv), golden 61 lines byte-identical, pyflakes clean, ast clean.

## Step 9 — run_insertion → `urlab/skills/bnc.py` (2026-08-28)

- `run_insertion(asm, adm_ctl, refs, T_tool0_conn, peck=..., ...)` moved whole: guard/settle/
  hold defaults now come from the bag (`asm.guard`, new `asm.settle_s`/`asm.hold_s`/`asm.tare`),
  decim + peck numbers read from `asm.spec` (run.log_decimation, collection.peck_*), observation
  logging through `skills.estimate.observe`, its retract through the module-local
  `retract_from`. Two call sites → `bnc_skills.run_insertion(asm, ...)`.
- Dead names dropped from the app: `_retract_ref` import, `hold_shared` local, `s_ret` alias
  (the `retract=True` signature pin re-pointed at skills source; the wiggle-observation slice
  test reads estimator_eval.py, not bnc — no change). App: 3616 lines.

Gates: stable-14 (venv), golden 61 lines byte-identical, pyflakes clean, ast clean.
**This step first physically relocates servo-loop code → HARDWARE H1 checkpoint due at the
next bench session (dry-run cannot validate force/servo behaviour).**

## Step 10 — leaf helpers → `urlab/skills/bnc.py` (2026-08-28)

- Moved `phase_gate`, `end_reset_with_snapshot`, `axis_offset_base`, `place_scatter`,
  `aligned_place_pose` as `f(asm, ...)`. New `ReorientRecoverySpec` (place_offsets kept as a
  raw mapping so the read-site defaults stay where they were; snap_to_ground). Bag grows
  gates_on / out_dir / ground_z / place_rng. The `gates_on = ...` app line stays VERBATIM
  (test-pinned) with `asm.gates_on` fed on the next line. axis_offset_base now derives
  cl_axis_off from `spec.collar_clocking.axis_offset_mm` itself; R_clock read from
  `asm.frames.R_clock`. skills imports fixed to the real homes: `..skills.reset` (not
  apps.reset) and `..skills.pick.connector_axis_height_m` (now unused in the app import and
  dropped there). App: 3499 lines.
- Pins co-migrated: def-count + body slices (place_scatter/aligned_place_pose/axis_offset_base)
  and the dry-run-gate line → skills source; scatter call-count and the ESCAPE/collar
  `phase_gate(asm, ...` / `axis_offset_base(asm)` call texts updated; app-side substring pins
  (`aligned_place_pose(` ≥ 3, reorient slice, place_after_failed_engage markers) survive the
  rename untouched. Baseline-failing aligned-place test still fails for its ORIGINAL reason
  (45° tuning-pin drift), not an anchor error.
- Recovery note: a partial run left domain updated and cut bodies lost with the app written —
  reconstructed the five bodies from the session's own reads. Cut lookahead standardized on
  `(?=\n    \S)`.

Gates: stable-14 (venv), golden 61 lines byte-identical, pyflakes clean, ast clean.

## Step 11 — tug_verify_in_place → `urlab/skills/bnc.py` (2026-08-28)

- NEW EXTRACTION PATTERN for the big bodies: the function moves VERBATIM behind a
  locals-at-head block (`robot, cfg, frames_t = asm.robot, asm.cfg, asm.frames`; controllers,
  pace aliases, and the tv_* numbers re-derived from `asm.spec.tug_verify`). Body text —
  and therefore nearly every source pin — survives unchanged. Its internal escape call went
  back to the module-local `clocking_retract(asm, ...)`.
- Bag grows `adm_tug` (None when tug is off, as before) and `clock_rows` (the clocking.csv
  list, now shared through the bag). Dead app locals dropped (tv_thresh_m, tv_extract_m, the
  adm_tug alias); tv_force/tv_time stay — the collar gate text still uses them. App: 3391 lines.
- Pins: both tug body slices (`def tug_verify_in_place(` → `def engage_insertion(`) re-anchored
  at skills source ending `def celebrate(`; the T_clk frame-invariant tuple keeps cc/cl
  app-side with tug checked in skills; the order/banned/terminated pins inside the body pass
  untouched thanks to the locals-at-head form.

Gates: stable-14 (venv), golden 61 lines byte-identical, pyflakes clean, ast clean.

## Step 12 — the in-hand belief onto the bag + place_after_failed_engage → skills (2026-08-28)

- **`asm.T_ftip_conn`**: the mutable fingertip→connector belief swept onto the bag (22 sites —
  the initial build, the per-cycle nominal reset, the estimator correction, and every reader).
  `T_ftip_conn_nominal` / `_catalogue` keep their names (test-pinned).
- `place_after_failed_engage(asm, miss_ref, T_tool0_conn_held)` moved verbatim behind
  locals-at-head (fail_retract numbers from `spec.engage.fail_retract`, place clearances from
  `spec.disassembly.place`, `_guarded = guarded` alias keeps the body text); its
  retract/aligned-place calls now module-local. Bag grows no_prompts / q_pick (fed at the two
  pick-pose assignments). Dead fail_retract locals dropped. App: 3283 lines.
- Pins: the recovery-body slice re-anchored at skills source (end `def celebrate(`); every
  in-body marker (retract-first ordering, return-counts, input-before-motion, no-phase_gate)
  passes untouched.

Gates: stable-14 (venv), golden 61 lines byte-identical, pyflakes clean, ast clean.

## Step 13 — both clockings → `urlab/skills/bnc.py` (2026-08-28)

- `connector_clocking(asm)` (~230 lines) and `collar_clocking(asm, T_base_conn, screw_deg=None,
  cl_rot=None, sp_on=None, unlocking=False)` (~670) moved VERBATIM behind locals-at-head; the
  def-time defaults `cl_rot=cl_rot, sp_on=sp_on` became None-sentinels resolved in the head from
  spec (explicit callers unchanged). Sweep geometry stays validated app-side (its error paths
  return False pre-motion, and the startup log reads it) and reaches the behaviour as
  `asm.cc_geometry`; bag also grows adm_cc/guard_cc/adm_cl/guard_cl/guard_push and
  `asm.T_tool0_conn_engaged` (fed at the cc call site — the anchor disassembly's unlock uses).
- App dropped ~20 dead cc_*/cl_*/sp_* locals; their WHY-comment blocks (grasp clock, retract vs
  retreat, wall standoff, tare-off, wrist-3 margin) moved into the skills heads. Imports
  trimmed; `_ScrewAdvance`/`_wrap_near`/`_fit_turn` survive as explicit re-export aliases
  because the suite still imports them from the app module. App: 2348 lines; skills: 1654.
- Pins: engage-body slices re-anchored on `def locate_target_visually(`; all cc/cl body slices
  (six collar→traj_ref, two cc→cl pairs each for body/cable, the tilt-gate knob, the
  stopped-binding count, the wrap-branch ban) re-anchored at skills source; the T_clk
  frame-invariant checks now all run against skills. Baseline clocking tests still fail on
  their ORIGINAL KeyError ('initial_connector_frame' config drift), not on anchors.

Gates: stable-14 (venv), golden 61 lines byte-identical, pyflakes clean, ast clean.

## Step 14 — disassembly → `urlab/skills/bnc.py` (2026-08-28)

- `disassembly(asm, state, screw_deg, T_base_conn_d)` verbatim behind locals-at-head (dis_*/cl_*
  re-derived from spec; collar/place/scatter/gate calls module-local). Dead app locals dropped;
  `_retreat_state` kept as a re-export alias (test-pinned import). App: 2175 lines.
- Pins: the celebrate-ordering call-site pin → `bnc_skills.disassembly(`; the gate-name roster
  and place-count/scatter/fallback pins now read the app+skills union (the gates and the
  scattered place live in the behaviours).

Gates: stable-14 (venv), golden 61 lines byte-identical, pyflakes clean, ast clean.

## Step 15 — locate_target_visually + reorient_recovery → skills (2026-08-28)

- `locate_target_visually(asm, q_return)` behind a head reading the new **VisualTargetSpec**
  (servo_refine / view_joints_deg / return_home_after / max_shift_mm/_deg; None disables the
  shift gate, as before); `reorient_recovery(asm)` behind a head reading the extended
  **ReorientRecoverySpec** (enabled, square fingertip_in_connector override,
  release_clearance_mm, approach_mm, settle_s). Bag grows the pick stack
  (scanner/geom/check/recovery/recorder/confirm), targets/tname/vt_rig and
  `asm.T_ftip_conn_catalogue`. The visual-source and rig validation stays app-side
  (parse-time, pre-motion). App: 1966 lines; skills: 2088.
- Pins: eng-slices re-anchored on the `# ---- RESET + PICK` flow comment (their old end anchor
  moved); reorient/settle/catalogue/T_conn-reuse pins → skills source; the visual-target
  call-site and function-extent pins updated; the place-recovery slice tightened to end at
  `def locate_target_visually(` (locate/reorient now sit between it and celebrate, and they
  legitimately use phase_gate). Spec equality: visual_target + reorient fields added.

Gates: stable-14 (venv), golden 61 lines byte-identical, pyflakes clean, ast clean.

## Step 16 — engage_insertion (+ traj_ref + the engage report helpers) → skills (2026-08-28)

- **engage_insertion(asm)** (~400 lines) verbatim behind locals-at-head: everything numeric
  re-derived from `spec.engage` (contact/confirm nested specs included); the WIGGLE RIG stays
  built and validated app-side (its error paths return False pre-motion) and reaches the
  behaviour as `asm.engage_rig` = (wiggle, amp, frq, scale, pv, pw); adm_en/guard_en on the
  bag. **traj_ref(asm, row, T_tool0_conn, commit=False)** promoted from a flow-nested def —
  the anchor arithmetic every reference goes through. `asm.dense` and the rolling IK seed
  `asm.seed_q` (7-site sweep) joined the bag.
- The module-level report machinery (DIM_KEYS, ENGAGE_TRACE_COLS, _save_engage_trace,
  _report_engage_trace, _num, _engage_report) moved to skills/bnc.py whole; the app keeps
  re-export aliases for the four names tests import from it. App: 1405 lines; skills: 2675.
- Pins: three engage-body slices → skills (`def engage_insertion(` → `def disassembly(`);
  the report test's log capture rewired to the SKILLS logger (the report now speaks as
  'bnc-behaviors'); anchor/commit-count pins updated to the `bnc_skills.traj_ref(asm, ...`
  call texts; engage-speed and REPORTED-ONLY pins split between skills (mechanism) and app
  (the RETIRED-key validation, which stays pre-motion in the app).

Gates: stable-14 EXACT set diff (venv), golden 61 lines byte-identical, pyflakes clean, ast
clean. Servo-critical code moved again — the H1 bench checkpoint covers this step too.

## Step 17 — setup → `skills.setup(asm)`; build_and_run IS the algorithm now (2026-08-28)

- The entire construction/validation half of build_and_run (~620 lines: spec-side validation
  with pre-motion `return False` paths, controllers, estimator, scanner stack, wiggle rig,
  target anchoring, out_dir/RNGs) moved VERBATIM into `setup(asm)` in skills/bnc.py. Its head
  binds `log = urlog.get('bnc-assembly')` so the startup narration keeps the app's voice —
  the golden trace pins those lines under that logger name. A publish block at its tail puts
  the last flow-shared objects on the bag (estimator/commit, the commit controller pair,
  noise_rng, col_mode/sweep_offsets, live_path, mats, debug_match).
- build_and_run now reads: construct the bag → `bnc_skills.setup(asm)` → a single
  "tuning read ONCE" block (bag unpack + plain spec derivations) → the reset/pick/assemble
  ALGORITHM (~600 lines of flow) → the end reset. **App: 822 lines.** skills/bnc.py: 3300.
  Dead app imports pruned; `_clocking_plan`/`_compliance_override` kept as re-export aliases
  (test-pinned imports).
- Pins: ~16 setup-content pins re-anchored by UNIONING the app+skills sources at their read
  sites (supersets — every flow pin still resolves); the two validated-before-motion ordering
  pins now assert the text lives in setup and `bnc_skills.setup(asm)` precedes the first
  reset; the catalogue-belief and RETIRED-key pins follow their code. A first-match deletion
  briefly stripped three lines from engage's head (same text as setup's copies) — restored;
  lesson: dedupe deletions must anchor on the LAST occurrence when setup shares derivation
  text with a behaviour head.

Gates: stable-14 EXACT (venv), golden 61 lines byte-identical, pyflakes clean, ast clean.

## Step 18 — the re-export alias block retired (2026-08-28)

- Every test that imported moved names from `urlab.apps.bnc_assembly` now imports the real
  home: `urlab.domain` (CLOCK_STATES/UNCLOCK_STATES, advance/retreat_state, clocking_plan,
  compliance_override, path_time, wrap_near, fit_turn), `urlab.robot.detectors`
  (AnyGuard, ScrewAdvance), `urlab.skills.bnc` (_engage_report, ENGAGE_TRACE_COLS,
  _report_engage_trace). `insertion_tester.py` (a donor app) repointed its detector import at
  robot.detectors, and the shared-owner test pin now names `.robot.detectors` as the owner.
  The whole alias block and its supporting imports left the app. Six dead `src =` reads left
  behind by earlier pin unions removed from the tests. **App: 804 lines.**

Gates: stable-14 (venv), golden 61 lines byte-identical, pyflakes clean, ast clean.

## Milestone — the code port is COMPLETE (2026-08-28)

`apps/bnc_assembly.py` 4,327 -> 804 lines: the algorithm only. `urlab/skills/bnc.py` holds
setup + all 20 behaviours (~3,300 lines); `urlab/domain.py` the specs/bag/math (684);
`urlab/robot/detectors.py` the termination detectors. Docs: `docs/bnc_assembly.md` (as-built
architecture) written; `docs/refactor_plan.md` marked executed-with-no-trees.

Remaining, deliberately deferred:
- **YAML schema swap** (replace the legacy `assembly.…` schema with per-behaviour top-level
  blocks; verify `legacy_translate(old) == load(new)`; migrate the ~36 config-path tests;
  generate docs/bnc_config_migration.md; delete the translator). Deferred pending a word with
  the operator: configs/bnc_assembly.yaml is actively edited on the robot machine and a schema
  swap mid-flight would collide with those merges.
- **Bench checkpoints H1–H4**: engage/insertion/clocking/retract code physically moved;
  dry-run cannot validate force control. Run the standard bench validation before the next
  hardware campaign.

## Step 19 — the YAML schema swap (2026-08-28)

Three gated phases:

1. **Spec completion**: `TrajectorySpec` (csv/angles_deg/resolutions/standoff) and the
   run-level keys (target_frame, gate_between_behaviors, live_plot,
   release_retract_distance_mm, debug_match) joined the spec; every remaining raw
   `assembly.` read in code became a spec read (still driving the OLD file through the
   translator). Gated: stable-14, golden byte-identical.
2. **The swap**: configs/bnc_assembly.yaml's `assembly:` block dissolved into per-behaviour
   top-level blocks mirroring BncSpec (run:/trajectory: regroup the flats;
   success_tolerance.{pos_mm,rot_deg} → run.success_{pos,rot}_*; the flat
   stiffness/mass/damping and guard override keys nest under compliance:/force_guard: in
   their five blocks), all comments riding along. `BncSpec.from_config` now reads the blocks
   DIRECTLY through `strip_derived_units` (drops the units-normaliser's derived SI siblings
   and explicit nulls — null still means inherit/default, exactly as the translator's
   `_present` behaved). **Proof**: the parsed BncSpec is `==` to the pre-swap snapshot.
   setup() refuses a config still carrying `assembly:` (points at
   docs/bnc_config_migration.md). The characterization harness drives
   `--set run.target_source=kinematic`; the golden stayed byte-identical — no refresh needed.
3. **Test + prose migration**: ~45 test config-reads re-pointed at the new paths (nested
   override reads included; the estimator_eval parity test flattens bnc's nested overrides
   before comparing; `widest_cable_leg`/`_resolved_wiggle` take the new shapes);
   `legacy_translate`/`_present`/`_split_overrides` DELETED; tests/test_bnc_spec.py rewritten
   against the new schema (field-for-field load, raw-yaml agreement, unknown-key dotted-path,
   strip_derived_units unit test, defaults-on-dataclass); 46 in-code `assembly.X` message
   mentions rewritten to the new paths. docs/bnc_config_migration.md generated;
   docs/bnc_assembly.md updated.

Gates: stable-14 EXACT set diff (venv), golden 61 lines byte-identical, pyflakes clean
(pre-existing noise only), ast clean. domain.py 616 lines. THE REFACTOR PLAN IS NOW FULLY
EXECUTED — remaining work is bench validation (H1–H4) only.
