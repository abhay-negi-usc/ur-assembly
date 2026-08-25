# Refactor plan — behaviours, objects, trees

*Drafted 2026-08-24. Written to be read cold: if you are an assistant or a new engineer picking
this up with no prior context, everything you need to start is in this file. A working plan, not a
spec — where you disagree, change it and keep the reasoning visible.*

---

## 0. Orientation — what this repo is

`ur-assembly` drives a **UR10e arm with a Robotiq 2F-85 gripper and a wrist-mounted RealSense D405**
to do **cable assembly**: find a cable lying on a bench, pick it up by its connector, and mate that
connector into a fixed socket. The flagship task is a **BNC** connector, which is a bayonet fitting
— it must be pushed in, rotated to catch its pins, then have a locking collar twisted.

### Domain glossary (you will not understand the code without this)

| term | meaning |
|---|---|
| **connector** | the rigid plug on the end of the cable — the thing that gets grasped and mated |
| **junction** | where the thin cable meets the fat connector. The detector finds THIS point; it is the origin of the estimated pose |
| **socket / target** | the fixed receptacle the connector mates into. Its pose is recorded in `configs/frames.yaml` under `targets:` |
| **engage** | drive the connector into the socket until it meets resistance. Ends the moment axial force is met — that is NORMAL, not a failure |
| **clocking** | rotating the connector to catch the bayonet pins ("connector clocking"), then rotating the locking **collar** ("collar clocking") |
| **mate** | the recorded pose where connector and socket are fully joined |
| **belief** | where the code *thinks* the connector is relative to the fingers, after grasping it. Distinct from where it was *commanded* to be |
| **fingertip** | the frame at the gripper pads, 183 mm along tool0 +Z. Most grasp geometry is expressed here |
| **tool0** | the robot flange frame — the UR's own tool frame with a zero TCP offset |
| **state machine** | `engaged → seated → locked`. Engage produces *engaged*, connector clocking *seated*, collar clocking *locked* |

### Repo map

```
urlab/
  apps/         28 files — entry points. `python -m urlab.apps.bnc_assembly`
                bnc_assembly.py is the flagship (2995-line build_and_run — the refactor target)
  skills/       17 files — reusable task logic: pick, scan, ground_pick, manifold, trajectory
  robot/         8 files — hardware: arm (RTDE), gripper (Modbus), robot (frames), guard,
                           admittance, collision (pybullet ground check), gripper_kinematics
  perception/   11 files — camera, aruco, sam3 wrapper, junction geometry, cable_trace_graph,
                           connector/cable_recon (multi-view fusion), tag (coloured markers)
configs/        22 YAML. One per app + shared: cables.yaml, frames.yaml, _common.yaml
tests/          test_smoke.py, test_behaviors.py, test_junction.py, test_cable_tag.py
docs/           13 per-app docs (none for bnc_assembly) + this file
analysis/       offline studies + "BNC Assembly Failure Modes.xlsx" (the behaviour list this
                refactor is meant to make real)
```

**External dependency worth knowing:** SAM3 segmentation comes from an installed `sam3` package
(model + weights). The *geometry* on top of its masks was vendored into `urlab/perception/` on
2026-08-24 and is no longer in a sibling checkout.

---

## 1. Objectives

In priority order. Every proposal below serves one of these; if it serves none, drop it.

1. **A behaviour is a callable that reports why it ended.** One function per row of the
   failure-modes spreadsheet, returning a typed `Outcome`. Today none of them can be called
   independently.
2. **A task is a behaviour tree.** An app becomes ~20 lines of tree construction over shared
   leaves, so a new cable variant (BNC vs banana) is a different tree, not a new 3000-line script.
3. **The things being manipulated are objects.** A cable knows its own dimensions and what its
   grasp counts mean; today it is dissolved into 13 config paths.
4. **Failures are trackable.** Each behaviour names its terminating condition, so the spreadsheet's
   *Failure Mode* and *Num. Occurrences* columns can be filled from logs rather than by hand.
5. **A newcomer can adapt it.** Typos in config fail loudly, dry runs are geometrically real, and
   safety guards cannot silently disable themselves.

---

## 2. How to work on this

### Verify every change

```bash
python -m pytest tests/ -q                 # the suite
python -m pyflakes urlab/ tests/           # undefined names, unused imports
python -c "import ast,sys;[ast.parse(open(f,encoding='utf-8').read()) for f in sys.argv[1:]]" FILE
```

**Always confirm a file still parses after a scripted edit.** On 2026-08-24 a patch applied twice,
corrupted `urlab/robot/collision.py` into an `IndentationError`, and — because `skills/pick.py`
caught bare `Exception` and downgraded to a warning — the ground-collision guard silently switched
itself off and the arm drove a path into the floor. A two-second `ast.parse` would have caught it.

### The test baseline — 11 KNOWN failures. Do not chase them.

As of 2026-08-24 the suite is **199 passed, 6 skipped, 11 failed**. These 11 pre-date this work and
are unrelated to it (mostly `bnc_assembly.yaml` tuning values that drifted from what tests pin):

```
test_behaviors.py::test_commanded_speeds_are_clamped_to_the_controller_ceiling
test_behaviors.py::test_the_aligned_place_pose_is_flat_and_on_the_socket_heading
test_smoke.py::test_bnc_assembly_shares_the_tuned_estimator
test_smoke.py::test_tool_frames_shared_yaml_source
test_smoke.py::test_preload_force_is_a_spike_not_a_press
test_smoke.py::test_a_clocking_stroke_follows_the_arc_and_not_the_chord
test_smoke.py::test_the_target_and_the_belief_describe_the_same_connector_datum
test_smoke.py::test_wiggle_station_grid_and_the_retuned_excitation_stay_runnable
test_smoke.py::test_the_collar_is_grasped_axially_and_turned_by_a_wrist_twist
test_smoke.py::test_the_connector_sweep_rocks_between_absolute_roll_positions
test_smoke.py::test_payload_and_joint_acceleration_are_fleet_wide_constants
```

**Before and after any change, diff the FAILED list — not the count.** A refactor is clean when the
set is unchanged. (They are worth fixing eventually, but separately: each is a real assertion about
tuning, and silencing them would lose the assertion.)

### House style — match it, it is deliberate

- **Comments say WHY, not what.** They record bench facts that cost real time: a force-vs-speed
  table, why a limit is axial rather than a magnitude, a frame sign error that put a grasp 91.4 mm
  out. **Never delete these to "tidy up."** Removing only literal duplicates is fine.
- Docstrings lead with the consequence, then the mechanism.
- Units live in key names: `xyz_mm` / `rpy_deg` vs `xyz` / `rpy` (m/rad). Never both for one triple.
- Frames: rpy is **extrinsic XYZ**; quaternions are `[x,y,z,w]`; RTDE speaks rotation *vectors* and
  the UR `base` frame, which differs from ROS `base_link` by 180° about Z (`BASE_LINK_FROM_UR_BASE`).
  Those conversions live in `urlab/transforms.py` and **nowhere else** — keep it that way.

### Ground rules

- **Never `git commit` without asking.** Stage, summarise, then ask.
- Hardware behaviour changes cannot be tested here (no robot, no torch/CUDA, no `cv2` in the default
  env). Say plainly what was verified and what was not.
- `pybullet` IS installed and the collision/FK tests really run. `cv2` is not — cv2-dependent tests
  use `pytest.importorskip('cv2')`.

### Start here — the first task, with acceptance criteria

**Extract `Cable`** (§5.1). It is pure data plus arithmetic, needs no robot, and is fully
unit-testable today.

Done when:
1. `urlab/domain/cable.py` defines `Cable` with `load`, `grasp_band`, `axis_height_m`,
   `held_belief`, `classify`.
2. `Robotiq2F85.grasp_result` is gone; call sites use `cable.classify(gripper.position())`.
3. `apply_cable_profile` still writes the same 13 config paths (compatibility shim) so nothing else
   breaks yet.
4. New tests cover: the count band derived from diameters; `classify` returning
   `ok`/`missed`/`empty` at the band edges; a cable with no tag.
5. The FAILED set is still exactly the 11 above.

---

## 3. Why — the problem, measured

Not opinion; these were counted on 2026-08-24.

| | |
|---|---|
| `bnc_assembly.build_and_run` | **2995 lines, 21 nested closures, 0 independently callable** |
| guarded free-space moves in that function | 16 |
| admittance `ramp` / `hold` / `warmup`+tare sites | 10 / 8 / 11 |
| `phase_gate` sites | 11 |
| config reads with a silent default (`.get`, `get_path`) | **1175** |
| YAML configs / largest app config | 22 / **1414 lines** (`bnc_assembly.yaml`) |
| per-app docs | 13 — **none for `bnc_assembly`**, the flagship app |
| config paths one cable is splattered across | **13**, in 3 unrelated namespaces |
| collision-check call sites outside `skills/pick.py` | **0** |

Two sentences of diagnosis:

**The structure does not match the reasoning.** The comments in this codebase are unusually good —
they describe discrete behaviours with named termination conditions and explain *why* each limit is
what it is. The code is one function with 21 closures. Most of this refactor is making the
structure say what the comments already say.

**Nothing can be tested.** There are zero unit tests on any assemble behaviour. That is why the
collar-clocking geometry bugs of 2026-08-24 were only findable on the bench, and why a corrupted
`collision.py` could silently disable the ground guard and drive the arm at the floor.

---

## 4. Target architecture

```
L4  trees        py_trees composites; an app IS its tree (~20 lines)
L3  behaviours   one callable per row of the failure-modes sheet; returns Outcome
L2  templates    CompliantDrive · GuardedMove   (the shapes the code actually repeats)
L1  primitives   guarded_move ✓ · admittance.ramp ✓ · screw_ramp (promote from closure)
L0  domain       Robot · Tool · Cable · Assembly · Socket
```

Build **L0 first**, then L2/L3, then L4. L1 already exists.

---

## 5. L0 — domain objects

### 5.1 `Cable` — the biggest single win

`config.apply_cable_profile()` currently **dissolves** a cable across 13 config paths
(`gripper.*_counts`, `grasp_check.*`, `junction_in_fingertip`, `cable_tag.color`). Afterwards
nothing downstream can ask *which cable is this* — it sees loose numbers.

The smoking gun is `Robotiq2F85.grasp_result(groove_counts, empty_counts, faces_max_counts,
tolerance, detect_empty, groove_max_counts)`: **six parameters, every one cable-derived.** The
gripper is doing the cable's job with the cable's data.

```python
@dataclass(frozen=True)
class Cable:
    name: str
    connector_diameter_mm: tuple[float, float]
    cable_diameter_mm: float
    junction_in_fingertip: np.ndarray        # T_fingertip_junction
    tag: TagMatcher                          # coloured marker, or disabled

    @classmethod
    def load(cls, name, cables_yaml) -> 'Cable': ...

    def grasp_band(self, gripper) -> CountBand      # via gripper kinematics + groove depth
    def axis_height_m(self) -> float                # half its greatest diameter
    def held_belief(self, T_conn_ftip) -> np.ndarray
    def classify(self, measured_counts) -> Literal['ok', 'missed', 'empty']
```

Then `gripper.close(); cable.classify(gripper.position())` reads as what it is: the gripper reports
counts, the **cable** knows what those counts mean about itself.

Keep `apply_cable_profile` as a compatibility shim writing the same paths until call sites migrate.

### 5.2 `Tool` — the stack bolted to tool0

One physical assembly currently described in four places: `frames.yaml` (`fingertip`),
`robot.payload` (mass/CoG — what FT taring depends on), `collision.ToolModel` (spacer/body/finger
geometry), `tool_frames.py`.

```python
class Tool:
    fingertip: np.ndarray        # T_tool0_fingertip
    payload: Payload
    collision: ToolModel
    gripper: Robotiq2F85
```

### 5.3 `Assembly` + `ClockState` — a home for the state machine

`state` is a bare local string plus module-level `_advance_state` / `_retreat_state`. It has real
invariants (engaged → seated → locked, no skipping) and deserves an owner.

```python
class Assembly:
    cable: Cable
    socket: Socket               # recorded mate pose + frame
    trajectory: Trajectory
    state: ClockState
    def advance(self, expected): ...   # asserts we are where we think, then advances
    def retreat(self, expected): ...
```

### 5.4 `Robot` — small additions only

`Robot` is **already a good object** (28 methods: `pose`, `tool0`, `camera`, `fingertip`,
`move_frame`, `register_frame`, `target`). Do not churn it.

- Add properties where a method is really an attribute read: `robot.q`, `robot.wrench`.
- Add `robot.tool` (3.2).
- Keep every `move_*` a **method** — they have side effects and can fail. A property that moves a
  robot is a trap.

---

## 6. The three types everything else hangs off

```python
@dataclass(frozen=True)
class Outcome:
    behavior: str
    status: str        # complete | limit | guard | refused | aborted | failed
    ended_by: str      # "axial force 5.1 N >= 5.0 N for 0.10 s"
    conditions: dict   # EVERY condition's value vs its own limit at the stop
    ok: bool

class TerminationSet:              # generalises _AnyGuard
    def check(self) -> str | None  # name of the condition that fired
    def snapshot(self) -> dict     # what _engage_report prints — for ANY behaviour

@dataclass
class RunContext:
    robot: Robot            # owns arm, tool, gripper, camera, frames
    assembly: Assembly      # owns cable, socket, trajectory, state
    cfg: Config             # tuning only, never identity
```

`RunContext` is the enabler: the 21 closures exist because ~100 locals have nowhere to live, and
several are **mutated** across closures (`state`, `T_ftip_conn`, IK seeds). Making that explicit is
the precondition for extracting anything.

`Outcome` pays for three things at once: per-behaviour failure tracking (the sheet's *Failure Mode*
column becomes `Outcome.ended_by`), the "print what terminated each behaviour" requirement, and
structured CSV rows for occurrence counting.

**Make `Outcome` the ONLY failure vocabulary.** Today there are four: `return False`, raise,
log-and-continue, and status strings (`'ok'|'missed'|'empty'|'unreachable'`).

---

## 7. L2 — the one abstraction the code has earned

Every compliant behaviour is the same shape: **a reference path `r(t)`, a set of termination
conditions each with its own limit, and a report naming which fired.** Three-quarters of it already
exists — guards are `ForceGuard`-shaped (`check`/`reset`/`tripped_by`), `_AnyGuard` ORs them,
`_engage_report` prints all conditions against their own limits. It is just trapped inside engage.

```python
CompliantDrive(ref_fn, terminations: TerminationSet, admittance) -> Outcome
GuardedMove(target, guard, interp, collision='check'|'exempt') -> Outcome
```

`CompliantDrive` absorbs all **10** ramp sites. Engage, connector clocking, collar clocking and tug
become *a reference function plus a termination list* — the parts that genuinely differ — instead
of four hand-rolled servo loops.

> **Do not build `ApproachActRetract` yet.** The standoff → align → act → retract skeleton does
> recur (pickup, collar clocking, disassembly, place), but the "act" differs enough that the
> template would grow a dozen hooks. Extract the four concrete sequences first; if they still look
> alike afterwards the abstraction will be obvious and correctly shaped. A leaky base class is
> harder to remove than the duplication it replaced.

---

## 8. L3 — behaviour catalogue

One callable per row of `analysis/BNC Assembly Failure Modes.xlsx`. **Bold = does not exist as a
function today.**

| task | behaviours |
|---|---|
| Estimate Target | `sweep_markers` · `servo_refine` · `fuse_target_pose` · **`anchor_target`** |
| Estimate Cable | `capture_view` · `detect_cables` · **`score_tags`** · **`select_cable`** · **`project_to_ground`** · **`lift_to_axis`** |
| Pick | `move_standoff` · `align_fingertip` · `close_on_connector` · **`classify_grasp`** · **`reseat_retry`** · `lift_verified` |
| Re-place | `move_pre_place` · `place` · `open_gripper` · `return_to_view` |
| Assemble | `engage` · `clock_connector` · `clock_collar` · `tug_verify` · `release_escape` · `clocking_retract` |
| Disassemble | 10 rows, currently **one** function |
| Cross-cutting | **`verify_collision_model`** · **`check_path_clear`** · `phase_gate` |

### Corrections to the sheet (2026-08-24 review)

- **Row 22 "Clock connector" duplicates row 20.** There is exactly one `connector_clocking()` call;
  it has an internal retry loop, not two behaviours. Row 21 "Push in connector" is really
  `engage_insertion()` = row 19.
- **Row 26** says "Close gripper on connector" during collar clocking — it closes on the **collar**.
- Row 15 typo: "standoff pose for placement place".

### Sub-behaviours

**Prewind is a child of clocking, not a peer.** `_wrist3_window` / `_fit_turn` are a *precondition*
of the turn — they decide whether the requested rotation fits the wrist's remaining range and refuse
before anything moves.

```
ClockCollar
├── PrewindWrist3(target_rotation)   ← precondition; refuses with the arm parked
├── ApproachCollar
├── GraspCollar
└── TurnCollar
```

One parameterised `PrewindWrist3` leaf, reused by `ClockConnector`, `ClockCollar` and the
disassembly unclock (sign flipped). Keep the existing **second** prewind check after the advance and
before the fingers close — it is deliberate, because at that point the fingers are still open and
nothing is clamped.

---

## 9. L4 — an app is its tree

```python
def bnc_tree(ctx):
    return Sequence("bnc assembly", memory=True, children=[
        LocateTarget(ctx),
        Retry(PickCable(ctx), attempts=3, fallback=ReorientRecovery(ctx)),
        Sequence("mate", children=[
            Engage(ctx), ClockConnector(ctx), ClockCollar(ctx), TugVerify(ctx)]),
        ReleaseEscape(ctx),
    ])

def banana_tree(ctx):                      # same leaves, fewer of them
    return Sequence("banana assembly", memory=True, children=[
        LocateTarget(ctx), Retry(PickCable(ctx), attempts=3),
        Engage(ctx), ReleaseEscape(ctx)])   # no clocking, no collar
```

Three decisions taken deliberately:

1. **Blocking leaves first, not `RUNNING`.** py_trees assumes tick-based semantics, but these
   behaviours block for seconds (engage runs ~14 s in one `while` loop). True async leaves would buy
   preemption and mid-motion abort, but require restructuring every servo loop into a state machine.
   Start blocking; convert individual leaves later where preemption actually matters.
2. **Safety as decorators, not leaves.** `phase_gate`, collision checks and prewind must not be
   children that can be forgotten. `@collision_checked`, `@gated("ENGAGE")`, or a guard subtree — so
   the policy is declared once and is structurally impossible to omit.
3. **The blackboard IS the `RunContext`.** Do not invent a second state mechanism.

Payoff for failure-mode tracking: the tree is *data*. Each tick logs leaf + `Outcome`, so occurrence
counting is a log query, and py_trees renders the tree to DOT — a picture of the decomposition the
spreadsheet describes in words.

---

## 10. Migration order — strangler fig, never a rewrite

Each step leaves the app runnable and adds tests. **Write a characterization test before each
extraction** — there is currently no safety net.

1. **`Cable`** + move `grasp_result` → `Cable.classify`. Pure data + arithmetic, fully unit-testable
   with no robot. Keep `apply_cable_profile` as a shim. *Worth doing even if you stop here.*
2. **`Tool`**, **`Assembly`** + `ClockState`.
3. **`RunContext`** — now trivial, it is just (1) and (2).
4. **`Outcome` + `TerminationSet`**; move `_engage_report` onto `TerminationSet.snapshot()`. Engage
   keeps working; every other behaviour gains the report for free.
5. **Extract leaf closures** that are already near-pure: `aligned_place_pose`, `axis_offset_base`,
   `retract_from`, `release_escape`, `clocking_retract`.
6. **`engage_insertion` → `CompliantDrive`.** Best understood, already has the report.
7. Then `connector_clocking`, then `collar_clocking` (~670 lines), then `disassembly`.
8. Delete the closures; `build_and_run` becomes a sequence.
9. **py_trees adapter** — thin, once 1–8 are done.

---

## 11. Cross-cutting fixes

### 11.1 Collision coverage is NOT universal — findings of 2026-08-24

Every collision call site is in `skills/pick.py`. Coverage by motion type:

| motion | checked? |
|---|---|
| `move_j` | **only** via `GraspController.align/descend/lift` |
| `move_l`, `move_to`, `move_frame_to` | **no** |
| `servo_j` / `servo_l` (compliance) | **no** |
| `force_mode` | **no** |
| all 16 guarded moves in `bnc_assembly.py` | **no** |

Some of that is defensible — `check_path` samples a joint interpolation, meaningless for a 125 Hz
servo reference, and contact behaviours *intend* to touch the socket. But it is **implicit**: nothing
says "these are deliberately unchecked", so a reader reasonably assumes they are.

**Make coverage declared, not inherited.** `GuardedMove(collision='check'|'exempt', exempt_reason=…)`
— exemption requires a written reason, so it is a decision on the record (the way
`fingertip_margin_mm` is documented rather than hidden). Free-space moves default to `check`. Report
coverage at startup: *"12 behaviours checked, 6 exempt (reasons: …)"*.

**Sampling density.** `path_samples: 25`. On a large reconfigure that is ~168 mm of tool travel
between consecutive checks. The floor is a half-space, so nothing can tunnel *through* it — the real
risk is a dip below the plane that starts and ends between two samples. Depth varies smoothly, so
the dangerous case (driving the wrist deep into the bench) spans several samples and IS caught; what
can slip through is a shallow graze on a long move.

- `margin_mm` is **0**. Raising it to ~10 mm is the free defence against under-sampling.
- Better: **adaptive sampling** — sample until max tool travel between samples is under ~20 mm, so
  density scales with the move. Still cheap (once per move, not per servo cycle).
- Report the resolution, not the sample count: *"checked at ≤20 mm tool spacing"* is auditable.

**Also note the model's scope**: ground plane + arm/tool bodies only. The socket fixture, bench
furniture and the cable itself are **not** in it. "Clear" means clear of the floor and of itself.

### 11.2 Config schema that rejects unknown keys — highest newcomer impact

**1175 silent-default reads** means a misspelled key does nothing, forever, silently. Three
instances of exactly this in one day (2026-08-24): duplicate YAML keys where the last silently won,
retired keys ignored, and a `min_fraction` → `min_pixels` rename that would have silently reverted
to a default in any config missed.

Declare each block as a dataclass with a validator that **fails on unknown keys**. It turns the
worst bug class here — *the config says one thing, the robot does another* — into a startup error
with a line number.

### 11.3 Make dry-run geometrically real

```python
def fk(self, q):
    if self.dry_run:
        return rtde_to_matrix([0.4, 0.0, 0.4, 0.0, 3.14, 0.0])   # a FIXED pose
```

Its own docstring admits *"every geometric result in a dry run is therefore meaningless."* But
`collision.fk_links` is a **validated** UR10e chain — it agrees with UR's published URDF to
0.0000 mm. Point dry-run `fk` at it and dry runs become geometrically correct: whole trees become
developable offline, and the test suite can exercise whole tasks. A handful of lines, large payoff.

### 11.4 Fail closed — never degrade silently

Hit twice on 2026-08-24: the collision model swallowed a `SyntaxError` and ran the arm unguarded on
one warning line; the ground-plane estimate has no residual check at all.

Rule with teeth: each tree declares `required_capabilities = ['collision', 'ft_sensor']`, checked at
startup, and **a guard that cannot run stops the machine.** A warning in a busy log is not a safety
mechanism. (`skills/pick.py` was changed to raise rather than warn — keep that direction.)

### 11.5 Documentation — the missing conceptual layer

13 per-app docs exist and the in-code WHY-comments are excellent, but there is no map, and the
flagship app has no doc. What a newcomer cannot get today:

- **One conceptual page**: frame conventions (superbly written inside `transforms.py`, invisible
  from outside), the data flow camera → detection → pose → grasp → assemble, the state machine.
- **Recipes**: *add a cable* (a YAML entry + a `Cable`), *add a behaviour* (a function + a leaf),
  *add a robot*.

### 11.6 `python -m urlab.doctor`

Optional deps and external checkouts are everywhere (pybullet, cv2, sam3, pymodbus, pyrealsense2,
RTDE). One command reporting what is installed, what is missing, whether the robot answers, and
whether the frame catalogue loads. The `getForwardKinematics` register failure of 2026-08-24 would
have been a one-line diagnosis.

---

## 12. What NOT to do

- **Do not make `transforms.py` object-oriented.** Pure math on matrices. A `Pose` class adds
  ceremony and an allocation per operation with no invariant to protect.
- **Do not build a `Behavior` class hierarchy** before py_trees needs it. Functions returning
  `Outcome` first.
- **Do not create a class per YAML block.** Objects belong where there is *identity and invariants*
  (a cable is a physical thing with rules about what its counts mean), not merely grouped data.
  Test: if a proposed class has no methods beyond `__init__`, it is a dataclass at best.
- **Do not strip the WHY-comments.** They are the most valuable thing in the repo — the force-vs-
  speed table, the `solvePnP` marker-size warning, the `+91.4 mm` frame-sign history. Remove only
  literal duplicates.
- **Do not rewrite `Robot`.** It is already well factored.

---

## 13. Open questions

1. **Async leaves** — is mid-motion preemption worth restructuring the servo loops, or is
   stop-on-guard sufficient? (Bench question, not a design one.)
2. **Compliance-phase collision** — `check_path` cannot model a 125 Hz reference. Live proximity per
   servo cycle, or a precomputed swept-volume envelope? Interim: `exempt` with a stated reason plus
   the force guard as the real protection.
3. **Where does `Socket` come from** — the recorded mate in `frames.yaml targets:`, or the visual
   marker estimate? Both exist (`post_engage_frame: target | believed`). The object should make the
   choice explicit rather than a config string read in three places.
4. **Something applied patches twice** across ≥6 files and broke `collision.py` outright
   (2026-08-24). Unexplained. Cleaning up the results does not stop recurrence.

---

## 14. Evidence appendix

Measured 2026-08-24, so they need not be re-derived.

- `build_and_run`: 2995 lines, 21 closures. `collar_clocking` ~670 lines; `disassembly` covers 10
  sheet rows in one function.
- Repeated shapes: 16 guarded moves · 10 ramps · 8 holds · 11 warmup/tare · 11 phase_gate ·
  2 `_AnyGuard`.
- `Robot` 28 methods / 0 properties · `URArm` 35 · `Robotiq2F85` 20.
- `apply_cable_profile` writes 13 config paths. `grasp_result` takes 6 cable-derived parameters.
- Silent config defaults: 1081 `.get(` + 94 `get_path(` = **1175**.
- Collision: all call sites in `skills/pick.py`; `path_samples: 25`; ~168 mm tool travel between
  samples on a large move; `margin_mm: 0`.
- UR10e envelope (used by the not-a-pose check): reachable ≤ **1539 mm**; link sum **1775 mm**;
  workspace diameter **3078 mm**. `fk_links` agrees with UR's URDF to **0.0000 mm**.
