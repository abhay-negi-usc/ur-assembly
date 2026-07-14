# urlab — a ROS-free UR10e + Robotiq + RealSense stack

This branch (`python-dev`) is the `dev` ROS2 workspace refactored into plain Python. The robot is
driven over **RTDE** (`ur_rtde`), the gripper over **Modbus** (`pymodbus`), and the camera over
**pyrealsense2** — no ROS graph, no `controller_manager`, no `move_group`, no tf2, no launch
files. One `pip install`, one `python -m urlab.apps.<demo>`.

The ROS packages still exist, unchanged, on `dev` and `main`. This branch replaces them; it does
not depend on them.

## Why it is smaller

The ROS version was ~6,700 lines across 14 packages, much of it plumbing: cv_bridge conversions,
PoseArray assembly, tf2 broadcasters/listeners, controller-switch choreography, `rclpy` spin
loops, per-package launch files and `package.xml`/`setup.py`. Removing ROS removed all of it. What
remained — the actual control laws, the perception math, the assembly geometry — was already pure
`numpy`/`opencv`, and is now in one importable library.

Three changes did more than relocate code; they deleted whole failure classes:

| Concern | ROS | Here | What went away |
|---|---|---|---|
| **IK** | MoveIt KDL, a local solver → random-seed retries on error -31 | `getInverseKinematics(pose, qnear)` — the controller's **analytic** solver | The retry loop; IK either has a solution or it doesn't, and you get the branch nearest where you are. |
| **Compliance** | `admittance_controller` loaded inactive, parameterised over a service, **activated** (which deactivated the JTC and caused a joint-0 velocity fault, patched with a reference-holding dance) | `forceMode()` — a mode of the running controller | The controller install, the switch, the reference streaming, and `_hold_reference` entirely. |
| **Staleness** | tf2's 10 s cache expired slowly-republished transforms; "fixed" three times by widening `tf_cache_s` | `FrameGraph` keeps the latest transform forever; the **caller** states a `max_age` | The class of bug where a good estimate fails to look up because a buffer dropped it. |

The connector fusion also became simpler: SAM3 runs **in-process**, so a detection is captured,
run, and ingested synchronously — the elaborate image-stamp attribution the ROS scan needed (to
avoid crediting a lagged detection to the wrong viewpoint) is gone.

## Install

Pick the layer you need — the requirements are split so a math/planning environment doesn't drag
in `ur_rtde` or `pyrealsense2`:

```bash
pip install -r requirements/core.txt          # math + fusion + --dry-run (no robot/camera/torch)
pip install -r requirements/robot.txt         # + drive the arm & gripper (RTDE + Modbus)
pip install -r requirements/perception.txt    # + camera & ArUco (RealSense + opencv)
pip install -r requirements/all.txt           # everything for a hardware run
pip install -r requirements/dev.txt           # core + perception + pytest (to run tests/)
```

or the packaged extras: `pip install -e .[robot,perception]`. Core (`numpy scipy pyyaml`) is enough
to import the library, run the tests, and plan with `--dry-run`.

**One-command environment setup.** Both paths pin Python to **3.12** (`ur_rtde` and `pyrealsense2`
ship wheels only through cp312):

```bash
# conda
conda env create -f environment.yml        # run from the repo root; then: conda activate urlab

# venv (creates .venv/ and installs)
./setup-venv.sh                             # Linux/macOS or Git Bash   (arg: core|robot|perception|all|dev)
.\setup-venv.ps1                            # Windows PowerShell        (same optional arg)
```

Both read the same `requirements/` files, so conda and venv never drift. Pass a layer name to
either script (default `all`) for a lighter env, e.g. `./setup-venv.sh core`.

**SAM3 is not installed by any of these** — it runs from your `sam3-abhay` checkout under its own
venv, because it needs pins this environment must not inherit (`torch==2.4.1+cu121` for the Pascal
GPU, the gated model weights, the 6 GB memory workaround). Point `sam3.repo_path` at that checkout;
urlab imports `cable_neck_core` from it at runtime. See `requirements/perception.txt` for the full
note.

### Robot prerequisites

- **`ur_rtde` uploads its own control script**, so the External Control URCap must **not** be
  playing and the pendant must be in **Remote Control**. (This is the opposite of the ROS driver,
  which needed External Control running.)
- The gripper is on a serial port (`/dev/ttyUSB0`); prefer a stable `/dev/serial/by-id/...` path.
- Set `robot.ip`, `camera.serial_no`, and `sam3.repo_path` in the config (or `--set`).

## Run

Every demo: `python -m urlab.apps.<name> [--config <name>] [--set k=v] [--dry-run] [--yes] [--debug]`.

```bash
python -m urlab.apps.cartesian --dry-run              # plan every move, no robot
python -m urlab.apps.cable_pick_place                 # real run
python -m urlab.apps.cable_pick_place --set scan.min_good_views=6 --set robot.ip=192.168.1.10
```

`--set key=value` overrides **any** config key, nested with dots (values parse as YAML). This
replaces the ROS `-p name:=value`, which only reached a handful of hand-declared parameters.

## Demos

Each demo has its own page in [`docs/`](docs/) — purpose, sequence, key config knobs, run command,
and caveats.

| ROS package | Replacement | Docs | Notes |
|---|---|---|---|
| `ur_cartesian_demo` | `urlab.apps.cartesian` | [cartesian](docs/cartesian.md) | |
| `ur_gripper_demo` | `urlab.apps.gripper` | [gripper](docs/gripper.md) | counts, not radians |
| `ur_pick_place_demo` | `urlab.apps.pick_place` | [pick_place](docs/pick_place.md) | ArUco fiducial pick-place |
| `ur_visual_servo_demo` | `urlab.apps.visual_servo` | [visual_servo](docs/visual_servo.md) | |
| `ur_admittance_demo` | `urlab.apps.admittance_hold` | [admittance_hold](docs/admittance_hold.md) | forceMode, no controller |
| `ur_kinematic_assembly_demo` | `urlab.apps.kinematic_assembly` | [kinematic_assembly](docs/kinematic_assembly.md) | |
| `ur_uncertain_assembly_sampling` | `urlab.apps.uncertain_sampling` | [uncertain_sampling](docs/uncertain_sampling.md) | data-collection CSV |
| `ur_cable_pick_place_demo` | `urlab.apps.cable_pick_place` | [cable_pick_place](docs/cable_pick_place.md) | |
| `ur_cable_touch_pick_place_demo` | `urlab.apps.cable_touch_pick_place` | [cable_touch_pick_place](docs/cable_touch_pick_place.md) | |
| `ur_cable_pick_assemble_demo` | `urlab.apps.cable_pick_assemble` | [cable_pick_assemble](docs/cable_pick_assemble.md) | |
| `ur_assembly_demo` | `pick_place` + the `insert` skill | — | fiducial pick + kinematic mate |
| `ur_vision_demo` | `urlab.perception.aruco` | — | a library, not a node |
| `ur_tf_demo` | `Robot`'s `hand_eye` edge | — | one static transform, not a package |
| `ur_gripper_bringup` | **not needed** | — | no URDF / controller_manager without ROS |

## Layout

```
urlab/
  transforms.py      pose math — the ONLY place rpy/quat/rotvec conventions and the
                     base_link<->UR-base bridge live
  frames.py          FrameGraph — the tf2 replacement, staleness is a caller concern
  config.py, log.py  config loading + logging/step-runner
  robot/             URArm (RTDE), Robotiq2F85 (Modbus), Robot facade, ForceGuard
  perception/        RealSenseCamera, ArUco, SAM3 adapter, ConnectorEstimator (multi-view fusion)
  skills/            reusable behaviours: servo, scan, pick, touch, insert, trajectory
  apps/              one thin script per demo
configs/             one yaml per demo (+ assembly_trajectory.csv)
tests/               offline math/geometry tests (no robot/camera/torch)
```

A demo is a short script that **composes skills** — they hold no references to each other, so any
two combine (scan + touch + insert) without one being a base class of the other. This is the
"modules and skills abstracted into common functions" the refactor was for; the ROS version
expressed the same sharing through a four-deep inheritance chain
(`PickPlace → CablePickPlace → CableTouchPickPlace / CablePickAssemble`).

## Conventions (get these wrong and everything is silently wrong)

- **rpy is EXTRINSIC XYZ** (`R = Rz(yaw) Ry(pitch) Rx(roll)`), matching the ROS stack — so every
  angle in the configs carried over untouched. CAD tools usually report intrinsic XYZ.
- **quaternions are `[x, y, z, w]`**.
- **Poses are `T_a_b`** ("b in a"); compose left-to-right: `T_a_c = T_a_b @ T_b_c`.
- **`tool0`** is the pendant all-zeros TCP — **not** `flange` (which is ~90° twisted).
- RTDE speaks the **UR `base`** frame (a 180° Z turn from ROS `base_link`) and **axis-angle**
  rotation vectors; both conversions are confined to `transforms.rtde_to_matrix` / `matrix_to_rtde`.

## Tests

```bash
python -m pytest tests/ -q       # or: python tests/test_smoke.py
```

Covers the pure-computation layers offline: the transform conventions, `FrameGraph` staleness, the
config loader, and the multi-view connector fusion against a synthetic ground truth (recovers the
origin to <5 mm and the axis to <5° on clean synthetic views).

## Hardware checklist still open

These are physical-calibration items the code cannot settle for you:

- `grasp_check.closed_counts` (228) — confirm against what the gripper reports at full closure.
  The rad→counts fudge (`full_close_rad`, permanently "TODO: verify me" in the ROS configs) is
  **gone**: the check now compares to a number the hardware reports directly.
- `grasp_check.detect_empty` — leave `false` until you have confirmed a seated cable reports
  `gOBJ=2` and an empty close reports `gOBJ=3` (see `robot/gripper.py`).
- `touch.contact_z_offset_m` — tune on the real connector.
- `assembly.target` — measure by jogging to a good mate and reading the chosen frame off the robot.
- **hand-eye `rpy`** — the configs carry `[0, 0, 0]` (identity), matching the `dev` branch exactly.
  The `dev` plan noted identity puts a detected marker ~0.6 m too low and proposed
  `[-1.5708, 0, -1.5708]` (an optical-frame rotation), but that change was never committed and is
  unverified. Left at identity here for faithful parity; resolve it as a hand-eye calibration on
  hardware and set `hand_eye.rpy` in the configs (or `--set hand_eye.rpy='[-1.5708,0,-1.5708]'`).
