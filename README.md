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
| **Compliance** | `admittance_controller` loaded inactive, parameterised over a service, **activated** (which deactivated the JTC and caused a joint-0 velocity fault, patched with a reference-holding dance) | **software admittance** over `servoL` (`robot/admittance.py`) — the same spring-mass-damper law, as a plain servo loop. `forceMode()` is used only where free-floating *is* the goal (`admittance_hold`): it has **no restoring stiffness**, so it cannot track a trajectory. | The controller install, the switch, the reference streaming, and `_hold_reference` entirely. |
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

**SAM3 is not installed by any of the above**, and it is not a separate process. urlab imports the
SAM3 detector **in-process** (see the in-process note above), so `torch` and the `sam3` package must
live in the **same** environment that runs urlab — added *on top of* your urlab env, not in a
parallel venv. The `### SAM3 setup` steps below are what actually make a cable demo run; the
`torch==2.4.1+cu121` pin, the gated weights, and the 6 GB memory workaround are why they are kept
out of `requirements/` (rationale in `requirements/perception.txt`).

### SAM3 setup

The detector geometry lives in your `sam3-abhay` checkout under `scripts/`; at runtime urlab adds
that `scripts/` dir to `sys.path` (via `sam3.repo_path`) and imports `cable_neck_core` /
`cable_neck_diameter`, which in turn `import torch` and `from sam3 import build_sam3_image_model`.
So the env that runs a cable demo needs **both** the `sam3` package importable (`pip install -e`)
**and** `torch` + the model weights present.

Into your **activated urlab env** (the same `.venv`/conda env the demos run in):

```bash
# 1. PyTorch. This pin is for the Pascal GTX 1060 (6 GB) -- the last wheels with sm_61 kernels.
#    Newer GPU: follow sam3-abhay/README.md (torch 2.7+/cu128). No GPU: swap cu121 -> cpu (slow).
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 2. The sam3 PACKAGE. Its pyproject does NOT pin torch (the pin above is safe) but DOES pin
#    numpy < 2, so it downgrades numpy and orphans a numpy-2-built scipy -- realign scipy (step 3).
pip install -e ../sam3-abhay          # = your sam3.repo_path checkout
# sam3 imports these at module load but omits them from its pyproject (its model_builder pulls in
# the whole zoo -- image+video+tracking -- so their deps must all resolve, even for image inference):
pip install einops pycocotools psutil

# 3. Hold numpy/scipy at a matching numpy<2 pair. Without this, scipy imports fail with
#    "module 'numpy' has no attribute 'long'" -- the whole env is numpy<2 once SAM3 is added.
pip install "numpy>=1.26,<2" "scipy<1.14"

# 4. Gated checkpoints: request access at https://huggingface.co/facebook/sam3, then log in:
huggingface-cli login                 # paste an HF token (or: export HF_TOKEN=hf_...)
```

Verify the import chain before launching the whole demo (arm + camera):

```bash
python -c "import torch; from sam3 import build_sam3_image_model; \
           print('sam3 OK', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Notes:

- **Keep `sam3-abhay` current.** The detector *scripts* come from the checkout, not from pip, so a
  `git pull` in `sam3-abhay` is how you get new/fixed detectors (e.g. `cable_neck_diameter.py`,
  which `sam3.mode: junction` and `scan.mode: reconstruction` require). Re-run `pip install -e` only
  if the `sam3` package itself changes — not for script edits.
- **`sam3.mode` picks the detector** (`neck` | `junction` | `tip`). `neck`/`tip` need only
  `cable_neck_core`; `junction` (the default in the cable configs) also needs `cable_neck_diameter`.
- **sam3 under-declares its runtime deps** — `model_builder` imports the whole model zoo at load, so
  `einops`, `pycocotools`, and `psutil` (step 2) are all required even for image inference (`triton`
  is pulled in by the torch wheel). If a later run raises `ModuleNotFoundError` for another package,
  install just that one — but do **not** install the guarded/lazy/eval-only ones its code also
  mentions: `xformers` (guarded, falls back to torch SDPA), `decord`/`torchcodec` (video only),
  `hydra`/`omegaconf` (training + the lazy multiplex builder), `detectron2` (agent/eval — an install
  nightmare), or `open_clip` (a docstring reference, not a real import). None are reached by
  `build_sam3_image_model`.
- **First run** downloads the checkpoint (a few GB) and takes ~30 s to load — the log says
  "Loading SAM3…".
- **Docker:** CUDA needs the container started with `--gpus all` (nvidia-container-toolkit),
  *separate* from the camera's USB passthrough (`-v /dev/bus/usb:/dev/bus/usb`, not `--device
  /dev/video*` — the pip `pyrealsense2` wheel uses the libusb backend). If the check above prints
  `cuda False`, add `--gpus all` or install the CPU torch wheel.

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
| — (new) | `urlab.apps.monitor` | [monitor](docs/monitor.md) | **live frame monitor** — read-only, works with pendant freedrive |
| — (new) | `urlab.apps.gripper_control` | — | **interactive gripper jog** — type a count, moves there, prints the measured count |
| — (new) | `urlab.apps.marker_calibration` | — | **fiducial rig calibration** — measures the target's pose in each ArUco marker's frame; paste the printed block into `configs/frames.yaml` `marker_rigs:` and set `assembly.target_source: visual` to let `bnc_assembly` find the socket by looking at it |

## Layout

```
urlab/
  transforms.py      pose math — the ONLY place rpy/quat/rotvec conventions and the
                     base_link<->UR-base bridge live
  frames.py          FrameGraph — the tf2 replacement, staleness is a caller concern
  config.py, log.py  config loading + logging/step-runner
  robot/             URArm (RTDE), Robotiq2F85 (Modbus), ForceGuard, and the Robot class:
                     registers the gripper/camera, holds the FRAME REGISTRY
                     (register_frame with parent chains, resolved through live FK), and the
                     MOTION PRIMITIVES (move_joints / move_cartesian / move_relative) every
                     behavior builds on
  perception/        RealSenseCamera, ArUco, SAM3 adapter, ConnectorEstimator (multi-view fusion)
  skills/            reusable behaviours: servo, scan, pick, touch, insert, trajectory
  behaviors/         py_trees behavior-tree layer: a library of common robot behaviors
                     (guarded moves, admittance ramps, gripper, operator gates) the apps
                     compose their run trees from
  apps/              one thin script per demo
configs/             one yaml per demo (+ assembly_trajectory.csv)
tests/               offline math/geometry tests (no robot/camera/torch)
```

A demo is a short script that **composes skills** — they hold no references to each other, so any
two combine (scan + touch + insert) without one being a base class of the other. The refactored
demos (calibration_check, wiggle_sampling, marker_calibration, cable_pick_estimate_assemble)
express their run sequence as a **py_trees behavior tree** built from `urlab/behaviors` — the
tree is logged at `--debug` before the arm moves, so the whole procedure is visible up front.

The layering is: **Robot motion primitives → behaviors → scripts**. A new script is a Robot,
some registered frames, and a chain of behaviors — see `urlab/apps/behavior_demo.py` for the
complete worked example (safe to run: `python -m urlab.apps.behavior_demo --dry-run`):

```python
def build(cfg, robot, camera, args):
    robot.load_frame_catalogue()                       # everything in configs/frames.yaml
    robot.register_frame('probe', T, parent='fingertip')
    return bt.chain('demo',
        ('reset', robot, cfg),
        ('open_gripper', robot),
        ('move_relative', robot, delta, 'advance', {'expressed_in': 'probe'}),
        ('close_gripper', robot))

main = bt.app('Demo', 'my_config', build)
``` This is the
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
- **The `Rz(π)` base bridge applies to WRENCHES too, not just poses** — `arm.wrench()` applies it, so
  everything downstream is `base_link`. Miss it on a new RTDE value and `x`/`y` come out **negated**
  while `z` looks fine. Anything reading a *magnitude* (the force guard, `force()`, `torque()`) is
  blind to it, so it stays hidden until something reads *components* — which is exactly how it
  reached hardware once already. **Symptom key: some axes wrong ⇒ FRAME error; all axes wrong ⇒ SIGN
  error.** A global sign flip can never fix a per-axis split.

## Tests

```bash
python -m pytest tests/ -q       # or: python tests/test_smoke.py
```

Covers the pure-computation layers offline: the transform conventions, `FrameGraph` staleness, the
config loader, and the multi-view connector fusion against a synthetic ground truth (recovers the
origin to <5 mm and the axis to <5° on clean synthetic views).

## Hardware checklist still open

These are physical-calibration items the code cannot settle for you:

- `grasp_check.groove_counts` / `empty_counts` / `faces_max_counts` (225 / 228 / 223) — the three
  fingertip closure levels. Confirm each against what the gripper reports: a **seated** cable stops
  at the groove count (225 — SUCCESS, the middle band, *not* full closure), an **empty** close goes
  to 228, a cable on the **flat faces** stalls at ≤ 223 (a miss). The rad→counts fudge
  (`full_close_rad`, permanently "TODO: verify me" in the ROS configs) is **gone**: these compare to
  numbers the hardware reports directly.
- `grasp_check.detect_empty` — now **POSITION**-based (228 vs 225 are distinct), so it defaults
  **on** and is reliable; the old UNVERIFIED gOBJ path is gone.
- `pickup.mode` — `position` (stiff) by default. Set `compliance` for a compliant grasp descent
  (software admittance, same law as the assembly insert); validate the params on hardware.
- `touch.contact_z_offset_mm` — tune on the real connector.
- `assembly.target` — measure by jogging to a good mate and reading the chosen frame off the robot.
- **hand-eye `rpy`** — the configs carry `[0, 0, 0]` (identity), matching the `dev` branch exactly.
  The `dev` plan noted identity puts a detected marker ~0.6 m too low and proposed
  `[-1.5708, 0, -1.5708]` (an optical-frame rotation), but that change was never committed and is
  unverified. Left at identity here for faithful parity; resolve it as a hand-eye calibration on
  hardware and set `hand_eye.rpy` in the configs (or `--set hand_eye.rpy='[-1.5708,0,-1.5708]'`).
