# ur_gripper_bringup

Brings up a **UR10e + Robotiq 2F‑85 together under one `controller_manager`** on ROS2 Jazzy, so
the arm and gripper controllers run **simultaneously without conflict**.

## Why this exists

Running `ur_control.launch.py` (arm) and `robotiq_control.launch.py` (gripper) separately
**conflicts**: both start a node named `/controller_manager` and both read the global
`/robot_description`, so the gripper's manager grabs the ur10e URDF and crashes on a second RTDE
connection. The fix is a **single** controller_manager driven by a **combined URDF** that holds
both ros2_control blocks — that's what this package provides, using the UR driver's official
`description_launchfile` hook (the "my_robot_cell" pattern).

```
world → ur10e arm (ur_robot) → flange → coupler_link → robotiq 2f-85
        └ ur_ros2_control (URPositionHardwareInterface)
                                          └ robotiq ros2_control (RobotiqGripperHardwareInterface)
```

## Layout

| File | Role |
|---|---|
| [urdf/ur10e_with_2f85.urdf.xacro](urdf/ur10e_with_2f85.urdf.xacro) | Combined description: UR arm + UR ros2_control + coupler + 2F‑85 + gripper ros2_control. |
| [launch/rsp.launch.py](launch/rsp.launch.py) | robot_state_publisher for the combined URDF (fed to the driver via `description_launchfile`). |
| [launch/ur_gripper_control.launch.py](launch/ur_gripper_control.launch.py) | Top-level: starts the driver + spawns gripper controllers + preflight warnings. |
| [config/coupler.yaml](config/coupler.yaml) | `com_port`, `gripper_enabled`, and the flange→gripper **coupler** transform. |
| [config/gripper_controllers.yaml](config/gripper_controllers.yaml) | Gripper controllers (with the position-only interface fix). |
| `config/ur10e_calibration.yaml` | This robot's `ur_calibration` kinematics (you extract it — see below). |

## Prerequisites

- UR driver + `ur_description` + `ur_client_library` (standard UR ROS2 install).
- `ros2_robotiq_gripper` built in this workspace (`robotiq_description` provides the gripper
  macros). See `ur_gripper_demo`'s README for the build (incl. the `tylerjw/serial` dep).
- Gripper on **`/dev/ttyUSB1`** (the container needs `--device=/dev/ttyUSB1`), powered (24 V).
- `pip install pymodbus` (optional — only for the gripper preflight probe; skipped if absent).

## Configure

Edit [config/coupler.yaml](config/coupler.yaml):
- `com_port` — defaults to `/dev/ttyUSB1`.
- `coupler.xyz` / `coupler.rpy` — the **flange → gripper-mount** transform.
- `gripper_enabled` — set `false` to bring up the **arm only** (omits the gripper from the URDF
  and skips its controllers).

### Robot calibration (always used)

Every UR arm ships from the factory with a **unique kinematic calibration** (small per-joint DH
corrections). Without it, `robot_state_publisher`'s FK — and therefore every `base_link → tool`
pose and MoveIt IK — is off from the real robot by **mm–cm** (this is exactly why
`ros2 run tf2_ros tf2_echo base flange` disagrees with the pendant's TCP readout).

This launch **defaults to using it**: it looks for
[`config/ur10e_calibration.yaml`](config/) and forwards it into the description. If the file is
missing it falls back to the generic ur10e kinematics and prints a **loud warning** (poses will
be off).

Extract it **once** from your robot (needs the robot reachable at its IP), writing it straight
into this package's config so it installs on the next build:

```bash
ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:=192.168.125.2 \
  target_filename:="$(ros2 pkg prefix ur_gripper_bringup)/share/ur_gripper_bringup/config/ur10e_calibration.yaml"
# (or write it into src/ur_gripper_bringup/config/ and colcon build so it's version-controlled)
```

Override the path per-launch with `kinematics_params_file:=/abs/path.yaml`.

> **MoveIt must match.** `/compute_ik` (move_group) has to use the **same** calibration or its
> IK and the driver's FK disagree. Launch it with
> `kinematics_params_file:=<same file>` — see the pick-and-place / cartesian demo READMEs.

## Build & run

```bash
cd /abhay_ws/ur-assembly
colcon build --packages-select ur_gripper_bringup --symlink-install
source install/setup.bash

# Mock first (no robot, no serial) -- validates the combined description + single CM:
ros2 launch ur_gripper_bringup ur_gripper_control.launch.py use_mock_hardware:=true

# Real robot (e-stop in hand; start External Control on the pendant):
ros2 launch ur_gripper_bringup ur_gripper_control.launch.py
```

Verify one controller_manager runs both:
```bash
ros2 node list | grep controller_manager        # exactly ONE, no "shared name" warning
ros2 control list_controllers                    # scaled_joint_trajectory_controller AND
                                                 # robotiq_gripper_controller both active
ros2 action list | grep grip                     # /robotiq_gripper_controller/gripper_cmd
```

Then the existing demos run on top, **simultaneously**:
```bash
ros2 run ur_gripper_demo gripper_demo            # gripper cycles
# ...and an arm move at the same time -- no conflict.
ros2 launch ur_vision_demo aruco_demo.launch.py  # perception (no ros2_control, never conflicts)
ros2 launch ur_tf_demo tf_streaming.launch.py
```

## Notes / known gaps

- **Coupler is identity** until you fill `config/coupler.yaml` (TODO in the xacro + a launch
  warning). The gripper will appear coincident with `tool0` until then.
- **Missing gripper:** the preflight prints a warning if nothing answers on `com_port`, but the
  gripper hardware is still loaded from the URDF — if it can't initialize, the combined
  controller_manager may fail (taking the arm with it). For arm-only, set
  `gripper_enabled: false`, or use `use_mock_hardware:=true` to test without hardware.
- **Gripper-controller spawn** is delayed 12 s so the controller_manager and gripper hardware
  are up first; if your machine is slow and the spawn races, increase the `TimerAction` period.
- **Version sensitivity:** the `ur_robot`/`ur_ros2_control` macro args and include paths track
  your installed `ur_robot_driver`/`ur_description` (Jazzy). If xacro errors on an unknown arg,
  diff against your installed `ur.urdf.xacro` / `ur.ros2_control.xacro` and adjust.
- Frames use **no tf_prefix** (`base_link`, `tool0`, …) to match `ur_tf_demo`/`ur_cartesian_demo`.
