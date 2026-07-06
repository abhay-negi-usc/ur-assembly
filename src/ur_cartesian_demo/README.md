# ur_cartesian_demo

A simple demonstration of **cartesian position control** for a UR10e on **ROS2 Jazzy**.

Starting from the robot's current ("initial") TCP pose, the node steps the TCP by:

- **±30 mm** along **X**, **Y**, **Z**
- **±30°** about **roll**, **pitch**, **yaw**

returning to the initial pose **between every move**. The full 12-move sequence is run **once
per motion frame** — by default in the **world** frame and then the **tool0** frame — so you
can compare cartesian motion expressed in a fixed world frame vs. the moving tool frame
(24 moves total). The motion frame defines the axes for both the translations and the
rotations; recentering is identical regardless of frame.

## How it works

`ur_controllers` 3.8.0 no longer ships a Cartesian trajectory controller, so this demo does
the Cartesian→joint mapping itself:

1. Read the current TCP pose from **tf2** (`reference_frame` → `tip_frame`).
2. For each ±step (expressed in the current motion frame's axes), build the target pose and
   solve it with **MoveIt's `/compute_ik`** service.
   IK uses the driver's **calibrated** URDF and is seeded with the initial joint state for
   solution continuity.
3. Execute the joint goal on the already-active **`scaled_joint_trajectory_controller`** via
   `FollowJointTrajectory` — smooth, time-parameterized, speed-scaled motion.
4. **Recenter** by commanding the *recorded initial joint configuration* directly (no IK), so
   the robot returns exactly to the start.

No controller switching and no extra controllers are needed — just the default UR controller
set plus a running MoveIt.

> ⚠️ This is a minimal demo. Keep the e-stop in hand on real hardware; defaults are slow
> (4 s/move) and prompt before each move.

## Prerequisites

- ROS2 **Jazzy** with the Universal Robots ROS2 driver and **MoveIt** for UR
  (`ur_moveit_config`).
- `tf_transformations`:
  ```bash
  sudo apt install ros-jazzy-tf-transformations
  ```

Start, in separate terminals:

**1) The driver** —

Simulation (recommended first):
```bash
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur10e
# or the real driver against fake hardware:
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.125.2 use_fake_hardware:=true
```
Real UR10e (use this robot's calibration, or FK/IK will be off by mm–cm):
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.125.2 \
  kinematics_params_file:=$(ros2 pkg prefix ur_gripper_bringup)/share/ur_gripper_bringup/config/ur10e_calibration.yaml
# then start the External Control program on the teach pendant
```
(See `ur_gripper_bringup/README.md` to extract the calibration file once.)

**2) MoveIt** (provides `/compute_ik`) — ⚠️ pass the **same** calibration as the driver so IK
and FK agree:
```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10e \
  kinematics_params_file:=$(ros2 pkg prefix ur_gripper_bringup)/share/ur_gripper_bringup/config/ur10e_calibration.yaml
```

Sanity checks:
```bash
ros2 service list | grep compute_ik                       # /compute_ik present
ros2 control list_controllers | grep scaled_joint_trajectory   # -> active
```

## Build

```bash
cd ~/abhay_ws/ur-assembly
colcon build --packages-select ur_cartesian_demo --symlink-install
source install/setup.bash
```

## Run

```bash
# Conservative defaults: 4 s/move, prompts before each move (good for real hardware)
ros2 run ur_cartesian_demo cartesian_pose_demo

# Sim / hands-off: no prompts, faster moves
ros2 run ur_cartesian_demo cartesian_pose_demo --ros-args \
  -p confirm_each_move:=false -p move_duration_s:=2.0
```

Launch-file alternative (use `confirm_each_move:=false` here — `input()` prompts only work
under `ros2 run` in an interactive terminal):
```bash
ros2 launch ur_cartesian_demo cartesian_pose_demo.launch.py \
  confirm_each_move:=false move_duration_s:=2.0
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `planning_group` | `ur_manipulator` | MoveIt group used for IK. |
| `reference_frame` | `base_link` | Frame the IK target poses are sent in (must be in tf). |
| `tip_frame` | `tool0` | Controlled TCP / IK tip link. |
| `motion_frames` | `['world', 'tool0']` | Frames whose axes define the deltas; sequence runs once per frame. |
| `joint_names` | UR 6 joints | Joint order for the trajectory goal. |
| `controller_action` | `/scaled_joint_trajectory_controller/follow_joint_trajectory` | Active joint controller's action. |
| `linear_step_m` | `0.030` | Linear step (m) → ±30 mm. |
| `angular_step_deg` | `30.0` | Angular step (deg) → ±30°. |
| `move_duration_s` | `4.0` | Time per move (s). Lower for sim. |
| `settle_s` | `0.5` | Pause after each return-to-center. |
| `ik_timeout_s` | `2.0` | Per-call IK timeout. |
| `avoid_collisions` | `true` | Reject self-colliding IK solutions. |
| `confirm_each_move` | `true` | Prompt on the console before each move. |

> **Motion frames:** each name in `motion_frames` is looked up in tf relative to
> `reference_frame` to get its axes, then both the linear and angular deltas are expressed in
> those axes. `world` gives fixed base-axis motion; `tool0` (the default controlled frame, and
> the controller's all-zeros TCP that matches the pendant) gives motion about the tool frame
> (Z out the tool). Avoid `flange` as the tip/motion frame — it shares tool0's origin and Z but
> its X/Y are rotated 90° about the tool axis, so flange X/Y moves won't match the pendant TCP.
> If a frame isn't found in tf, the demo warns and falls back to `reference_frame` axes. Test a
> single frame with e.g. `-p motion_frames:="['tool0']"`.

## Verify it works

1. **Sim:** open RViz, run with `confirm_each_move:=false move_duration_s:=2.0`, and watch the
   TCP step ±30 mm / ±30° on each axis and return to start between moves, in both the world and
   tool0 frames.
2. **Real UR10e:** rerun with defaults (prompts, 4 s/move), e-stop ready, stepping through each
   move and confirming the teach-pendant pose readout returns to the start each time.
3. If a move logs `IK failed`, that axis target was unreachable from the start pose (e.g. wrist
   near a singularity); the demo skips it and continues. Try a different start configuration.
