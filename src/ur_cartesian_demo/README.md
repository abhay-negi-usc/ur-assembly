# ur_cartesian_demo

A simple demonstration of **cartesian position control** for a UR10e on **ROS2 Jazzy**.

Starting from the robot's current ("initial") TCP pose, the node steps the TCP by:

- **±10 mm** along **X**, **Y**, **Z**
- **±10°** about **roll**, **pitch**, **yaw**

returning to the initial pose **between every move** (12 moves total).

## How it works

`ur_controllers` 3.8.0 no longer ships a Cartesian trajectory controller, so this demo does
the Cartesian→joint mapping itself:

1. Read the current TCP pose from **tf2** (`reference_frame` → `tip_frame`).
2. For each ±step, build the target pose and solve it with **MoveIt's `/compute_ik`** service.
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
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=yyy.yyy.yyy.yyy use_fake_hardware:=true
```
Real UR10e:
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=<ROBOT_IP>
# then start the External Control program on the teach pendant
```

**2) MoveIt** (provides `/compute_ik`) —
```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10e
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
| `reference_frame` | `base_link` | Frame target poses are expressed in. |
| `tip_frame` | `tool0` | Controlled TCP / IK tip link. |
| `joint_names` | UR 6 joints | Joint order for the trajectory goal. |
| `controller_action` | `/scaled_joint_trajectory_controller/follow_joint_trajectory` | Active joint controller's action. |
| `linear_step_m` | `0.010` | Linear step (m) → ±10 mm. |
| `angular_step_deg` | `10.0` | Angular step (deg) → ±10°. |
| `move_duration_s` | `4.0` | Time per move (s). Lower for sim. |
| `settle_s` | `0.5` | Pause after each return-to-center. |
| `ik_timeout_s` | `2.0` | Per-call IK timeout. |
| `avoid_collisions` | `true` | Reject self-colliding IK solutions. |
| `rotate_in_tool_frame` | `true` | RPY about the tool axes (`true`) or base axes (`false`). |
| `confirm_each_move` | `true` | Prompt on the console before each move. |

## Verify it works

1. **Sim:** open RViz, run with `confirm_each_move:=false move_duration_s:=2.0`, and watch the
   TCP step ±10 mm / ±10° on each axis and return to start between moves.
2. **Real UR10e:** rerun with defaults (prompts, 4 s/move), e-stop ready, stepping through each
   move and confirming the teach-pendant pose readout returns to the start each time.
3. If a move logs `IK failed`, that axis target was unreachable from the start pose (e.g. wrist
   near a singularity); the demo skips it and continues. Try a different start configuration.
