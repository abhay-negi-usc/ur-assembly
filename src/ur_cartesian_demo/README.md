# ur_cartesian_demo

A simple demonstration of **cartesian position control** for a UR10e on **ROS2 Jazzy**.

Starting from the robot's current ("initial") TCP pose, the node steps the TCP by:

- **±10 mm** along **X**, **Y**, **Z**
- **±10°** about **roll**, **pitch**, **yaw**

returning to the initial pose **between every move** (12 moves total).

Commands go to the UR driver's `pose_based_cartesian_traj_controller` via the
`cartesian_control_msgs/action/FollowCartesianTrajectory` action — the controller/robot
solves the inverse kinematics, so this script never computes joint angles.

The current TCP pose is read from **tf2** (`base` → `tool0`), which works on both the real
robot and in simulation. (`tcp_pose_broadcaster` is intentionally *not* used — it is
non-functional in simulation on Jazzy.)

> ⚠️ The cartesian trajectory controller does **not** check reachability (IK) or collisions.
> It is meant for short, simple test motions like this. Keep the e-stop in hand on real
> hardware.

## Prerequisites

- ROS2 **Jazzy** with the Universal Robots ROS2 driver installed (provides
  `ur_robot_driver`, the cartesian controllers, and `cartesian_control_msgs`).
- `tf_transformations`:
  ```bash
  sudo apt install ros-jazzy-tf-transformations
  ```

Start the driver first.

**Simulation (recommended first):**
```bash
ros2 launch ur_simulation_gz ur_sim_control.launch.py ur_type:=ur10e
# or, with the real driver against fake hardware:
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=yyy.yyy.yyy.yyy use_fake_hardware:=true
```

**Real UR10e:**
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=<ROBOT_IP>
```
(then start the External Control program on the teach pendant).

## Activate the cartesian controller

`pose_based_cartesian_traj_controller` is loaded but **inactive** by default, and it conflicts
with the default `scaled_joint_trajectory_controller` (only one may be active). Activate it:

```bash
ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller \
  --activate pose_based_cartesian_traj_controller
```

Verify:
```bash
ros2 control list_controllers          # pose_based_cartesian_traj_controller -> active
ros2 action list | grep follow_cartesian_trajectory
```

> Alternatively pass `-p auto_switch_controllers:=true` and the node will switch for you on
> start and restore `scaled_joint_trajectory_controller` on exit.

## Build

```bash
cd ~/abhay_ws/ur-assembly
colcon build --packages-select ur_cartesian_demo
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

There is also a launch file (handy for sim / non-interactive runs):
```bash
ros2 launch ur_cartesian_demo cartesian_pose_demo.launch.py \
  confirm_each_move:=false move_duration_s:=2.0
```
> Note: `confirm_each_move:=true` reads from the console with `input()`, which only works when
> launched via `ros2 run` in an interactive terminal. Use `confirm_each_move:=false` with the
> launch file.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `controller_name` | `pose_based_cartesian_traj_controller` | Action namespace to command. |
| `base_frame` | `base` | Reference frame for target poses. |
| `tip_frame` | `tool0` | Controlled TCP frame. |
| `linear_step_m` | `0.010` | Linear step (m) → ±10 mm. |
| `angular_step_deg` | `10.0` | Angular step (deg) → ±10°. |
| `move_duration_s` | `4.0` | Time per move (s). Lower for sim. |
| `settle_s` | `0.5` | Pause after each return-to-center. |
| `rotate_in_tool_frame` | `true` | RPY about the tool axes (`true`) or base axes (`false`). |
| `confirm_each_move` | `true` | Prompt on the console before each move. |
| `auto_switch_controllers` | `false` | Switch controllers on start / restore on exit. |

## Verify it works

1. **Sim:** open RViz, run with `confirm_each_move:=false move_duration_s:=2.0`, and watch the
   TCP step ±10 mm / ±10° on each axis and return to start between moves.
2. `ros2 control list_controllers` shows the cartesian controller active.
3. **Real UR10e:** rerun with defaults (prompts, 4 s/move), e-stop ready, stepping through each
   move and confirming the pose readout on the teach pendant returns to the start each time.
