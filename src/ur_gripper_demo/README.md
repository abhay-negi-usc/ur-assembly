# ur_gripper_demo

A demo of a **Robotiq 2F-85** gripper on a UR control box, on **ROS2 Jazzy**, via the
ros2_control [`robotiq_gripper_controller`](https://github.com/PickNikRobotics/ros2_robotiq_gripper).

**Gripper only — this node never commands the arm.**

It cycles the gripper through a sweep of positions (open → 25% → 50% → 75% → closed → open)
by sending `control_msgs/action/GripperCommand` goals to
`/robotiq_gripper_controller/gripper_cmd`.

## How it works

For the 2F-85 the GripperCommand `position` runs **0.0 (open) → 0.8 (closed)**, and
`max_effort` sets the grip force. The node interpolates from `open_position` to
`closed_position` by a "closedness" fraction, waits for each goal's result, and logs
`reached_goal` / `stalled` / `position`. A **stall is a normal success** — it just means the
fingers met an object before reaching the commanded position.

## Prerequisites

The gripper must already be running as a ros2_control controller (e.g. via
`ros2_robotiq_gripper`, launched standalone or alongside the UR driver). Confirm:

```bash
ros2 control list_controllers | grep -i grip      # robotiq_gripper_controller -> active
ros2 action list | grep -i grip                   # /robotiq_gripper_controller/gripper_cmd
```

If the action name differs, pass it with `-p action_name:=...`.

> If the gripper isn't activated, you may need to (re)activate it once. ros2_robotiq_gripper
> exposes `/robotiq_activation_controller/reactivate_gripper` (`std_srvs/Trigger`); run the
> demo with `-p activate_first:=true` to call it first, or:
> `ros2 service call /robotiq_activation_controller/reactivate_gripper std_srvs/srv/Trigger`

## Build & run

```bash
cd ~/abhay_ws/ur-assembly
colcon build --packages-select ur_gripper_demo --symlink-install
source install/setup.bash

ros2 run ur_gripper_demo gripper_demo
# or
ros2 launch ur_gripper_demo gripper_demo.launch.py cycles:=3
```

Quick manual test without this package:
```bash
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd \
  control_msgs/action/GripperCommand "{command: {position: 0.8, max_effort: 50.0}}"
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `action_name` | `/robotiq_gripper_controller/gripper_cmd` | GripperCommand action server. |
| `open_position` | `0.0` | Position for fully open. |
| `closed_position` | `0.8` | Position for fully closed. |
| `max_effort` | `50.0` | Grip force/effort per goal. |
| `dwell_s` | `1.5` | Pause between positions (s). |
| `cycles` | `1` | Number of open→closed→open cycles. |
| `fractions` | `[0.0, 0.25, 0.5, 0.75, 1.0, 0.0]` | "Closedness" steps per cycle (0=open, 1=closed). |
| `activate_first` | `false` | Call the reactivate service before cycling. |
| `activation_service` | `/robotiq_activation_controller/reactivate_gripper` | Trigger service for (re)activation. |

## Notes

- If open/closed look **inverted** for your controller config, swap `open_position` and
  `closed_position` (e.g. `-p open_position:=0.8 -p closed_position:=0.0`).
- To just open and close (no intermediate steps): `-p fractions:="[0.0, 1.0]"`.
- Keep clear of the fingers — the gripper closes with force.
