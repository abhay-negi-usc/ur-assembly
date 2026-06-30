# ur_gripper_demo

A demo of a **Robotiq 2F-85** gripper on a UR control box, on **ROS2 Jazzy**, via the
ros2_control [`robotiq_gripper_controller`](https://github.com/PickNikRobotics/ros2_robotiq_gripper).

**Gripper only — this node never commands the arm.**

It cycles the gripper through a sweep of positions (open → 25% → 50% → 75% → closed → open)
by sending **`control_msgs/action/ParallelGripperCommand`** goals to
`/robotiq_gripper_controller/gripper_cmd` — the action type used by the modern
`parallel_gripper_action_controller/GripperActionController` (the deprecated
`position_controllers`/`effort_controllers` `GripperActionController`, which used the older
`control_msgs/action/GripperCommand`, are *not* what this driver uses).

## How it works

A `ParallelGripperCommand` goal carries a `sensor_msgs/JointState` whose `position[0]` is the
target joint angle for `robotiq_85_left_knuckle_joint`. For the 2F-85 that runs **0.0 (open)
→ ~0.8 (closed)**. The node interpolates from `open_position` to `closed_position` by a
"closedness" fraction, waits for each goal's result, and logs `reached_goal` / `stalled` /
`position`. A **stall is a normal success** — the fingers met an object before reaching the
commanded position.

`velocity` and `effort` are included in the goal but are only honored if the hardware exposes
those command interfaces. With the stock Robotiq config the gripper is **position-commanded
only** (see the config fix below), so `max_effort` / `max_velocity` are currently ignored and
the gripper uses its default force/speed.

## Prerequisites

### 1. Robotiq driver built from source
`ros2_robotiq_gripper` has no Jazzy binary, so it's built in this workspace. It also needs the
`serial` library (no rosdep key), cloned separately:
```bash
cd /abhay_ws/ur-assembly/src
git clone https://github.com/PickNikRobotics/ros2_robotiq_gripper.git
git clone -b ros2 https://github.com/tylerjw/serial.git
cd /abhay_ws/ur-assembly
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-up-to robotiq_driver robotiq_description robotiq_controllers
source install/setup.bash
```

### 2. Serial device visible
The gripper is on a USB→RS-485 adapter at **`/dev/ttyUSB1`** (verify with
`ls -l /dev/serial/by-id/`). If you run inside Docker, the container must be started with
`--device=/dev/ttyUSB1` (it can't be hot-added to a running container). The gripper also needs
its **24 V power** — the USB adapter carries data only.

### 3. One-time controller-config fix
The stock `robotiq_controllers.yaml` makes the controller claim **effort and velocity command
interfaces** the real hardware doesn't export, so it fails to activate with:
`Unable to activate ... 'robotiq_85_left_knuckle_joint/effort' is not available`. Blank those
two lines:
```bash
for f in \
  /abhay_ws/ur-assembly/install/robotiq_description/share/robotiq_description/config/robotiq_controllers.yaml \
  /abhay_ws/ur-assembly/src/ros2_robotiq_gripper/robotiq_description/config/robotiq_controllers.yaml ; do
  sed -i 's|max_effort_interface:.*|max_effort_interface: ""|; s|max_velocity_interface:.*|max_velocity_interface: ""|' "$f"
done
```

### 4. Gripper bringup (standalone)
> ⚠️ The stock `robotiq_control.launch.py` starts its **own** `/controller_manager` and reads
> the global `/robot_description`, so it **cannot run at the same time as the UR driver** —
> they collide (the gripper manager grabs the ur10e URDF and crashes on a second RTDE
> connection). For this gripper-only demo, **stop the UR driver first**. Running arm + gripper
> together is handled by a separate integrated bringup (see *Running with the arm* below).

```bash
# stop the UR driver if it's running, then:
ros2 launch robotiq_description robotiq_control.launch.py com_port:=/dev/ttyUSB1
```
Confirm the controller is up and the action type is right:
```bash
ros2 control list_controllers | grep -i grip      # robotiq_gripper_controller -> active
ros2 action info /robotiq_gripper_controller/gripper_cmd -t   # ... [control_msgs/action/ParallelGripperCommand]
```
If it shows `inactive`, activate it: `ros2 control set_controller_state robotiq_gripper_controller active`.

## Build & run

```bash
cd /abhay_ws/ur-assembly
colcon build --packages-select ur_gripper_demo --symlink-install
source install/setup.bash

ros2 run ur_gripper_demo gripper_demo
# or
ros2 launch ur_gripper_demo gripper_demo.launch.py cycles:=3
```

Quick manual test without this package (`ParallelGripperCommand`, goal is a
`sensor_msgs/JointState`):
```bash
ros2 action send_goal /robotiq_gripper_controller/gripper_cmd \
  control_msgs/action/ParallelGripperCommand \
  "{command: {name: [robotiq_85_left_knuckle_joint], position: [0.8]}}"   # 0.0 opens, 0.8 closes
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `action_name` | `/robotiq_gripper_controller/gripper_cmd` | ParallelGripperCommand action server. |
| `joint_name` | `robotiq_85_left_knuckle_joint` | Joint name put in the JointState goal. |
| `open_position` | `0.0` | Joint position for fully open. |
| `closed_position` | `0.8` | Joint position for fully closed. |
| `max_effort` | `50.0` | Grip force in the goal (ignored unless HW exposes an effort interface). |
| `max_velocity` | `0.5` | Speed in the goal (ignored unless HW exposes a velocity interface). |
| `dwell_s` | `1.5` | Pause between positions (s). |
| `cycles` | `1` | Number of open→closed→open cycles. |
| `fractions` | `[0.0, 0.25, 0.5, 0.75, 1.0, 0.0]` | "Closedness" steps per cycle (0=open, 1=closed). |
| `activate_first` | `false` | Call the reactivate service before cycling. |
| `activation_service` | `/robotiq_activation_controller/reactivate_gripper` | Trigger service for (re)activation. |

## Running with the arm (future)

Because the stock gripper bringup can't share a graph with the UR driver, simultaneous
arm + gripper operation needs them under **one controller_manager**: a combined URDF (UR10e +
2F-85 with both `ros2_control` hardware blocks) and a combined controllers yaml, launched via
the UR driver with `description_file:=` / `controllers_file:=`. That lives in a separate
`ur_gripper_bringup` package (planned). This `ur_gripper_demo` node works **unchanged** against
that setup — it only needs the `gripper_cmd` action, regardless of which manager owns it.

## Notes

- If open/closed look **inverted** for your config, swap the positions
  (e.g. `-p open_position:=0.8 -p closed_position:=0.0`).
- Just open and close (no intermediate steps): `-p fractions:="[0.0, 1.0]"`.
- If `gripper_demo` hangs on *"Waiting for gripper action server"* while the action **is**
  listed, it's an action-**type** mismatch — confirm with
  `ros2 action info /robotiq_gripper_controller/gripper_cmd -t` (must be
  `ParallelGripperCommand`).
- Keep clear of the fingers — the gripper closes with force.
