# ur_admittance_demo

An **admittance-control** demo for a UR10e on **ROS2 Jazzy**, using the ros2_control
[`admittance_controller`](https://control.ros.org/jazzy/doc/ros2_controllers/admittance_controller/doc/userdoc.html).

The demo holds the robot's initial pose as a reference; you **push or pull the TCP by hand**
and it yields to the force, then springs back per the configured Cartesian
mass/damping/stiffness (`F = M·a + D·v + S·(x − x_d)`).

> ⚠️ **Real robot only.** Admittance reacts to the wrist **force-torque sensor**. On fake
> hardware / sim the FT reads zero, so there is nothing to comply with. Run this on the real
> UR10e with the **e-stop in hand** — the arm is intentionally compliant and will move when
> touched.

## How it works

`admittance_controller` is **not** part of the UR driver's controller set, so you install it,
load it onto the running `controller_manager` with the config in
[`config/ur_admittance_controller.yaml`](config/ur_admittance_controller.yaml), and switch to
it. It:

1. Reads a **joint-space reference** on `<controller>/joint_references`
   (`trajectory_msgs/JointTrajectoryPoint`), does FK to a desired Cartesian pose.
2. Reads the wrist FT sensor (`tcp_fts_sensor/force.*`, `torque.*`) and applies Cartesian
   admittance in the `control.frame` (`tool0`).
3. IKs the resulting compliant pose back to joint **position** commands (the same UR interface
   `scaled_joint_trajectory_controller` uses — so the two conflict; only one is active).

The [`admittance_hold_demo`](ur_admittance_demo/admittance_hold_demo.py) node just captures the
initial joint configuration and republishes it as the constant reference, so the hold target =
where the arm started.

## Prerequisites

Install the controller and the KDL kinematics plugin (not pulled in by the UR driver):

```bash
sudo apt update
sudo apt install ros-jazzy-admittance-controller ros-jazzy-kinematics-interface-kdl
```

Start the **real** driver (FT sensor must be live). Pass this robot's calibration so FK matches
the hardware (extract it once — see `ur_gripper_bringup/README.md`):

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.125.2 \
  kinematics_params_file:=$(ros2 pkg prefix ur_gripper_bringup)/share/ur_gripper_bringup/config/ur10e_calibration.yaml
# then start the External Control program on the teach pendant
```

Confirm the FT sensor interfaces and the joint command interface exist (names used in the
YAML — adjust the YAML if yours differ):

```bash
ros2 control list_hardware_interfaces | grep -E 'tcp_fts_sensor|position|velocity'
ros2 topic echo /force_torque_sensor_broadcaster/wrench --once   # sanity: nonzero when pushed
```

## Bring up the admittance controller

Load it onto the running controller_manager (inactive), then switch to it:

```bash
# from the workspace, after building + sourcing (see below)
ros2 run controller_manager spawner admittance_controller \
  --param-file $(ros2 pkg prefix ur_admittance_demo)/share/ur_admittance_demo/config/ur_admittance_controller.yaml \
  --controller-manager /controller_manager --inactive

ros2 control switch_controllers \
  --deactivate scaled_joint_trajectory_controller \
  --activate admittance_controller

ros2 control list_controllers | grep admittance     # -> active
```

To stop / hand control back:
```bash
ros2 control switch_controllers \
  --deactivate admittance_controller \
  --activate scaled_joint_trajectory_controller
```

## Build & run the hold demo

```bash
cd ~/abhay_ws/ur-assembly
colcon build --packages-select ur_admittance_demo --symlink-install
source install/setup.bash

ros2 run ur_admittance_demo admittance_hold_demo
# or: ros2 launch ur_admittance_demo admittance_hold_demo.launch.py
```

Now push/pull the TCP — it should give and return. `Ctrl-C` stops the node (the controller
keeps holding the last reference; switch back to the JTC to end compliance).

## Tuning (`config/ur_admittance_controller.yaml`)

`admittance.{mass, damping_ratio, stiffness}` are 6-vectors `[x, y, z, rx, ry, rz]`:

- **stiffness** (N/m, Nm/rad): lower → softer / easier to push, larger excursions. Set an axis
  to `0` for free-floating (hand-guiding) on that axis. Default `200`/`15`.
- **mass** (kg, kg·m²): higher → more sluggish/heavy feel.
- **damping_ratio**: `1.0` ≈ critically damped; lower → springier/overshoot.
- **selected_axes**: set `false` to make an axis rigid (no compliance).

After editing, re-load the controller (unload + spawn again, or restart it).

## Things to verify for your setup

| YAML field | Default | Check |
|---|---|---|
| `ft_sensor.name` | `tcp_fts_sensor` | `ros2 control list_hardware_interfaces \| grep fts` |
| `ft_sensor.frame.id` | `tool0` | FT measurement frame in your URDF |
| `command_interfaces` | `[position]` | UR exposes `position` + `velocity`; switch if you prefer velocity |
| `kinematics.base` / `tip` | `base_link` / `tool0` | match your URDF frames |
| `gravity_compensation` | `force: 0.0` | set a real CoG/force if NOT using pendant payload comp |

If pushing produces no motion: check the wrench is nonzero when you push, that
`admittance_controller` is **active**, and that stiffness isn't so high the excursion is tiny.
