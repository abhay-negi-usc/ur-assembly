# ur_kinematic_assembly_demo

A **kinematic** assembly demo for the UR10e — **no vision, no gripper**. The part is rigidly
attached to the flange; all poses are **ground truth** from config, and the assembly path is a
**CSV trajectory**. The robot goes to a stand-off, then executes the trajectory under **position**
or **admittance** control.

## What it does

1. **Stand-off** — move so `tool0` is at the assembled pose backed off `standoff_distance_m`
   along `standoff_axis` (in the assembled tool0 frame). Position-controlled.
2. **Execute the assembly trajectory** — IK each CSV waypoint (chained seeds → continuous joint
   path) and run it, in the selected `control_mode`:
   - **position** — one multi-point `JointTrajectory` to `scaled_joint_trajectory_controller`.
   - **admittance** — switch to the ros2_control `admittance_controller`, apply the tunable
     mass/damping/stiffness, tare the F/T, and stream the waypoints as joint references while the
     arm yields to contact — with a **force-guarded stop** (`max_force_n` / `max_torque_nm`).
3. **Wind-down** — one of:
   - **Disassembly** (`disassemble_after: true`) — run the trajectory in **reverse** (extraction, in
     the same `control_mode`), then go to the stand-off, then the initial pose.
   - **Simple retract** (`return_home_after: true`, if not disassembling) — straight to the stand-off,
     then home.

### Frames — everything is TOOL0

There is **no object frame to define**. `assembled_pose` is the **tool0** pose in base when
assembled — the value you can poll directly. The CSV rows are **tool0** poses **relative to that
assembled pose** (`T_assembled_tool0`), and each is commanded as
`tool0 = assembled_pose · T_assembled_tool0`. (The part is still rigidly on the flange and rides
along; you just author the path in tool0 terms, which is what you can measure.)

## Trajectory CSV

[config/assembly_trajectory.csv](config/assembly_trajectory.csv) — columns `x,y,z,roll,pitch,yaw`
(meters, radians unless `trajectory_angles_deg: true`), one waypoint per row, in order. Lines
starting with `#` and a header row are ignored. Each row is a **tool0** pose relative to the
assembled pose. Start near the stand-off; end at the assembled pose (last row ≈ all zeros). The
bundled example is a 5 cm straight insertion along the assembled tool0 +Z.

## Control mode

`control_mode: position` (default) or `admittance`.

**Admittance** needs the ros2_control `admittance_controller` **loaded (inactive)** first — see
`ur_admittance_demo`:
```bash
sudo apt install ros-jazzy-admittance-controller ros-jazzy-kinematics-interface-kdl
ros2 run controller_manager spawner admittance_controller \
  --param-file $(ros2 pkg prefix ur_admittance_demo)/share/ur_admittance_demo/config/ur_admittance_controller.yaml \
  --controller-manager /controller_manager --inactive
```
The demo switches to it and back itself. The **tunable control parameters** live under
`admittance:` in this demo's yaml (`mass`, `damping_ratio`, `stiffness`, `selected_axes`). With
`apply_params: true` they're set on the controller's own node (`/<controller>/set_parameters`, names
`admittance.*`) **before** it's activated. Per the `admittance_controller` source, all four are
dynamic (not `read_only`) and `enable_parameter_update_without_reactivation` defaults to `true`, so
they apply reliably — this isn't the pre‑Humble "params on /controller_manager" layout.

## Bring up the robot (no gripper, no camera)

This demo uses **no vision and no gripper** — don't launch `ur_vision_demo`, `ur_tf_demo`, or the
gripper bringup. With the gripper and camera physically removed, bring up the **bare arm**. `tool0`
is the flange TCP — the frame the whole demo is authored in (`assembled_pose` and the trajectory are
all tool0 poses), so the attached part never enters the kinematic chain.

### One command (recommended)

This package ships an arm-only bring-up that starts the driver **and** move_group (**and**,
optionally, the admittance controller) with this robot's calibration wired in:

```bash
# position mode:
ros2 launch ur_kinematic_assembly_demo arm_bringup.launch.py

# admittance mode (also loads admittance_controller, inactive):
ros2 launch ur_kinematic_assembly_demo arm_bringup.launch.py load_admittance:=true
```
Then **Play the External Control program on the pendant**.

Args: `robot_ip` (192.168.125.2), `ur_type` (ur10e), `kinematics_params_file` (defaults to
`ur_gripper_bringup`'s `ur10e_calibration.yaml`; **falls back to generic kinematics with a warning**
if missing), `launch_moveit` (true), `load_admittance` (false), `launch_rviz` (false → MoveIt RViz).

### Or step by step

Let `CAL=$(ros2 pkg prefix ur_gripper_bringup)/share/ur_gripper_bringup/config/ur10e_calibration.yaml`
(your extracted calibration — see `ur_gripper_bringup/README.md`).

**1. Arm driver — arm only, with calibration** (brings up `scaled_joint_trajectory_controller` +
`force_torque_sensor_broadcaster`):
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.125.2 \
  kinematics_params_file:=$CAL
# then Play the External Control program on the pendant
```

**2. move_group** (same calibration), for `/compute_ik`:
```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10e kinematics_params_file:=$CAL
```

**3. (admittance mode only)** load the admittance controller inactive — see the section above.

> Alternative one-shot for the arm: `ur_gripper_bringup` already defaults to the calibration — set
> `gripper_enabled: false` in its `config/coupler.yaml` and launch it; it brings up **arm-only**
> (omits the gripper from the URDF and its controllers).

Sanity checks:
```bash
ros2 control list_controllers | grep scaled_joint_trajectory   # ...] active
ros2 service list | grep compute_ik                            # present
# admittance mode:
ros2 control list_controllers | grep admittance                # loaded (inactive)
ros2 topic echo /force_torque_sensor_broadcaster/wrench --once  # nonzero when you push the tool
```

## Build & run

```bash
cd /abhay_ws/ur-assembly
colcon build --packages-select ur_kinematic_assembly_demo --symlink-install
source install/setup.bash

ros2 run ur_kinematic_assembly_demo kinematic_assembly     # prompts render under `ros2 run`
```

## Configure — [config/kinematic_assembly.yaml](config/kinematic_assembly.yaml)

| Param | Meaning |
|---|---|
| `assembled_pose` (xyz/rpy) | **tool0** pose in `base` when assembled (poll `base_link → tool0`; trajectory end) |
| `standoff_distance_m` / `standoff_axis` | Stand-off = assembled backed off along this (assembled tool0 frame) |
| `trajectory_csv` / `trajectory_angles_deg` | The waypoint CSV / whether its angles are degrees |
| `control_mode` | `position` or `admittance` |
| `waypoint_dt_s` | Time per trajectory segment |
| `disassemble_after` | After assembling, run the trajectory in reverse (extraction) → stand-off → home |
| `return_home_after` | If not disassembling, retract to stand-off + home when done |
| `admittance.mass/damping_ratio/stiffness/selected_axes` | Tunable compliance (6-vectors) |
| `admittance.apply_params` | Push those to the controller at runtime |
| `admittance.max_force_n` / `max_torque_nm` | Force-guarded stop (0 disables) |

## Polling the tool / assembled pose

Useful for **capturing or verifying the assembled pose** — e.g. jog the arm (or run an assembly) to
a known-good mate, then read where `tool0` ended up. `tf2_echo` only *reads* the tf tree, so
something must be **publishing** it first: that's `robot_state_publisher`, started by the driver,
which turns the robot's live joint angles into the `base_link → … → tool0` transforms. Two terminals:

**Terminal 1 — bring up the arm** (publishes the tf; you do NOT need to Play External Control just to
*read* the pose — the driver streams joint states over RTDE as soon as it connects):
```bash
cd /abhay_ws/ur-assembly && source install/setup.bash
ros2 launch ur_kinematic_assembly_demo arm_bringup.launch.py launch_moveit:=false
```

**Terminal 2 — poll:**
```bash
cd /abhay_ws/ur-assembly && source install/setup.bash
ros2 topic hz /tf                                   # confirm the tree is ticking
ros2 run tf2_ros tf2_echo base_link tool0           # tool0 pose in base, ~1 Hz (add a number for Hz)
```
Use `base` instead of `base_link` to match the pendant convention (they differ by 180° about Z).

**Paste it directly.** `assembled_pose` in the yaml **is** the `tool0` pose in base — the same thing
`tf2_echo base_link tool0` prints — so at a successful mate just copy the reading straight into
`assembled_pose` (xyz + the RPY). No object-to-tool0 conversion, nothing else to know.

**Poll `base_link`, not `base`** — the yaml's `base_frame` is `base_link`, and `base` is rotated
180° about Z from it; pasting a `base` reading would flip the assembled pose 180° about Z.

**As a topic instead of a terminal read:** with the driver up, `ur_tf_demo` streams `base → tool` as
a `PoseStamped`:
```bash
ros2 launch ur_tf_demo tf_streaming.launch.py publish_static_tf:=false   # skips the (absent) camera tf
```

> For accurate readings the driver must run with this robot's **calibration** (the `arm_bringup`
> launch wires it in) — otherwise `tool0` is off from the real TCP by mm–cm.

## Safety

- Moves the arm autonomously; **e-stop in hand**. `confirm_each_step: true` prompts before each move.
- **Admittance mode applies force.** Start with a conservative `max_force_n` and moderate stiffness.
  On any abort/Ctrl-C the node **switches back to position control** so the arm isn't left compliant.
- No collision avoidance (`avoid_collisions: false`, no planning scene) — keep the workspace clear.
- IK is chained for continuity, but a bad waypoint can still be unreachable; it aborts and names the
  failing waypoint index.
