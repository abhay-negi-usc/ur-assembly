# ur_kinematic_assembly_demo

A **kinematic** assembly demo for the UR10e — **no vision, no gripper**. The part is rigidly
attached to the flange; all poses are **ground truth** from config, and the assembly path is a
**CSV trajectory**. The robot goes to a stand-off, then executes the trajectory under **position**
or **admittance** control.

## What it does

1. **Stand-off** — move so the held object is at the assembled pose backed off `standoff_distance_m`
   along `standoff_axis` (in the assembled-object frame). Position-controlled.
2. **Execute the assembly trajectory** — IK each CSV waypoint (chained seeds → continuous joint
   path) and run it, in the selected `control_mode`:
   - **position** — one multi-point `JointTrajectory` to `scaled_joint_trajectory_controller`.
   - **admittance** — switch to the ros2_control `admittance_controller`, apply the tunable
     mass/damping/stiffness, tare the F/T, and stream the waypoints as joint references while the
     arm yields to contact — with a **force-guarded stop** (`max_force_n` / `max_torque_nm`).
3. **Wind-down** (optional, `return_home_after`) — retract to the stand-off, return home.

### Frames

Config gives `held_object_pose` = held object w.r.t. `tool0`, and `assembled_pose` = held object in
`base` when assembled. The CSV rows are held-object poses **relative to the assembled pose**
(`T_assembled_held`). Each is commanded as `tool0 = T_base_assembled · T_assembled_held ·
inv(held_object_pose)`.

## Trajectory CSV

[config/assembly_trajectory.csv](config/assembly_trajectory.csv) — columns `x,y,z,roll,pitch,yaw`
(meters, radians unless `trajectory_angles_deg: true`), one waypoint per row, in order. Lines
starting with `#` and a header row are ignored. Start near the stand-off; end at the assembled pose
(last row ≈ all zeros). The bundled example is a 5 cm straight insertion along the assembled +Z.

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

## Prerequisites

- Arm driver / bringup with `scaled_joint_trajectory_controller` **active** (pass this robot's
  calibration — see `ur_gripper_bringup`).
- `move_group` for `/compute_ik`.
- (admittance only) `admittance_controller` loaded inactive; a live wrist F/T sensor.

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
| `held_object_pose` (xyz/rpy) | Held object w.r.t. `tool0` (ground truth) |
| `assembled_pose` (xyz/rpy) | Held object in `base` when assembled (ground truth; trajectory end) |
| `standoff_distance_m` / `standoff_axis` | Stand-off = assembled backed off along this (assembled frame) |
| `trajectory_csv` / `trajectory_angles_deg` | The waypoint CSV / whether its angles are degrees |
| `control_mode` | `position` or `admittance` |
| `waypoint_dt_s` | Time per trajectory segment |
| `return_home_after` | Retract to stand-off + home when done |
| `admittance.mass/damping_ratio/stiffness/selected_axes` | Tunable compliance (6-vectors) |
| `admittance.apply_params` | Push those to the controller at runtime |
| `admittance.max_force_n` / `max_torque_nm` | Force-guarded stop (0 disables) |

## Safety

- Moves the arm autonomously; **e-stop in hand**. `confirm_each_step: true` prompts before each move.
- **Admittance mode applies force.** Start with a conservative `max_force_n` and moderate stiffness.
  On any abort/Ctrl-C the node **switches back to position control** so the arm isn't left compliant.
- No collision avoidance (`avoid_collisions: false`, no planning scene) — keep the workspace clear.
- IK is chained for continuity, but a bad waypoint can still be unreachable; it aborts and names the
  failing waypoint index.
