# ur_assembly_demo

Fiducial-guided **pick-and-assemble** for the UR10e + Robotiq 2F-85. It **picks exactly like
[`ur_pick_place_demo`](../ur_pick_place_demo/)** (this node subclasses that one and reuses its whole
pick pipeline), then mates the held part into a fiducial-located target using **compliance
(admittance) control** for the final insertion.

## Sequence

1. **Pick** (identical to pick-and-place, up through lift): detect → visual approach → estimate →
   grasp → close → lift. Honors `blind_pick` and all the pick params.
2. **View pose** — move so the camera sits at `assembly.view_pose` (camera w.r.t. base) to see the
   target assembly's marker.
3. **Detect target** — read the target marker and compute the target object pose:
   `T_base_targetobj = T_base_targetmarker · inv(target_object_marker)`.
4. **Pre-align** — move the held part to `prealign_distance_m` back from the assembled pose along
   `assembly.approach_axis` (target-object frame), at the assembled orientation.
5. **Stand-off** — move to `standoff_distance_m` (just before contact).
6. **Compliant mate** — **tare** the wrist F/T, **switch** to the `admittance_controller`, and ramp
   a joint reference from the stand-off to the **assembled pose** (`assembly.assembled_pose` = held
   object w.r.t. target object) while the arm yields to contact forces.
7. **Release** — open the gripper.
8. **Retract** — switch back to position control, then retract to the stand-off → view pose → home.

The held object's pose relative to `tool0` is fixed once grasped
(`T_tool0_held = grasp_tcp_offset · inv(object_grasp)`), so every assembly target is commanded as
`tool0 = T_base_targetobj · transl(axis·d) · assembled_pose · inv(T_tool0_held)`.

## Prerequisites (all running)

1. **Integrated bringup** (arm + gripper, one controller_manager) — `ur_gripper_bringup`.
2. **move_group** for `/compute_ik` — pass this robot's calibration (see `ur_gripper_bringup`).
3. **Vision + hand-eye tf** — `ur_vision_demo` + `ur_tf_demo` (both the picked object's marker and
   the **target** marker must be detectable / published as `camera1_marker_<id>`).
4. **admittance_controller LOADED (inactive)** on the controller_manager, for the mate:
   ```bash
   sudo apt install ros-jazzy-admittance-controller ros-jazzy-kinematics-interface-kdl
   ros2 run controller_manager spawner admittance_controller \
     --param-file $(ros2 pkg prefix ur_admittance_demo)/share/ur_admittance_demo/config/ur_admittance_controller.yaml \
     --controller-manager /controller_manager --inactive
   ```
   The demo switches to it (and back) itself; you only need it **loaded**. Tune stiffness/damping
   in `ur_admittance_demo/config/ur_admittance_controller.yaml`.

## Configure

Everything is in [config/assemble.yaml](config/assemble.yaml) — the **full pick config** (same
schema as `pick_place.yaml`) plus an `assembly:` section:

| Param | Meaning |
|---|---|
| `assembly.view_pose` (xyz/rpy) | Camera pose w.r.t. base for viewing the target marker |
| `assembly.target_marker.id` | Marker fixed to the target assembly (≠ the picked object's marker) |
| `assembly.target_object_marker` (xyz/rpy) | Target marker pose **in the target-object frame** |
| `assembly.assembled_pose` (xyz/rpy) | Held-object pose **in the target-object frame** when mated (the insertion goal) |
| `assembly.approach_axis` | Mating axis in the target-object frame (separation direction) |
| `assembly.prealign_distance_m` / `standoff_distance_m` | Distances back along the axis |
| `assembly.compliance.*` | admittance/position controller names, switch + F/T-tare services, reference topic, insertion ramp time/rate/hold |
| `assembly.compliance.max_force_n` / `max_torque_nm` | **force-guarded stop**: halt the insertion when the tared contact force/torque reaches this (0 disables) |
| `assembly.compliance.wrench_topic` | F/T wrench topic to monitor (default `/force_torque_sensor_broadcaster/wrench`) |

## Build & run

```bash
cd /abhay_ws/ur-assembly
colcon build --packages-select ur_pick_place_demo ur_assembly_demo --symlink-install
source install/setup.bash

ros2 run ur_assembly_demo assemble        # prompts render under `ros2 run`
```

With `confirm_each_step: true` (default) it prompts before every motion — **use `ros2 run`**. Keep
the **e-stop in hand**.

## Safety & notes

- **The mate applies force.** Compliance behavior (how hard it pushes, how much it yields) is set by
  the admittance stiffness/damping in `ur_admittance_demo`'s yaml. Start soft.
- On any abort/Ctrl-C the node **switches back to position control** so the arm isn't left compliant.
- **Force-guarded stop:** the insertion ramps the reference from stand-off to assembled over
  `insertion_time_s`, but **stops advancing the moment the tared contact wrench reaches
  `max_force_n` / `max_torque_nm`**, then holds there (part seated). The wrench is read *after* the
  F/T tare, so it reflects contact only. Set both to `0` to disable and drive to the full reference.
  Start with a conservative `max_force_n` and low admittance stiffness.
- **No collision avoidance** (`avoid_collisions: false`, no planning scene) — keep the workspace clear.
- Requires two distinct markers in view at their respective stages: the picked object's marker
  (during pick) and the target marker (at the view pose).
