# ur_pick_place_demo

A **fiducial-guided pick-and-place** demo for the UR10e + Robotiq 2F‑85 (ROS2 Jazzy). The
object carries an ArUco marker; the robot detects it, computes the object and grasp poses, and
runs an approach → grasp → lift → place → release sequence. **Place = pick pose + a configured
offset.**

## How it works

```
T_base_marker   = tf lookup base_link -> camera1_marker_<id>     (vision + hand-eye)
T_base_object   = T_base_marker * inv(object_marker)             (object pose in base)
T_base_grasp    = T_base_object * object_grasp                   (grasp-TCP target in base)
tool0 target    = T_base_grasp * inv(grasp_tcp_offset)           (IK solves for tool0)
```

Each arm move is MoveIt `/compute_ik` for `tool0` + a `FollowJointTrajectory` goal to
`scaled_joint_trajectory_controller`; the gripper uses the `ParallelGripperCommand` action. The
grasp pose is expressed for a **grasp‑TCP between the fingers**; since IK solves for `tool0`,
each grasp‑TCP target is converted to a `tool0` target via `inv(grasp_tcp_offset)`.

Sequence: `open → pre-grasp → grasp → close → lift → pre-place → place → open → retreat → home`
(home = the joint configuration captured at start).

## Prerequisites (all running)

1. **Integrated bringup** (arm + gripper, one controller_manager):
   `ros2 launch ur_gripper_bringup ur_gripper_control.launch.py`
   The bringup defaults to this robot's calibration
   (`ur_gripper_bringup/config/ur10e_calibration.yaml`); see that package's README to extract it.
2. **move_group** for `/compute_ik` (arm-only MoveIt config is fine — IK targets `flange`).
   ⚠️ It **must** use the **same** calibration as the bringup, or the driver's FK and MoveIt's
   IK disagree and grasps land off:
   ```
   ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10e \
     kinematics_params_file:=$(ros2 pkg prefix ur_gripper_bringup)/share/ur_gripper_bringup/config/ur10e_calibration.yaml
   ```
3. **Vision + hand-eye tf** so the marker resolves in `base_link`:
   `ros2 launch ur_vision_demo aruco_demo.launch.py`
   `ros2 launch ur_tf_demo tf_streaming.launch.py`

Sanity check before running: with the object's marker in view,
`ros2 run tf2_ros tf2_echo base_link camera1_marker_<id>` should return a pose.

## Configure

Everything is in [config/pick_place.yaml](config/pick_place.yaml):

| Param | Meaning | Default |
|---|---|---|
| `marker.id` / `dictionary` / `size_m` | The object's marker (match `ur_vision_demo`) | 0 / DICT_4X4_50 / 0.05 |
| `object_marker` (xyz/rpy) | Marker pose in the object frame (`T_object_marker`) | z = 0.25 m |
| `object_grasp` (xyz/rpy) | Grasp‑TCP pose in the object frame (`T_object_grasp`) | identity |
| `grasp_tcp_offset` (xyz/rpy) | Grasp point between fingers, relative to `tool0` | z = 0.16 m ⚠️ **set to your hardware** |
| `approach_distance_m` / `approach_axis` | Pre-grasp stand-off (grasp frame) | 0.10 m / −Z |
| `lift_distance_m` / `lift_axis` | Lift after grasp (base frame) | 0.10 m / +Z |
| `place_offset_xyz` / `place_offset_rpy` | Place = pick shifted by this (base frame) | +0.20 m Y |
| `gripper.*` | action, joint, open/closed positions, effort, velocity | 0.0 / 0.8 / 50 / 0.5 |
| `move_duration_s`, `settle_s`, `confirm_each_step` | timing + safety | 4 s, 0.5 s, true |

⚠️ **`grasp_tcp_offset` is the one you must set** — it's the distance from `tool0` to the
fingertip contact point (≈ coupler 0.016 + the 2F‑85 base-to-fingertip length). The default
0.16 m is approximate; measure it (or read it off the URDF) for accurate grasps.

## Build & run

```bash
cd /abhay_ws/ur-assembly
colcon build --packages-select ur_pick_place_demo --symlink-install
source install/setup.bash

ros2 launch ur_pick_place_demo pick_place_demo.launch.py
# or, hands-off:  ros2 run ur_pick_place_demo pick_place --ros-args -p ...
```

With `confirm_each_step: true` (default) it prompts before every motion — **use `ros2 run`**
in an interactive terminal so the prompts render (or set it false for the launch file). Keep
the **e‑stop in hand**.

## Notes / safety

- **Reachability** — if a pose logs `IK failed`, it's unreachable from the current
  configuration (e.g. grasp too low / behind the robot); adjust the object position,
  `place_offset`, or approach/lift distances.
- **Collisions** — IK uses the arm-only MoveIt model, so it won't avoid the table/object; the
  approach/retreat stand-offs are what keep the path clean. Start with generous distances.
- **Grip width** — `gripper.closed_position` (0.8 = fully closed) should be tuned to the object
  so it grasps without over-closing; a stall is a normal grasp outcome.
- **One-shot** — the node performs a single pick-and-place then exits.
