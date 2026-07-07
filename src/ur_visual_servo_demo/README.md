# ur_visual_servo_demo

**Eye-in-hand position-based visual servoing (PBVS)** for the UR10e. The wrist-mounted camera is
driven to a desired pose relative to a detected ArUco marker — directly in front of it, facing it
head-on, at a stand-off distance — and then **tracks** it: move the marker and the arm follows.

## How it works

Each control iteration (`rate_hz`):

```
1. read T_base_marker from tf                      (ur_vision_demo + ur_tf_demo)
2. desired camera pose:
     T_base_cam_des = T_base_marker · [Rx(cam_rpy_in_marker) | (0,0,standoff_m)]
3. pose error (current camera -> desired):
     within pos_deadband & ang_deadband  -> HOLD (no motion)
4. else step a clamped `gain` fraction toward the desired pose:
     T_cam_cmd    = interpolate(T_base_cam, T_base_cam_des, gain, max steps)   # slerp + clamp
     T_base_tool0 = T_cam_cmd · inv(T_tool0_camera)                            # back-solve tool0
     IK (retry, KDL) -> scaled_joint_trajectory_controller
```

The desired camera pose in the marker frame is `Rx(π)` at `standoff_m` along the marker normal —
optical axis facing the marker, centered. It's a proportional controller: it closes a fraction of
the remaining error each step (clamped for safety) so it converges smoothly and then holds inside
the deadband. **When the marker leaves view, the arm holds** (stops commanding motion).

This is the same servo geometry the pick-and-place demo uses for its approach — here it runs as a
continuous, standalone tracking loop.

## Prerequisites (all running)

1. **Arm driver / integrated bringup** with `scaled_joint_trajectory_controller` **active**
   (pass this robot's calibration — see `ur_gripper_bringup`).
2. **move_group** for `/compute_ik`:
   `ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10e kinematics_params_file:=<your calibration>`
3. **Vision + hand-eye tf** so the marker resolves in `base_link` and `tool0 → camera` is published:
   `ros2 launch ur_vision_demo aruco_demo.launch.py`
   `ros2 launch ur_tf_demo tf_streaming.launch.py`

Sanity check: with the marker in view,
`ros2 run tf2_ros tf2_echo base_link camera1_marker_<id>` returns a pose.

## Configure

Everything is in [config/visual_servo.yaml](config/visual_servo.yaml):

| Param | Meaning | Default |
|---|---|---|
| `marker.id` / `frame` | marker to servo on (match `ur_vision_demo`) | 0 |
| `standoff_m` | desired camera↔marker distance (hold point) | 0.20 m |
| `cam_rpy_in_marker` | desired camera orientation in the marker frame | [π,0,0] |
| `gain` | fraction of pose error closed per iteration | 0.4 |
| `max_linear_step_m` / `max_angular_step_deg` | per-iteration clamps (safety) | 0.02 m / 15° |
| `pos_deadband_m` / `ang_deadband_deg` | within → hold (no jitter) | 5 mm / 1.5° |
| `rate_hz` / `move_duration_s` | loop rate / time per corrective move | 2 Hz / 0.8 s |
| `marker_max_age_s` | older tf = "not in view" (tf2 caches the last one) | 0.5 s |
| `ik_timeout_s` / `ik_attempts` / `avoid_collisions` | IK behavior | 2 s / 12 / false |
| `confirm_start` / `confirm_each_move` | prompt once before the loop / before every move | true / false |
| `max_iterations` | 0 = until Ctrl-C; >0 = stop after N | 0 |

> `cam_rpy_in_marker: [π,0,0]` assumes the marker's **+Z faces the camera** (standard). If the arm
> drives the camera to the wrong side, flip this. `standoff_m` should be within reach and inside the
> camera's usable range.

## Build & run

```bash
cd /abhay_ws/ur-assembly
colcon build --packages-select ur_visual_servo_demo --symlink-install
source install/setup.bash

# hands-on (prompts render under ros2 run) -- keep the e-stop in hand:
ros2 run ur_visual_servo_demo visual_servo

# or the launch file (set confirm_start/confirm_each_move false first for hands-off):
ros2 launch ur_visual_servo_demo visual_servo_demo.launch.py
```

Then slowly move the marker — the arm re-centers to keep it in front of the camera at `standoff_m`.

## Safety notes

- This demo **moves the arm autonomously and continuously** — the riskiest demo in this workspace.
  Start with the **e-stop in hand**, a low `gain`, small `max_*_step`, and a slow marker.
- For a cautious first run set `confirm_each_move: true` (prompts before every correction — not true
  servoing, but lets you step through it).
- Motion is clamped per iteration and holds on marker loss, but there is **no obstacle avoidance**
  (`avoid_collisions: false`, no planning scene) — keep the workspace clear.
- `tool0` is the IK/control frame (matches the pendant), consistent with the other demos.
