# ur_tf_demo

Transform handling for the UR assembly cell, the **standard ROS2 way** (tf2). It streams, as
`geometry_msgs/PoseStamped` in the **robot base frame**:

- the **tool pose** — `base_link → tool0` — on `/pose_streamer/tool_pose`
- each **detected fiducial marker** — `base_link → <marker>` — on
  `/pose_streamer/markers/<marker_frame>`
- all markers together as a `PoseArray` on `/pose_streamer/marker_poses` (for RViz)

## How it works (and why tf2)

Rather than hand-composing transforms, the `pose_streamer` node runs a
`tf2_ros.TransformListener` and calls `lookup_transform(base, target)`. tf2 composes the chain
published by the existing nodes:

```
base_link ──(UR driver / robot_state_publisher)──▶ tool0
tool0 ──(hand-eye static transform, here)────────▶ camera1_color_optical_frame   # eye-in-hand
camera1_color_optical_frame ──(ur_vision_demo)───▶ camera1_marker_<id>
```
(For a fixed/world camera — eye-to-hand — the hand-eye static transform hangs off `base_link`
instead; set `parent_frame: base_link` in the config.)

So `base_link → tool0` and `base_link → camera1_marker_<id>` both resolve through one tf tree —
no manual matrix math, and any consumer can also just use tf directly. This node simply
**republishes** the lookups as pose topics for convenience.

The piece that ties the camera tree to the robot tree is the **hand-eye transform**
(`base_link → camera optical frame`), published here from
[`config/hand_eye.yaml`](config/hand_eye.yaml). Until you put a real calibration there, the
marker→base poses will be wrong (but `base→tool0` is always correct).

## Prerequisites

These must be running so the tf chain is complete:
- **UR driver** — provides `base_link → … → tool0`:
  `ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=192.168.125.2`
- **Vision demo** — provides `camera → marker`:
  `ros2 launch ur_vision_demo aruco_demo.launch.py`

(`base→tool0` streams with just the UR driver; markers need the vision demo + a marker in view.)

## Configure the hand-eye transform

Edit [`config/hand_eye.yaml`](config/hand_eye.yaml): set `base_frame`/`tool_frame`, the
`marker_frame_prefixes` (must match `ur_vision_demo`'s `<camera>_marker_`), and one `cameras:`
entry per camera with `parent_frame` (`tool0` for eye-in-hand, `base_link` for a fixed camera),
`camera_frame`, and the `parent_frame → camera_frame` extrinsics (`xyz` in meters, `rpy` in
radians) from your hand-eye calibration.

> Don't have a calibration yet? You can eyeball/measure a rough `xyz`+`rpy` to sanity-check the
> pipeline, then swap in the calibrated values (e.g. from `easy_handeye2`).

## Build & run

```bash
cd /abhay_ws/ur-assembly
colcon build --packages-select ur_tf_demo --symlink-install
source install/setup.bash

ros2 launch ur_tf_demo tf_streaming.launch.py
```

Watch the streams:
```bash
# tool pose in the base frame
ros2 topic echo /pose_streamer/tool_pose

# all markers in the base frame
ros2 topic echo /pose_streamer/marker_poses

# a specific marker
ros2 topic echo /pose_streamer/markers/camera1_marker_0
```
Cross-check against raw tf:
```bash
ros2 run tf2_ros tf2_echo base_link tool0
ros2 run tf2_ros tf2_echo base_link camera1_marker_0
```
RViz: Fixed Frame `base_link`, add **TF** and **PoseArray** (`/pose_streamer/marker_poses`).

## Parameters (`pose_streamer`)

| Parameter | Default | Description |
|---|---|---|
| `base_frame` | `base_link` | Reference frame for all output poses. |
| `tool_frame` | `tool0` | Tool frame to stream. |
| `marker_frame_prefixes` | `['camera1_marker_']` | tf frames matching these prefixes are streamed as markers. |
| `publish_rate_hz` | `10.0` | Lookup/publish rate. |
| `max_marker_age_s` | `1.0` | Skip markers whose latest tf is older than this (out of view). |

## Notes

- **Marker discovery is dynamic** — markers are found by scanning tf for frames matching the
  prefixes, so markers appear/disappear as they enter/leave view (older than `max_marker_age_s`
  are dropped). No need to pre-list marker IDs.
- **Multiple cameras:** add a `cameras:` entry (with its own hand-eye extrinsics) and the
  matching `<camera>_marker_` prefix; all cameras' markers stream into the same base frame.
- **`base` vs `base_link`:** UR publishes both (`base` is rotated 180° about Z). This uses
  `base_link` (the URDF root). Change `base_frame` if you want poses in `base`.
- If `base→tool0` warns "no transform", the UR driver isn't running or uses a different base
  frame name; if markers never appear, check the hand-eye transform exists
  (`ros2 run tf2_ros tf2_echo base_link camera1_color_optical_frame`) and that the vision demo
  sees a marker.
