# ur_cable_pick_place_demo

Cable pick‑and‑place for the UR10e + Robotiq 2F‑85. It's **`ur_pick_place_demo` with the single
fiducial detection replaced by a multi‑view scan** that feeds the **SAM3 cable‑connector pose
estimator** (external nodes in the `sam3-abhay` repo). This node subclasses `PickPlace`, so all the
IK / trajectory / gripper / grasp machinery is reused; it needs **no torch** — the SAM3 nodes run
separately and are coupled only through TF.

## Sequence

```
open → scan (multi-view) → estimate connector pose → grasp-align → grasp → close → lift
     → pre-place → place → open → retreat → home
```

1. **Scan** — the robot moves the **camera** to each `scan.camera_poses` view (camera‑w.r.t.‑base),
   holding still for `scan.dwell_s` so the SAM3 detector processes a clean frame and the pose
   estimator accumulates that view. The views must have **parallax** (the camera translates/rotates
   between them) while keeping the cable in frame.
2. **Estimate** — the SAM3 `connector_pose_node` fuses the views and broadcasts TF
   `base_link → connector`. This demo reads it and builds the grasp.
3. **Pick & place** — identical to `ur_pick_place_demo` from the grasp on.

### Connector frame convention

The SAM3 estimator's connector axis (its only well‑determined rotational DOF) is measured; this demo
rebuilds a full, deterministic grasp frame from it plus your "up" assumption:

- **x** = the cable‑connector **axis** (SAM3 measures it; it's that TF's z‑column),
- **z** = robot base **+Z** (`connector_up_axis`), re‑orthogonalized ⟂ x — the *"cable z coincident
  with base z"* assumption,
- **y** = `z × x` (right‑handed, horizontal).

### Fingertip frame — the grasp reference

A **fingertip frame** is defined relative to the gripper (`fingertip_grasp`, w.r.t. the
`grasp_tcp_offset` frame): translation `[0, 12.54, 181.65] mm` and axes **+x = gripper −y**,
**+y = gripper −x** (⟹ **+z = gripper −z**). The grasp is commanded so this fingertip frame
**coincides with the connector frame** — `tool0 = T_base_connector · inv(T_tool0_fingertip)` — so the
fingertip *is* the grasp reference the pick machinery uses (approach/lift are relative to it). The
gripper (fingers‑center) frame is kept as `T_tool0_gripper`. With `publish_fingertip_tf: true` the
node broadcasts `tool0 → fingertip` so you can confirm in RViz that it lands on the `connector` frame
at grasp. `connector_grasp` is now just an optional offset of the fingertip target from the connector
(default identity).

> **Euler convention:** the config `rpy` is **extrinsic XYZ** (`sxyz`). Those axes are `[180, 0, 90]°`
> in the **intrinsic‑XYZ** (moving‑frame) convention CAD tools report, which equals **`[180, 0, -90]°`**
> = `[π, 0, -π/2]` here — so the config uses `-π/2`, not `+π/2`, for the yaw.

## Prerequisites (all running)

1. **Integrated bringup** (arm + gripper): `ros2 launch ur_gripper_bringup ur_gripper_control.launch.py`
2. **move_group** for `/compute_ik` (with your calibration — see `ur_gripper_bringup`).
3. **Hand‑eye tf** so `base_link → camera` is published: `ros2 launch ur_tf_demo tf_streaming.launch.py`
4. **RealSense** publishing color image + `camera_info`.
5. **SAM3 nodes** (from the `sam3-abhay` repo), started separately — they produce the `connector` TF:
   ```bash
   # detector (needs torch + rclpy; ~1-2 s/frame):
   python scripts/cable_neck_ros_node.py --ros-args \
     -p image_topic:=/camera1/camera/color/image_raw
   # fusion -> base_link -> connector (plain rclpy, no torch):
   python scripts/connector_pose_node.py --ros-args \
     -p world_frame:=base_link -p connector_frame:=connector \
     -p necks_topic:=/cable_neck_detector/necks \
     -p camera_info_topic:=/camera1/camera/color/camera_info
   ```

## Configure — [config/cable_pick_place.yaml](config/cable_pick_place.yaml)

| Param | Meaning |
|---|---|
| `connector_frame` | TF the SAM3 estimator broadcasts (match its `connector_frame`) |
| `connector_max_age_s` / `connector_wait_s` | freshness of the estimate / how long to wait after the scan |
| `connector_up_axis` | base axis the connector z aligns to (base +Z assumption) |
| `connector_grasp` (xyz/rpy) | optional offset of the fingertip target from the connector (default identity) |
| `grasp_tcp_offset` (xyz/rpy) | gripper (fingers‑center) frame, relative to `tool0` |
| `fingertip_grasp` (xyz/rpy) | fingertip frame w.r.t. the gripper; the **grasp reference** (xyz `[0, 12.54, 181.65] mm`, rpy `[π,0,-π/2]` sxyz) |
| `publish_fingertip_tf` | broadcast `tool0 → fingertip` for RViz verification |
| `scan.camera_poses` | list of camera‑w.r.t.‑base views to visit (**set for your cell**) |
| `scan.dwell_s` | hold time per view (≥ SAM3 inference, ~2 s) |
| `save_scan_images` / `debug_image_topic` | save the SAM3 overlay per view / Node 1's `~/debug_image` |
| `data_dir` / `scan_images_subdir` | where overlays go: `<data_dir>/<subdir>/<timestamp>/view_NN.png` |
| `approach_*` / `lift_*` / `place_offset_*` / `gripper.*` | same as `ur_pick_place_demo` |

**Scan overlays:** at each view the demo saves Node 1's annotated frame (masks + neck + orientation
arrow — the same output as the `sam3-abhay` CLI) to
`data/cable_pick_place/<run timestamp>/view_NN.png`, so you get a visual record of what SAM3 saw.
Run `cable_neck_ros_node` with `publish_debug:=true`, and run this demo from the workspace root so
`data/` resolves there.

## Build & run

```bash
cd /abhay_ws/ur-assembly
colcon build --packages-select ur_pick_place_demo ur_cable_pick_place_demo --symlink-install
source install/setup.bash

ros2 run ur_cable_pick_place_demo cable_pick_place     # prompts render under `ros2 run`
```

## Notes & caveats

- **Set `scan.camera_poses` for your cell.** The bundled examples look down at ~`[0.4, 0, *]`; they
  must keep *your* cable in view and provide parallax (the fusion won't triangulate without it).
- **SAM3 timing:** the detector runs ~1–2 s/frame and drops frames while busy, so `dwell_s` must be
  long enough to get one clean, static frame per view.
- **Estimator history:** `connector_pose_node` accumulates a rolling window of views. For a clean run
  restart it (or let its window roll over) so a previous cable's views don't bias the estimate.
- **Grasp clocking:** with the base‑Z‑up assumption the connector frame is fully determined, so the
  grasp is repeatable. If a cable ever tilts far from horizontal, revisit `connector_up_axis`.
- No collision avoidance (`avoid_collisions: false`) — keep the workspace clear; **e‑stop in hand**.
