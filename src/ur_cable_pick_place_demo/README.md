# ur_cable_pick_place_demo

Cable pick‑and‑place for the UR10e + Robotiq 2F‑85. It's **`ur_pick_place_demo` with the single
fiducial detection replaced by a multi‑view scan** that feeds the **SAM3 cable‑connector pose
estimator** (external nodes in the `sam3-abhay` repo). This node subclasses `PickPlace`, so all the
IK / trajectory / gripper / grasp machinery is reused; it needs **no torch** — the SAM3 nodes run
separately and are coupled only through TF.

## Sequence

```
open → scan (multi-view) → estimate connector pose → grasp-align → grasp → close → [grasp check]
     ↳ SHORT (cable not seated in fingertip groove) → open (drop) → return to initial pose → retry
     → lift → pre-place → place → open → retreat → home
```

1. **Scan** — the robot sweeps the **camera** through a set of views **relative to its pose at the
   start of the scan** (jog the robot so the cable is in view first). Each view is `scan.offsets[i]`
   applied in the **camera frame** and clamped to `scan.relative_bounds`, holding still for
   `scan.dwell_s` so the SAM3 detector processes a clean frame and the pose estimator accumulates
   that view. The offsets must give **parallax** (the camera translates between them) while keeping
   the cable in frame — no absolute cell coordinates to tune.
2. **Estimate** — the SAM3 `connector_pose_node` fuses the views and broadcasts TF
   `base_link → connector`. This demo reads it and builds the grasp.
3. **Grasp check** (`grasp_check.enabled`) — the grasp **commands a full close**, then reads the
   gripper position in **counts (0–255)** (converted from `/joint_states`). With the cable **seated in
   the fingertip groove** (or the fingers empty) the fingers reach `grasp_check.closed_counts` (~228);
   if the cable is caught **outside the groove** they stall **short** → failure. On failure the robot
   **opens (drops), returns to the initial pose, and retries** the whole scan→grasp sequence (up to
   `grasp_check.max_retries`). *(This detects only the not‑in‑groove failure — an empty pickup also
   reaches `closed_counts` and isn't distinguished yet.)*
4. **Pick & place** — identical to `ur_pick_place_demo` from the (successful) grasp on.

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

## Run — one process per terminal

Everything runs inside the Docker container; open a **new terminal per step** with
`docker exec -it jazzy-dev bash`. Only **Terminal 5** (the SAM3 detector) uses the SAM3 venv — all
the others use plain system Python. (One-time: build the workspace, and set up the SAM3 venv per the
`sam3-abhay` repo.)

**Build once** (any terminal):
```bash
source /opt/ros/jazzy/setup.bash
cd /abhay_ws/ur-assembly
colcon build --packages-select ur_pick_place_demo ur_cable_pick_place_demo --symlink-install
```

**Terminal 1 — arm + gripper bringup** (then press *Play* on the pendant's External Control program):
```bash
source /opt/ros/jazzy/setup.bash && source /abhay_ws/ur-assembly/install/setup.bash
ros2 launch ur_gripper_bringup ur_gripper_control.launch.py
```

**Terminal 2 — move_group** (`/compute_ik`, with this robot's calibration):
```bash
source /opt/ros/jazzy/setup.bash
CAL=$(ros2 pkg prefix ur_gripper_bringup)/share/ur_gripper_bringup/config/ur10e_calibration.yaml
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur10e kinematics_params_file:=$CAL
```

**Terminal 3 — RealSense camera** (frame `camera1_color_optical_frame`, and `publish_tf:=false` so it
doesn't double‑parent that frame — the hand‑eye tf owns it):
```bash
source /opt/ros/jazzy/setup.bash
ros2 launch realsense2_camera rs_launch.py \
  camera_name:=camera1 serial_no:=_218622272137 \
  enable_depth:=false enable_color:=true publish_tf:=false
# -> /camera1/color/image_raw + /camera1/color/camera_info, frame camera1_color_optical_frame
```

**Terminal 4 — hand‑eye tf** (publishes `tool0 -> camera1_color_optical_frame`):
```bash
source /opt/ros/jazzy/setup.bash && source /abhay_ws/ur-assembly/install/setup.bash
ros2 launch ur_tf_demo tf_streaming.launch.py
```

**Terminal 5 — SAM3 detector (Node 1)** — the ONLY terminal that activates the SAM3 venv (torch/GPU):
```bash
source /opt/sam3_venv/bin/activate && source /opt/ros/jazzy/setup.bash
python /abhay_ws/sam3-abhay/scripts/cable_neck_ros_node.py --ros-args \
  -p image_topic:=/camera1/color/image_raw -p publish_debug:=true
```

**Terminal 6 — SAM3 fusion (Node 2)** — plain system Python; broadcasts `base_link -> connector`:
```bash
source /opt/ros/jazzy/setup.bash
python /abhay_ws/sam3-abhay/scripts/connector_pose_node.py --ros-args \
  -p world_frame:=base_link -p connector_frame:=connector \
  -p necks_topic:=/cable_neck_detector/necks \
  -p camera_info_topic:=/camera1/color/camera_info
```

**Terminal 7 — the cable pick‑and‑place demo** (from the workspace root, so `data/` lands there):
```bash
cd /abhay_ws/ur-assembly
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 run ur_cable_pick_place_demo cable_pick_place
```

> The frame names must line up: RealSense publishes images in `camera1_color_optical_frame` (Terminal
> 3), the hand‑eye tf connects `base_link → camera1_color_optical_frame` (Terminal 4), Node 1 copies
> that frame onto `/cable_neck_detector/necks`, and Node 2 looks up `base_link ← camera1_color_optical_frame`
> to fuse — then the demo reads the resulting `base_link → connector`.

*(Optional Terminal 8 — RViz to watch the frames: `rviz2`, then add TF and check `fingertip` lands on
`connector` at grasp.)*

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
| `scan.offsets` | per‑view offsets from the **start** camera pose, in the **camera frame** (xyz m, rpy rad); give parallax |
| `scan.relative_bounds` | max \|offset\| (camera frame) — a safety clamp so the camera stays near the start pose |
| `scan.dwell_s` | hold time per view (≥ SAM3 inference, ~2 s) |
| `save_scan_images` / `debug_image_topic` | save the SAM3 overlay per view / Node 1's `~/debug_image` |
| `data_dir` / `scan_images_subdir` | where overlays go: `<data_dir>/<subdir>/<timestamp>/view_NN.png` |
| `speed.max_joint_velocity_rad_s` / `speed.max_cartesian_velocity_m_s` | cap arm velocity — each move's duration scales with its size (more restrictive wins; `0` = that cap off; **both `0` → fixed `move_duration_s`**). Runtime override: `-p max_joint_velocity:=` / `-p max_cartesian_velocity:=` |
| `speed.min_move_duration_s` | floor so tiny moves aren't near‑instantaneous |
| `grasp_check.enabled` | detect a not‑in‑groove grasp from the gripper counts and recover/retry (commands a full close) |
| `grasp_check.closed_counts` / `tolerance_counts` | counts (0–255) at full closure / tolerance — reading `>= closed_counts − tolerance` is OK, short is a failure |
| `grasp_check.full_close_rad` | knuckle joint value at 255 counts (radians→counts conversion; verify against `/joint_states`) |
| `grasp_check.settle_s` / `grasp_check.max_retries` | settle before reading / extra retries after the first attempt |
| `approach_*` / `lift_*` / `place_offset_*` / `gripper.*` | same as `ur_pick_place_demo` |

**Scan overlays:** at each view the demo saves Node 1's annotated frame (masks + neck + orientation
arrow — the same output as the `sam3-abhay` CLI) to
`data/cable_pick_place/<run timestamp>/view_NN.png`, so you get a visual record of what SAM3 saw.
Run `cable_neck_ros_node` with `publish_debug:=true`, and run this demo from the workspace root so
`data/` resolves there.

## Notes & caveats

- **Jog the camera onto the cable first, then run.** The scan is **relative to the start pose**, so
  there are no cell coordinates to set — just position the camera so the cable is in view. Tune
  `scan.offsets` (camera‑frame) for enough parallax and `scan.relative_bounds` to keep the sweep safe;
  the fusion won't triangulate without translation between views.
- **SAM3 timing:** the detector runs ~1–2 s/frame and drops frames while busy, so `dwell_s` must be
  long enough to get one clean, static frame per view.
- **Estimator history:** `connector_pose_node` accumulates a rolling window of views. For a clean run
  restart it (or let its window roll over) so a previous cable's views don't bias the estimate.
- **Grasp clocking:** with the base‑Z‑up assumption the connector frame is fully determined, so the
  grasp is repeatable. If a cable ever tilts far from horizontal, revisit `connector_up_axis`.
- **Grasp check calibration:** the check works in **counts (0–255)**, converted from the joint via
  `grasp_check.full_close_rad`. Verify the conversion once: fully close the gripper and read
  `robotiq_85_left_knuckle_joint` (`ros2 topic echo /joint_states`) — that value should map to ~255,
  so set `full_close_rad` to it (the 2F‑85 knuckle upper limit, ~0.8). The node prints the live counts
  each check, so confirm a full close reports ~`closed_counts` (228) and a not‑in‑groove catch reports
  lower. The grasp deliberately **commands a full close** (not `gripper.closed_position`) so the stall
  position is cable‑determined; force is still bounded by `gripper.max_effort`. Set
  `grasp_check.enabled: false` to disable detection/recovery. Empty‑pickup detection is future work.
- **Speed caps:** `speed.max_joint_velocity_rad_s` / `speed.max_cartesian_velocity_m_s` bound how fast
  the arm moves (the more restrictive wins). This times each move so velocity stays under the cap —
  which also prevents the joint‑velocity‑limit protective stops a too‑short fixed `move_duration_s`
  can trigger. They bound the *average* speed of a single‑point move (peak can be ~1.5×), so set them
  conservatively. Set both to `0` to revert to the fixed `move_duration_s`. Tune per run without
  editing the config: `ros2 run ur_cable_pick_place_demo cable_pick_place --ros-args -p max_joint_velocity:=0.3`.
- No collision avoidance (`avoid_collisions: false`) — keep the workspace clear; **e‑stop in hand**.
