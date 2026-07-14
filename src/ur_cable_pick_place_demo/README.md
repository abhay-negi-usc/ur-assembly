# ur_cable_pick_place_demo

Cable pick‑and‑place for the UR10e + Robotiq 2F‑85. It's **`ur_pick_place_demo` with the single
fiducial detection replaced by a multi‑view scan** that feeds the **SAM3 cable‑connector pose
estimator** (external nodes in the `sam3-abhay` repo). This node subclasses `PickPlace`, so all the
IK / trajectory / gripper / grasp machinery is reused; it needs **no torch** — the SAM3 nodes run
separately and are coupled only through TF.

## Sequence

```
open → scan (multi-view) → estimate connector pose → return to initial pose
     → grasp-align → grasp → close → [grasp check]
     ↳ SHORT (cable not seated in fingertip groove) → open (drop) → return to initial pose → retry
     → lift → pre-place → place → open → retreat → home
```
> **The estimate happens *before* returning home, on purpose.** `connector_pose_node` only republishes
> the connector TF while the camera can still **see** the cable. Leave the view first and it stops
> refreshing, ages past `connector_max_age_s`, and the read fails. Reading it at the last scan view
> captures `T_base_grasp` as a **base‑frame** pose, which stays valid however the arm moves after.

1. **Scan (two phases)** — all views are **relative to the camera's pose at the start of the scan**
   (jog the robot so the cable is in view first — no absolute cell coordinates to tune). Each view
   holds still for `scan.dwell_s` so SAM3 gets a clean, static frame.
   - **Seed views** (`scan.offsets`) — swept along the camera's own image axes and clamped to
     `scan.relative_bounds`, each tilted 10° back toward the start view so the cable stays framed.
     These supply the translation **parallax** needed to triangulate the connector's **origin**.
   - **Refine views** (`scan.refine`) — once a first estimate exists, these translate the camera along
     the **measured connector y axis** and aim it exactly at the measured connector origin. This is
     the *only* motion that sharpens the **axis** — see [Why the refine views go along y](#why-the-refine-views-go-along-y).
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

**`connector_pose_node` publishes the connector frame directly**, and this demo consumes that TF
**as‑is**. (It used to publish the axis in *z* with arbitrary x/y and let the demo rebuild the frame —
which meant the published TF looked wrong in RViz and hid whether the axis was actually right.)

- **x** = the cable‑connector **axis** (the one rotational DOF the multi‑view fusion measures),
- **z** = **up** (that node's `up_axis` param, default base **+Z**), re‑orthogonalized ⟂ x — the
  *"cable z coincident with base z"* assumption, which pins down the roll the fusion cannot measure,
- **y** = `z × x` (right‑handed, horizontal).

Every axis is meaningful, so `ros2 run tf2_ros tf2_echo base_link connector` shows the **real** frame.
The grasp is this frame plus `connector_grasp` (default identity).

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

## Run — one command (recommended)

`cable_stack.launch.py` starts the **whole stack** (bringup, move_group, RealSense, hand‑eye tf, and
both SAM3 nodes) so you only need **two terminals**:

```bash
# Terminal A -- the whole stack
source /opt/ros/jazzy/setup.bash && source /abhay_ws/ur-assembly/install/setup.bash
ros2 launch ur_cable_pick_place_demo cable_stack.launch.py
#   ... then press PLAY on the pendant's External Control program.

# Terminal B -- the demo (kept separate: it prompts on stdin, which doesn't work under `ros2 launch`)
cd /abhay_ws/ur-assembly
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 run ur_cable_pick_place_demo cable_pick_place
```

The launch file runs the **SAM3 detector under the venv's interpreter** (via `ExecuteProcess`, since a
normal `Node` action would use system Python, which has no torch) and staggers startup so each piece
comes up in dependency order.

Skip anything you already have running, or point it at different hardware:
```bash
ros2 launch ur_cable_pick_place_demo cable_stack.launch.py bringup:=false camera:=false
ros2 launch ur_cable_pick_place_demo cable_stack.launch.py camera_serial:=_123456789012
```

| Launch arg | Default | Purpose |
|---|---|---|
| `bringup` / `moveit` / `camera` / `handeye` / `sam3` | `true` | toggle each component off if it's already up |
| `camera_name` / `camera_serial` | `camera1` / `_218622272137` | `camera_name` sets the image **frame** (`<name>_color_optical_frame`) — must match the hand‑eye tf |
| `image_topic` / `camera_info_topic` | `/camera/camera1/color/...` | realsense2_camera nests topics under *namespace* **and** *name* — **not** `/camera1/...` |
| `sam3_python` / `sam3_scripts` | `/opt/sam3_venv/bin/python` / `/abhay_ws/sam3-abhay/scripts` | where the venv + SAM3 nodes live |
| `neck_diameter` / `tf_cache_s` | `0.0034` / `60.0` | connector diameter; TF history (must exceed SAM3's inference latency) |

## Run — manually, one process per terminal (for debugging)

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
```
> **Topics are nested, frames are not.** realsense2_camera puts topics under *camera_namespace* **and**
> *camera_name*, so you get **`/camera/camera1/color/image_raw`** and **`/camera/camera1/color/camera_info`**
> — **not** `/camera1/...`. The **frame** is still `camera1_color_optical_frame` (from `camera_name`),
> which is what must match the hand‑eye tf. Confirm with `ros2 topic list | grep image_raw`.

**Terminal 4 — hand‑eye tf** (publishes `tool0 -> camera1_color_optical_frame`):
```bash
source /opt/ros/jazzy/setup.bash && source /abhay_ws/ur-assembly/install/setup.bash
ros2 launch ur_tf_demo tf_streaming.launch.py
```

**Terminal 5 — SAM3 detector (Node 1)** — the ONLY terminal that activates the SAM3 venv (torch/GPU):
```bash
source /opt/sam3_venv/bin/activate && source /opt/ros/jazzy/setup.bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 /abhay_ws/sam3-abhay/scripts/cable_neck_ros_node.py --ros-args \
  -p image_topic:=/camera/camera1/color/image_raw -p publish_debug:=true \
  -p adaptive:=true -p confidence_floor:=0.2
```
> **`adaptive:=true` is the important one.** With a fixed threshold, SAM3 flips between labelling the
> connector "cable" and vice versa; whichever class comes up empty kills the neck entirely, because a
> neck **is** the cable/connector contact. Adaptive mode runs SAM3 **once** at `confidence_floor`, then
> searches the threshold *pair* in software, keeping the most confident masks that still yield a valid
> neck. It costs **no extra inference** — the threshold is only a filter on per‑mask scores. The log
> shows what it chose: `thr: cable=0.91 conn=0.18 (eff 0.18, 1 combos)`.

**Terminal 6 — SAM3 fusion (Node 2)** — plain system Python; broadcasts `base_link -> connector`:
```bash
source /opt/ros/jazzy/setup.bash
python3 /abhay_ws/sam3-abhay/scripts/connector_pose_node.py --ros-args \
  -p world_frame:=base_link -p connector_frame:=connector \
  -p necks_topic:=/cable_neck_detector/necks \
  -p camera_info_topic:=/camera/camera1/color/camera_info \
  -p neck_diameter:=0.0034 \
  -p up_axis:="[0.0, 0.0, 1.0]" \
  -p tf_cache_s:=60.0
```
> `tf_cache_s` is **required** on a slow GPU. Necks carry the **image** timestamp, so they arrive one
> whole inference late (~6 s here). Node 2 looks up the camera pose *at that stamp*, and tf2's default
> **10 s** buffer isn't enough headroom → `"Lookup would require extrapolation into the past"`.

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

## Why the refine views go along `y`

The multi‑view fusion recovers the connector **axis** as the intersection of back‑projected planes:
view *k*'s 2D neck line back‑projects to a plane containing the camera **centre** `C_k` and the 3D axis
line through `P`, with normal

```
n_k  ∝  a × (C_k − P)          ⟂ to both the axis and the viewing ray
```

and the axis is the **null space of the stacked `n_k`**. So a new view only helps if it produces a
*different* `n_k` — which requires moving the camera **out of the current plane**, i.e. along `n` itself.
With **x** = the axis and **z** ≈ the viewing direction, that is exactly **`y = z × x`** — the connector's
own y axis.

The corollary is the part that bites:

| Camera translation | Helps the **origin**? | Helps the **axis**? |
|---|---|---|
| ⟂ axis, along connector **y** | ✅ | ✅ **the only one that does** |
| **∥ axis** | ✅ | ❌ **zero** — `C` stays inside the same plane, reproducing the same `n_k` |
| along the viewing ray (toward/away) | ❌ | ❌ |

**Camera *rotation* contributes nothing at all** — both the triangulation ray and the back‑projected
plane depend only on the camera *centre*, not its orientation. The 10° tilt on the seed views is purely
a field‑of‑view device, not an information source.

Because the seed views sweep the camera's **image** axes blindly, roughly **half of them land parallel
to the cable and do nothing for the axis**. The refine phase fixes that: once the seed views give a
first estimate, it reads the measured axis, computes `y`, and spends its motion there — aiming the
camera exactly at the now‑known connector origin (an exact look‑at, so no tilt approximation is needed).

## Timing — everything is sized around SAM3's inference latency

**This is the single most important thing to get right.** SAM3 is slow on a pre‑Ampere GPU (~**6 s per
frame** on a GTX 1060), and *every* timeout in the pipeline has to be sized above that. The defaults
below assume ~6 s; **measure your own** from the cadence of Node 1's `necks=…` log lines and scale.

| Setting | Where | Default | Why it must be long |
|---|---|---|---|
| `tf_cache_s` | **Node 2** (CLI param) | **60 s** | Necks carry the **image** stamp, so they arrive one inference late. Node 2 looks up the camera pose *at that stamp* — tf2's default **10 s** buffer is too short → `"extrapolation into the past"`. |
| `tf_cache_s` | **demo** yaml | **120 s** | Same disease, the *consumer* side. Node 2 republishes the connector only ~once per inference, so with tf2's 10 s default the transform **expires from the demo's buffer between publishes** and the lookup fails even though Node 2 is publishing fine. A long cache is safe — it only keeps transforms *available*; staleness is enforced separately by `connector_max_age_s`. |
| `scan.dwell_s` | demo yaml | **10 s** | Must **exceed** one inference, or the robot moves to the next view before SAM3 has processed a clean, *static* frame from this one. |
| `connector_max_age_s` | demo yaml | **30 s** | The TF's age at read time swings across the whole republish interval (~6–12 s), and **any view where the cable isn't detected extends it**. With a 16‑view (~4 min) scan, a couple of consecutive misses near the end is normal — too tight and a perfectly good estimate is thrown away as "stale". |
| `connector_wait_s` | demo yaml | **40 s** | After the scan, the fusion still needs time to produce (or refresh) the estimate. |
| `move_timeout_s` | demo yaml | 60 s | Unrelated to SAM3 — catches an accepted trajectory that never executes (pendant not playing, e‑stop, speed slider at 0). |

**Scan budget:** 12 seed + 4 refine = **16 views × (10 s dwell + ~4 s move) ≈ 4 minutes.** If you move SAM3 to a
faster GPU, `dwell_s` is the dominant term — drop it and everything else can come down proportionally.

Two subtleties worth knowing, because they caused real bugs here:

- **Node 2 stamps its output with `now`, not the image time.** The connector is a *static object pose* —
  it answers "where is the cable," not "where was it 6 s ago." Stamping it with the (stale) image time
  forced every consumer — `tf2_echo`, RViz, this demo — into a 6‑second time‑travel lookup that fails.
  The image stamp is still used **internally** to fetch the camera pose at *capture* time, which is the
  part that genuinely must be time‑accurate.
- **A slow detector doesn't hurt accuracy, only throughput.** Each neck is paired with the camera pose
  at its own capture time, so the triangulation stays correct no matter how far behind the detector runs.

## Configure — [config/cable_pick_place.yaml](config/cable_pick_place.yaml)

| Param | Meaning |
|---|---|
| `connector_frame` | TF the SAM3 estimator broadcasts (match its `connector_frame`) |
| `connector_max_age_s` / `connector_wait_s` | freshness of the estimate / how long to wait after the scan — **both sized around SAM3's latency**; see [Timing](#timing--everything-is-sized-around-sam3s-inference-latency) |
| *(connector frame convention)* | **not here** — built by `connector_pose_node` (its `up_axis` param); this demo reads the TF as‑is |
| `connector_grasp` (xyz/rpy) | optional offset of the fingertip target from the connector (default identity) |
| `grasp_tcp_offset` (xyz/rpy) | gripper (fingers‑center) frame, relative to `tool0` |
| `fingertip_grasp` (xyz/rpy) | fingertip frame w.r.t. the gripper; the **grasp reference** (xyz `[0, 12.54, 181.65] mm`, rpy `[π,0,-π/2]` sxyz) |
| `publish_fingertip_tf` | broadcast `tool0 → fingertip` for RViz verification |
| `scan.offsets` | **seed views** — per‑view offsets from the **start** camera pose, in the **camera frame** (xyz m, rpy rad); give parallax for the **origin** |
| `scan.relative_bounds` | max \|offset\| (camera frame) — a safety clamp so the camera stays near the start pose |
| `scan.refine.enabled` / `offsets_m` / `max_offset_m` | **refine views** — signed distances along the *measured* connector **y** axis, aimed at the connector; the only motion that sharpens the **axis** |
| `scan.dwell_s` | hold time per view — **must exceed one SAM3 inference** (~6 s on a pre‑Ampere GPU); see [Timing](#timing--everything-is-sized-around-sam3s-inference-latency) |
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
  grasp is repeatable. If a cable ever tilts far from horizontal, revisit `connector_pose_node`'s
  `up_axis` param (that's where the "up" assumption now lives).
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
