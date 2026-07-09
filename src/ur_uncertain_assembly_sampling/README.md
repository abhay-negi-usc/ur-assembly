# ur_uncertain_assembly_sampling

Repeatedly performs **assembly under uncertainty** and **disassembly** on the UR10e, logging the
assembly data to a CSV. It **subclasses [`ur_kinematic_assembly_demo`](../ur_kinematic_assembly_demo/)**,
so all the geometry is the same — **all trajectory is the held object w.r.t. the target object**,
and everything is anchored from the polled `assembled_pose` (tool0) + `held_object_pose`.

## What each trial does

1. **Resample** the ideal assembly trajectory (the CSV) to a fixed spatial resolution
   (`translational_resolution_m` / `rotational_resolution_deg`) → dense waypoints.
2. **Chunk** — take the first `chunk_fraction` of it (`1.0` = the whole assembly).
3. **Perturb** each waypoint, **in the target‑object frame** (`perturbed = bias · noise · ideal`):
   - **bias** — one uniform draw per trial (constant across the chunk): a systematic offset,
   - **noise** — a fresh uniform draw per waypoint: jitter.

   Execute that perturbed chunk under **admittance**, **recording continuously at the F/T sensor
   rate** (a row is written on every wrench message while assembling). A **force‑ AND torque‑guarded
   stop** (`max_force_n` / `max_torque_nm`) ends the chunk the instant either threshold is exceeded.
4. **Disassemble (not logged)** — find the **closest pose on the ideal dense trajectory** (weighted
   translation + rotation, `closest_pose_rot_weight_mm_per_deg` mm per degree), go to it, then run
   the **ideal reverse** trajectory from there back to the start (no added error).

Repeat for `num_trials`. The CSV is **flushed after each trial** (a crash keeps completed trials).

## Control mode

`control_mode: admittance` (default) is required for meaningful contact sampling — the demo
deliberately drives a **misaligned** part into contact, and position control would rigidly jam it.
Admittance yields to contact and the **force/torque‑guarded stop** ends a chunk that binds, then the
part is extracted along the ideal reverse path. Needs the `admittance_controller` **loaded
(inactive)** — see `ur_admittance_demo`.

## The CSV

One **timestamped** file per run (`<csv_path stem>_YYYYmmdd_HHMMSS.csv`), **appended incrementally**
(rows streamed straight to disk — never accumulated in memory) and **flushed after each trial**.
**Assembly rows only** — disassembly is not recorded. Columns:

| Group | Columns |
|---|---|
| index | `trial`, `timestamp` (s) |
| **actual** `tool0` in `base_link` | `x y z`, `qx qy qz qw`, `yaw_deg pitch_deg roll_deg` |
| **actual** held object in target object | `x y z`, `qx qy qz qw`, `yaw_deg pitch_deg roll_deg` |
| **commanded** `tool0` in `base_link` | `x y z`, `qx qy qz qw`, `yaw_deg pitch_deg roll_deg` |
| **commanded** held object in target object | `x y z`, `qx qy qz qw`, `yaw_deg pitch_deg roll_deg` |
| F/T in the **tool0** frame | `fx fy fz tx ty tz` |
| F/T in the **held‑object** frame | `fx fy fz tx ty tz` |

Rotations are given as **both** a quaternion and **ZYX Euler angles in degrees** (`yaw, pitch, roll`
— the standard robotics convention). *Actual* poses are the live robot state (tool0 from tf, held
pose composed through `held_object_pose`), so they reflect compliance/contact deflection; *commanded*
is the current perturbed waypoint. The held‑frame wrench is the tool0 wrench transformed through
`inv(held_object_pose)` (rotation **+** moment‑arm).

### F/T frame (from the UR driver source)

The UR `force_torque_sensor_broadcaster` (Jazzy) publishes on **`/force_torque_sensor_broadcaster/wrench`**
(hardcoded `~/wrench`; the config's `topic_name: ft_data` is ignored — the broadcaster has no such
param) in the **`tool0_controller`** frame. The node reads the message's `frame_id` and transforms
the wrench to `tool0` via tf (`tool0_controller` ≈ `tool0`), then to the held‑object frame — so the
two logged wrenches are correct regardless. A smoother `~/wrench_filtered` topic also exists; set
`admittance.wrench_topic` to it if you prefer filtered data.

## Configure — [config/uncertain_assembly_sampling.yaml](config/uncertain_assembly_sampling.yaml)

The full kinematic‑assembly config (`assembled_pose`, `held_object_pose`, stand‑off, trajectory CSV,
admittance) **plus** a `sampling:` block:

| Param | Meaning |
|---|---|
| `num_trials` | Number of perturbed‑assembly + disassembly cycles |
| `chunk_fraction` | Fraction of the assembly trajectory to execute (1 = full insert) |
| `translational_resolution_m` / `rotational_resolution_deg` | Resample spacing of the ideal path |
| `bias` | Per‑trial uniform bias half‑widths `[x,y,z,roll,pitch,yaw]` (m, deg), target frame (systematic) |
| `noise` | Per‑waypoint uniform noise half‑widths `[x,y,z,roll,pitch,yaw]` (m, deg), target frame (jitter) |
| `closest_pose_rot_weight_mm_per_deg` | Disassembly closest‑pose metric weight (mm per degree; 1.0 = 1 mm ≡ 1°) |
| `random_seed` | 0 = nondeterministic; >0 seeds numpy for reproducible trials |
| `csv_path` | Output CSV stem (relative → cwd); a `_YYYYmmdd_HHMMSS` timestamp is appended per run |

The force/torque **guard thresholds** are `admittance.max_force_n` / `admittance.max_torque_nm`.

## Prerequisites & bring-up

Same as the kinematic demo (arm driver + move_group + a live F/T; admittance mode also needs the
admittance controller loaded). One command:

```bash
ros2 launch ur_kinematic_assembly_demo arm_bringup.launch.py load_admittance:=true
# then Play External Control on the pendant
```

## Build & run

```bash
cd /abhay_ws/ur-assembly
colcon build --packages-select ur_kinematic_assembly_demo ur_uncertain_assembly_sampling --symlink-install
source install/setup.bash

ros2 run ur_uncertain_assembly_sampling uncertain_assembly_sampling   # prompts render under `ros2 run`
```

`confirm_each_step: true` prompts at each trial boundary — set it `false` for an unattended run.

## Safety

- Repeatedly drives a **misaligned part into contact** — **e‑stop in hand**, `control_mode:
  admittance`, conservative `max_force_n`, and small `bias_*` / `noise_*` to start.
- On any abort/Ctrl‑C the node **switches back to position control** and closes the CSV.
- No collision avoidance (`avoid_collisions: false`) — keep the workspace clear.
- The wrench frame is handled from the message `frame_id` (UR publishes `tool0_controller`), so the
  logged F/T is correct without assuming a frame — see the F/T‑frame note above.
