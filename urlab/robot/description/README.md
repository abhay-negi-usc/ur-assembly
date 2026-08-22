# Robot description

Geometry for the ground-collision check in `urlab/robot/collision.py`. Nothing here is used for
motion — the controller owns the real kinematics — so a discrepancy costs a wrong *clearance*,
not a wrong move. That is still worth catching, which is why the two cross-checks below exist.

## `ur10e/`

| | |
|---|---|
| Source | [UniversalRobots/Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) @ `rolling` |
| License | BSD-3-Clause — see `ur10e/LICENSE`, which must stay with the meshes |
| Contents | `ur10e.urdf` (generated), `meshes/collision/*.stl` (upstream, 7 files, ~430 KB) |

**The URDF is generated, not copied.** Upstream ships the description as *xacro*, which needs a
ROS toolchain to expand; there is no ROS in this workspace and no plain `.urdf` in that repo. So
`fetch_ur10e.py` assembles one from Universal Robots' own published parameter files:

- `config/ur10e/default_kinematics.yaml` — the joint origins (this *is* the calibrated chain)
- `config/ur10e/visual_parameters.yaml` — where each collision mesh sits within its link
- `urdf/ur_macro.xacro` — the link/joint topology and the `flange` / `tool0` frames

Every **number** is upstream's; only the assembly is ours.

```
python -m urlab.robot.description.fetch_ur10e
```

Re-run to refresh, then diff — a silent upstream change to the kinematics shows up there.

### Two independent cross-checks

The reason to trust it is that two descriptions written from *different* upstream files agree:

1. **Offline.** `ur10e.urdf` is built from the published **joint origins**; `collision.fk_links`
   is built from the published **DH table**. `tests/test_behaviors.py` asserts their `tool0`
   agrees over random configurations — it comes out at ~6e-8 m, which is pybullet's float32
   precision, not a modelling difference.
2. **On hardware.** `GroundCollisionModel.verify_against_controller()` compares both against
   `getForwardKinematics` at start-up and logs an error past 1 mm. A dry run reports
   **UNVERIFIED** rather than pretending to have checked.

### What is *not* from upstream

The **80 mm spacer** and the **Robotiq 2F-85** are not in this description. They are modelled as
parametric capsules in `collision.ToolModel`, sized from `pickup.collision.tool` in the app
config and anchored on the repo's own calibrated `fingertip_grasp` (183 mm along tool0). Swapping
in a real 2F-85 mesh would be a strict improvement; the fingertip **intersection allowance**
(see `collision.py`) would still have to be applied to the finger bodies alone.

### Nominal, not per-robot

These are UR's nominal figures (`hash: calib_5119701370761913513`). A specific arm's factory
calibration differs by a fraction of a millimetre — far inside the margins this model works to,
and `verify_against_controller` measures the difference on the real machine anyway.
