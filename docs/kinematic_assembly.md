# kinematic_assembly — CSV-trajectory assembly

Port of `ur_kinematic_assembly_demo`. Follow a CSV trajectory of held-object poses (relative to
the target object) from a stand-off to the mate, optionally under compliance, then disassemble in
reverse and return home. No perception.

```
stand-off -> follow the trajectory (position or forceMode) -> mate
  -> [disassemble in reverse | return home]
```

**The anchoring trick.** The assembled `tool0` pose is measured ground truth (jog to a good mate,
read `base_link → tool0` off the robot). The target-object frame is never measured — it is
back-derived so the LAST CSV row maps exactly onto that assembled pose. So the last row is always
the mate by construction. Do not try to measure the target frame independently.

```bash
python -m urlab.apps.kinematic_assembly [--dry-run] [--yes]
```

Config: `configs/kinematic_assembly.yaml` (+ `configs/assembly_trajectory.csv`)

| key | meaning |
|---|---|
| `assembled_pose` | tool0 in base at the mate — **measure this** |
| `held_object_pose` | the held part relative to tool0 |
| `standoff_distance_mm` / `standoff_axis` | pre-position back-off, in the target frame |
| `trajectory_csv` / `trajectory_angles_deg` | the waypoints (x,y,z,r,p,y) and their angle unit |
| `control_mode` | `position` or `admittance` (forceMode) |
| `disassemble_after` | run the trajectory in reverse before homing |
| `admittance.max_force_n` / `max_torque_nm` | contact-stop limits (compliant mode) |

Waypoint IK is chained (each seeds the next), so the whole path stays on one IK branch.
