# pick_place — fiducial pick-and-place

Port of `ur_pick_place_demo`. Detect an ArUco marker, visually approach it, estimate the grasp
from the close view, grasp, and place.

```
open -> detect + align/centre -> approach to standoff -> estimate grasp
  -> [blind: home then open-loop grasp | servo: closed-loop to grasp]
  -> close -> lift -> pre-place -> place -> open -> retreat -> home
```

```bash
python -m urlab.apps.pick_place [--dry-run] [--yes]
```

Config: `configs/pick_place.yaml`

| key | meaning |
|---|---|
| `marker.id` | the ArUco id to pick (11) |
| `aruco.marker_size_m` / `dictionary` | marker geometry / family |
| `object_marker`, `object_grasp` | marker→object and object→grasp transforms |
| `grasp_tcp_offset` | grasp TCP relative to tool0 |
| `servo_standoff_m` / `servo_step_m` | visual approach standoff and step |
| `blind_pick` | `true`: home then grasp open-loop; `false`: closed-loop visual servo |
| `place_offset_xyz` | place = grasp shifted by this |

Grasp geometry and the place sequence come from the shared `pick` + `servo` skills — the same ones
the cable demos use.
