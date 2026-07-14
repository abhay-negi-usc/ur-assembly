# gripper — Robotiq 2F-85 cycle

Port of `ur_gripper_demo`. Steps the gripper through a list of fractions (0 = open, 1 = closed),
dwelling at each. A **stall is success** — the fingers meeting an object before the commanded
position is the normal outcome of a grasp, and the gripper reports it (`gOBJ`), so the demo keeps
going.

Positions are in **counts (0–255)** natively over Modbus — no radians, no `full_close_rad`
conversion (that was a ROS-side fudge; it's gone).

```bash
python -m urlab.apps.gripper [--dry-run]
```

Config: `configs/gripper.yaml`

| key | meaning |
|---|---|
| `gripper.port` | serial device (`/dev/ttyUSB0`; prefer `/dev/serial/by-id/...`) |
| `gripper.open_counts` / `closed_counts` | stroke endpoints (0 / 255) |
| `gripper.force_counts` / `speed_counts` | grip force / speed (0–255) |
| `gripper_demo.fractions` | the sweep, e.g. `[0, 0.25, 0.5, 0.75, 1, 0]` |
| `gripper_demo.cycles` / `dwell_s` | repeats and pause per step |

No arm, no camera. Verifies the Modbus link and gripper activation.
