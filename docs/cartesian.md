# cartesian — Cartesian jog

Port of `ur_cartesian_demo`. Nudges the tool ±along each of x/y/z and ±about roll/pitch/yaw around
a captured home pose, returning to home between every nudge. Runs the sequence once per motion
frame (`world` = base axes, `tool0` = tool axes). Every target is relative to the home pose, never
the previous one, so errors do not accumulate.

```bash
python -m urlab.apps.cartesian [--dry-run] [--yes]
```

Config: `configs/cartesian.yaml`

| key | meaning |
|---|---|
| `linear_step_mm` | translation nudge size (0.03) |
| `angular_step_deg` | rotation nudge size (30) |
| `motion_frames` | frames to jog in, in order (`[world, tool0]`) |
| `speed.*` | joint/Cartesian velocity caps |

No gripper, no camera. Good first check that RTDE is connected and moving.
