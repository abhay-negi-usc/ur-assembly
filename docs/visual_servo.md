# visual_servo — continuous PBVS to a marker

Port of `ur_visual_servo_demo`. Continuously drives the camera to a fixed standoff, square to an
ArUco marker (position-based visual servo). Holds when the marker is out of view or already within
the deadband. Runs until Ctrl-C.

```bash
python -m urlab.apps.visual_servo [--dry-run]
```

Config: `configs/visual_servo.yaml`

| key | meaning |
|---|---|
| `marker.id` | the marker to track (11) |
| `standoff_m` | distance to hold on the marker normal (0.10) |
| `cam_rpy_in_marker` | camera orientation in the marker frame (`[pi,0,0]` = look back down the normal) |
| `servo_gain` / `servo_max_linear_step_m` / `servo_max_angular_step_deg` | clamped step per iteration |
| `pos_deadband_m` / `ang_deadband_deg` | hold band |
| `rate_hz` | loop rate |
| `marker_max_age_s` | a marker older than this counts as "not in view" |

No gripper. The control law is the shared `servo` skill.
