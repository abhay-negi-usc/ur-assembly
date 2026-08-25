# cable_touch_pick_place — vision x/y/yaw + touch z

Port of `ur_cable_touch_pick_place_demo`. Uses vision and touch each for what it is good at: the
camera looks down, so the connector's **z is the camera's weak depth axis** — but its x, y and yaw
are lateral in the image and well determined. So vision gives x/y/yaw and a **touch measures z**.
The probe is the gripper's own fingers, fully closed (no extra hardware). Roll and pitch are
assumed zero (the connector lies flat), which is what makes one contact enough to pin the height.

```
close (fingers = probe) -> scan -> estimate tip (x,y,yaw) -> align over the touch point
  -> descend to contact (z) -> retract -> open -> align -> grasp (vision xy + touched z)
  -> [grasp check] -> lift -> place -> home
```

**TIP pipeline.** Uses SAM3 `mode: tip`, which is classification-free (it unions the cable and
connector masks), so it survives SAM3 labelling the whole assembly "cable" — the failure that
starves the neck pipeline.

**The touch is position-controlled**, one small step at a time, with the force checked *before*
each step — so it stops the instant the threshold is crossed and overshoot is bounded by one step.

```bash
python -m urlab.apps.cable_touch_pick_place [--dry-run] [--yes]
```

Config: `configs/cable_touch_pick_place.yaml`

| key | meaning |
|---|---|
| `sam3.mode` | `tip` (classification-free); `curve_px` = arc walked back for the axis |
| `touch_offset` / `grasp_offset` | where to touch / grasp, relative to the tip (default: same point) |
| `touch.force_n` | contact threshold — VERY low (3 N); it's a probe, not a push |
| `touch.step_mm` / `max_descent_mm` | 1 mm steps, give up past this depth |
| `touch.contact_z_offset_mm` | added to the contact z for the grasp — **tune on the real part** |
| `align.*` | closed-loop align deadbands over the touch point |
| `scan.*` | same multi-view machinery as cable_pick_place (refines the tip axis) |

**KEEP THE E-STOP IN HAND** — a position-controlled descent into a rigid object builds force fast.
