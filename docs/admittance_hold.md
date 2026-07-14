# admittance_hold — compliant in place

Port of `ur_admittance_demo`. Makes the arm compliant where it stands: it holds its pose but
yields to a push and springs back. Push it around by hand; it hand-guides.

In ROS this needed the `admittance_controller` (a separate C++ plugin the Python node fed a
constant reference to). Here it is **UR `forceMode`** with all six axes compliant and zero target
wrench — one call, no controller to load, no switch, and none of the joint-0 velocity fault the
controller activation caused.

```bash
python -m urlab.apps.admittance_hold [--dry-run]
```

Config: `configs/admittance_hold.yaml`

| key | meaning |
|---|---|
| `admittance.selected_axes` | which axes yield (`[1,1,1,1,1,1]` = all) |
| `admittance.force_limits` | compliant-axis speed caps (m/s, rad/s) |
| `admittance.damping` / `gain_scaling` | forceMode tuning |
| `admittance.hold_duration_s` | `0` = until Ctrl-C |

Tares the F/T first, so residual tool weight isn't read as a push. **KEEP THE E-STOP IN HAND** —
a mis-tared sensor makes the arm drift.
