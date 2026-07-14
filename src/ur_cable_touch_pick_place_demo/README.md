# ur_cable_touch_pick_place_demo

Cable **touch‑then‑pick** for the UR10e + Robotiq 2F‑85. It subclasses `ur_cable_pick_place_demo` (so
the scan, fingertip frame, grasp check and speed caps are all reused), but it gets the connector's pose
a fundamentally different way.

## The idea: use each sensor only for what it's actually good at

| Quantity | Source | Why |
|---|---|---|
| **x, y** | vision | **Lateral** in the image — the well‑determined directions of a multi‑view fit |
| **yaw** | vision | From the connector axis, which the scan's refine orbit is built to sharpen |
| **z** | **touch** | The camera looks **down**, so z *is* the camera's **depth** direction — the **weakest** axis of a monocular fit (error grows as `Z²/baseline`; several mm here) |
| **roll, pitch** | **assumed zero** | The connector lies flat. This assumption is what makes a **single** touch sufficient — with roll/pitch free, one contact point couldn't pin down the height |

The **probe is the gripper itself with the fingers fully closed** — no extra hardware.

## Sequence

```
close gripper (fingers = probe) → scan (multi-view) → estimate TIP (x, y, yaw only)
  → servo-align over the touch point while hovering
  → descend in 1 mm steps until a very low force trips  →  contact z  ⇒  the connector's z
  → retract → open gripper → align → grasp → [grasp check] → lift
  → pre-place → place → open → retreat → home
```

The **touch point** is defined relative to the **tip** (`touch_offset`, default **5 cm along the tip's
−x**, i.e. back along the cable from the tip). The **grasp point defaults to the same place** — on
purpose: z is only *known* where you touched, so grasping anywhere else re‑introduces the height
uncertainty the touch just removed.

## Perception — classification‑free (this is the point)

SAM3 routinely labels the whole assembly **"cable"** and returns **no connector mask at all**. That
**starves the neck pipeline completely**: `compute_necks()` iterates over *connector* masks, so zero
connectors ⇒ zero necks, no matter how good the cable mask is.

The **tip pipeline refuses to depend on that classification**:

1. **Union** the cable and connector masks into one `cable_and_connector` object.
2. Find that shape's two **ends** — its *geodesic* diameter (geodesic, not Euclidean: a cable is a
   **curve** and can double back, so two points can be adjacent in the image yet far apart along it).
3. Pick the **connector end** — nearest the connector mask if SAM3 produced one (best evidence),
   otherwise the **thicker** end (a connector is fatter than the cable it terminates).
4. Walk **back along the curve** by a predefined length (`curve_px`) and take `tip − back` as the
   **axis**. A fixed arc length is far more stable than the local tangent at the very tip, which is
   dominated by mask noise.

`~/tips` uses the **same message convention as `~/necks`**, so `connector_pose_node` fuses it
**unchanged** — including its RANSAC/consensus **outlier rejection**, which is what discards background
cables and connectors.

## Run

```bash
# Terminal A — the whole stack (SAM3 TIP detector + fusion → base_link → connector_tip)
ros2 launch ur_cable_touch_pick_place_demo touch_stack.launch.py
#   ...then press PLAY on the pendant.

# Terminal B — the demo (separate: it prompts on stdin, which doesn't work under `ros2 launch`)
ros2 run ur_cable_touch_pick_place_demo cable_touch_pick_place
```

Build first:
```bash
colcon build --packages-select ur_pick_place_demo ur_cable_pick_place_demo \
                              ur_cable_touch_pick_place_demo --symlink-install
source install/setup.bash
```

## Configure — [config/cable_touch_pick_place.yaml](config/cable_touch_pick_place.yaml)

| Param | Meaning |
|---|---|
| `connector_frame` | the TF the fusion publishes (`connector_tip`). The inherited scan keys off this, so the **refine orbit sharpens the tip's axis** for free |
| `touch_offset` | where to touch, **in the tip frame** (its **x = the axis**, pointing out of the cable → **−x = back along the cable**) |
| `grasp_offset` | where to grasp, in the tip frame. **Keep it equal to `touch_offset`** unless you have a reason |
| `touch.force_n` | **3 N** — a probe, not a push. Raise only if the F/T noise floor triggers it spuriously |
| `touch.step_m` | **1 mm** — bounds the force overshoot past the trigger |
| `touch.max_descent_m` | abort rather than drive deeper than this (60 mm) |
| `touch.contact_z_offset_m` | added to the contact z to get the **grasp** z. The probe touches the **top** of the connector, so a small **negative** value drops the fingertip to its centre — **tune on the real part** |
| `touch.tare_before` | zero the F/T just before descending. **Essential** — the tool weight is ≫ 3 N |
| `align.*` | closed‑loop align deadbands; the tip is **re‑read each iteration**, so an improving estimate is tracked rather than committed to once |

## Safety — read this

The descent is **position‑controlled**, not compliant. The force is checked **before each step**, so
the overshoot past the trigger is bounded by **one step (~1 mm)** — but a position‑controlled move into
a **rigid** object builds force fast. That's why `step_m` must stay small and `force_n` low. The UR's
protective stop is the backstop.

**Keep `confirm_each_step: true` and the e‑stop in hand**, especially on the first runs.

Two failure modes worth knowing:
- **"No contact within 60 mm"** → the vision x/y is wrong (the probe missed the cable entirely), or
  `force_n` is below the F/T noise floor so it should have tripped and didn't. It **aborts** rather
  than driving deeper.
- **Instant contact at step 0** → the F/T wasn't tared, or `force_n` is *below* the noise floor. Check
  the tare succeeded in the log.

The log prints the **vision‑vs‑touch disagreement** on every run:
```
CONTACT after 12 step(s): force 3.14 N >= 3.00 N. probe z=0.0431 m (descended 12.0 mm).
connector z = 0.0431 m. Vision said 0.0518 m -- off by +8.7 mm.
```
That number is the whole justification for this demo. If it's consistently small, monocular z was fine
and you don't need the touch; if it's several mm (expected), the touch is buying you real accuracy.

## Frame names — three different things

`tip_frame` (**`tool0`**, the robot flange the IK solves for) ≠ **fingertip** (the grasp reference *and*
the probe, ~18 cm out) ≠ **`connector_tip`** (the end of the *cable*, from vision). They are unrelated;
don't let the shared word "tip" mislead you.
