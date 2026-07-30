# cable_pick_place — multi-view cable pick-and-place

Port of `ur_cable_pick_place_demo`. Scan the cable from several viewpoints so SAM3's neck
detections fuse into a 3D connector pose, grasp there, and place. The grasp check (in counts)
catches a cable that did not seat in the fingertip groove and retries the whole scan→grasp.

```
open -> scan (multi-view) -> estimate connector -> return home -> grasp-align -> grasp
  -> close -> [grasp check] -> lift -> pre-place -> place -> open -> retreat -> home
```

**In-process SAM3.** Unlike the ROS version, the detector and the multi-view fusion run in this
process — the scan captures a frame, runs SAM3 on it, and ingests the result synchronously. So
there is no detection topic, no TF, and no staleness to tune, and the image-stamp attribution the
ROS scan needed is gone.

Two independent choices decide how the grasp pose is found: **which 2D detector** runs
(`sam3.mode`) and **how its detections become a 3D pose** (`scan.mode`).

**Detector method (`sam3.mode`).** All emit the same `(u, v, yaw)` detection, so the fusion is
identical whichever you pick:
- **`junction`** (default) — the cable/connector junction by the **diameter-profiling** method
  (`cable_neck_diameter`): unions both prompts, traces the assembly, and puts the junction where the
  constant-diameter cable ends. Classification-free (survives SAM3 calling everything "cable"), one
  junction per frame, **no** adaptive mode; `sam3.min_contrast` rejects a weak diameter step. This
  is the only detector that also exposes the full cable **skeleton**, so it is **required** for
  `scan.mode: reconstruction`.
- **`neck`** — the **same physical junction** via `cable_neck_core`; iterates over connector masks,
  so a mislabelled connector can starve it, but it has the adaptive-threshold mode. `--set
  sam3.mode=neck`.
- **`tip`** — the cable's free end (used by the touch demo).

**Pose estimator (`scan.mode`).** — *this is the "which junction pose estimator" switch.*
- **`fuse`** (default) — the `ConnectorEstimator` triangulates the junction point across views
  (RANSAC scored by distinct views) and takes the axis from the null space of the back-projected
  image lines. Simple and robust; the axis **tilt** (out of the image plane) and the ray **depth**
  are its weak quantities.
- **`reconstruction`** — additionally reconstructs the 3D **cable centreline** (correspondence by a
  monotone epipolar march anchored at the junction) and **refines** the junction pose from it: the
  origin becomes the fitted curve's terminus and the axis becomes the curve's 3D **tangent**, so the
  tilt is *measured*, not inferred. Optionally pins the roll from the cable's osculating plane
  (`reconstruction.use_curve_plane_roll`, needs a curved cable). Needs `sam3.mode: junction`; tuned
  by the `reconstruction:` block; stops when the cross-view reprojection error **and** the
  view-to-view origin shift are both under threshold. Enable with `--set scan.mode=reconstruction`.
  It saves a `recon_NN.png` figure next to each overlay: two 3D views (different azimuths) of the
  reconstructed cable points in **base-frame axes**, with the refined junction triad.

**Only close views are fused.** In both modes a detected view is fused into the final pose **only**
while the camera is within `scan.max_view_distance_m` of the cable — farther detections still steer
the approach in (via the rough origin) but are kept out of the fit, because their depth error grows
as Z². (Previously every view was fused with equal weight.)

**The scan geometry** (documented in `urlab/perception/connector.py`): only camera *translation*
adds information; orbiting the cable axis is what sharpens the axis (hence the refine phase); and
depth error grows as Z², so the scan steps closer after each good view. It refuses to fit rather
than trust too few views or too little parallax.

```bash
python -m urlab.apps.cable_pick_place [--dry-run] [--yes] [--set scan.min_good_views=6]
```

Config: `configs/cable_pick_place.yaml`

| key | meaning |
|---|---|
| `sam3.repo_path` | your sam3-abhay checkout (urlab imports the detector module from it) |
| `sam3.mode` | `junction` (default) \| `neck` \| `tip` — see below; `adaptive` + `confidence_floor` (neck only) tune the threshold per image |
| `scan.mode` | `fuse` (default) \| `reconstruction` — **which pose estimator**; reconstruction needs `sam3.mode: junction` |
| `scan.min_good_views` / `max_passes` | keep scanning until this many close views detect |
| `scan.max_view_distance_m` | a view is **fused** only within this range of the cable (farther ones only steer the approach) |
| `scan.offsets` / `relative_bounds` | the viewpoints (camera frame) and their safety clamp |
| `scan.approach.min_distance_m` | the **view floor** (inside the D405's ~70–500 mm range) |
| `scan.refine.orbit_deg` / `max_orbit_deg` / `min_height_m` | axis-sharpening orbit + its guards |
| `connector_estimator.*` | RANSAC / parallax / workspace gates for the fuse-mode fit |
| `reconstruction.*` | curve reconstruction + its two convergence thresholds (`scan.mode: reconstruction`) |
| `<cable>.junction_in_fingertip` (cables.yaml) | the junction pose wrt the **fingertip** at grasp — the fingertip is posed at detected junction @ its inverse before closing (replaces `connector_grasp` / `junction_offset_m`) |
| `fingertip_grasp` | the fingertip relative to tool0 (the grasp reference) |
| `grasp_check.closed_counts` | full-closure counts; stalling short = failed grasp |

Needs the SAM3 environment (see `requirements/perception.txt`). `min_height_m` **must** stay below
`approach.min_distance_m` or every orbit view is rejected once the camera closes in.
