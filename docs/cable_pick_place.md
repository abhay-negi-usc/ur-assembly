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
| `sam3.repo_path` | your sam3-abhay checkout (urlab imports `cable_neck_core` from it) |
| `sam3.mode` | `neck` here; `adaptive` + `confidence_floor` tune the threshold per image |
| `scan.min_good_views` / `max_passes` | keep scanning until this many views detect |
| `scan.offsets` / `relative_bounds` | the viewpoints (camera frame) and their safety clamp |
| `scan.approach.min_distance_m` | the **view floor** (200 mm — inside the D405's ~70–500 mm range) |
| `scan.refine.orbit_deg` / `max_orbit_deg` / `min_height_m` | axis-sharpening orbit + its guards |
| `connector_estimator.*` | RANSAC / parallax / workspace gates for the fusion |
| `connector_grasp` | fingertip offset from the connector frame at grasp |
| `fingertip_grasp` | the fingertip relative to tool0 (the grasp reference) |
| `grasp_check.closed_counts` | full-closure counts; stalling short = failed grasp |

Needs the SAM3 environment (see `requirements/perception.txt`). `min_height_m` **must** stay below
`approach.min_distance_m` or every orbit view is rejected once the camera closes in.
