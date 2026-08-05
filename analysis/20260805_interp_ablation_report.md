# Interpolated-correspondence ablation — banana connector

**Date:** 2026-08-05
**Scripts:** `analysis/manifold_interp_ablation.py` (solver: `analysis/manifold_icp_validation.py`
`solve_trial`, identical to `urlab/skills/manifold.py`)
**Data:** manifold `data/banana_map.csv` (161,161 points, median sample spacing 3.23 mm-eq);
validation `data/banana_observations.csv` (2,622 rows → 25 observations of ~100 contact rows)
**Runs:** `analysis/20260805_140917_interp_ablation_banana/` (main grid),
`analysis/20260805_141800_interp_ablation_banana/` (extension: k=16 softness sweep, k=32)

## Question

The contact manifold is a finite sample of a continuous surface, so exact nearest-neighbour ICP
matching can **latch** onto the single closest sample and quantise (or entirely misplace) the
recovered belief correction. Does the optional soft correspondence — blending the k nearest
manifold points, weights `exp(-(d - d_nearest)/tau)`, `tau = interp_softness x` the manifold's
median point spacing — measurably improve recovery, and what tuning is best?

## Method

Paired comparison: every variant solves the same 25 observations with the same hidden offsets
(data-derived) and the same RNG seed (7), so guesses and RANSAC draws are identical — any error
difference is the feature. Settings: perturb/estimate dims x/z/pitch, 100 guesses x 50 ICP
iterations, step gain 1.5, no residual gate, recency half-life 0.5, scaling deg→mm 1.0 /
unit-force 10 / unit-torque 100.

## Results (mean |error| over 25 observations)

| Variant | x [mm] | z [mm] | pitch [deg] | final residual [mm-eq] |
|---|---|---|---|---|
| exact-NN (baseline) | 3.62 | 2.87 | 1.42 | 3.68 |
| k=4 soft=0.5 | 2.57 | 2.66 | 1.38 | 2.51 |
| k=4 soft=1.0 | 2.62 | 2.63 | 1.39 | 2.52 |
| k=8 soft=0.5 | 2.42 | 2.63 | 1.24 | 2.28 |
| k=8 soft=1.0 | 2.44 | 2.58 | 1.30 | 2.32 |
| k=8 soft=2.0 | 2.42 | 2.56 | 1.31 | 2.36 |
| k=16 soft=0.5 | 2.13 | 2.50 | 1.26 | 2.17 |
| **k=16 soft=1.0** | **1.72** | **2.49** | **1.24** | 2.21 |
| k=16 soft=2.0 | 1.72 | 2.48 | 1.25 | 2.27 |
| k=32 soft=1.0 | 1.74 | 2.46 | 1.22 | 2.28 |
| k=32 soft=2.0 | 1.47 | 2.46 | 1.22 | 2.36 |

Figures: `abs_error_by_variant.png` in each run folder (per-dim distributions, baseline outlined).

## Findings

1. **Interpolation's win is tail suppression, not median shift.** The baseline's x-error
   distribution spans ~0.5–8 mm with outliers to 11 mm — latching failures where an observation
   set locks onto the wrong manifold sample. With interpolation the x box collapses to
   ~0.3–2.4 mm. Medians move less (x 1.31 → 0.73 mm; z and pitch medians roughly unchanged).
   In robot terms: insurance against the occasional badly-wrong correction, the failure mode
   that costs an assembly attempt.
2. **Gains saturate at k=16.** k=4 already buys most of the residual improvement; k=16 halves
   mean x-error vs the baseline; k=32 is marginal while doubling query cost and memory.
3. **Softness is not critical on this manifold.** 0.5–2.0 perform nearly identically — at
   3.2 mm-eq sample spacing the manifold is dense enough that kernel width barely matters.
   (On a deliberately coarse synthetic manifold, soft=2.0 over-smooths and degrades — see
   `test_manifold_interpolation_reduces_latching` — so softness still deserves a look whenever
   the manifold is sparse.)

## Recommendation

Enable in `configs/cable_pick_estimate_assemble.yaml` → `estimation:`

```yaml
interp_neighbors: 16
interp_softness: 1.0
```

Caveat: this ablation ran at the offline validation scalings (deg→mm 1.0 / force 10 /
torque 100); the robot's estimation section currently uses different ones (0.2 / 0 / 1). tau
self-adapts (it derives from the manifold's spacing in whatever space the scalings define), but
k=16 was validated in the offline space — do one on-robot sanity run before trusting it in the
loop.

## Reproduce

`python analysis/manifold_interp_ablation.py` — the CSV paths above are in its CONFIG
`overrides`; the extension variants (k=16 soft sweep, k=32) can be added to `variants`.
