"""Robotiq 2F-85 finger-gap <-> fingertip-advance relation (analytic, from the open linkage).

WHY: the 2F-85's fingertips do not close in a fixed plane -- each pad rides a CIRCLE around its
inner-knuckle (spring-link) pivot while staying parallel, so the pads ADVANCE along the gripper
approach axis as the fingers close (~28 mm over the full 85 mm stroke). A grasp closed to one gap
therefore puts the fingertip at a DIFFERENT distance from the flange than a grasp at another gap;
this module gives that relation in closed form so grasp/descent depths can be compensated per
object diameter.

SOURCE of the linkage constants: the MuJoCo Menagerie 2F-85 model
(google-deepmind/mujoco_menagerie, robotiq_2f85/2f85.xml). Chosen over the ROS URDFs (e.g.
PickNikRobotics/ros2_robotiq_gripper) DELIBERATELY: Menagerie models the true four-bar LOOP
CLOSURE, and its spring-link geometry reproduces the full ~85 mm stroke across the driver's
0..0.8 rad range; the URDFs' mimic-joint parallelogram (inner joints = +-driver angle)
understrokes (~71 mm) -- good enough to visualise, wrong to measure from.

THE MODEL (per finger; y = lateral from the gripper centreline, z = forward from the mounting
face, both in metres). The pad-carrying follower attaches to the spring link at radius R from
the pivot (Y0, Z0), and the pad keeps a fixed orientation, so the pad face moves on

    y(psi) = Y0 + R cos(psi) + W        z(psi) = Z0 + R sin(psi) + H

with psi swinging PSI_OPEN -> PSI_CLOSED as the gripper closes. gap = 2 y, and eliminating psi:

    z(gap) = Z0 + sqrt(R^2 - (gap/2 - Y0 - W)^2) + H

W lumps every fixed LATERAL offset from the follower origin to the actual pad face and H every
fixed FORWARD offset. CUSTOM FINGERTIPS CHANGE W AND H (ours have a groove): calibrate them by
measuring (gap, fingertip z) at one or two known closures. The defaults are the STOCK pads --
W chosen so the faces meet exactly (gap 0) at the driver limit (it then equals Menagerie's pad
+ silicone offsets, a consistency check), H = 0. DIFFERENCES z(g1) - z(g2) are H-free, so
depth COMPENSATION between two gaps needs no H calibration at all.
"""

import numpy as np

# Menagerie robotiq_2f85/2f85.xml: spring_link body at (y, z) = (0.0132, 0.0609) from the base
# mounting face; follower attached at (0.055, 0.0375) in the spring-link frame; driver (and,
# through the loop closure, the spring link) swings 0..0.8 rad.
SPRING_PIVOT_LATERAL_M = 0.0132
SPRING_PIVOT_FORWARD_M = 0.0609
FOLLOWER_RADIUS_M = float(np.hypot(0.055, 0.0375))          # 0.066568
PSI_OPEN_RAD = float(np.arctan2(0.0375, 0.055))             # 0.59870 (fully open)
PSI_CLOSED_RAD = PSI_OPEN_RAD + 0.8                         # 1.39870 (fully closed)

# Stock pads: the lateral pad-face offset that makes gap(PSI_CLOSED) exactly 0. Comes out to
# -24.6 mm = Menagerie's pad offset (-18.9 mm) plus its silicone thickness -- a cross-check.
STOCK_PAD_LATERAL_M = -(SPRING_PIVOT_LATERAL_M
                        + FOLLOWER_RADIUS_M * float(np.cos(PSI_CLOSED_RAD)))

# Full-open gap with the stock pads: ~87 mm (spec 85 mm; the margin is pad compression).
GAP_MAX_M = 2.0 * (SPRING_PIVOT_LATERAL_M + FOLLOWER_RADIUS_M * float(np.cos(PSI_OPEN_RAD))
                   + STOCK_PAD_LATERAL_M)


def pad_forward_from_gap(gap_m, pad_lateral_m=STOCK_PAD_LATERAL_M, pad_forward_m=0.0):
    """Forward position (m, along the approach axis from the 2F-85 mounting face) of the pad
    face when the fingers are at `gap_m`. Larger when more closed -- the pads ADVANCE.

    `pad_lateral_m` / `pad_forward_m` are the fixed offsets of YOUR pad face from the follower
    origin (calibrate for custom fingertips; defaults = stock pads). Raises ValueError for a gap
    the linkage cannot reach with those pads."""
    u = gap_m / 2.0 - SPRING_PIVOT_LATERAL_M - pad_lateral_m        # = R cos(psi)
    u_min = FOLLOWER_RADIUS_M * float(np.cos(PSI_CLOSED_RAD))
    u_max = FOLLOWER_RADIUS_M * float(np.cos(PSI_OPEN_RAD))
    if not (u_min - 1e-9 <= u <= u_max + 1e-9):
        lo = 2.0 * (SPRING_PIVOT_LATERAL_M + u_min + pad_lateral_m)
        hi = 2.0 * (SPRING_PIVOT_LATERAL_M + u_max + pad_lateral_m)
        raise ValueError(f'gap {gap_m * 1000:.1f} mm is outside the linkage stroke '
                         f'[{lo * 1000:.1f}, {hi * 1000:.1f}] mm for these pads')
    u = float(np.clip(u, u_min, u_max))
    return (SPRING_PIVOT_FORWARD_M + float(np.sqrt(FOLLOWER_RADIUS_M ** 2 - u * u))
            + pad_forward_m)


def gap_from_counts(counts, counts_open, counts_closed, gap_at_open_m=GAP_MAX_M):
    """Finger gap (m) from a gripper position in COUNTS, anchored on THIS gripper's measured
    endpoints: `counts_open` <-> `gap_at_open_m`, `counts_closed` <-> gap 0 (the counts where
    YOUR pads meet -- e.g. 228, not 255, with custom fingertips). Linear in counts, which is
    Robotiq's own ~0.4 mm/count convention; the residual nonlinearity through the four-bar is
    well under the repeatability of the drive."""
    frac = (float(counts_closed) - float(counts)) / (float(counts_closed) - float(counts_open))
    return max(0.0, min(1.0, frac)) * float(gap_at_open_m)


def pad_forward_from_counts(counts, counts_open, counts_closed, gap_at_open_m=GAP_MAX_M,
                            pad_lateral_m=STOCK_PAD_LATERAL_M, pad_forward_m=0.0):
    """pad_forward_from_gap of gap_from_counts -- the one-call version."""
    return pad_forward_from_gap(gap_from_counts(counts, counts_open, counts_closed,
                                                gap_at_open_m),
                                pad_lateral_m, pad_forward_m)
