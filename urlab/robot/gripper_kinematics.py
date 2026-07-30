"""Robotiq 2F-85 finger gap <-> fingertip forward position -- CALIBRATED on our gripper.

WHY: the 2F-85's fingertips do not close in a fixed plane -- each fingertip rides a CIRCLE
around its inner-knuckle (spring-link) pivot while staying parallel, so the tips ADVANCE along
the gripper approach axis as the fingers close. A grasp closed to one gap therefore puts the
fingertip at a DIFFERENT distance from the flange than a grasp at another gap; this module gives
that relation so grasp/descent depths can be compensated per object diameter.

THE MODEL (per finger; y = lateral from the centreline, z = forward, metres):

    gap/2 = A + R cos(psi)        z = B + R sin(psi)        psi = PSI0 + K * counts

i.e. the fingertip point moves on a circle of radius R, traversed LINEARLY in encoder counts,
until the pads meet at psi_closed = arccos(-A/R) (~counts 216 on our unit); commanding further
(216..255) only squeezes the pads -- the geometry FREEZES there. Eliminating psi:

    z(gap) = B + sqrt(R^2 - (gap/2 - A)^2)

CALIBRATION (2026-07-30, OUR gripper with the custom grooved fingertips; supersedes the
constants first derived from the MuJoCo Menagerie robotiq_2f85 model). Measured
(encoder, finger separation mm, delta height mm):

    3: (83.56, 93.66)   50: (67.03, 99.03)   100: (47.80, 104.15)
    150: (27.19, 105.90)   200: (7.36, 106.42)   230: (0.00, 106.42)

Least-squares circle + linear-angle fit reproduces every row within 0.6 mm gap / 0.5 mm z.
The fitted R = 55.0 mm differs from the Menagerie linkage radius (66.6 mm): the custom
fingertips' CONTACT point does not coincide with the stock pad reference (and the pads flex), so
it rides a smaller effective circle -- the measurement wins. Full-stroke tip advance: 12.8 mm.

DATUM: z is in the CALIBRATION datum (the "delta height" measurement), NOT flange-absolute.
DIFFERENCES z(g1) - z(g2) are datum-free -- that is what depth compensation needs. For an
absolute flange distance add one measured constant via `datum_offset_m`.

QUIRK worth knowing: gap 0 lies slightly PAST the circle's apex (psi_closed ~ 96.5 deg > 90),
so z(gap) is not strictly monotonic -- it peaks at gap = 2A ~ 12.4 mm and dips ~0.3 mm by gap 0
(the measured 200/230 rows are flat for exactly this reason).
"""

import numpy as np

# ---- Calibrated constants (fit residuals < 0.5 mm; see the docstring table) ----
R_M = 0.054976                    # effective fingertip circle radius
APEX_LATERAL_M = 0.006212         # A: circle-centre lateral offset (apex at gap = 2A)
DATUM_FORWARD_M = 0.051645        # B: circle-centre forward position, in the calibration datum
PSI0_RAD = float(np.radians(49.225))          # fingertip angle at encoder 0
K_RAD_PER_COUNT = float(np.radians(0.2186))   # angle per encoder count (linear -- fit +-0.3 deg)

# The pads meet (gap 0) here; beyond, the pads squeeze and the geometry freezes.
PSI_CLOSED_RAD = float(np.arccos(-APEX_LATERAL_M / R_M))              # ~96.5 deg
COUNTS_CLOSED = (PSI_CLOSED_RAD - PSI0_RAD) / K_RAD_PER_COUNT         # ~216
# Widest commandable gap (encoder 0).
GAP_MAX_M = 2.0 * (APEX_LATERAL_M + R_M * float(np.cos(PSI0_RAD)))    # ~84.2 mm

# Per-fingertip V-GROOVE depth. An object SEATED IN THE GROOVE stalls the fingers at a FLAT-FACE
# separation of (object width - 2 * groove), so width <-> counts needs this on top of the circle
# model. Calibrated from the banana connector (diameter 9.55-10.7 mm stalling at counts 213/207
# -> 4.11/3.43 mm per side); the +-0.35 mm spread is the current model uncertainty. This is an
# EFFECTIVE depth: the stall data behind it was taken at the working grip force, so the typical
# PAD COMPRESSION (all separations in this module are zero-compression values; real grasps
# squeeze the pads, which is desired for grip pressure) is absorbed here on average.
GROOVE_DEPTH_M = 0.00377


def gap_from_counts(counts):
    """Finger separation (m) at an encoder value, from the calibrated circle + linear-angle
    model (NOT a linear mm/count approximation -- the measured mapping bends from 0.35 to
    0.41 mm/count through the stroke). Saturates at 0 past COUNTS_CLOSED (~216): commanding
    216..255 only squeezes the pads."""
    psi = PSI0_RAD + K_RAD_PER_COUNT * min(float(counts), COUNTS_CLOSED)
    return max(0.0, 2.0 * (APEX_LATERAL_M + R_M * float(np.cos(psi))))


def counts_from_gap(gap_m):
    """The encoder value at which closure reaches finger separation `gap_m` -- the inverse of
    gap_from_counts. NOTE this predicts the FLAT-FACE separation: an object seated in the
    fingertip GROOVE stalls LATER (at higher counts) by the combined groove depth, so measured
    grasp bands (cables.yaml) sit above this prediction -- compare the two to estimate the
    groove depth, not to replace the measured bands."""
    if not (-1e-9 <= gap_m <= GAP_MAX_M + 1e-9):
        raise ValueError(f'gap {gap_m * 1000:.1f} mm is outside the calibrated stroke '
                         f'[0, {GAP_MAX_M * 1000:.1f}] mm')
    u = float(np.clip(gap_m, 0.0, GAP_MAX_M)) / 2.0 - APEX_LATERAL_M
    psi = float(np.arccos(u / R_M))
    return (psi - PSI0_RAD) / K_RAD_PER_COUNT


def counts_from_width(width_m, groove_depth_m=GROOVE_DEPTH_M):
    """The encoder value at which the fingers stall on an object of physical `width_m` SEATED IN
    THE GROOVES: counts_from_gap(width - 2 * groove). An object thinner than 2 * groove (e.g. a
    bare cable) disappears into the grooves entirely -- the prediction saturates at COUNTS_CLOSED
    and the true stall lands somewhere in the pad-flex region beyond (measured, not modelled)."""
    return counts_from_gap(max(0.0, float(width_m) - 2.0 * groove_depth_m))


def width_from_counts(counts, groove_depth_m=GROOVE_DEPTH_M):
    """Physical width (m) of an object seated in the grooves when the fingers stalled at
    `counts` -- the PAYLOAD WIDTH readback: gap_from_counts + 2 * groove. Floors at 2 * groove
    past free closure (~counts 216), where thin objects are no longer resolvable."""
    return gap_from_counts(counts) + 2.0 * groove_depth_m


def pad_forward_from_gap(gap_m, datum_offset_m=0.0):
    """Forward fingertip position (m, calibration datum + `datum_offset_m`) at finger gap
    `gap_m`. Larger when more closed -- the tips ADVANCE by 12.8 mm over the full stroke.
    Differences between two gaps are datum-free. Raises ValueError outside the reachable
    stroke [0, GAP_MAX_M]."""
    if not (-1e-9 <= gap_m <= GAP_MAX_M + 1e-9):
        raise ValueError(f'gap {gap_m * 1000:.1f} mm is outside the calibrated stroke '
                         f'[0, {GAP_MAX_M * 1000:.1f}] mm')
    u = float(np.clip(gap_m, 0.0, GAP_MAX_M)) / 2.0 - APEX_LATERAL_M
    return DATUM_FORWARD_M + float(np.sqrt(R_M ** 2 - u * u)) + datum_offset_m


def pad_forward_from_counts(counts, datum_offset_m=0.0):
    """Forward fingertip position (m) at an encoder value -- rises to the apex (~counts 186),
    dips ~0.3 mm to pad contact, then freezes past COUNTS_CLOSED. The one-call composition for
    depth compensation from the gripper state."""
    psi = PSI0_RAD + K_RAD_PER_COUNT * min(float(counts), COUNTS_CLOSED)
    return DATUM_FORWARD_M + R_M * float(np.sin(psi)) + datum_offset_m
