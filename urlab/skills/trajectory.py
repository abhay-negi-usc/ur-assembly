"""Trajectory tooling for the kinematic-assembly and uncertain-sampling demos.

All pure numpy, lifted verbatim from KinematicAssembly / UncertainAssemblySampling (the audit
flagged these as already ROS-free): CSV loading, resampling, the target-frame anchoring trick,
per-trial perturbation, and the closest-pose disassembly snap.

THE ANCHORING TRICK (worth understanding before you touch it). The ground truth is the
ASSEMBLED tool0 pose in base -- you jog the arm to a good mate and read it off the robot. The
target OBJECT frame is never measured; it is back-derived so that the LAST trajectory row maps
EXACTLY onto that assembled pose:

    T_base_targetobj = assembled_pose @ held_object_pose @ inv(traj[-1])

Every row then maps to a tool0 command via
    T_base_tool0(row) = T_base_targetobj @ T_targetobj_held(row) @ inv(T_tool0_held)
and the last row collapses back to assembled_pose by construction. That self-consistency is the
whole point -- do not try to measure the target frame independently.
"""

import csv
import itertools
import math

import numpy as np

from .. import log as urlog
from ..transforms import (
    inverse, matrix_to_quat, matrix_to_xyzrpy, slerp_matrix, xyzrpy_to_matrix)

log = urlog.get('trajectory')


def load_csv(path, angles_deg=False):
    """Load an (x, y, z, roll, pitch, yaw) trajectory as a list of 4x4 T_targetobj_held.

    Skips blanks, '#' comments, and any row whose first six cells do not all parse as floats --
    which is how the header line is silently dropped."""
    mats = []
    with open(path, 'r') as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith('#'):
                continue
            try:
                vals = [float(x) for x in row[:6]]
            except ValueError:
                continue
            if len(vals) < 6:
                continue
            rpy = [math.radians(a) for a in vals[3:6]] if angles_deg else vals[3:6]
            mats.append(xyzrpy_to_matrix(vals[0:3], rpy))
    return mats


def anchor_target(assembled_pose, held_object_pose, traj_last):
    """T_base_targetobj, back-derived so traj[-1] maps exactly onto the assembled pose."""
    T_base_held_assembled = assembled_pose @ held_object_pose
    return T_base_held_assembled @ inverse(traj_last)


def tool0_at(T_base_targetobj, T_targetobj_held, T_tool0_held):
    """A trajectory row (held-obj-vs-target) reduced to a tool0 pose in base."""
    return T_base_targetobj @ T_targetobj_held @ inverse(T_tool0_held)


def resample(mats, res_t=0.001, res_r_deg=1.0):
    """Densify so consecutive poses are within res_t metres and res_r_deg degrees.

    n = max(1, ceil(dp / res_t), ceil(dtheta / res_r)) subdivisions per segment; LERP position,
    SLERP orientation. The geodesic angle is 2*arccos(|qa.qb|) (shortest path)."""
    res_r = math.radians(res_r_deg)
    dense = [mats[0]]
    for a, b in zip(mats[:-1], mats[1:]):
        dp = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
        dot = min(1.0, abs(float(np.dot(matrix_to_quat(a), matrix_to_quat(b)))))
        dtheta = float(2.0 * np.arccos(dot))
        n = 1
        if res_t > 0:
            n = max(n, int(np.ceil(dp / res_t)))
        if res_r > 0:
            n = max(n, int(np.ceil(dtheta / res_r)))
        for k in range(1, n + 1):
            dense.append(slerp_matrix(a, b, k / n))
    return dense


def delta_from(vec):
    """A 4x4 pose delta from a 6-vector [x, y, z (m), roll, pitch, yaw (DEG)]."""
    v = np.asarray(vec, dtype=float)
    return xyzrpy_to_matrix(v[:3], np.radians(v[3:6]))


def random_delta(lower, upper, rng):
    """A uniform random pose delta with per-DOF LOWER/UPPER bounds [x,y,z (m), r,p,y (deg)].

    Bounds are absolute, NOT half-widths -- so a range need not be centred on zero (e.g. a connector
    that always sags can be given [-4, -1] mm rather than a symmetric +/-)."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if np.any(hi < lo):
        raise ValueError(f'uncertainty upper < lower on DOF {list(np.where(hi < lo)[0])}')
    return delta_from(lo + rng.uniform(0.0, 1.0, 6) * (hi - lo))


def _axis_values(lo, hi, step):
    """Grid values from lo to hi INCLUSIVE, ~`step` apart. The step is nudged so both endpoints are
    hit exactly (a range that isn't a whole multiple of `step` would otherwise silently drop its
    upper end). A degenerate DOF (lo == hi) contributes a single value."""
    if math.isclose(lo, hi):
        return [lo]
    if step <= 0:
        raise ValueError(f'grid_resolution must be > 0 for a DOF spanning [{lo}, {hi}]')
    n = max(1, int(round(abs(hi - lo) / step)))
    return [lo + i * (hi - lo) / n for i in range(n + 1)]


def grid_deltas(lower, upper, resolution):
    """Every combination of the per-DOF grid values -- the full Cartesian product, in order.

    Deterministic and exhaustive: `len()` IS the trial count. DOFs whose lower == upper contribute
    one value, so an all-zero DOF costs nothing. Returns a list of 4x4 deltas."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if np.any(hi < lo):
        raise ValueError(f'uncertainty upper < lower on DOF {list(np.where(hi < lo)[0])}')
    res = np.asarray(resolution, dtype=float)
    axes = [_axis_values(lo[i], hi[i], res[i]) for i in range(6)]
    return [delta_from(v) for v in itertools.product(*axes)]


PERTURB_FRAMES = ('connector', 'held', 'target')


def perturb(dense, bias, noise_lower, noise_upper, rng, frame='connector'):
    """The dense poses with a bias (ONE delta for the whole trial -- a systematic offset) plus noise
    (a fresh draw per waypoint -- jitter).

    `bias` is an already-chosen 4x4 delta, so the caller decides how it was picked: a random draw
    (`random_delta`) or the next point of an exhaustive sweep (`grid_deltas`).

    WHICH FRAME the bias acts in is a choice of ERROR SOURCE, and the two model different physics:

    frame='connector' (or 'held', the default): perturbed[i] = dense[i] @ bias @ noise_i
        (RIGHT multiply, per waypoint). This is IN-HAND POSE ERROR -- the part is held wrong in the
        gripper by `bias`, expressed along the PART's own axes. The robot still executes its NOMINAL
        motion (it does not know the grasp is off), so the travel direction stays along the TARGET's
        axis while the part rides through the insertion tilted/offset. A pitch bias tilts the part,
        it does NOT rotate the approach direction -- that is correct here, not a bug.

    frame='target': perturbed[i] = bias @ noise_i @ dense[i] (LEFT multiply). This is TARGET / SOCKET
        POSE ERROR -- the robot's belief about where the mate is, is wrong by `bias`, expressed along
        the TARGET's axes. The whole approach is rigidly misaimed, so the part drives in along its
        OWN (rotated) axis and a rotational bias also translates it by (R_bias - I) @ p.
        (The trajectory's last row is the identity mate, so this is already the rigid,
        mate-anchored misalignment -- anchor @ bias @ anchor^-1 with anchor = I.)

    Both reach a DIFFERENT path but the same class of endpoint, so pick by the error you mean to
    study; uncertain_sampling exposes it as `sampling.perturb_frame`."""
    if frame not in PERTURB_FRAMES:
        raise ValueError(f'perturb frame {frame!r} must be one of {PERTURB_FRAMES}')
    if frame in ('held', 'connector'):
        return [c @ bias @ random_delta(noise_lower, noise_upper, rng) for c in dense]
    return [bias @ random_delta(noise_lower, noise_upper, rng) @ c for c in dense]


def closest_index(actual, dense, rot_weight_mm_per_deg=1.0):
    """Index of the dense pose closest to `actual`, by cost = |dp|_mm + w * dtheta_deg."""
    p = actual[:3, 3]
    qa = matrix_to_quat(actual)
    best_i, best_cost = 0, float('inf')
    for i, d in enumerate(dense):
        dt_mm = float(np.linalg.norm(d[:3, 3] - p)) * 1000.0
        dot = min(1.0, abs(float(np.dot(qa, matrix_to_quat(d)))))
        dr_deg = math.degrees(2.0 * math.acos(dot))
        cost = dt_mm + rot_weight_mm_per_deg * dr_deg
        if cost < best_cost:
            best_i, best_cost = i, cost
    return best_i


def pose_fields(T):
    """[x, y, z, qx, qy, qz, qw, yaw_deg, pitch_deg, roll_deg] for a CSV log row."""
    q = matrix_to_quat(T)
    _, rpy = matrix_to_xyzrpy(T)
    roll, pitch, yaw = rpy
    return [T[0, 3], T[1, 3], T[2, 3], q[0], q[1], q[2], q[3],
            math.degrees(yaw), math.degrees(pitch), math.degrees(roll)]


def ik_chain(arm, poses, seed):
    """IK a sequence of tool0 poses, each seeded from the previous solution -- keeps the whole path
    on one IK branch. Returns the joint list, or None on the first unreachable pose."""
    out = []
    q = list(seed)
    for i, pose in enumerate(poses):
        q = arm.ik(pose, q)
        if q is None:
            log.error('IK failed at waypoint %d/%d.', i + 1, len(poses))
            return None
        out.append(q)
    return out
