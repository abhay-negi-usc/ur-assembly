"""Estimate-loop I/O shared by the assembly apps: observation rows and their CSV form.

These lived as private helpers in apps/cable_pick_estimate_assemble and were imported by
sibling apps -- app-to-app coupling on underscore names. One home, public names; the donor
apps re-export under their old names.
"""

import csv

import numpy as np

from .manifold import FORCE_COLS, POSE_COLS, TORQUE_COLS
from ..transforms import inverse, matrix_to_xyzrpy


def corr_to_m(T_corr_mm):
    """The estimator's correction (translation in mm) -> a metre-based transform."""
    T = np.array(T_corr_mm, dtype=float)
    T[:3, 3] /= 1000.0
    return T


def observe(robot, T_tool0_conn, T_base_tconn):
    """One observation row: believed connector-wrt-target [mm, deg 6-vec] + raw wrench in the
    believed connector frame [N, Nm]."""
    T_base_tool0 = robot.tool0()
    T_base_conn = T_base_tool0 @ T_tool0_conn
    rel = inverse(T_base_tconn) @ T_base_conn
    xyz, rpy = matrix_to_xyzrpy(rel)
    # Flange pose handed over: wrench_in moves the moment's reference point off the flange,
    # and re-reading the pose there would pair the wrench with a different cycle.
    w = robot.arm.wrench_in(T_base_conn, T_base_tool0)
    return list(xyz * 1000.0) + list(np.degrees(rpy)) + list(w)


def save_observations(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(POSE_COLS + FORCE_COLS + TORQUE_COLS)
        w.writerows(rows)
