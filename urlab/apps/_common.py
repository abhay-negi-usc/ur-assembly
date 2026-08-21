"""Shared helpers for the app scripts.

Every app used to carry private copies of the same plumbing: segment timing for compliant
ramps, timestamped output directories, ETA formatting, the tare callback, guarded free-space
moves, and EOF-safe operator prompts.  They live here once, so the apps stay short and the
copies cannot drift.
"""

import math
import os
from datetime import datetime, timedelta

from .. import config as urconfig
from .. import log as urlog
from ..transforms import pose_error

log = urlog.get('app')


def seg_time(A, B, v_mm_s, w_deg_s=None, min_s=0.0):
    """Duration for a compliant reference ramp A -> B, paced so that neither the translation
    speed (mm/s) nor -- when `w_deg_s` is given -- the rotation speed (deg/s) is exceeded.
    Never shorter than `min_s` (typically one servo cycle)."""
    lin_m, ang_rad = pose_error(A, B)
    t_lin = (lin_m * 1000.0 / v_mm_s) if v_mm_s > 0 else 0.0
    t_ang = (math.degrees(ang_rad) / w_deg_s) if (w_deg_s or 0.0) > 0 else 0.0
    return max(t_lin, t_ang, min_s)


def tare_fn(robot, cfg_section, key='tare_before', default=True):
    """The standard mid-warmup tare callback, or None when disabled in the config.

    Must not block (the servo stream is live while it runs), hence settle=False."""
    if bool((cfg_section or {}).get(key, default)):
        return lambda: robot.arm.zero_ft(settle=False)
    return None


def guarded(robot, guard, move_fn):
    """Run a free-space move with the force guard armed as a canceller.

    A trip here means the arm hit something UNEXPECTED (contact phases read the guard
    themselves, where a trip means 'seated')."""
    guard.reset()
    robot.arm.add_guard(guard)
    try:
        ok = move_fn()
    finally:
        robot.arm.clear_guards()
    if not ok and guard.tripped_by:
        log.error('Force guard tripped during a free-space move (%s) -- hit something '
                  'unexpected.', guard.tripped_by)
    return ok


def ask(prompt, abort_answers=('q', 'quit', 'n', 'no'), on_eof=True):
    """EOF-safe operator prompt. Returns False if the user aborts; `on_eof` is the answer when
    stdin is closed (piped/unattended runs)."""
    try:
        answer = input(prompt)
    except EOFError:
        return on_eof
    return answer.strip().lower() not in abort_answers


def experiment_dir(cfg, name):
    """data/experiments/<name>_<stamp>/ -- one directory per run, created here."""
    out = os.path.join(cfg.get('data_dir', 'data'), 'experiments',
                       f'{name}_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out, exist_ok=True)
    return out


def run_dir(cfg, path, per_cable=True):
    """<path>[/<cable>]/run_<stamp>/ -- a timestamped run directory under a configured root
    (NOT created; the caller decides when)."""
    path = urconfig.resolve(cfg, path)
    cable = cfg.get('cable') if per_cable else None
    if cable:
        path = os.path.join(path, str(cable))
    return os.path.join(path, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')


def fmt_dur(seconds):
    """'h:mm:ss' (or 'm:ss' under an hour) for log lines."""
    seconds = int(max(0.0, seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f'{h}:{m:02d}:{sec:02d}' if h else f'{m}:{sec:02d}'


def eta_clock(seconds_from_now):
    """Wall-clock time `seconds_from_now` ahead -- the 'done ~HH:MM:SS' in progress lines."""
    return (datetime.now() + timedelta(seconds=max(0.0, seconds_from_now))).strftime('%H:%M:%S')
