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
    """Run a free-space move with the force guard armed as a canceller (the canonical
    implementation lives next to ForceGuard in robot/guard.py)."""
    from ..robot.guard import guarded_move
    return guarded_move(robot, guard, move_fn)


def prompts_off(cfg):
    """True when `skip_prompts` (or --no-prompts) is set: run without asking ANYTHING.

    Stronger than confirm_each_step / --yes, which only silence the per-step gates and leave
    the deliberate interlocks -- the reset, the pre-contact stand-off, the operator's success
    call -- still asking. This silences those too, and each falls back to the same behaviour a
    dry run uses (the tolerance check decides success).

    THE ONE PROMPT IT DOES NOT TOUCH is the cable labelling: which cable to pick is an input,
    not a confirmation, and there is no sane default for it -- a run that guessed would grab
    an arbitrary cable. skills/ground_pick asks regardless."""
    return bool(cfg.get('skip_prompts', False))


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


def pose_fields_mm(T):
    """skills.trajectory.pose_fields with the translation in MILLIMETRES (matching the _mm
    column names the sampling CSVs use)."""
    from ..skills import trajectory as traj
    f = traj.pose_fields(T)
    return [f[0] * 1000.0, f[1] * 1000.0, f[2] * 1000.0] + f[3:]


def fmt_dur(seconds):
    """'h:mm:ss' (or 'm:ss' under an hour) for log lines."""
    seconds = int(max(0.0, seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f'{h}:{m:02d}:{sec:02d}' if h else f'{m}:{sec:02d}'


def eta_clock(seconds_from_now):
    """Wall-clock time `seconds_from_now` ahead -- the 'done ~HH:MM:SS' in progress lines."""
    return (datetime.now() + timedelta(seconds=max(0.0, seconds_from_now))).strftime('%H:%M:%S')
