"""The multisine wiggle -- ONE implementation, shared by every app that superimposes one.

WHY THIS EXISTS. wiggle_sampling, bnc_assembly's engage and estimator_eval's observation passes all
want the same excitation, and until now two of them had their own copy. Copies drift: the
frame convention, the taper and the speed-cap policy had already diverged between the two that
existed, which means observations collected by one app were NOT comparable with a map built by
another -- the exact failure the map/eval parity work exists to prevent. The waveform, the frame it
acts in, the validation and the servo loop all live here now, so "the same wiggle" is enforced by
construction rather than by remembering.

THE WAVEFORM

    offset_i(t) = env(t) * A_i * sin(2*pi*f_i*t + phi_i)          i over (x, y, z, roll, pitch, yaw)

with `env` a raised cosine ramping 0 -> 1 -> 0 over `taper_s` at each end.

THE FRAME. The offset is applied by RIGHT-MULTIPLICATION in the held part's own frame:

    T_ref(t) = T_base_anchor @ Delta(offset(t))

so a tilted part rocks about ITS own axes, not the target's. This is the same convention as
trajectory.perturb(frame='connector') and uncertain_sampling._axial_ref, and it is the one
wiggle_sampling was collected under. bnc_assembly's engage previously ADDED the offset to a
target-frame 6-vector instead, which is a different motion once the part is misaligned.

FOUR THINGS THAT SILENTLY BREAK A WIGGLE, all refused before the arm moves:

  * amplitude with no frequency -- a constant offset wearing a wiggle's name.
  * aliasing -- a tone above a quarter of the reference rate is reconstructed as a slower one.
    The run looks correct and excites a frequency nobody chose.
  * a degenerate frequency ratio -- two axes at a simple ratio retrace one closed Lissajous curve,
    so the probe sweeps a LINE through the box instead of filling it, and a rank estimate cannot
    tell that from a real constraint.
  * exceeding the speed cap. This RAISES rather than dilating the clock. bnc_assembly used to
    stretch the waveform to fit under its cap, which is defensible when the wiggle is a means to
    an end -- but it makes the delivered frequency a function of the amplitude, so an amplitude
    sweep confounds radius with spectrum, and two apps with different caps deliver different
    excitations while claiming the same config. One policy, and it is to refuse.

NOTE ON THE DELIVERED RATE. The waveform clock is driven from the caller's wall clock, not from an
assumed control period. An earlier run assumed dt = 1/reference_rate_hz while the servo loop
actually cycled at ~332 Hz, so every commanded frequency came out 2.65x high. Passing real elapsed
time makes the delivered frequency the configured one whatever the loop does.
"""
import math

import numpy as np

from .. import log as urlog
from ..transforms import xyzrpy_to_matrix

log = urlog.get('wiggle')

DIMS = ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')


class WiggleError(ValueError):
    """A configuration that would produce something other than the wiggle asked for."""


def from_shared(cfg, local=None, ref_name='wiggle_sampling.yaml', label=''):
    """The wiggle block from the config where it is TUNED, with any local keys layered on top.

    The axes, amplitudes, frequencies and phases are an experimental setting that was arrived at
    once, against real hardware, and every app that superimposes a wiggle wants THAT one -- an app
    carrying its own copy is an app whose observations stop being comparable the first time the
    tuned one changes. So the numbers live in configs/wiggle_sampling.yaml and are read at run
    time, the same way estimator_eval already reads uncertain_sampling.yaml for contact physics.

    `local` (the app's own `wiggle:` block, if any) is merged OVER the shared one, so a deliberate
    override is still possible and is visible as a diff against the shared values.

    Returns (block, source) where source names where the numbers came from, for logging."""
    import os as _os

    import yaml as _yaml

    from .. import config as _urconfig
    shared = {}
    src = 'local only'
    try:
        p = _urconfig.resolve(cfg, _os.path.join('configs', ref_name))
        if not _os.path.exists(p):
            p = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(
                _os.path.abspath(__file__)))), 'configs', ref_name)
        if _os.path.exists(p):
            shared = ((_yaml.safe_load(open(p)) or {}).get('wiggle') or {})
            src = ref_name
    except Exception as exc:                       # noqa: BLE001 -- advisory, never fatal
        log.info('%scould not read %s (%s); using local values only',
                 f'[{label}] ' if label else '', ref_name, exc)
    keep = ('amplitude', 'frequency_hz', 'phase_deg', 'taper_s',
            'max_speed_mm_s', 'max_rotation_deg_s')
    out = {k: shared[k] for k in keep if k in shared}
    for k, v in (local or {}).items():
        if k in out and out[k] != v:
            src += f' (+local {k})'
        out[k] = v
    return out, src


def _vec(block, key, default=0.0):
    b = (block or {}).get(key) or {}
    if not isinstance(b, dict):
        raise WiggleError(f'wiggle {key} must be a mapping keyed by {list(DIMS)}')
    bad = [k for k in b if k not in DIMS]
    if bad:
        raise WiggleError(f'wiggle {key} has unknown axes {bad}; expected {list(DIMS)}')
    return [float(b.get(d, default)) for d in DIMS]


class Wiggle:
    """A validated multisine. `None` from `from_cfg` means "no oscillation configured"."""

    def __init__(self, amp, frq, pha_deg, taper_s=0.0, label=''):
        self.amp = [float(v) for v in amp]
        self.frq = [float(v) for v in frq]
        self.pha = [math.radians(float(v)) for v in pha_deg]
        self.taper_s = float(taper_s)
        self.label = label
        self.live = [i for i in range(6) if abs(self.amp[i]) > 0.0]

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_cfg(cls, block, label=''):
        block = block or {}
        amp = _vec(block, 'amplitude')
        if not any(abs(v) > 0.0 for v in amp):
            return None
        w = cls(amp, _vec(block, 'frequency_hz'), _vec(block, 'phase_deg'),
                float(block.get('taper_s', 0.0) or 0.0), label)
        w.validate(rate_hz=float(block.get('sample_rate_hz', 0) or 0),
                   cap_v=block.get('max_speed_mm_s'),
                   cap_w=block.get('max_rotation_deg_s'),
                   scale=float(block.get('amplitude_scale', 1.0) or 1.0))
        return w

    # ------------------------------------------------------------------ validation
    def validate(self, rate_hz=0.0, cap_v=None, cap_w=None, scale=1.0, duration_s=None):
        tag = f'[{self.label}] ' if self.label else ''
        for i in self.live:
            if self.frq[i] <= 0.0:
                raise WiggleError(
                    f'{tag}amplitude.{DIMS[i]} is non-zero but its frequency is 0 -- that is a '
                    'constant OFFSET, not an oscillation. Put a constant offset in the pose.')
        fmax = max(self.frq[i] for i in self.live)
        if rate_hz and rate_hz < 4.0 * fmax:
            raise WiggleError(
                f'{tag}the reference is rebuilt at {rate_hz:.0f} Hz but the fastest axis is '
                f'{fmax:.2f} Hz -- below 4x the sampled sine ALIASES into a slower one, and the '
                'run looks correct while exciting a frequency nobody chose.')
        pv, pw = self.peak_speed(scale)
        if cap_v and pv > float(cap_v):
            raise WiggleError(
                f'{tag}peak {pv:.1f} mm/s exceeds the {float(cap_v):.1f} mm/s cap. Lower the '
                'amplitude or the frequency -- this is NOT dilated away, because dilating ties '
                'the delivered spectrum to the amplitude.')
        if cap_w and pw > float(cap_w):
            raise WiggleError(f'{tag}peak {pw:.1f} deg/s exceeds the {float(cap_w):.1f} deg/s cap.')
        if len(self.live) >= 2:
            orbit, slowest = self.orbit_s()
            if orbit < 3.0 * slowest:
                raise WiggleError(
                    f'{tag}frequencies {[self.frq[i] for i in self.live]} Hz close their orbit '
                    f'every {orbit:.1f} s against a slowest single-axis period of {slowest:.1f} s '
                    '-- too simple a ratio, so the probe traces a LINE instead of filling the box.')
            if duration_s and duration_s < orbit:
                log.warning('%sthe burst is %.1f s but the orbit closes every %.1f s -- each one '
                            'sees %.0f%% of the pattern, so windows from different bursts are not '
                            'comparable.', tag, duration_s, orbit, 100 * duration_s / orbit)
        if duration_s:
            for i in self.live:
                if self.frq[i] * duration_s < 1.0:
                    log.warning('%s%s completes %.2f cycles in %.1f s -- under one cycle acts as a '
                                'constant OFFSET, not a wiggle.', tag, DIMS[i],
                                self.frq[i] * duration_s, duration_s)
        if self.taper_s > 0 and duration_s and 2 * self.taper_s >= duration_s:
            raise WiggleError(f'{tag}taper_s {self.taper_s:.2f} x2 does not fit inside '
                              f'{duration_s:.2f} s -- it would never reach full amplitude.')

    # ------------------------------------------------------------------ description
    def peak_speed(self, scale=1.0):
        tw = math.tau
        pv = max([abs(self.amp[i]) * scale * tw * self.frq[i] for i in range(3)
                  if self.frq[i] > 0], default=0.0)
        pw = max([abs(self.amp[i]) * scale * tw * self.frq[i] for i in range(3, 6)
                  if self.frq[i] > 0], default=0.0)
        return pv, pw

    def orbit_s(self):
        """(orbit period, slowest single-axis period). The Lissajous figure closes at 1/gcd."""
        g = 0
        for i in self.live:
            g = math.gcd(g, int(round(self.frq[i] * 1000.0)))
        orbit = (1000.0 / g) if g else 0.0
        return orbit, 1.0 / min(self.frq[i] for i in self.live)

    def describe(self, scale=1.0):
        live = ', '.join(f'{DIMS[i]} {self.amp[i] * scale:+.2f}@{self.frq[i]:.2f}Hz'
                         for i in self.live)
        pv, pw = self.peak_speed(scale)
        orbit, _ = self.orbit_s()
        return (f'{live}; peak {pv:.2f} mm/s / {pw:.2f} deg/s'
                + (f'; orbit closes every {orbit:.1f} s' if orbit else ''))

    # ------------------------------------------------------------------ the waveform
    def envelope(self, t, duration):
        """Raised cosine, 0 -> 1 -> 0 over taper_s at each end.

        Without it the multisine steps to full amplitude at t=0 (the phases are not all zero, so
        sin(phi) != 0), and a step on the reference is an impulse into the contact -- a transient
        with nothing to do with the constraint geometry."""
        if self.taper_s <= 0.0 or not duration:
            return 1.0
        if t < self.taper_s:
            return 0.5 * (1.0 - math.cos(math.pi * t / self.taper_s))
        if t > duration - self.taper_s:
            return 0.5 * (1.0 - math.cos(math.pi * max(0.0, duration - t) / self.taper_s))
        return 1.0

    def offset6(self, t, duration=None, scale=1.0):
        """The 6-vector excitation (mm / deg) at time t."""
        env = self.envelope(t, duration) * scale
        out = np.zeros(6)
        for i in self.live:
            if self.frq[i] > 0.0:
                out[i] = env * self.amp[i] * math.sin(2.0 * math.pi * self.frq[i] * t + self.pha[i])
        return out

    def delta(self, t, duration=None, scale=1.0):
        """The 4x4 factor to RIGHT-multiply onto the held part's pose."""
        v = self.offset6(t, duration, scale)
        return xyzrpy_to_matrix(v[:3] / 1000.0, np.radians(v[3:]))

    def tapered(self, t, duration):
        return self.envelope(t, duration) < 0.999


def run(adm, wig, anchor_fn, duration, dt, guard=None, on_step=None, scale=1.0,
        on_ref=None, clock=None):
    """Drive `wig` for `duration` seconds, stepping the reference once per servo cycle.

    `anchor_fn(delta4x4) -> tool0 pose` is the caller's anchoring: a station for
    wiggle_sampling, a trajectory point for engage, the believed seat for estimator_eval. Keeping
    it a callback is what lets one waveform serve all three without any of them re-implementing it.

    `clock` (default time.monotonic) supplies REAL elapsed time, so the delivered frequency is the
    configured one even when the servo loop does not run at its nominal rate.

    Returns ('done'|'seated', last_reference)."""
    import time as _t
    if clock is None:
        # A DRY RUN HAS NO REAL TIMING to be faithful to, and a real clock would make a dry pass of
        # a 50-minute protocol take 50 minutes. Advance a virtual clock by dt instead, which is
        # exactly what the old assumed-dt loop did -- correct offline, and never used on hardware.
        dry = bool(getattr(getattr(adm, 'arm', None), 'dry_run', False))
        if dry:
            _n = [0]

            def clock():
                _n[0] += 1
                return (_n[0] - 1) * dt
        else:
            clock = _t.monotonic
    t0 = clock()
    prev = anchor_fn(np.eye(4))
    last = prev
    while True:
        t = clock() - t0
        if t >= duration:
            break
        cur = anchor_fn(wig.delta(t, duration, scale))
        if on_ref is not None:
            on_ref(cur, t, wig.tapered(t, duration))
        res = adm.ramp(prev, cur, dt, guard, on_step=on_step)
        prev = last = cur
        if res == 'seated':
            return 'seated', last
    return 'done', last
