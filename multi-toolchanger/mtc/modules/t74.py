"""t74 -- the T74 tile motor: PID position control on an encoder. Firmware: firmware/t74.cpp,
control law: firmware/t74_control.h. Its pins are in config/t74.yaml (build-time checked).

HOW IT IS TUNED. The motor and its load are modelled as a DC motor driving an inertia:

    tau * y'' + y' = K * (u - friction)       y in encoder counts, u the PWM (-255..255)

`t74_identify` measures K (speed per PWM), tau (the mechanical time constant -- it grows with the
load's inertia) and the friction, by stepping the PWM and fitting that model to the encoder.
pid_gains() then puts all three closed-loop poles at -w, `bandwidth` in config/t74.yaml:

    Kp = 3 tau w^2 / K      Ki = tau w^3 / K      Kd = (3 tau w - 1) / K

-- critically damped AT THE IDENTIFIED LOAD. Identify with the HEAVIEST load the tile will
carry: lighter loads then stay close to critically damped (a damping ratio of at least 0.87),
while a load heavier than the identified one would ring. The integral term holds an unbalanced
load at the target; the friction term keeps the gearbox from making it hunt.

Wire protocol (all positions in encoder counts from the zero; this file converts degrees):

    'J<kp>,<ki>,<kd>,<K>,<tau>,<friction>\\n' -> "t74 gains ok"
    'L<vmax>,<amax>,<band>,<maxerr>,<homespeed>,<hold>\\n' -> "t74 limits ok"
                                      hold 1: hold after a move; 0: release once settled
    'R<counts>\\n' / 'A<counts>,<turn>\\n'  -> "t74 move from <p> to <g>",
                                      A with turn > 0 (counts per tile turn): the shortest way
                                      round, to the nearest angle equal to <counts> mod <turn>
                                      later "t74 done pos <p> goal <g>"
    'H'  -> "t74 homing", "t74 homed at raw <n>", later "t74 done ..."
    'Z' -> "t74 zero"     'S' -> "t74 halt pos <p>"     'X' -> "t74 off pos <p>"
    'O<pwm>\\n' -> "t74 open <pwm> pos <p>"
    'I<pwm1>,<pwm2>,<ms>\\n' -> "t74 id start <raw>", "t74 id <ms> <raw>"..., "t74 id done"
    'E' -> "t74 pos <p> goal <g> err <e> u <u> mode <m> homed <0|1> index <n> missed <n> raw <n>"

Problems arrive as "t74 rejected: ..." (a command refused) or "t74 FAULT ..." (motor stopped).
"""

import json
import math
import os
import re
import time

from ..base import ToolChangerError, parse_int
from ..registry import Module, Param

CALIBRATION_FILE = 't74_calibration.json'   # measured state, beside config/t74.yaml
CONTROL_HZ = 500                             # must match t74.cpp

REJECTED = 't74 rejected:'
FAULT = 't74 FAULT'


#  =====   the maths   =====
def pid_gains(K, tau, w):
    """(kp, ki, kd, note) placing all three closed-loop poles at -w (rad/s).

    Units follow the board: kp in PWM per count, ki per count-second, kd per count/s. K is
    signed, so the gains come out with whatever sign makes the loop negative feedback.
    Kd would go negative below w = 1/(3 tau): it is clamped to 0 there and `note` says so."""
    kp = 3 * tau * w * w / K
    ki = tau * w ** 3 / K
    kd = (3 * tau * w - 1) / K
    note = ''
    if 3 * tau * w < 1:
        kd = 0.0
        note = (f'bandwidth {w:g} rad/s is below 1/(3 tau) = {1 / (3 * tau):.1f} rad/s, so the '
                f'derivative gain would be negative: it is clamped to 0, and the response is '
                f'no longer exactly critically damped. Raise the bandwidth.')
    elif w > CONTROL_HZ / 10:
        note = (f'bandwidth {w:g} rad/s is more than a tenth of the {CONTROL_HZ} Hz control '
                f'rate; the sampled loop will not behave like the model. Lower it.')
    return kp, ki, kd, note


def _solve(a, b):
    """Solve the small linear system a x = b (Gaussian elimination with partial pivoting)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ToolChangerError('identification fit is singular -- did the motor move?')
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit_steps(samples, step_s):
    """Fit the motor model to a two-step identification run.

    `samples` is [(seconds, count)], PWM pwm1 for the first `step_s` seconds and pwm2 for the
    next. Within each step the speed relaxes exponentially (time constant tau) towards that
    step's steady speed, the first from rest and the second from the first's steady speed:

        step 1:  x = a1 + v1 (t - tau (1 - e^(-t/tau)))
        step 2:  x = a2 + v2 (s - tau (1 - e^(-s/tau))) + v1 tau (1 - e^(-s/tau)),   s = t - T

    For a given tau that is linear in (a1, v1, a2, v2), so least squares solves it exactly, and
    a golden-section search finds the tau with the smallest residual.

    Returns (v1, v2, tau, rms) -- speeds in counts/s, tau in s, rms residual in counts."""
    T = step_s
    pts = [(t, float(x)) for t, x in samples if 0 <= t < 2 * T]
    if len(pts) < 20:
        raise ToolChangerError(f'only {len(pts)} identification samples -- too few to fit')

    def basis(t, tau):
        if t < T:
            g = 1 - math.exp(-t / tau)
            return [1, t - tau * g, 0, 0]
        s = t - T
        g = 1 - math.exp(-s / tau)
        return [0, tau * g, 1, s - tau * g]

    def solve(tau):
        ata = [[0.0] * 4 for _ in range(4)]
        atb = [0.0] * 4
        for t, x in pts:
            row = basis(t, tau)
            for i in range(4):
                atb[i] += row[i] * x
                for j in range(4):
                    ata[i][j] += row[i] * row[j]
        p = _solve(ata, atb)
        sse = sum((x - sum(r * q for r, q in zip(basis(t, tau), p))) ** 2 for t, x in pts)
        return p, sse

    lo, hi = 1e-3, T            # tau must leave each step time to settle
    phi = (math.sqrt(5) - 1) / 2
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fc, fd = solve(c)[1], solve(d)[1]
    for _ in range(60):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo)
            fc = solve(c)[1]
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo)
            fd = solve(d)[1]
    tau = (lo + hi) / 2
    (a1, v1, a2, v2), sse = solve(tau)
    return v1, v2, tau, math.sqrt(sse / len(pts))


def model_from_fit(pwm1, pwm2, v1, v2):
    """(K, friction) from the steady speeds at two PWMs of the same sign.

    speed = K (pwm - friction * sign(pwm)), so the slope between the two points is K and the
    PWM where the line meets zero speed is the friction."""
    K = (v2 - v1) / (pwm2 - pwm1)
    friction = (pwm1 - v1 / K) * (1 if pwm1 > 0 else -1)
    return K, max(0.0, friction)


#  =====   calibration file   =====
def load_calibration(path):
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise ToolChangerError(f'could not read {path}: {exc}')
    return data if isinstance(data, dict) else {}


def save_calibration(path, data):
    data = dict(data, saved=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2)
        fh.write('\n')


#  =====   the device   =====
class T74:
    """Blocking control of the T74. Angles are tile degrees; the board works in counts."""

    def __init__(self, board, settings):
        self.board = board
        self.tag = board.tag
        self.s = settings
        self.bandwidth = settings.bandwidth
        self.hold = settings.hold_after_move     # t74_hold changes it for the session
        self.cal_path = os.path.join(settings.config_dir, CALIBRATION_FILE)
        self.cal = load_calibration(self.cal_path)
        #  The board forgets everything on reset, and opening the port resets it: send it all.
        if self.turn_counts:
            self._send_limits()
        if self.identified:
            self._send_gains(quiet=True)

    #  ---- calibration ----
    @property
    def turn_counts(self):
        """Signed encoder counts per FORWARD tile turn, or None before t74_calibrate."""
        tc = self.cal.get('turn_counts')
        return tc if isinstance(tc, int) and abs(tc) >= 100 else None

    @property
    def identified(self):
        return all(isinstance(self.cal.get(k), (int, float))
                   for k in ('model_k', 'model_tau', 'friction')) and self.cal['model_k'] != 0

    def _need_counts(self):
        if not self.turn_counts:
            raise ToolChangerError('t74 has no counts per tile turn yet: run t74_calibrate (or '
                                   't74_counts N if you know it)')
        return self.turn_counts

    def _need_model(self):
        if not self.identified:
            raise ToolChangerError('t74 has no motor model yet, so no gains: run t74_identify '
                                   'with the heaviest load mounted')

    def deg(self, counts):
        return counts * 360.0 / self._need_counts() + 0.0   # + 0.0: no "-0.000"

    def counts(self, degrees):
        return int(round(degrees * self._need_counts() / 360.0))

    def _save(self, **changes):
        self.cal.update(changes)
        save_calibration(self.cal_path, self.cal)
        rel = os.path.relpath(self.cal_path)
        print(f'   saved to {self.cal_path if rel.startswith("..") else rel}')

    #  ---- talking to the board ----
    def _ask(self, cmd, expect):
        """Send `cmd`, return the regex match of the reply `expect`; refusals and faults raise."""
        final, lines = self.board.exchange(
            cmd, lambda ln: re.match(expect, ln) or ln.startswith((REJECTED, FAULT)))
        for note in lines[:-1]:
            print(f'   {note}')
        if final.startswith((REJECTED, FAULT)):
            raise ToolChangerError(final)
        return re.match(expect, final)

    def _wait(self, cmd, expect, timeout):
        """Wait for `expect` after a command was accepted; a fault raises."""
        try:
            final, lines = self.board.collect(
                cmd, lambda ln: re.match(expect, ln) or ln.startswith(FAULT), timeout)
        except KeyboardInterrupt:
            self.halt()
            raise
        for note in lines[:-1]:
            if not note.startswith('t74 id '):
                print(f'   {note}')
        if final.startswith(FAULT):
            raise ToolChangerError(final)
        return re.match(expect, final), lines

    def _send_limits(self):
        tc = abs(self.turn_counts)
        per_deg = tc / 360.0
        band = max(1.0, self.s.hold_band_deg * per_deg)
        home = self.s.home_speed_dps * per_deg * (1 if self.turn_counts > 0 else -1)
        self._ask(f'L{self.s.max_speed_dps * per_deg:.1f},{self.s.max_accel_dps2 * per_deg:.1f},'
                  f'{band:.1f},{self.s.max_error_deg * per_deg:.1f},{home:.1f},'
                  f'{1 if self.hold else 0}\n',
                  r't74 limits ok')

    def _send_gains(self, quiet=False):
        K, tau, fr = self.cal['model_k'], self.cal['model_tau'], self.cal['friction']
        kp, ki, kd, note = pid_gains(K, tau, self.bandwidth)
        self._ask(f'J{kp:.6g},{ki:.6g},{kd:.6g},{K:.6g},{tau:.6g},{fr:.4g}\n', r't74 gains ok')
        if note:
            print(f'   WARNING: {note}')
        if self.turn_counts:
            top = abs(K) * (255 - fr) * 360.0 / abs(self.turn_counts)
            if self.s.max_speed_dps > 0.8 * top:
                print(f'   WARNING: max_speed_dps {self.s.max_speed_dps:g} is near or above what '
                      f'the motor can do at full PWM (~{top:.0f} deg/s): moves will saturate and '
                      f'lag. Lower it in config/t74.yaml.')
        if not quiet:
            print(f'{self.tag}t74 gains: kp {kp:.4g}  ki {ki:.4g}  kd {kd:.4g}  (bandwidth '
                  f'{self.bandwidth:g} rad/s; triple pole at -{self.bandwidth:g})')
        return kp, ki, kd

    def _move_timeout(self, distance_counts):
        tc = abs(self._need_counts())
        v = self.s.max_speed_dps * tc / 360.0
        a = self.s.max_accel_dps2 * tc / 360.0
        return abs(distance_counts) / v + v / a + 5.0 / self.bandwidth + 3.0

    #  ---- moves ----
    def _move(self, cmd, label):
        self._need_model()
        m = self._ask(cmd, r't74 move from (-?\d+) to (-?\d+)')
        start, goal = int(m.group(1)), int(m.group(2))
        print(f'{self.tag}{label}: {self.deg(start):.2f} -> {self.deg(goal):.2f} deg')
        done, _ = self._wait(cmd, r't74 done pos (-?\d+) goal (-?\d+)',
                             self._move_timeout(goal - start))
        pos = int(done.group(1))
        print(f'{self.tag}{label}: at {self.deg(pos):.3f} deg, error '
              f'{self.deg(pos - goal):+.3f} deg -- {self._after()}')
        return self.deg(pos)

    def move(self, degrees):
        """Turn the tile `degrees` (negative = reverse) from where it is meant to be, then hold.

        Relative to the previous target while holding, so a run of moves does not accumulate
        each one's small error. Returns the final angle from zero."""
        return self._move(f'R{self.counts(degrees)}\n', 't74_move')

    def goto(self, degrees):
        """Turn the tile to `degrees` from the zero (home, t74_zero, or power-on), then hold.

        The shortest way round: 300 deg from 0 is reached by turning back 60, through the
        zero. The board picks the direction (from the same base as a relative move), so the
        final angle is `degrees` give or take whole turns. Returns it."""
        return self._move(f'A{self.counts(degrees)},{abs(self._need_counts())}\n', 't74_goto')

    def home(self):
        """Run forward to the encoder index pulse, make it zero, and hold there."""
        self._need_model()
        self._ask('H', r't74 homing')
        tc = abs(self._need_counts())
        guard = 5 * self.s.encoder_ppr / (self.s.home_speed_dps * tc / 360.0)
        self._wait('H', r't74 homed at raw (-?\d+)', guard + 3.0)
        self._wait('H', r't74 done pos (-?\d+)', self._move_timeout(4 * self.s.encoder_ppr))
        print(f'{self.tag}t74_home: homed; zero is the index pulse -- {self._after()}')
        if tc > 6 * self.s.encoder_ppr:
            print(f'   note: the encoder is on the motor side ({tc / (4 * self.s.encoder_ppr):.1f} '
                  f'encoder turns per tile turn), so its index repeats every '
                  f'{360.0 * 4 * self.s.encoder_ppr / tc:.1f} deg of tile. Home is a repeatable '
                  f'MOTOR position, not a unique tile angle.')

    def zero(self):
        """Make the current position zero."""
        self._ask('Z', r't74 zero')
        print(f'{self.tag}t74_zero: this is now 0 deg')

    def halt(self):
        """Brake to a stop and hold there (open-loop runs just stop)."""
        m = self._ask('S', r't74 halt pos (-?\d+)')
        if self.turn_counts:
            print(f'{self.tag}t74: halted at {self.deg(int(m.group(1))):.2f} deg')

    def stop(self):
        """Release the motor: no more holding torque."""
        m = self._ask('X', r't74 off pos (-?\d+)')
        where = f' at {self.deg(int(m.group(1))):.2f} deg' if self.turn_counts else ''
        print(f'{self.tag}t74_stop: motor released{where}')
        return int(m.group(1))

    def safe_stop(self):
        """What q and a failed sequence call. HALT, not release: an unbalanced load must not be
        dropped because someone pressed q."""
        self.halt()

    def check_line(self, line):
        if line.startswith(FAULT):
            raise ToolChangerError(line)

    def report(self):
        """The board's state, in degrees where it has them. Returns the parsed fields."""
        m = self._ask('E', r't74 pos .*')
        f = dict(zip(*[iter(m.group(0).split()[1:])] * 2))
        if self.turn_counts:
            print(f"{self.tag}t74: at {self.deg(int(f['pos'])):.3f} deg, goal "
                  f"{self.deg(int(f['goal'])):.3f} deg, error {self.deg(int(f['err'])):+.3f} deg, "
                  f"u {f['u']}, {f['mode']}, homed {'yes' if f['homed'] == '1' else 'no'}, "
                  f"index pulses {f['index']}, missed edges {f['missed']}")
        else:
            print(f'{self.tag}{m.group(0)}')
        return f

    #  ---- calibration and identification ----
    def calibrate(self, turns):
        """Count encoder counts per tile turn: run forward while you watch a mark go round."""
        pwm = self.s.calibrate_pwm
        print('CALIBRATION: the tile turns forward at PWM %d until you press Enter.' % pwm)
        input('  1. Mark the tile and line the mark up with a fixed reference. Enter to start. ')
        m = self._ask(f'O{pwm}\n', r't74 open -?\d+ pos (-?\d+)')
        start = int(m.group(1))
        try:
            what = 'once' if turns == 1 else f'for the {turns}th time'
            input(f'  2. Press Enter the moment the mark comes back to the reference {what}. ')
        finally:
            end = self.stop()
        tc = int(round((end - start) / turns))
        if abs(tc) < 100:
            raise ToolChangerError(f'only {end - start} counts over {turns} turn(s): is the '
                                   f'encoder connected (A on D2, B on D3)?')
        self._save(turn_counts=tc)
        print(f'{self.tag}t74_calibrate: {tc} counts per tile turn = '
              f'{abs(tc) / (4 * self.s.encoder_ppr):.2f} encoder turns (the gear ratio, if '
              f'encoder_ppr matches the DIP switches)')
        self._send_limits()

    def set_counts(self, counts):
        """Set the counts per tile turn by hand (4 x PPR x gear ratio, signed)."""
        self._save(turn_counts=int(counts))
        self._send_limits()
        print(f'{self.tag}t74_counts: {counts} counts per tile turn')

    def identify(self, pwm1, pwm2, seconds):
        """Step the PWM to pwm1, then pwm2, and fit the motor model to the encoder."""
        if (pwm1 > 0) != (pwm2 > 0) or pwm1 == pwm2:
            raise ToolChangerError('identify needs two different PWMs of the same sign')
        ms = int(round(seconds * 1000))
        m = self._ask(f'I{pwm1},{pwm2},{ms}\n', r't74 id start (-?\d+)')
        raw0 = int(m.group(1))
        print(f'{self.tag}t74_identify: PWM {pwm1} for {seconds:g} s, then {pwm2} for '
              f'{seconds:g} s ...')
        done, lines = self._wait('I', r't74 id (done|overflow)', 2 * seconds + 5)
        if done.group(1) == 'overflow':
            raise ToolChangerError('the board could not send the samples fast enough; '
                                   'try again (or shorter steps)')
        samples = []
        for ln in lines:
            s = re.match(r't74 id (\d+) (-?\d+)$', ln)
            if s:
                samples.append((int(s.group(1)) / 1000.0, int(s.group(2)) - raw0))
        v1, v2, tau, rms = fit_steps(samples, seconds)
        moved = abs(v1) * seconds
        if moved < 50:
            raise ToolChangerError(f'the motor barely moved at PWM {pwm1} ({moved:.0f} counts): '
                                   f'raise both PWMs above the friction')
        K, friction = model_from_fit(pwm1, pwm2, v1, v2)
        self._save(model_k=K, model_tau=tau, friction=friction, identify_pwm=[pwm1, pwm2],
                   identify_seconds=seconds, identify_rms_counts=round(rms, 1))
        per_deg = f' = {K * 360 / self.turn_counts:.3f} deg/s per PWM' if self.turn_counts else ''
        print(f'{self.tag}t74_identify: K {K:.2f} counts/s per PWM{per_deg}, tau {tau * 1000:.1f} '
              f'ms, friction {friction:.1f} PWM (fit rms {rms:.1f} counts)')
        if rms > 0.02 * moved:
            print(f'   WARNING: the fit is poor (rms {rms:.0f} of {moved:.0f} counts moved). Check '
                  f'"missed edges" in t74_status, and that nothing stopped the tile.')
        self._send_gains()
        return K, tau, friction

    def _after(self):
        return 'holding' if self.hold else 'released'

    def set_hold(self, on=None):
        """Hold after each move (True) or release once settled (False); None toggles.

        The board does the releasing, the moment the move has settled within hold_band_deg.
        A relative move afterwards still counts from the last target, not from wherever an
        unbalanced load let the tile drift. Returns the new state."""
        self.hold = (not self.hold) if on is None else bool(on)
        if self.turn_counts:
            self._send_limits()             # the flag travels with the limits
        state = ('on -- moves end holding the target' if self.hold
                 else 'off -- the motor is released once each move settles')
        print(f'{self.tag}t74_hold: {state}')
        return self.hold

    def tune(self, bandwidth=None):
        """Show the gains, or recompute them for another bandwidth for this session."""
        if bandwidth is not None:
            self.bandwidth = bandwidth
        self._need_model()
        self._send_gains()
        print(f'   critically damped at the identified load (tau '
              f'{self.cal["model_tau"] * 1000:.1f} ms): errors die away like '
              f'(1 + wt + (wt)^2/2) e^(-wt), ~{6.6 / self.bandwidth:.2f} s to 1%.')
        if bandwidth is not None and bandwidth != self.s.bandwidth:
            print(f'   for this session only -- put `bandwidth: {bandwidth:g}` in config/t74.yaml '
                  f'to keep it')


#  =====   the module   =====
def _number(text, lo, hi, what, unit):
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise ToolChangerError(f'{what} wants a number of {unit}, got {text!r}')
    if not lo <= value <= hi or value != value:
        raise ToolChangerError(f'{what} {text} is out of range {lo:g}..{hi:g} {unit}')
    return value


def _check_number(lo, hi):
    def check(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
            return f'must be a number from {lo:g} to {hi:g}'
        return None
    return check


def _check_int(lo, hi):
    def check(value):
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            return f'must be a whole number from {lo} to {hi}'
        return None
    return check


MODULE = Module('t74', 'T74 tile motor: PID position control on an encoder (firmware/t74.cpp)',
                device=T74)

MODULE.pin('rpwm', 5, needs='pwm')        # IBT-2 forward
MODULE.pin('lpwm', 6, needs='pwm')        # IBT-2 reverse
MODULE.pin('r_en', 7)
MODULE.pin('l_en', 8)
MODULE.pin('enc_a', 2, needs='interrupt') # every edge of A and B is counted, so both need
MODULE.pin('enc_b', 3, needs='interrupt') #   the Uno's two hardware interrupts
MODULE.pin('enc_x', 4, needs='pcint')     # the index pulse
MODULE.claim('timer1')                    # the 500 Hz control loop

MODULE.setting('bandwidth', 15.0, _check_number(0.5, 50))      # rad/s: the triple pole
MODULE.setting('max_speed_dps', 60.0, _check_number(1, 3600))   # tile deg/s, profile cruise
MODULE.setting('max_accel_dps2', 240.0, _check_number(1, 36000))
MODULE.setting('hold_band_deg', 0.05, _check_number(0, 10))     # "in position" half-width
MODULE.setting('max_error_deg', 10.0, _check_number(0.1, 360))  # following error -> fault
MODULE.setting('home_speed_dps', 20.0, _check_number(1, 360))
MODULE.setting('calibrate_pwm', 100, _check_int(1, 255))
MODULE.setting('identify_pwm1', 60, _check_int(1, 255))
MODULE.setting('identify_pwm2', 120, _check_int(1, 255))
MODULE.setting('identify_seconds', 1.0, _check_number(0.1, 10))
MODULE.setting('encoder_ppr', 500, _check_int(1, 10000))        # must match the DIP switches
MODULE.setting('hold_after_move', True,
               lambda v: None if isinstance(v, bool) else 'must be true or false')

MODULE.kind('degrees', lambda t, cfg, n: _number(t, -3600, 3600, n, 'degrees'),
            lambda cfg: 'tile degrees, -3600..3600, decimals allowed; negative = reverse')
MODULE.kind('turns', lambda t, cfg, n: parse_int(t, 10, n, 'turns'),
            lambda cfg: 'whole tile turns to count over, 1..10 (more is more exact)')
MODULE.kind('counts', lambda t, cfg, n: parse_int(t, 10_000_000, n, 'counts'),
            lambda cfg: 'signed encoder counts per forward tile turn (4 x PPR x gear ratio)')
MODULE.kind('bandwidth', lambda t, cfg, n: _number(t, 0.5, 50, n, 'rad/s'),
            lambda cfg: 'closed-loop pole, rad/s (0.5..50): higher is faster and stiffer')
MODULE.kind('pwm', lambda t, cfg, n: parse_int(t, 255, n, 'PWM'),
            lambda cfg: 'PWM, -255..255; both identify PWMs must have the same sign')
def _on_off(text, cfg, name):
    word = str(text).strip().lower()
    if word in ('on', 'true', 'yes', '1'):
        return True
    if word in ('off', 'false', 'no', '0'):
        return False
    raise ToolChangerError(f'{name} wants on or off, got {text!r}')


MODULE.kind('on_off', _on_off, lambda cfg: 'on or off; leave it out to toggle',
            lambda cfg: ['on', 'off'])
MODULE.kind('step_seconds', lambda t, cfg, n: _number(t, 0.1, 10, n, 'seconds'),
            lambda cfg: 'seconds per identification step, 0.1..10')


#  With hold_after_move on, moves end HOLDING, and closing the port resets the board, which
#  releases the motor. So from the one-shot CLI they keep the connection open until q or Ctrl-C
#  (runs_on), like `drive`. With it off the move ends released anyway, and the CLI just exits.
HOLDS = dict(timed=True, runs_on=lambda args, cfg: cfg.settings['t74'].hold_after_move)


@MODULE.command(Param('DEG', 'degrees'), **HOLDS)
def cmd_t74_move(s, deg):
    """Turn the tile DEG degrees from its current target, then hold there (see t74_hold).

    The board follows a trapezoidal profile (max_speed_dps, max_accel_dps2 in config/t74.yaml)
    under PID control, and reports when it is within hold_band_deg. It then KEEPS HOLDING --
    t74_stop releases it -- unless t74_hold is off, when the board releases it right away.
    Relative to the previous target, so errors do not add up over a run of moves. Press q to
    brake to a stop."""
    s.dev('t74').move(deg)


@MODULE.command(Param('DEG', 'degrees'), **HOLDS)
def cmd_t74_goto(s, deg):
    """Turn the tile to DEG degrees from the zero, the shortest way round, then hold (t74_hold).

    It never turns more than half a turn: from 10 deg, `t74_goto 300` turns back 70 deg through
    the zero rather than forward 290 (exactly half a turn may go either way). Angles are the same
    a whole turn apart, so DEG 300 and -60 are the same place, and t74_status may report the
    tile at -60 or 660 after a few trips round.

    The zero is where the board powered up, where t74_zero was given, or the index pulse after
    t74_home. Opening the port resets the board, so a zero lasts one connection. Press q to
    brake to a stop and hold."""
    s.dev('t74').goto(deg)


@MODULE.command(**HOLDS)
def cmd_t74_home(s):
    """Run forward to the encoder index pulse, make it zero, and hold there (see t74_hold).

    With the encoder on the motor shaft the index repeats about every 18 deg of tile, so this
    is a repeatable motor position, not a unique tile angle."""
    s.dev('t74').home()


@MODULE.command()
def cmd_t74_zero(s):
    """Make the current position 0 deg."""
    s.dev('t74').zero()


@MODULE.command()
def cmd_t74_halt(s):
    """Brake to a stop and hold where it ends up (what q does)."""
    s.dev('t74').halt()


@MODULE.command()
def cmd_t74_stop(s):
    """Release the motor: no holding torque, so an unbalanced load can turn the tile."""
    s.dev('t74').stop()


@MODULE.command()
def cmd_t74_status(s):
    """Report the position, target, error, PWM, mode and encoder health."""
    s.dev('t74').report()


@MODULE.command(timed=True, in_sequence=False)
def cmd_t74_watch(s):
    """Print the position five times a second until q -- turn the shaft by hand to test wiring.

    The count should change smoothly both ways, missed edges should stay 0, and index pulses
    should go up by one per encoder shaft turn."""
    dev = s.dev('t74')
    while True:
        dev.report()
        s.wait(0.2)


@MODULE.command(Param('TURNS', 'turns', optional=True), in_sequence=False)
def cmd_t74_calibrate(s, turns=1):
    """Count the encoder counts per tile turn, watching a mark on the tile.

    Runs forward open loop at calibrate_pwm; press Enter when the mark has come round TURNS
    times (more turns divide your timing error). Saved to config/t74_calibration.json and
    reloaded on every connect. Needed once per gearbox, not per tile."""
    s.dev('t74').calibrate(turns)


@MODULE.command(Param('COUNTS', 'counts'), in_sequence=False)
def cmd_t74_counts(s, counts):
    """Set the counts per tile turn by hand, if you know the gear ratio exactly."""
    s.dev('t74').set_counts(counts)


@MODULE.command(Param('PWM1', 'pwm', optional=True), Param('PWM2', 'pwm', optional=True),
                Param('SECONDS', 'step_seconds', optional=True), in_sequence=False, timed=True)
def cmd_t74_identify(s, pwm1=None, pwm2=None, seconds=None):
    """Measure the motor model (K, tau, friction) and compute the gains. HEAVIEST LOAD ON.

    Runs PWM1 for SECONDS, then PWM2 for SECONDS, open loop, and fits the model to the encoder
    -- so the tile turns forward (roughly half a turn with the defaults: make sure it can).
    Defaults come from config/t74.yaml. Do this with the heaviest load the tile will carry:
    the gains are critically damped for the load they were identified with, and a heavier one
    would ring. Saved to config/t74_calibration.json."""
    st = s.cfg.settings['t74']
    s.dev('t74').identify(pwm1 if pwm1 is not None else st.identify_pwm1,
                          pwm2 if pwm2 is not None else st.identify_pwm2,
                          seconds if seconds is not None else st.identify_seconds)


@MODULE.command(Param('STATE', 'on_off', optional=True))
def cmd_t74_hold(s, state=None):
    """Toggle holding after each move, or set it with on/off. Default: hold_after_move.

    On (the default in config/t74.yaml): every move, goto and home ends holding the target
    under PID, so an unbalanced load cannot back-drive the tile; t74_stop releases it. Off: the
    board releases the motor as soon as each move settles within hold_band_deg -- less heat,
    but the load can drift. A relative move still counts from the last target either way.
    Lasts this session; set hold_after_move in config/t74.yaml to change the default."""
    s.dev('t74').set_hold(state)


@MODULE.command(Param('BANDWIDTH', 'bandwidth', optional=True))
def cmd_t74_tune(s, bandwidth=None):
    """Show the PID gains, or recompute them for another bandwidth (this session only).

    The gains place all three closed-loop poles at -BANDWIDTH rad/s. Higher is faster and
    stiffer against an unbalanced load, but asks for more PWM and amplifies encoder noise."""
    s.dev('t74').tune(bandwidth)
