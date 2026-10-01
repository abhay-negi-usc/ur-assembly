#!/usr/bin/env python3
"""A fake multi-toolchanger board on a pty, so the driver can be tested with no hardware.

    ./fake_board.py          run the driver against the simulation and assert the protocol

Opening the real port pulls DTR, resets the Arduino and swings the servo, so this exists to
exercise the driver without moving anything. It speaks exactly the lines firmware/ prints.
"""

import contextlib
import io
import json
import math
import os
import pathlib
import pty
import re
import sys
import tempfile
import threading
import time

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from mtc import Interrupted, Session, ToolChanger, ToolChangerError, load_config  # noqa: E402
from mtc.base import MAX_RUN_S  # noqa: E402
from mtc.config import PROTOCOL  # noqa: E402
from mtc.modules.screwdrive import duty_for_rpm  # noqa: E402
from mtc.modules.t74 import fit_steps, pid_gains  # noqa: E402
from mtc.session import complete_line, render_help  # noqa: E402

FIRMWARE = HERE / 'firmware'
THRESH = int(re.search(r'const int thresh\s*=\s*(\d+)',
                       (FIRMWARE / 'coupler.cpp').read_text()).group(1))
MAX_RUN_MS = int(re.search(r'const unsigned long maxRunMs\s*=\s*(\d+)',
                           (FIRMWARE / 'screwdrive.cpp').read_text()).group(1))
RAW_TOOL = 400        # what the sensor reads with a tool in front of it (below thresh)
RAW_EMPTY = 1023      # ... and with nothing there (railed, as measured on the cell)

BOARD = {'dc': 0}     # what the simulated screwdrive is doing, for the tests to inspect
ALL = ['coupler', 'relay', 'screwdrive']


def read_args(fd, count):
    """screwdrive.cpp's readArgs(): exactly `count` comma-separated '-'?digits up to the
    newline, always draining to it. None if malformed."""
    nums, text, ok = [], b'', True
    while True:
        b = os.read(fd, 1)
        end = not b or b in b'\r\n'
        if end or b == b',':
            if text in (b'', b'-') or len(nums) >= count:
                ok = False
            else:
                nums.append(int(text))
            text = b''
            if end:
                break
            continue
        if (b.isdigit() or (b == b'-' and not text)) and len(text) < 8:
            text += b
        else:
            ok = False
    return nums if ok and len(nums) == count else None


class FakeT74:
    """The T74 firmware's protocol over a motor whose true model is known, so the host's
    identification fit can be checked against the truth. Moves are not PID-simulated here
    (firmware/test/ does that, on the real control law): the position just follows the profile
    to the goal, so the host's side of every exchange can be exercised."""

    K, TAU, FRICTION = -120.0, 0.06, 35.0     # the "true" motor: counts/s per PWM, s, PWM
    INDEX_EVERY = 2000                        # counts per encoder turn: 4 x 500 PPR

    def __init__(self, say):
        self.say = say
        self.raw = 0.0                # encoder count
        self.zero = 0
        self.gains = None             # last J: kp, ki, kd, K, tau, friction
        self.limits = None            # last L: vmax, amax, band, maxerr, homespeed
        self.goal = 0.0
        self.motion = None            # (t0, start, goal, seconds) of a move in progress
        self.open = None              # (t0, start, pwm) of an open-loop run
        self.mode = 'off'
        self.goal_kept = False        # released on a goal by hold-off: R counts from it
        self.gen = 0
        self.fault_next_move = False
        self.commands = []            # every command letter seen, for the tests

    def _where(self):
        now = time.time()
        if self.motion:
            t0, start, goal, dur = self.motion
            f = min(1.0, (now - t0) / dur)
            return start + (goal - start) * f
        if self.open:
            t0, start, pwm = self.open
            speed = self.K * (abs(pwm) - self.FRICTION) * (1 if pwm > 0 else -1)
            return start + speed * (now - t0)
        return self.raw

    def _freeze(self):
        self.raw = self._where()
        self.motion = self.open = None
        self.gen += 1

    def _move(self, goal):
        if not self.gains or not self.limits:
            self.say('t74 rejected: no gains yet -- the host sends them; run t74_identify first')
            return
        self._freeze()
        start, vmax, amax = self.raw, self.limits[0], self.limits[1]
        dur = (abs(goal - start) / vmax + vmax / amax) * 0.2 + 0.05   # faster than life
        self.motion = (time.time(), start, goal, dur)
        self.mode, self.goal = 'hold', goal
        self.say(f't74 move from {round(start) - self.zero} to {round(goal) - self.zero}')
        gen, fault = self.gen, self.fault_next_move
        self.fault_next_move = False

        def finish():
            if self.gen != gen:
                return
            if fault:
                self._freeze()
                self.mode = 'off'
                self.say('t74 FAULT following error too large -- jammed, overloaded, or the '
                         'gains have the wrong sign (re-run t74_identify)')
                return
            self.raw, self.motion = goal, None
            self.goal_kept = self.limits[5] == 0
            self.say(f't74 done pos {round(goal) - self.zero} goal {round(goal) - self.zero}')
            if self.limits[5] == 0:
                self.mode = 'off'                  # released once settled, goal kept
        threading.Timer(dur, finish).start()

    def identify(self, pwm1, pwm2, ms):
        """The samples the real board would stream: two steps of the true model, quantized."""
        self.say(f't74 id start {round(self.raw)}')
        T, x, v, t = ms / 1000.0, self.raw, 0.0, 0.0
        dt = 0.0005
        for step, pwm in enumerate((pwm1, pwm2)):
            target = self.K * (pwm - self.FRICTION)
            for _ in range(int(T / dt)):
                v += dt * (target - v) / self.TAU
                x += v * dt
                t += dt
                tick = round(t / dt)
                if tick % 8 == 0:                       # every 4 ms
                    self.say(f't74 id {round(t * 1000)} {int(x // 1)}')
        self.raw = x
        self.say('t74 id done')

    def handle(self, ch, fd):
        self.commands.append(ch)
        if ch in 'JLRAOI':
            text = b''
            while True:
                b = os.read(fd, 1)
                if not b or b in b'\r\n':
                    break
                text += b
            try:
                nums = [float(v) for v in text.decode().split(',')]
            except ValueError:
                nums = []
        if ch == 'J':
            self.gains = nums
            self.say('t74 gains ok')
        elif ch == 'L':
            self.limits = nums
            self.say('t74 limits ok')
        elif ch == 'R':
            base = self.goal if self.mode == 'hold' or self.goal_kept else self._where()
            self._move(round(base + nums[0]))
        elif ch == 'A':
            base = self.goal if self.mode == 'hold' or self.goal_kept else self._where()
            d = self.zero + nums[0] - base
            if len(nums) > 1 and nums[1] > 0:       # as the firmware: the nearest equivalent
                d = math.fmod(d, nums[1])
                d = d - nums[1] if d > nums[1] / 2 else d + nums[1] if d <= -nums[1] / 2 else d
            self._move(round(base + d))
        elif ch == 'H':
            if not self.gains or not self.limits:
                self.say('t74 rejected: no gains yet')
                return
            self._freeze()
            self.say('t74 homing')
            direction = 1 if self.limits[4] > 0 else -1
            at = (math.floor(self.raw / self.INDEX_EVERY) + (1 if direction > 0 else 0)) \
                * self.INDEX_EVERY
            gen = self.gen

            def found():
                if self.gen == gen:
                    self.zero, self.raw, self.goal, self.mode = at, at, at, 'hold'
                    self.say(f't74 homed at raw {at}')
                    self.say('t74 done pos 0 goal 0')
            threading.Timer(0.1, found).start()
        elif ch == 'Z':
            self._freeze()
            self.zero = round(self.raw)
            self.say('t74 zero')
        elif ch == 'S':
            self._freeze()
            if self.mode == 'hold':
                self.goal = self.raw
            else:
                self.mode = 'off'
            self.say(f't74 halt pos {round(self.raw) - self.zero}')
        elif ch == 'X':
            self._freeze()
            self.goal_kept = False
            self.mode = 'off'
            self.say(f't74 off pos {round(self.raw) - self.zero}')
        elif ch == 'O':
            self._freeze()
            self.open, self.mode = (time.time(), self.raw, int(nums[0])), 'open'
            self.say(f't74 open {int(nums[0])} pos {round(self.raw) - self.zero}')
        elif ch == 'I':
            self._freeze()
            self.identify(int(nums[0]), int(nums[1]), int(nums[2]))
        elif ch == 'E':
            pos = round(self._where())
            self.say(f't74 pos {pos - self.zero} goal {round(self.goal) - self.zero} err 0 u 0 '
                     f'mode {self.mode} homed 0 index 0 missed 0 raw {pos} '
                     f'hold {int(self.limits[5]) if self.limits else 1}')


T74S = []   # the FakeT74 of each simulated t74 board, for the tests to inspect


def board(fd, tool_present, modules=ALL, proto=PROTOCOL, name=None):
    """Mimic firmware/ built with `modules`: one ASCII byte in (plus numbers after the
    screwdrive's), lines out. Bytes for a module that is not built in are ignored, as on the
    board. modules=None mimics firmware from before protocol 2, which cannot say what it has;
    `name` mimics build_flash.sh --name."""
    status = 1 if tool_present[0] else 0     # setup(): status = checkTool() ? 1 : 0
    relay = False
    bypassed = False
    run = [0]             # generation of the current timed run/ramp; bumped to cancel it

    #  delayed a beat: the driver opens the pty after this thread starts, and a banner written
    #  before it is listening is a banner it never sees
    time.sleep(0.2)
    identity = f"modules={','.join(modules)} proto={proto}" if modules is not None else None
    if identity and name:
        identity = f'name={name} {identity}'

    os.write(fd, f"toolchanger ready {identity or ''}".strip().encode() + b"\r\n")
    has = set(modules if modules is not None else ALL)
    t74 = FakeT74(lambda text: os.write(fd, text.encode() + b"\r\n"))
    T74S.append(t74)

    def say(text):
        os.write(fd, text.encode() + b"\r\n")

    def raw():
        return RAW_TOOL if tool_present[0] else RAW_EMPTY

    def signal(ok):
        say(f"{status} confirmed!" if ok else f"emergency stop raw={raw()} thresh={THRESH}")

    def later(seconds, value, message):
        def fire(gen=run[0]):
            if run[0] == gen:     # not replaced by a later command
                BOARD['dc'] = value
                say(message)
        threading.Timer(seconds, fire).start()

    while True:
        try:
            c = os.read(fd, 1)
        except OSError:
            return
        if not c:
            return
        ch = c.decode('ascii', errors='replace')

        if ch == '?':
            if identity:                       # old firmware ignores it
                say(identity)
        elif (ch in 'srb' or ch.isdigit()) and 'coupler' not in has:
            pass
        elif ch == 'm' and 'relay' not in has:
            pass
        elif ch in 'dtpa' and 'screwdrive' not in has:
            pass
        elif ch in 'JLRAHZSXOIE':
            if 't74' in has:
                t74.handle(ch, fd)
        elif ch == 's':
            say("tool present" if tool_present[0] else "tool absent")
            signal(status == 0 or tool_present[0])
        elif ch == 'r':
            say(f"raw {raw()} thresh {THRESH} tool {'yes' if tool_present[0] else 'no'} "
                f"status {status} bypass {'on' if bypassed else 'off'}")
        elif ch == 'b':
            bypassed = not bypassed
            say("sensor bypass ON -- grip is NOT verified" if bypassed else "sensor bypass off")
        elif ch.isdigit():
            new = int(ch)
            if new != status:
                say(f"changed status from {status} to {new}")
                status = new
            if bypassed:
                say("(sensor bypassed)")
                signal(True)
            elif status > 0:
                signal(tool_present[0])
            else:
                say(f"released, tool {'still in the changer' if tool_present[0] else 'gone'}")
                signal(True)
        elif ch == 'm':
            relay = not relay
            say("Motor On" if relay else "Motor Off")
        elif ch == 'd':
            run[0] += 1
            args = read_args(fd, 1)
            if args is None:
                say("screwdrive rejected a malformed speed, stopping")
                args = [0]
            BOARD['dc'] = max(-100, min(100, args[0]))
            say(f"screwdrive {BOARD['dc']}%")
        elif ch in 'tpa':
            run[0] += 1
            args = read_args(fd, 3 if ch == 'a' else 2)
            if args is None or not 0 < args[-1] <= MAX_RUN_MS:
                say("screwdrive rejected a malformed run, stopping")
                BOARD['dc'] = 0
                say("screwdrive 0%")
                continue
            ms = args[-1]
            if ch == 't':
                BOARD['dc'] = max(-100, min(100, args[0]))
                say(f"screwdrive {BOARD['dc']}% for {ms} ms")
                later(ms / 1000, 0, "screwdrive run done")
            elif ch == 'p':
                BOARD['dc'] = max(-255, min(255, args[0]))
                say(f"screwdrive pwm {BOARD['dc']} for {ms} ms")
                later(ms / 1000, 0, "screwdrive run done")
            else:
                start, end = (max(-100, min(100, v)) for v in args[:2])
                BOARD['dc'] = start
                say(f"screwdrive ramp {start}% to {end}% over {ms} ms")
                later(ms / 1000, end, "screwdrive ramp done")     # and HOLDS end
        #  anything else (line endings, noise) is ignored, as on the board


#  =====   configs   =====
FULL = {
    'multitoolchanger.yaml': """
modules: [coupler, relay, screwdrive]
sequences:
  grab_and_spin:
    description: spans modules, so it lives in the main file
    steps: [hold, run 40 0.2, release]
  grip_then_spin:
    steps: [hold, run 40 5]
  tool_then_spin: [sequence coupler_cycle, sequence screwdrive_good]
  nested_grip: [stop, sequence grip_then_spin, stop]
""",
    'screwdrive.yaml': """
settings:
  max_rpm: 400
sequences:
  screwdrive_good:
    description: spin, ramp, stop
    steps: [rpm 200 0.2, wait 0.1, ramp 0 30 0.2, stop]
  screwdrive_list_form: [stop]
  screwdrive_slow: [run 50 5, stop]
  screwdrive_broken: [spin 3, rpm 900 1, run 40, sequence coupler_cycle, hold, stop]
  screwdrive_twice: [sequence screwdrive_good, sequence screwdrive_list_form]
  screwdrive_deep: [stop, sequence screwdrive_twice, sequence screwdrive_slow]
  screwdrive_loop_a: [stop, sequence screwdrive_loop_b]
  screwdrive_loop_b: [sequence screwdrive_loop_a]
  screwdrive_self: [sequence screwdrive_self]
  screwdrive_into_loop: [sequence screwdrive_loop_a]
  screwdrive_nests_broken: [stop, sequence screwdrive_broken]
  screwdrive_nests_missing: [sequence screwdrive_nope]
  unprefixed: [stop]
""",
    'coupler.yaml': """
sequences:
  coupler_cycle: [hold, wait 0.1, release]
""",
}

ONLY_SCREWDRIVE = {
    'multitoolchanger.yaml': "modules: [screwdrive]\n",
}


def write_config(files, tmp):
    os.makedirs(tmp, exist_ok=True)
    for name, text in files.items():
        (pathlib.Path(tmp) / name).write_text(text)
    return os.path.join(tmp, 'multitoolchanger.yaml')


def expect_error(fn, *args, contains=''):
    try:
        fn(*args)
    except ToolChangerError as exc:
        assert contains in str(exc), (contains, str(exc))
        return exc
    raise AssertionError(f'{getattr(fn, "__name__", fn)}{args!r} should have raised')


class FakeWatcher:
    """KeyWatcher stand-in: 'presses q' once `after` seconds have passed since it was entered."""
    after = None
    active = True

    def __enter__(self):
        self.t0 = time.time()
        return self

    def pressed(self):
        return self.after is not None and time.time() - self.t0 >= self.after

    def __exit__(self, *exc):
        return False


#  =====   checks   =====
def check_docs_match_firmware():
    """Fail if the servo angles quoted in the coupler module have drifted from coupler.cpp."""
    firmware = (FIRMWARE / 'coupler.cpp').read_text()
    lock = int(re.search(r'const int lockAngle\s*=\s*(\d+)', firmware).group(1))
    nolock = int(re.search(r'const int noLockAngle\s*=\s*(\d+)', firmware).group(1))
    module = (HERE / 'mtc' / 'modules' / 'coupler.py').read_text()
    assert MAX_RUN_S * 1000 == MAX_RUN_MS, 'MAX_RUN_S disagrees with screwdrive.cpp'
    for expected, pattern in ((nolock, r'0 = unlocked \(servo (\d+) deg\)'),
                              (lock, r'>0 = locked \(servo (\d+) deg\)'),
                              (lock, r'lockAngle, (\d+) deg'),
                              (nolock, r'noLockAngle, (\d+) deg')):
        found = re.search(pattern, module)
        assert found, f"coupler.py: nothing matched {pattern!r} -- fix this check"
        assert int(found.group(1)) == expected, (
            f"coupler.py says {found.group(1)} deg where coupler.cpp says {expected}")
    print(f"angles agree: locked {lock} deg, unlocked {nolock} deg")


def check_config_loading(tmp):
    """Which modules load, what is fatal, and what only marks a sequence."""
    cfg = load_config(write_config(FULL, os.path.join(tmp, 'full')))
    assert list(cfg.modules) == ['general', 'coupler', 'relay', 'screwdrive']
    assert cfg.settings['screwdrive'].max_rpm == 400
    seqs = cfg.sequences
    for good in ('screwdrive_good', 'screwdrive_list_form', 'coupler_cycle', 'grab_and_spin'):
        assert not seqs[good].errors, (good, seqs[good].errors)
    assert seqs['coupler_cycle'].source == 'coupler' and seqs['grab_and_spin'].source == 'main'
    errors = '\n'.join(seqs['screwdrive_broken'].errors)
    for problem in ("unknown command 'spin'", 'RPM 900 is out of range -400..400',
                    'usage: run PCT SECONDS', 'coupler_cycle is a coupler sequence',
                    'hold is a coupler command'):
        assert problem in errors, (problem, errors)
    assert len(seqs['screwdrive_broken'].errors) == 5, 'the last step (stop) is fine'

    #  nesting: allowed, two levels deep, and across modules only from the main file
    for good in ('screwdrive_twice', 'screwdrive_deep', 'tool_then_spin', 'nested_grip'):
        assert not seqs[good].errors, (good, seqs[good].errors)
    #  loops are refused -- every member names the loop -- and so is whatever runs into one
    assert 'runs itself: screwdrive_loop_a -> screwdrive_loop_b -> screwdrive_loop_a' in \
        seqs['screwdrive_loop_a'].errors[0], seqs['screwdrive_loop_a'].errors
    assert 'runs itself' in seqs['screwdrive_loop_b'].errors[0]
    assert 'runs itself: screwdrive_self -> screwdrive_self' in seqs['screwdrive_self'].errors[0]
    assert seqs['screwdrive_into_loop'].errors == [
        'step 1 (sequence screwdrive_loop_a): screwdrive_loop_a cannot run '
        '(`sequence screwdrive_loop_a` says why)'], seqs['screwdrive_into_loop'].errors
    #  a broken or missing nested sequence makes its caller unable to run too
    assert 'step 2 (sequence screwdrive_broken): screwdrive_broken cannot run' in \
        seqs['screwdrive_nests_broken'].errors[0]
    assert "no sequence called 'screwdrive_nope'" in seqs['screwdrive_nests_missing'].errors[0]
    assert 'must be named screwdrive_' in seqs['unprefixed'].errors[0]

    #  only what is listed exists
    small = load_config(write_config(ONLY_SCREWDRIVE, os.path.join(tmp, 'small')))
    assert list(small.modules) == ['general', 'screwdrive']
    assert 'hold' not in small.commands and 'motor' not in small.commands
    assert small.settings['screwdrive'].max_rpm == 500, 'no screwdrive.yaml: the default'
    text = render_help(small)
    assert '  coupler ' not in text and '  hold ' not in text and '  screwdrive ' in text
    assert complete_line('h', small) == ['help ']

    #  fatal: these would otherwise run with the wrong modules or the wrong numbers
    for files, contains in (
            ({'multitoolchanger.yaml': 'modules: [screwdrive, warpdrive]'},
             "no module called 'warpdrive'"),
            ({'multitoolchanger.yaml': 'modules: [general]'}, 'general is always loaded'),
            ({'multitoolchanger.yaml': 'modules: [relay, relay]'}, 'listed twice'),
            ({'multitoolchanger.yaml': 'modulez: [relay]'}, 'unknown key'),
            ({'multitoolchanger.yaml': 'modules: [screwdrive]',
              'screwdrive.yaml': 'settings: {max_rmp: 300}'}, 'unknown setting(s) max_rmp'),
            ({'multitoolchanger.yaml': 'modules: [screwdrive]',
              'screwdrive.yaml': 'settings: {max_rpm: 0}'}, 'max_rpm must be a number >= 1'),
    ):
        d = os.path.join(tmp, f'bad{len(os.listdir(tmp))}')
        expect_error(load_config, write_config(files, d), contains=contains)
    expect_error(load_config, os.path.join(tmp, 'nowhere.yaml'), contains='no config at')
    print('config loading checks passed')
    return cfg


def check_session(tc, cfg, tool_present):
    """The command layer: sequences, q, help and completion, against the fake board."""
    s = Session(cfg, tc, watcher=FakeWatcher)

    #  a broken sequence is refused while parsing, so nothing at all runs
    BOARD['dc'] = 0
    expect_error(s.execute, ['sequence', 'screwdrive_broken'], contains='cannot run')

    #  good ones run start to finish; rpm scales from THIS config's max_rpm
    assert s.execute(['sequence', 'screwdrive_good']) is True and BOARD['dc'] == 0
    assert s.execute(['sequence', 'coupler_cycle']) is True
    assert s.execute(['sequence', 'grab_and_spin']) is True
    #  nested ones run in full, the progress lines showing the path
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert s.execute(['sequence', 'tool_then_spin']) is True
    text = out.getvalue()
    for line in ('[tool_then_spin 1/2] sequence coupler_cycle',
                 '  [tool_then_spin 1/2 > coupler_cycle 3/3] release',
                 '[tool_then_spin 2/2] sequence screwdrive_good',
                 '  [tool_then_spin 2/2 > screwdrive_good 4/4] stop',
                 '  sequence screwdrive_good: done', 'sequence tool_then_spin: done'):
        assert line in text, (line, text)
    assert s._running == []

    #  a failing step aborts the rest, and the screwdrive is stopped
    tool_present[0] = False
    tc.coupler.release()
    expect_error(s.execute, ['sequence', 'grip_then_spin'],
                 contains='aborted at grip_then_spin 1/2')
    assert BOARD['dc'] == 0, 'the run step never started'
    #  ... also deep inside a nested one: the whole path, reported once, nothing after it runs
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exc = expect_error(s.execute, ['sequence', 'nested_grip'],
                           contains='sequence nested_grip aborted at nested_grip 2/3 > '
                                    'grip_then_spin 1/2 (hold)')
    assert str(exc).count('aborted') == 1, str(exc)
    assert '[nested_grip 3/3]' not in out.getvalue(), 'the step after the failure never runs'
    assert s._running == [], 'the nesting is unwound after a failure'
    tool_present[0] = True
    tc.coupler.release()

    #  q mid-run stops the motor at once, and the session carries on as normal
    FakeWatcher.after = 0.3
    t0 = time.time()
    exc = expect_error(s.execute, ['run', '60', '5'], contains='screwdrive stopped')
    assert isinstance(exc, Interrupted)
    assert time.time() - t0 < 1.5, 'q must not wait for the 5 s run to finish'
    assert BOARD['dc'] == 0, 'q stops the motor'
    assert tc.board.poll_hook is None, 'the q hook is removed after the command'
    FakeWatcher.after = None
    assert s.execute(['status']) is True, 'normal operation resumes after q'

    #  ... and mid-ramp, and inside a sequence, which names the step it stopped at
    FakeWatcher.after = 0.3
    expect_error(s.execute, ['ramp', '0', '80', '5'], contains='interrupted')
    assert BOARD['dc'] == 0
    exc = expect_error(s.execute, ['sequence', 'screwdrive_slow'],
                       contains='at screwdrive_slow 1/2: run 50 5')
    assert isinstance(exc, Interrupted) and BOARD['dc'] == 0
    #  q two levels down ends every sequence around it
    exc = expect_error(s.execute, ['sequence', 'screwdrive_deep'],
                       contains='sequence screwdrive_deep: interrupted')
    assert isinstance(exc, Interrupted) and BOARD['dc'] == 0
    assert re.search(r'\(at screwdrive_deep [23]/3 > screwdrive_\w+ \d/\d', str(exc)), str(exc)
    assert s._running == []
    FakeWatcher.after = 0.2
    expect_error(s.execute, ['wait', '5'], contains='interrupted')
    FakeWatcher.after = None

    #  help: the general commands and the modules; each module's commands; the sequences
    def listed(usage, text):
        return any(ln.startswith(f'  {usage} ') or ln == f'  {usage}' for ln in text.splitlines())

    text = render_help(cfg)
    for name, cmd in cfg.commands.items():
        assert listed(cmd.usage, text) == (cmd.module == 'general'), name
    for name in cfg.modules:
        assert name == 'general' or f'  {name} ' in text, name
    assert '3 commands' not in text and '6 commands, 13 sequences' in text, text
    assert 'sequences from' not in text
    for name in cfg.modules:
        part = render_help(cfg, name)
        for cmd in cfg.commands.values():
            assert listed(cmd.usage, part) == (cmd.module == name), (name, cmd.name)
        assert 'sequences from' not in part
    assert 'help coupler sequence' in render_help(cfg, 'coupler')
    every = render_help(cfg, 'sequence')
    assert all(n in every for n in cfg.sequences) and 'CANNOT RUN' in every
    assert 'multitoolchanger.yaml' in every, 'main-file sequences are listed too'
    mine = render_help(cfg, 'coupler', 'sequence')
    assert 'coupler_cycle' in mine and 'screwdrive_good' not in mine
    assert 'no sequences' in render_help(cfg, 'relay', 'sequence')
    expect_error(render_help, cfg, 'hold', 'sequence', contains='needs a module name')
    expect_error(s.execute, ['help', 'coupler', 'seqs'], contains='only thing after a module')
    assert '-400..400' in render_help(cfg, 'rpm')

    #  completion: commands first, then whatever the argument's kind offers
    c = complete_line
    assert c('', cfg) == sorted(n + ' ' for n in cfg.commands)
    assert c('seq', cfg) == ['sequence ']
    assert c('sequence coup', cfg) == ['coupler_cycle ']
    assert c('help s', cfg) == ['screwdrive ', 'sequence ', 'status ', 'stop ']
    assert c('help coupler s', cfg) == ['sequence ']
    assert c('rpm ', cfg) == [] and c('ramp 0 ', cfg) == [] and c('nope ', cfg) == []
    print('session checks passed')


def fake_port(tool_present=None, **board_kwargs):
    """A fresh simulated board on its own pty; returns the device path to open."""
    master, slave = pty.openpty()
    threading.Thread(target=board, args=(master, tool_present or [True]), kwargs=board_kwargs,
                     daemon=True).start()
    return os.ttyname(slave)


def open_quietly(*args, **kwargs):
    """ToolChanger(...), returning it and whatever it warned about on stderr."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        tc = ToolChanger(*args, settle=0.1, timeout=3, **kwargs)
    return tc, err.getvalue()


def check_detection(cfg):
    """The board says which modules it was built with, and that is what loads."""
    main_cpp = (FIRMWARE / 'main.cpp').read_text()
    assert int(re.search(r'const int PROTOCOL = (\d+);', main_cpp).group(1)) == PROTOCOL, \
        'PROTOCOL in mtc/config.py disagrees with firmware/main.cpp'

    #  a screwdrive-only board: no coupler commands, no coupler sequences
    tc, warned = open_quietly(fake_port(modules=['screwdrive']), config_path=cfg.path)
    try:
        assert tc.source == 'detected' and not warned, warned
        assert list(tc.config.modules) == ['general', 'screwdrive']
        assert 'hold' not in tc.config.commands and not hasattr(tc.devices, 'coupler')
        assert 'coupler_cycle' not in tc.config.sequences
        assert 'hold' in ' '.join(tc.config.sequences['grab_and_spin'].errors), \
            'a main sequence using an absent module cannot run'
        assert complete_line('ho', tc.config) == []
        assert 'coupler --' not in render_help(tc.config)
        assert Session(tc.config, tc).execute(['sequence', 'screwdrive_good']) is True
    finally:
        tc.close()

    #  the prompt: the flashed name, or mtc(modules) so unnamed boards still differ
    import multitoolchanger
    for kwargs, want in ((dict(modules=['relay', 'screwdrive']), 'mtc(relay,screwdrive)'),
                         (dict(modules=['screwdrive'], name='cleat'), 'cleat')):
        tc, warned = open_quietly(fake_port(**kwargs), config_path=cfg.path)
        try:
            got = multitoolchanger.prompt_name(Session(tc.config, tc))
            assert got == want and not warned, (got, want, warned)
        finally:
            tc.close()
    tc, _ = open_quietly(fake_port(modules=['relay']), config_path=cfg.path, detect=False)
    try:
        assert multitoolchanger.prompt_name(Session(tc.config, tc)).startswith('mtc(coupler'), \
            'without detect, the modules shown are the ones loaded'
    finally:
        tc.close()
    tc, _ = open_quietly(fake_port(modules=['relay'], name='flashed'), config_path=cfg.path,
                         name='end_effector')    # as --port end_effector from couplers.yaml
    try:
        assert multitoolchanger.prompt_name(Session(tc.config, tc)) == 'end_effector', \
            'a couplers.yaml name picked with --port wins over the flashed name'
    finally:
        tc.close()

    #  --no-detect: the config's list, with a warning naming the difference
    tc, warned = open_quietly(fake_port(modules=['screwdrive', 'relay']), cfg, detect=False)
    try:
        assert tc.source == 'config' and 'coupler' in tc.config.modules
        assert 'loads coupler but' in warned and 'time out' in warned, warned
    finally:
        tc.close()
    tc, warned = open_quietly(fake_port(modules=['screwdrive']),
                              load_config(cfg.path, modules=['screwdrive', 'relay']),
                              detect=False)
    tc.close()
    assert 'loads relay but' in warned, warned
    tc, warned = open_quietly(fake_port(), load_config(cfg.path, modules=['screwdrive']),
                              detect=False)
    tc.close()
    assert 'also has coupler, relay' in warned, warned

    #  firmware too old to say: the config list, with a warning, detect or not
    tc, warned = open_quietly(fake_port(modules=None), config_path=cfg.path)
    tc.close()
    assert tc.source == 'config' and tc.detected is None and 'Reflash' in warned, warned
    assert list(tc.config.modules) == ['general'] + ALL

    #  a module the board has but this script does not know: skipped, said so
    tc, warned = open_quietly(fake_port(modules=['screwdrive', 'warpdrive']), config_path=cfg.path)
    tc.close()
    assert list(tc.config.modules) == ['general', 'screwdrive'] and 'warpdrive' in warned

    #  another protocol: refused outright, and the port is closed again
    port = fake_port(proto=PROTOCOL - 1)
    expect_error(lambda: open_quietly(port, config_path=cfg.path),
                 contains=f'protocol {PROTOCOL - 1}')
    print('detection checks passed')


def check_control_law():
    """Build and run the firmware's control law against the simulated motor (firmware/test/)."""
    import shutil
    import subprocess
    if not shutil.which('g++'):
        print('control law simulation SKIPPED: no g++')
        return
    test = FIRMWARE / 'test' / 't74_control_test.cpp'
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, 't74_test')
        subprocess.run(['g++', '-O2', '-std=c++11', '-Wall', '-Wextra', '-Werror', '-o', exe,
                        str(test)], check=True)
        run = subprocess.run([exe], capture_output=True, text=True)
    assert run.returncode == 0, 'the control law simulation failed:\n' + run.stdout
    print('control law simulation passed (firmware/test/t74_control_test.cpp)')


def check_pins(tmp):
    """build_flash.sh's pin check (mtc/pins.py), without compiling anything."""
    from mtc.pins import check, defines, parse_pin

    def plan(files, modules):
        d = os.path.join(tmp, f'pins{len(os.listdir(tmp))}')
        cfg = load_config(write_config(dict({'multitoolchanger.yaml': 'modules: []'}, **files), d),
                          modules=modules)
        return check(cfg, modules), defines(cfg, modules)

    #  the defaults are the wiring the firmware always had
    problems, flags = plan({}, ['coupler', 'relay', 'screwdrive'])
    assert not problems, problems
    assert flags == ['-DPIN_COUPLER_SERVO=5', '-DPIN_COUPLER_SENSOR=17', '-DPIN_COUPLER_LED=13',
                     '-DPIN_RELAY_K1=3', '-DPIN_SCREWDRIVE_PWM=6', '-DPIN_SCREWDRIVE_DIR=7'], flags
    problems, flags = plan({}, ['t74'])
    assert not problems and '-DPIN_T74_ENC_X=4' in flags

    #  clashes, timers, capabilities -- each named
    problems = '\n'.join(plan({}, ['t74', 'coupler'])[0])
    assert 'coupler needs timer1, which t74 already has' in problems
    assert 'D5 is wanted by t74.rpwm and coupler.servo' in problems
    problems = plan({'screwdrive.yaml': 'pins: {pwm: 11, dir: 12}',
                     't74.yaml': 'pins: {enc_x: A0}'}, ['t74', 'screwdrive'])[0]
    assert not problems, 'repinned, the t74 and the screwdrive share a board'
    problems = '\n'.join(plan({'screwdrive.yaml': 'pins: {pwm: 9, dir: 12}'},
                              ['t74', 'screwdrive'])[0])
    assert 'runs on timer1 -- and t74 has taken timer1 over' in problems
    problems = '\n'.join(plan({'screwdrive.yaml': 'pins: {pwm: 7, dir: 0}'}, ['screwdrive'])[0])
    assert 'D7 has none' in problems and 'D0 is the serial link' in problems
    problems = '\n'.join(plan({'t74.yaml': 'pins: {enc_a: 4, enc_x: 2}'}, ['t74'])[0])
    assert 'needs a hardware interrupt, and D4 has none' in problems
    problems = '\n'.join(plan({'coupler.yaml': 'pins: {sensor: 8}'}, ['coupler'])[0])
    assert 'needs an analog input, and D8 is not one' in problems
    #  two pin-change pins on one port would need the same interrupt vector twice
    assert not plan({'t74.yaml': 'pins: {enc_x: A0}'}, ['t74'])[0]

    #  the config itself: unknown roles and pins the board does not have are fatal
    expect_error(plan, {'relay.yaml': 'pins: {k2: 4}'}, ['relay'], contains='unknown pin(s) k2')
    expect_error(plan, {'relay.yaml': 'pins: {k1: A7}'}, ['relay'], contains='not an Uno pin')
    assert [parse_pin(v, '') for v in (6, '6', 'D6', 'a3', 'A0')] == [6, 6, 6, 17, 14]
    print('pin checks passed')


def check_t74(tmp):
    """The t74 module against a fake board whose motor model is known."""
    #  the gains really do put all three poles at -w: compare the characteristic polynomial
    K, tau, w = -120.0, 0.06, 15.0
    kp, ki, kd, note = pid_gains(K, tau, w)
    got = [tau, 1 + K * kd, K * kp, K * ki]
    want = [tau, 3 * tau * w, 3 * tau * w ** 2, tau * w ** 3]
    assert all(abs(a - b) < 1e-9 * max(1, abs(b)) for a, b in zip(got, want)) and not note
    assert 'clamped' in pid_gains(K, tau, 1.0)[3], 'w below 1/(3 tau) is called out'

    d = os.path.join(tmp, 't74cfg')
    write_config({'multitoolchanger.yaml': 'modules: [t74]\n',
                  't74.yaml': 'settings: {max_speed_dps: 360, max_accel_dps2: 3600}\n'
                              'sequences:\n  t74_square: [t74_move 90, wait 0.1, t74_move -90]\n'},
                 d)
    with open(os.path.join(d, 't74_calibration.json'), 'w') as fh:
        fh.write('{"turn_counts": -40000}\n')

    tc, warned = open_quietly(fake_port(modules=['t74']), config_path=os.path.join(d,
                              'multitoolchanger.yaml'))
    fake = T74S[-1]
    try:
        assert tc.source == 'detected' and list(tc.config.modules) == ['general', 't74'], warned
        dev = tc.t74
        assert fake.limits and fake.limits[4] < 0, 'limits sent; homes forward = counts down'
        assert fake.gains is None, 'no model yet, so no gains sent'
        expect_error(dev.move, 90, contains='t74_identify')

        #  identification recovers the true model, and the gains it sends match pid_gains()
        K, tau, friction = dev.identify(60, 120, 1.0)
        assert abs(K / FakeT74.K - 1) < 0.03, K
        assert abs(tau / FakeT74.TAU - 1) < 0.10, tau
        assert abs(friction - FakeT74.FRICTION) < 2.0, friction
        sent = fake.gains
        kp, ki, kd, _ = pid_gains(K, tau, dev.bandwidth)
        want = (kp, ki, kd, K, tau)
        assert all(abs(a - b) < 1e-4 * max(1, abs(b)) for a, b in zip(sent, want)), (sent, want)
        saved = json.load(open(os.path.join(d, 't74_calibration.json')))
        assert saved['turn_counts'] == -40000 and abs(saved['model_k'] - K) < 1e-9

        #  moves report in degrees, relative moves add to the target, goto is from zero
        dev.zero()                              # identify turned it ~113 deg
        assert abs(dev.move(90) - 90) < 0.01
        assert abs(dev.move(-30) - 60) < 0.01
        assert abs(dev.goto(-45) + 45) < 0.01
        #  goto takes the shortest way round: through the zero, never more than half a turn
        assert abs(dev.goto(300) + 60) < 0.01          # -45 -> -60, not +345 forward
        assert abs(dev.goto(10) - 10) < 0.01           # -60 -> 10: forward 70 through zero
        assert abs(dev.goto(200) + 160) < 0.01         # 190 forward vs 170 back: back
        assert abs(dev.goto(-170) + 170) < 0.01        # 10 deg on: the same place, -170
        assert abs(dev.goto(540) + 180) < 0.01         # whole turns ignored: 10 back to -180
        dev.zero()
        assert abs(dev.move(10) - 10) < 0.01
        dev.home()
        f = dev.report()
        assert f['mode'] == 'hold' and f['pos'] == '0'

        #  hold after move: on by default; off, the board releases once settled, and a
        #  relative move still counts from the last target
        assert dev.hold is True and fake.limits[5] == 1
        assert dev.set_hold() is False and fake.limits[5] == 0, 'no argument toggles'
        before = dev.report()['pos']
        assert abs(dev.move(5) - (int(before) * 360 / -40000 + 5)) < 0.01
        assert fake.mode == 'off', 'released after the move'
        assert abs(dev.move(5) - (int(before) * 360 / -40000 + 10)) < 0.01, 'from the last goal'
        assert dev.set_hold(True) is True and fake.limits[5] == 1
        Session(tc.config, tc).execute(['t74_hold', 'off'])
        assert dev.hold is False
        Session(tc.config, tc).execute(['t74_hold', 'on'])
        assert dev.hold is True
        expect_error(Session(tc.config, tc).execute, ['t74_hold', 'maybe'], contains='on or off')
        assert complete_line('t74_hold o', tc.config) == ['off ', 'on ']

        #  a fault mid-move stops it and is an error, not a success
        fake.fault_next_move = True
        expect_error(dev.move, 45, contains='FAULT following error')

        #  a reconnect re-sends limits AND the gains: the board forgot them in the reset
        s = Session(tc.config, tc, watcher=FakeWatcher)
        assert s.execute(['sequence', 't74_square']) is True

        #  q halts -- HOLDS where it is -- rather than releasing an unbalanced load
        FakeWatcher.after = 0.05
        del fake.commands[:]
        expect_error(s.execute, ['t74_move', '3000'], contains='t74 stopped')
        assert 'S' in fake.commands and 'X' not in fake.commands, fake.commands
        FakeWatcher.after = None
        dev.stop()
        assert fake.mode == 'off'

        #  Ctrl-C and leaving let go: the motor is released, not left holding
        dev.goto(0)
        assert fake.mode == 'hold'
        assert tc.let_go() == ['t74'] and fake.mode == 'off'
        dev.goto(10)
        assert fake.mode == 'hold'
    finally:
        tc.close()
    assert fake.mode == 'off' and fake.commands[-1] == 'X', 'closing releases the motor'
    tc.close()                                  # twice is harmless: the port is already shut

    tc, _ = open_quietly(fake_port(modules=['t74']),
                         config_path=os.path.join(d, 'multitoolchanger.yaml'))
    tc.close()
    assert T74S[-1].gains is not None, 'gains re-sent on connect from the saved model'
    print('t74 checks passed')


def main():
    check_docs_match_firmware()
    check_control_law()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = check_config_loading(tmp)
        check_detection(cfg)
        check_pins(tmp)
        check_t74(tmp)
        check_board(cfg)

    print('\nALL DRIVER ASSERTIONS PASSED')


def check_board(cfg):

    tool_present = [True]  # what the proximity sensor "sees"; flip it mid-test
    tc = ToolChanger(fake_port(tool_present), cfg, settle=0.1, timeout=3, verbose=True)
    assert tc.source == 'detected' and tc.detected == ALL
    cfg = tc.config
    sd, cp = tc.screwdrive, tc.coupler
    try:
        #  the board booted with a tool present, so it is already locked
        assert cp.hold() is True
        assert cp.status() is True
        assert cp.probe() == RAW_TOOL

        assert tc.relay.toggle() is True, 'first toggle turns the relay on'
        assert tc.relay.toggle() is False

        #  the screwdrive: what comes back is what the BOARD says it is doing
        assert sd.drive(60) == 60
        assert sd.drive('-40') == -40, 'a reversal, from a string as the CLI passes it'
        assert sd.stop() == 0
        for bad in (150, -101, '6.5', 'fast', None, True):
            expect_error(sd.drive, bad)

        #  a garbled number on the wire must stop the motor AND be drained, not leak digits
        #  through as coupler commands -- "d12x3" leaving a '3' behind would be a hold
        sd.drive(30)
        final, lines = tc.board.exchange('d12x3\n', lambda ln: ln.startswith('screwdrive ') and
                                         ln.endswith('%'))
        assert final == 'screwdrive 0%' and 'rejected' in lines[0], lines
        assert cp.status() is True, 'no stray coupler command ran'

        #  timed runs: the board times them and says when it has stopped
        t0 = time.time()
        assert sd.run(50, 0.3) == 50
        assert time.time() - t0 >= 0.3, 'run() returns only once the board says it is done'
        assert sd.rpm(200, '0.2') == 200, 'rpm from strings, as the CLI passes them'
        assert sd.rpm(-400, 0.1) == -400
        assert sd.pwm(128, 0.1) == 128 and sd.pwm(-255, 0.1) == -255
        assert (duty_for_rpm(250, 500), duty_for_rpm(-500, 500),
                duty_for_rpm(200, 400), duty_for_rpm(500, 1000)) == (128, -255, 128, 128)
        for fn, bad in ((sd.rpm, (401, 1)), (sd.rpm, (100, 0)), (sd.rpm, (100, 3601)),
                        (sd.rpm, (100, 'soon')), (sd.pwm, (256, 1)), (sd.run, ('2.5', 1))):
            expect_error(fn, *bad)

        #  ramps: the board ramps, then HOLDS the end speed until told otherwise
        t0 = time.time()
        assert sd.ramp(0, 50, 0.3) == 50
        assert time.time() - t0 >= 0.3 and BOARD['dc'] == 50, 'ramp holds its end speed'
        assert sd.ramp(50, -50, 0.2) == -50 and BOARD['dc'] == -50, 'through zero'
        assert sd.ramp(-50, 0, 0.1) == 0 and BOARD['dc'] == 0, 'ramp to 0 finishes stopped'
        for bad in ((0, 101, 1), (0, 50, 0), (0, 'x', 1)):
            expect_error(sd.ramp, *bad)
        assert sd.run(30, 5, wait=False) == 30
        assert sd.stop() == 0, 'a new command replaces a timed run in progress'
        time.sleep(0.1)

        #  the coupler still behaves as it did on the single-device board
        assert cp.release() is True, 'release with the tool still present must confirm'
        tool_present[0] = False
        assert cp.hold() is False, 'hold with nothing to grip is an emergency stop'
        assert cp.release() is True
        assert cp.bypass() is True
        assert cp.hold() is True, 'bypassed, hold confirms without the sensor'
        assert cp.release() is True
        assert cp.bypass() is False

        tool_present[0] = True
        cp.release()
        check_session(tc, cfg, tool_present)
    finally:
        tc.close()


if __name__ == '__main__':
    main()
