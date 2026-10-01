#!/usr/bin/env python3
"""A fake multi-toolchanger board on a pty, so the driver can be tested with no hardware.

    ./fake_board.py          run the driver against the simulation and assert the protocol

Opening the real port pulls DTR, resets the Arduino and swings the servo, so this exists to
exercise the driver without moving anything. It speaks exactly the lines firmware/ prints.
"""

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
from mtc.modules.screwdrive import duty_for_rpm  # noqa: E402
from mtc.session import complete_line, render_help  # noqa: E402

FIRMWARE = HERE / 'firmware'
THRESH = int(re.search(r'const int thresh\s*=\s*(\d+)',
                       (FIRMWARE / 'coupler.cpp').read_text()).group(1))
MAX_RUN_MS = int(re.search(r'const unsigned long maxRunMs\s*=\s*(\d+)',
                           (FIRMWARE / 'screwdrive.cpp').read_text()).group(1))
RAW_TOOL = 400        # what the sensor reads with a tool in front of it (below thresh)
RAW_EMPTY = 1023      # ... and with nothing there (railed, as measured on the cell)

BOARD = {'dc': 0}     # what the simulated screwdrive is doing, for the tests to inspect


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


def board(fd, tool_present):
    """Mimic firmware/: one ASCII byte in (plus numbers after the screwdrive's), lines out."""
    status = 1 if tool_present[0] else 0     # setup(): status = checkTool() ? 1 : 0
    relay = False
    bypassed = False
    run = [0]             # generation of the current timed run/ramp; bumped to cancel it

    #  delayed a beat: the driver opens the pty after this thread starts, and a banner written
    #  before it is listening is a banner it never sees
    time.sleep(0.2)
    os.write(fd, b"toolchanger ready\r\n")

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

        if ch == 's':
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
  screwdrive_broken: [spin 3, rpm 900 1, run 40, sequence screwdrive_good, hold, stop]
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
                    'usage: run PCT SECONDS', 'sequence cannot be used in a sequence',
                    'hold is a coupler command'):
        assert problem in errors, (problem, errors)
    assert len(seqs['screwdrive_broken'].errors) == 5, 'the last step (stop) is fine'
    assert 'must be named screwdrive_' in seqs['unprefixed'].errors[0]

    #  only what is listed exists
    small = load_config(write_config(ONLY_SCREWDRIVE, os.path.join(tmp, 'small')))
    assert list(small.modules) == ['general', 'screwdrive']
    assert 'hold' not in small.commands and 'motor' not in small.commands
    assert small.settings['screwdrive'].max_rpm == 500, 'no screwdrive.yaml: the default'
    text = render_help(small)
    assert 'coupler --' not in text and '  hold ' not in text and 'screwdrive --' in text
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

    #  a failing step aborts the rest, and the screwdrive is stopped
    tool_present[0] = False
    tc.coupler.release()
    expect_error(s.execute, ['sequence', 'grip_then_spin'], contains='aborted at step 1/2')
    assert BOARD['dc'] == 0, 'the run step never started'
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
    exc = expect_error(s.execute, ['sequence', 'screwdrive_slow'], contains='at step 1/2')
    assert isinstance(exc, Interrupted) and BOARD['dc'] == 0
    FakeWatcher.after = 0.2
    expect_error(s.execute, ['wait', '5'], contains='interrupted')
    FakeWatcher.after = None

    #  help: every loaded command, grouped under its module, and sequences by source file
    text = render_help(cfg)
    for name in cfg.modules:
        assert f'{name} -- ' in text, name
    for name, cmd in cfg.commands.items():
        assert cmd.usage in text and cmd.summary, name
    assert 'sequences from' in text and 'screwdrive.yaml' in text and 'CANNOT RUN' in text
    assert '-400..400' in render_help(cfg, 'rpm')
    assert 'hold' in render_help(cfg, 'coupler') and 'drive' not in render_help(cfg, 'coupler')
    assert 'coupler_cycle' in render_help(cfg, 'coupler')

    #  completion: commands first, then whatever the argument's kind offers
    c = complete_line
    assert c('', cfg) == sorted(n + ' ' for n in cfg.commands)
    assert c('seq', cfg) == ['sequence ']
    assert c('sequence coup', cfg) == ['coupler_cycle ']
    assert c('help s', cfg) == ['screwdrive ', 'sequence ', 'status ', 'stop ']
    assert c('rpm ', cfg) == [] and c('ramp 0 ', cfg) == [] and c('nope ', cfg) == []
    print('session checks passed')


def main():
    check_docs_match_firmware()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = check_config_loading(tmp)

    master, slave = pty.openpty()
    tool_present = [True]  # what the proximity sensor "sees"; flip it mid-test
    threading.Thread(target=board, args=(master, tool_present), daemon=True).start()

    tc = ToolChanger(os.ttyname(slave), cfg, settle=0.1, timeout=3, verbose=True)
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

    print('\nALL DRIVER ASSERTIONS PASSED')


if __name__ == '__main__':
    main()
