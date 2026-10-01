#!/usr/bin/env python3
"""Drive the T74 motor (IBT-2 driver, Arduino Uno, firmware/main.cpp) from the terminal.

With an AMT10E2-V encoder on the drive shaft, moves are closed loop: the board stops the motor on
the encoder count. Without one, moves are timed estimates.

    ./t74.py                  interactive session: c, m, s, angles, e, h, g, q   (recommended)
    ./t74.py calibrate        time one full tile turn and count its encoder counts; saves both
    ./t74.py move 60          turn +60 deg (-30 turns back)
    ./t74.py home             run forward to the encoder index pulse and call it 0 deg
    ./t74.py goto 90          home, then go to 90 deg from the index (needs the encoder)
    ./t74.py watch            print the encoder 5x a second; turn the shaft by hand to test wiring
    ./t74.py counts 20480     set the counts per tile turn by hand (4 x PPR if 1:1 on the tile)
    ./t74.py speed 80         set the run speed (PWM 1..255); clears the turn time, so recalibrate
    ./t74.py status           show the saved calibration and speed, and what the board reports
    ./t74.py list             show the serial ports that look like an Arduino

Options: --port /dev/ttyACM0   --cal-file PATH   --baud 115200

OPENING THE PORT RESETS THE UNO, which erases the turn time, counts, speed and home from its
memory. This script saves the first three to a file and sends them back (P<pwm>, T<ms>, K<counts>)
every time it connects, so you only recalibrate when you actually need to, not on every run. Home
cannot be saved (the count restarts at 0 on reset), which is why `goto` homes first.

Ctrl-C always sends S (stop) before the script exits.

Wire protocol (firmware/main.cpp), 115200 baud, one command per line:

    C         start the calibration turn (motor runs forward)
    M         the mark has come back round: stop and store the motor-on time and counts
    S         stop now, motor disabled
    T<ms>     load a saved turn time, e.g. T4200
    K<counts> load saved encoder counts per tile turn, signed, e.g. K20480
    P<pwm>    set the run speed 1..255, e.g. P80 (a new speed clears the turn time)
    H         home: run forward to the encoder index pulse, that is 0 deg
    G<deg>    go to an absolute angle from home, e.g. G90
    E         print the encoder count and angle
    ?         print the turn time, counts and mode
    <deg>     relative move of 1..360 deg, negative = reverse, e.g. 60 or -30
"""

import argparse
import json
import os
import queue
import re
import sys
import threading
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is missing. Install it with:  pip install pyserial")


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CAL_FILE = os.path.join(HERE, 't74_calibration.json')
BANNER = ('T74 READY', 'TIMER MODE')                       #  printed by setup(), i.e. once per board reset
BOOT_TIMEOUT = 4.0                          #  bootloader + setup() after the port-open reset

CAL_RE = re.compile(r'motor-on time:\s*(\d+)\s*ms(?: at PWM\s*(\d+))?'
                    r'(?:, encoder:\s*(-?\d+) counts)?')
PWM_SET_RE = re.compile(r'Run PWM set:\s*(\d+)')
TURN_SET_RE = re.compile(r'Turn time set:\s*(\d+)\s*ms')
COUNTS_SET_RE = re.compile(r'Turn counts set:\s*(-?\d+)')
MOVE_RE = re.compile(r'power on for\s*(\d+)\s*ms')
MOVE_DONE = 'Timed move finished'
ENC_MOVE = 'Encoder move:'
ENC_DONE = 'Encoder move finished'
HOMED = 'Homed'
ENC_TIMEOUT = 60 + 2 + 3                    #  firmware move limit + settle + slack, seconds
STOPPED = 'STOPPED'
#  Replies that mean a command was refused. A move or T<ms> waiting on the board gives up on these.
REFUSALS = ('Already running', 'Calibrate first', 'Enter an angle', 'Move time out of range',
            'Turn time rejected', 'Type ', 'Command too long', 'Calibration rejected',
            'Calibration timed out', 'Send C first', 'PWM rejected', 'ENCODER FAULT',
            'Turn counts rejected', 'Goto needs')

ARDUINO_HINTS = ('arduino', 'ch340', 'ftdi', 'wch', 'usb serial', 'elegoo')


class T74Error(RuntimeError):
    pass


#  =====   calibration file   =====
def _load(path):
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(path, data):
    data['saved'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2)
        fh.write('\n')


def _load_int(path, key, lo, hi):
    try:
        value = int(_load(path)[key])
    except (KeyError, ValueError, TypeError):
        return None
    return value if lo <= value <= hi else None


def load_turn_ms(path):
    """The saved full-turn time in ms, or None if nothing usable is saved."""
    return _load_int(path, 'full_turn_ms', 500, 60000)


def load_turn_counts(path):
    """The saved signed encoder counts per tile turn, or None if no encoder was calibrated."""
    value = _load_int(path, 'turn_counts', -10000000, 10000000)
    return value if value is not None and abs(value) >= 100 else None


def load_pwm(path):
    """The saved run speed (PWM 1..255), or None to leave the firmware default."""
    return _load_int(path, 'run_pwm', 1, 255)


def save_turn_ms(path, ms, pwm=None, counts=None):
    data = _load(path)
    data['full_turn_ms'] = ms
    if pwm is not None:
        data['run_pwm'] = pwm
    if counts is not None:
        data['turn_counts'] = counts
    _save(path, data)
    print(f'  saved {ms} ms' + (f' at PWM {pwm}' if pwm is not None else '')
          + (f', {counts} encoder counts' if counts is not None else '') + f' to {path}')


def save_turn_counts(path, counts):
    data = _load(path)
    if data.get('turn_counts') == counts:
        return
    data['turn_counts'] = counts
    _save(path, data)
    print(f'  saved {counts} encoder counts per turn to {path}')


def save_pwm(path, pwm):
    """Save the run speed. A different speed drops the turn time, which only holds at the speed
    it was measured at."""
    data = _load(path)
    if data.get('run_pwm') == pwm:
        return
    data['run_pwm'] = pwm
    if data.pop('full_turn_ms', None) is not None:
        if data.get('turn_counts'):
            print('  speed changed: saved turn time cleared. Encoder moves still work; the turn '
                  'time is only used without the encoder.')
        else:
            print('  speed changed: saved turn time cleared, recalibrate before moving.')
    _save(path, data)
    print(f'  saved PWM {pwm} to {path}')


#  =====   which port   =====
def candidates():
    found = []
    for p in list_ports.comports():
        text = f"{p.manufacturer or ''} {p.product or ''} {p.description or ''}".lower()
        if any(k in text for k in ARDUINO_HINTS) or p.serial_number:
            found.append(p)
    return sorted(found, key=lambda p: p.device)


def describe_ports():
    found = candidates()
    if not found:
        print('No boards found. Check `ls /dev/ttyACM* /dev/ttyUSB*` and the USB cable.')
        return
    for p in found:
        print(f'  {p.device}   serial {p.serial_number or "(none)"}   '
              f'{(p.product or p.description or "").strip()}')


def find_port(port):
    if port:
        return port
    found = candidates()
    if len(found) == 1:
        return found[0].device
    if not found:
        raise T74Error('no Arduino found. Plug it in, or pass --port /dev/ttyACM0.')
    raise T74Error('more than one board is plugged in, pass --port. Found: '
                   + ', '.join(p.device for p in found))


#  =====   the board   =====
class T74:
    """One serial connection to the board. A background thread reads every line the board
    prints, echoes it, and queues it for whoever is waiting on a reply."""

    def __init__(self, port=None, baud=115200, cal_file=DEFAULT_CAL_FILE, echo=True):
        self.cal_file = cal_file
        self.echo = echo
        self.lines = queue.Queue()
        self.port = find_port(port)
        try:
            self.ser = serial.Serial(self.port, baud, timeout=0.1)
        except serial.SerialException as exc:
            if not os.path.exists(self.port):
                raise T74Error(f'{self.port} does not exist. Run `./t74.py list` to see the '
                               f'ports that do.') from exc
            raise T74Error(f'cannot open {self.port}: {exc}. Is the Arduino IDE serial monitor '
                           f'still open?') from exc
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    #  ---- plumbing ----
    def _read_loop(self):
        buf = b''
        while not self._stop.is_set():
            try:
                chunk = self.ser.read(256)
            except (serial.SerialException, OSError):
                break
            buf += chunk
            while b'\n' in buf:
                raw, buf = buf.split(b'\n', 1)
                line = raw.decode(errors='replace').strip()
                if not line:
                    continue
                if self.echo:
                    print(f'  board: {line}')
                m = CAL_RE.search(line)
                if m:
                    save_turn_ms(self.cal_file, int(m.group(1)),
                                 int(m.group(2)) if m.group(2) else None,
                                 int(m.group(3)) if m.group(3) is not None else None)
                m = COUNTS_SET_RE.search(line)
                if m:
                    save_turn_counts(self.cal_file, int(m.group(1)))
                m = PWM_SET_RE.search(line)
                if m:
                    save_pwm(self.cal_file, int(m.group(1)))
                self.lines.put(line)

    def send(self, text):
        self.ser.write((text + '\n').encode())
        self.ser.flush()

    def wait_for(self, predicate, timeout):
        """The first line for which predicate(line) is true. Refusals raise T74Error."""
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise T74Error('no reply from the board in time.')
            try:
                line = self.lines.get(timeout=left)
            except queue.Empty:
                continue
            if predicate(line):
                return line
            if line.startswith(REFUSALS):
                raise T74Error(line)

    def drain(self):
        while not self.lines.empty():
            self.lines.get_nowait()

    #  ---- session ----
    def connect(self):
        """Wait for the reset banner, then reload the saved speed, encoder counts and turn time.
        Returns the turn time in ms, or None if none was saved."""
        print(f'  connecting to {self.port} (this resets the Uno)...')
        try:
            self.wait_for(lambda l: l.startswith(BANNER), BOOT_TIMEOUT)
        except T74Error:
            #  No reset on open (e.g. some USB adapters): the board is up already, carry on.
            print('  (no boot banner seen; assuming the board is already running)')
        pwm = load_pwm(self.cal_file)
        if pwm is not None:
            self.set_pwm(pwm)
        counts = load_turn_counts(self.cal_file)
        if counts is not None:
            self.set_counts(counts)
        ms = load_turn_ms(self.cal_file)
        if ms is None:
            if counts is None:
                print('  no saved calibration: calibrate before moving.')
            return None
        self.send(f'T{ms}')
        self.wait_for(lambda l: TURN_SET_RE.search(l), 2.0)
        return ms

    def set_pwm(self, pwm):
        """Set the run speed. The reader thread saves it (and clears a stale turn time)."""
        self.drain()
        self.send(f'P{pwm}')
        try:
            self.wait_for(lambda l: PWM_SET_RE.search(l), 2.0)
        except T74Error as exc:
            if 'P<pwm>' not in str(exc) and str(exc).startswith('Type C'):
                raise T74Error(f'the board on {self.port} has old firmware without speed '
                               f'control. Flash it: cd firmware && '
                               f'PORT={self.port} ./build_flash.sh upload') from exc
            raise

    def set_counts(self, counts):
        """Load encoder counts per tile turn. The reader thread saves them."""
        self.drain()
        self.send(f'K{counts}')
        try:
            self.wait_for(lambda l: COUNTS_SET_RE.search(l), 2.0)
        except T74Error as exc:
            if str(exc).startswith('Type C') and 'K<counts>' not in str(exc):
                raise T74Error(f'the board on {self.port} has old firmware without encoder '
                               f'support. Flash it: cd firmware && '
                               f'PORT={self.port} ./build_flash.sh upload') from exc
            raise

    def stop(self):
        try:
            self.send('S')
        except (serial.SerialException, OSError):
            pass

    def move(self, degrees):
        """Relative move. Blocks until the board says the move is finished."""
        self._run(f'{degrees:g}')

    def goto(self, degrees):
        """Absolute move, degrees from home. Needs the encoder."""
        self._run(f'G{degrees:g}')

    def _run(self, command):
        self.drain()
        self.send(command)
        line = self.wait_for(lambda l: MOVE_RE.search(l) or l.startswith((ENC_MOVE, ENC_DONE)),
                             2.0)
        if line.startswith(ENC_DONE):
            return line                         #  already there
        if line.startswith(ENC_MOVE):
            return self.wait_for(lambda l: l.startswith((ENC_DONE, STOPPED)), ENC_TIMEOUT)
        move_ms = int(MOVE_RE.search(line).group(1))
        return self.wait_for(lambda l: l.startswith((MOVE_DONE, STOPPED)), move_ms / 1000 + 3.0)

    def home(self):
        """Run forward to the encoder index pulse; that becomes 0 deg until the next reset."""
        self.drain()
        self.send('H')
        self.wait_for(lambda l: l.startswith('Homing'), 2.0)
        return self.wait_for(lambda l: l.startswith((HOMED, STOPPED)), ENC_TIMEOUT)

    def encoder(self):
        """The board's one-line encoder report."""
        self.send('E')
        return self.wait_for(lambda l: l.startswith('Encoder:'), 2.0)

    def close(self):
        self._stop.set()
        self._reader.join(timeout=1)
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        time.sleep(0.05)
        self.close()


#  =====   commands   =====
def cmd_calibrate(board):
    print('\nCALIBRATION: the motor turns forward at the run speed until you tell it to stop.')
    input('  1. Put a mark on the tile and line it up with a fixed reference. Press Enter to start. ')
    board.drain()
    board.send('C')
    board.wait_for(lambda l: l.startswith('Calibration running'), 2.0)
    input('  2. Press Enter the moment the mark comes back to the reference (one full turn). ')
    board.send('M')
    line = board.wait_for(lambda l: CAL_RE.search(l), 2.0)
    try:
        #  With the encoder, the board then measures how far the motor coasts. Let it finish:
        #  leaving now would send S and cut that short.
        board.wait_for(lambda l: l.startswith(('Coast after', 'No encoder')), 3.0)
    except T74Error:
        pass
    m = CAL_RE.search(line)
    ms = int(m.group(1))
    counts = int(m.group(3)) if m.group(3) is not None else 0
    print(f'\n  One full turn = {ms} ms of motor-on time.')
    if counts:
        print(f'  Encoder: {counts} counts per turn, so moves are closed loop. If the encoder is on '
              f'the tile shaft 1:1, the exact value is 4 x PPR (20480 at the factory 5120 PPR): '
              f'set it with `./t74.py counts {"-" if counts < 0 else ""}20480`.')
    else:
        print('  No encoder counts: moves will be timed estimates.')
    print('  Try `./t74.py move 90` and check the mark.')


def cmd_status(board, ms):
    pwm = load_pwm(board.cal_file)
    counts = load_turn_counts(board.cal_file)
    print(f'  saved turn time: {ms if ms else "none"}, '
          f'encoder counts per turn: {counts if counts else "none"}, '
          f'speed: {pwm if pwm else "firmware default"} ({board.cal_file})')
    board.send('?')
    board.wait_for(lambda l: l.startswith('Turn time:'), 2.0)
    board.encoder()


def cmd_watch(board):
    print('  Turn the shaft by hand. The count should change smoothly, "missed edges" should stay 0,'
          '\n  and "index pulses" should go up by one per shaft turn. Ctrl-C to stop.')
    board.echo = False
    last = None
    while True:
        line = board.encoder()
        if line != last:
            print(f'  {line}')
            last = line
        time.sleep(0.2)


INTERACTIVE_HELP = """
  c        start calibration (motor turns forward)
  m        the mark is back at the reference: stop and save the turn time
  <angle>  relative move, e.g. 60, 15.5, -30   (1..360, negative = reverse)
  p <pwm>  set the run speed 1..255, e.g. p 80  (clears the turn time: recalibrate)
  e        encoder count and angle
  h        home to the encoder index pulse (that is 0 deg); also `home`
  g <deg>  go to an absolute angle from home, e.g. g 90
  k <n>    set encoder counts per tile turn, e.g. k 20480
  s        STOP now
  ?        board status
  help     this list
  q        quit (stops the motor)
"""


def cmd_interactive(board):
    print(INTERACTIVE_HELP)
    while True:
        try:
            text = input('t74> ').strip()
        except EOFError:
            return
        if not text:
            continue
        low = text.lower()
        if low in ('q', 'quit', 'exit'):
            return
        if low == 'help':
            print(INTERACTIVE_HELP)
            continue
        if low == 'home':
            low = 'h'
        if low in ('c', 'm', 's', '?', 'e', 'h'):
            board.send(low.upper())
            continue
        if low[0] in 'pkg':
            parse, example = {'p': (int, 'p 80'), 'k': (int, 'k 20480'), 'g': (float, 'g 90')}[low[0]]
            try:
                value = parse(low[1:])
            except ValueError:
                print(f'  type {low[0]} and a number, e.g. {example}')
                continue
            board.send(f'{low[0].upper()}{value:g}' if parse is float else f'{low[0].upper()}{value}')
            time.sleep(0.2)
            continue
        try:
            float(text)
        except ValueError:
            print('  type c, m, s, e, h, g, p, k, ?, q, or an angle such as 60')
            continue
        board.send(text)
        #  Replies are printed by the reader thread; give them a moment before the next prompt.
        time.sleep(0.2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog='See README.md for the per-tile procedure.')
    ap.add_argument('command', nargs='?', default='interactive',
                    choices=('interactive', 'calibrate', 'move', 'home', 'goto', 'watch', 'counts',
                             'speed', 'status', 'list'))
    ap.add_argument('value', nargs='?', type=float,
                    help='degrees for `move` (negative = reverse) and `goto`, PWM 1..255 for '
                         '`speed`, counts per turn for `counts`')
    ap.add_argument('--port', help='serial port, e.g. /dev/ttyACM0 (auto if only one board)')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--cal-file', default=DEFAULT_CAL_FILE,
                    help=f'where the turn time is saved (default {DEFAULT_CAL_FILE})')
    args = ap.parse_args(argv)

    if args.command == 'list':
        describe_ports()
        return 0
    if args.command == 'move':
        if args.value is None:
            ap.error('move needs an angle, e.g. `./t74.py move 60` or `./t74.py move -30`')
        if not 1 <= abs(args.value) <= 360:
            ap.error('angle must be 1..360 deg, positive or negative')
        if load_turn_ms(args.cal_file) is None and load_turn_counts(args.cal_file) is None:
            ap.error('no saved calibration. Run `./t74.py calibrate` first.')
    if args.command == 'goto':
        if args.value is None:
            ap.error('goto needs an angle from home, e.g. `./t74.py goto 90`')
        if load_turn_counts(args.cal_file) is None:
            ap.error('goto needs the encoder: run `./t74.py calibrate` with it connected first.')
    if args.command == 'counts':
        if args.value is None or args.value != int(args.value) or not 100 <= abs(args.value) <= 1e7:
            ap.error('counts needs a whole number of counts per tile turn, e.g. '
                     '`./t74.py counts 20480` (negative if forward counts down)')
    if args.command == 'speed':
        if args.value is None or args.value != int(args.value) or not 1 <= args.value <= 255:
            ap.error('speed needs a whole PWM from 1 to 255, e.g. `./t74.py speed 80`')

    try:
        with T74(args.port, args.baud, args.cal_file) as board:
            ms = board.connect()
            if args.command == 'calibrate':
                cmd_calibrate(board)
            elif args.command == 'move':
                board.move(args.value)
            elif args.command == 'home':
                board.home()
            elif args.command == 'goto':
                #  Opening the port reset the board and its home, so find the index again first.
                board.home()
                board.goto(args.value)
            elif args.command == 'watch':
                cmd_watch(board)
            elif args.command == 'counts':
                board.set_counts(int(args.value))
            elif args.command == 'speed':
                board.set_pwm(int(args.value))
            elif args.command == 'status':
                cmd_status(board, ms)
            else:
                cmd_interactive(board)
    except KeyboardInterrupt:
        print('\n  Ctrl-C: motor stopped.')
        return 130
    except T74Error as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
