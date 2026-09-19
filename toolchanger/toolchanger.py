#!/usr/bin/env python3
"""Drive the Arduino toolchanger (main.cpp) over serial from the terminal.

    ./toolchanger.py hold        # lock the ball bearings onto the tool   (sends '1')
    ./toolchanger.py release     # unlock, let the tool go                (sends '0')
    ./toolchanger.py status      # ask whether a tool is actually there   (sends 's')
    ./toolchanger.py motor       # toggle the motor relay K1 on/off       (sends 'm')
    ./toolchanger.py monitor     # just watch whatever the board prints
    ./toolchanger.py             # interactive prompt (hold/release/status/motor/quit)

Options: --port /dev/ttyACM0   --baud 9600   --timeout 5   --verbose

The wire protocol is exactly what main.cpp implements -- one ASCII byte per command:

    '0'..'9'  changeStatus(n).  0 = unlocked (servo 15 deg), >0 = locked (servo 50 deg).
              Board may print "changed status from X to Y", then the confirmation line.
    's'       report status without changing it.
    'm'       toggleMotorPower() -> "Motor On" / "Motor Off".

and the board answers with one of:

    "<n> confirmed!"   the proximity sensor AGREES with the commanded status -- success.
    "emergency stop"   the sensor DISAGREES. Asked to hold but nothing is gripped, or
                       asked to release but something is still detected.

Timing notes that matter: opening the port pulls DTR and RESETS the Arduino, so we wait
for it to boot before sending (--settle). loop() has a 200 ms delay and reads ONE byte
per pass, changeServo() blocks 500 ms and every checkTool() averages 10 analog reads at
10 ms each -- so a hold/release round trip takes the better part of a second.

Import it instead of running it to use the same thing from your UR code:

    from toolchanger import ToolChanger
    with ToolChanger('/dev/ttyACM0') as tc:
        tc.hold()
"""

import argparse
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is missing. Install it with:  pip install pyserial")


LOCKED = '1'
UNLOCKED = '0'
QUERY = 's'
MOTOR = 'm'

CONFIRM_SUFFIX = 'confirmed!'
EMERGENCY = 'emergency stop'


class ToolChangerError(RuntimeError):
    pass


def find_port():
    """Pick the most likely Arduino port, preferring a stable /dev/serial/by-id path."""
    ports = list(list_ports.comports())
    if not ports:
        raise ToolChangerError(
            "No serial ports found. Plug the Arduino in and check `ls /dev/ttyACM* /dev/ttyUSB*`.")

    def score(p):
        text = f"{p.manufacturer or ''} {p.product or ''} {p.description or ''}".lower()
        return (any(k in text for k in ('arduino', 'ch340', 'ftdi', 'wch', 'usb serial')),
                p.device.startswith('/dev/ttyACM'))

    best = max(ports, key=score)
    return best.device


class ToolChanger:
    """Blocking control of the toolchanger. `hold()` locks, `release()` unlocks."""

    def __init__(self, port=None, baud=9600, timeout=5.0, settle=2.0, verbose=False):
        self.port = port or find_port()
        self.baud = baud
        self.timeout = timeout
        self.verbose = verbose
        # read timeout is per-readline; the overall budget is enforced in _exchange()
        self.ser = serial.Serial(self.port, baud, timeout=0.3)
        # Opening the port toggles DTR, which resets the board. Anything we send during
        # the bootloader window is lost, so wait it out and drop the boot noise.
        time.sleep(settle)
        self.ser.reset_input_buffer()

    # ---------------------------------------------------------------- raw io
    def _send(self, byte):
        if self.verbose:
            print(f"  -> {byte!r}", file=sys.stderr)
        self.ser.write(byte.encode('ascii'))
        self.ser.flush()

    def _exchange(self, byte, terminal):
        """Send one command byte, collect lines until `terminal(line)` is true.

        Returns (final_line, all_lines). Informational chatter such as
        "changed status from 0 to 1" arrives first and is kept in all_lines."""
        self._send(byte)
        lines, deadline = [], time.time() + self.timeout
        while time.time() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode('ascii', errors='replace').strip()
            if not line:
                continue
            if self.verbose:
                print(f"  <- {line}", file=sys.stderr)
            lines.append(line)
            if terminal(line):
                return line, lines
        raise ToolChangerError(
            f"No reply to {byte!r} within {self.timeout}s on {self.port}. "
            f"Check the baud rate matches Serial.begin(9600) in main.cpp, that the sketch is "
            f"actually flashed, and that no serial monitor is holding the port."
            + (f" Got partial output: {lines}" if lines else ""))

    # ---------------------------------------------------------------- commands
    def _status_cmd(self, byte, label):
        def terminal(line):
            return line.endswith(CONFIRM_SUFFIX) or line == EMERGENCY

        final, lines = self._exchange(byte, terminal)
        for note in lines[:-1]:
            print(f"   {note}")
        if final == EMERGENCY:
            print(f"{label}: EMERGENCY STOP -- the sensor disagrees with the commanded state.")
            return False
        print(f"{label}: {final}")
        return True

    def hold(self):
        """Lock the bearings onto the tool (servo -> 50 deg)."""
        return self._status_cmd(LOCKED, 'hold')

    def release(self):
        """Unlock and let the tool go (servo -> 15 deg)."""
        return self._status_cmd(UNLOCKED, 'release')

    def status(self):
        """Ask the board whether the sensor agrees with its current status."""
        return self._status_cmd(QUERY, 'status')

    def motor(self):
        """Toggle relay K1. Returns True if the motor ended up ON."""
        final, _ = self._exchange(MOTOR, lambda ln: ln.startswith('Motor'))
        print(f"motor: {final}")
        return final == 'Motor On'

    def monitor(self):
        """Print whatever the board sends until Ctrl-C."""
        print(f"Monitoring {self.port} at {self.baud} baud. Ctrl-C to stop.")
        while True:
            raw = self.ser.readline()
            if raw:
                print(raw.decode('ascii', errors='replace').rstrip())

    def close(self):
        if self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def repl(tc):
    print(f"Connected to {tc.port}. Commands: hold, release, status, motor, quit")
    actions = {'hold': tc.hold, 'h': tc.hold,
               'release': tc.release, 'r': tc.release,
               'status': tc.status, 's': tc.status,
               'motor': tc.motor, 'm': tc.motor}
    while True:
        try:
            word = input('toolchanger> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if word in ('quit', 'exit', 'q'):
            return
        if not word:
            continue
        if word not in actions:
            print(f"  unknown: {word!r} -- try hold, release, status, motor, quit")
            continue
        try:
            actions[word]()
        except ToolChangerError as exc:
            print(f"  error: {exc}")


def main():
    ap = argparse.ArgumentParser(
        description='Drive the Arduino toolchanger over serial.',
        epilog="hold = clamp the ball bearings onto the tool, release = let it go.")
    ap.add_argument('command', nargs='?',
                    choices=['hold', 'release', 'status', 'motor', 'monitor'],
                    help='omit for an interactive prompt')
    ap.add_argument('--port', help='serial device (default: autodetect)')
    ap.add_argument('--baud', type=int, default=9600, help='must match Serial.begin() (default 9600)')
    ap.add_argument('--timeout', type=float, default=5.0, help='seconds to wait for a reply')
    ap.add_argument('--settle', type=float, default=2.0, help='seconds to wait for the board to boot')
    ap.add_argument('-v', '--verbose', action='store_true', help='show the raw bytes and lines')
    args = ap.parse_args()

    try:
        tc = ToolChanger(args.port, args.baud, args.timeout, args.settle, args.verbose)
    except serial.SerialException as exc:
        sys.exit(f"Cannot open the port: {exc}\n"
                 f"Check `ls /dev/ttyACM* /dev/ttyUSB*`, that you are in the dialout group, "
                 f"and that the Arduino IDE serial monitor is closed.")
    except ToolChangerError as exc:
        sys.exit(str(exc))

    try:
        if args.command is None:
            repl(tc)
        elif args.command == 'monitor':
            tc.monitor()
        else:
            ok = getattr(tc, args.command)()
            sys.exit(0 if ok else 2)
    except KeyboardInterrupt:
        print()
    except ToolChangerError as exc:
        sys.exit(str(exc))
    finally:
        tc.close()


if __name__ == '__main__':
    main()
