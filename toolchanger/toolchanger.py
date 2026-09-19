#!/usr/bin/env python3
"""Drive the Arduino toolchanger (main.cpp) over serial from the terminal.

    ./toolchanger.py hold        # lock the ball bearings onto the tool   (sends '1')
    ./toolchanger.py release     # unlock, let the tool go                (sends '0')
    ./toolchanger.py status      # ask whether a tool is actually there   (sends 's')
    ./toolchanger.py motor       # toggle the motor relay K1 on/off       (sends 'm')
    ./toolchanger.py probe       # print the raw proximity reading        (sends 'r')
    ./toolchanger.py calibrate   # measure thresh -- READ ONLY, servo never moves
    ./toolchanger.py monitor     # just watch whatever the board prints
    ./toolchanger.py             # interactive prompt (hold/release/status/motor/quit)

Options: --port /dev/ttyACM0   --baud 9600   --timeout 5   --verbose

The wire protocol is exactly what main.cpp implements -- one ASCII byte per command:

    '0'..'9'  changeStatus(n).  0 = unlocked (servo 50 deg), >0 = locked (servo 15 deg).
              The angles live in main.cpp as noLockAngle/lockAngle -- if you retune them
              there, these comments are the thing that goes stale.
              Board may print "changed status from X to Y", then the confirmation line.
    's'       report status without changing it, with no retry.
    'r'       print the raw averaged sensor reading, for calibrating thresh. Reads only.
    'm'       toggleMotorPower() -> "Motor On" / "Motor Off".

and the board answers with one of:

    "<n> confirmed!"   the proximity sensor AGREES with the commanded status -- success.
    "emergency stop raw=N thresh=M"
                       the sensor DISAGREES. Asked to hold but nothing is gripped, or asked
                       to release but something is still detected. raw is the reading that
                       caused it: when raw sits near thresh the threshold is mistuned rather
                       than the tool being absent -- run `calibrate` and edit main.cpp.

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
RAW = 'r'
MOTOR = 'm'

BANNER = 'toolchanger ready'   #  printed by setup(), i.e. once per board reset
CONFIRM_SUFFIX = 'confirmed!'
EMERGENCY = 'emergency stop'   #  the board appends " raw=N thresh=M"; match on the prefix


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

    def __init__(self, port=None, baud=9600, timeout=5.0, settle=3.0, verbose=False,
                 latch=False):
        self.port = port or find_port()
        self.baud = baud
        self.timeout = timeout
        self.verbose = verbose
        # read timeout is per-readline; the overall budget is enforced in _exchange()
        self.ser = serial.Serial(self.port, baud, timeout=0.3)
        if latch:
            self._clear_hupcl()
        # Opening the port toggles DTR, which resets the board. Anything we send during the
        # bootloader window is lost, so wait the reset out before sending.
        self._sync(settle)

    def _sync(self, settle):
        """Wait for the board's boot banner, then drop anything still buffered.

        Sleeping a fixed time and flushing early is not enough: a line that lands AFTER the
        flush but BEFORE the next command -- the tail of the previous session, or the banner
        from a slow boot -- is read as the reply to that command, and every answer after it is
        off by one. Waiting for the banner puts us at a known point in the stream.

        Seeing the banner also means the board has just run setup(), so motorActive is false
        and the relay is off, whatever state the previous session left behind."""
        #  setup() has to get through the bootloader, a 100 ms sensor average and a
        #  500 ms servo move before it can print, so the floor here is generous
        waited = max(settle, 2.5)
        deadline = time.time() + waited
        while time.time() < deadline:
            raw = self.ser.readline()
            if raw and BANNER in raw.decode('ascii', errors='replace'):
                self.booted = True
                break
        else:
            # An older sketch predates the banner, so this is a warning and not an error.
            self.booted = False
            if self.verbose:
                print(f"  (no {BANNER!r} within {waited}s -- older firmware?)", file=sys.stderr)
        self.ser.reset_input_buffer()

    def _clear_hupcl(self):
        """Stop the kernel dropping DTR when the port closes.

        Closing the port normally hangs up DTR, which resets the Arduino -- setup() then runs
        digitalWrite(relayK1, LOW) and the motor switches off the instant this process exits.
        Clearing HUPCL leaves the board running, so a latched relay stays latched.

        Note this gives up a dead-man switch: with the reset in place, a crashed or killed
        script always leaves the motor off. Latched, the motor keeps running until something
        turns it off or the board loses power."""
        try:
            import termios
            fd = self.ser.fileno()
            attrs = termios.tcgetattr(fd)
            attrs[2] &= ~termios.HUPCL      # index 2 is cflag
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except Exception as exc:  # not a tty, no termios, permissions -- never fatal
            print(f"  warning: could not keep the board alive across close ({exc}). It will "
                  f"reset on exit and the motor will switch off.", file=sys.stderr)

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
            return line.endswith(CONFIRM_SUFFIX) or line.startswith(EMERGENCY)

        final, lines = self._exchange(byte, terminal)
        for note in lines[:-1]:
            print(f"   {note}")
        if final.startswith(EMERGENCY):
            print(f"{label}: EMERGENCY STOP -- the sensor disagrees with the commanded state.")
            print(f"   board said: {final}")
            print(f"   {self._explain_emergency(final)}")
            return False
        print(f"{label}: {final}")
        return True

    def hold(self):
        """Lock the bearings onto the tool (servo -> lockAngle, 15 deg)."""
        return self._status_cmd(LOCKED, 'hold')

    def release(self):
        """Unlock and let the tool go (servo -> noLockAngle, 50 deg)."""
        return self._status_cmd(UNLOCKED, 'release')

    def status(self):
        """Ask the board whether the sensor agrees with its current status."""
        return self._status_cmd(QUERY, 'status')

    def motor(self):
        """Toggle relay K1. Returns True if the motor ended up ON."""
        final, _ = self._exchange(MOTOR, lambda ln: ln.startswith('Motor'))
        print(f"motor: {final}")
        return final == 'Motor On'

    #  a reading this far from the threshold is a clear verdict; anything nearer and the
    #  threshold itself is the thing in doubt, not the tool
    DECISIVE_MARGIN = 100

    @classmethod
    def _explain_emergency(cls, line):
        """Say whether an emergency stop means "no tool" or "threshold mistuned"."""
        try:
            fields = dict(part.split('=', 1) for part in line.split() if '=' in part)
            raw, thresh = int(fields['raw']), int(fields['thresh'])
        except (ValueError, KeyError):
            return ('Could not parse the reading. Run `./toolchanger.py probe` to see it.')

        if abs(raw - thresh) < cls.DECISIVE_MARGIN:
            return (f"raw {raw} is within {cls.DECISIVE_MARGIN} of thresh {thresh}, so the "
                    f"THRESHOLD is in doubt, not the tool. Run `./toolchanger.py calibrate`.")
        return (f"raw {raw} is a clear {abs(raw - thresh)} from thresh {thresh}, so the sensor "
                f"is confident: there is genuinely no tool to grip.")

    def probe(self):
        """Print the raw averaged sensor reading. Reads only -- the servo does not move."""
        final, _ = self._exchange(RAW, lambda ln: ln.startswith('raw '))
        print(f"probe: {final}")
        return self._parse_raw(final)

    @staticmethod
    def _parse_raw(line):
        """Pull N out of "raw N thresh M tool yes status 1"."""
        try:
            return int(line.split()[1])
        except (IndexError, ValueError):
            raise ToolChangerError(
                f"Could not read a raw value out of {line!r}. An older sketch that does not "
                f"implement 'r' is probably still flashed -- reflash with "
                f"firmware/build_flash.sh upload.")

    def calibrate(self):
        """Measure the sensor with and without a tool and recommend thresh/toolReadsHigh.

        Nothing here commands a status change, so the servo stays where it is."""
        print("Calibration reads the sensor only -- the servo will not move.\n")
        input("  1. MOUNT a tool, then press Enter...")
        mounted = self.probe()
        input("  2. REMOVE the tool, then press Enter...")
        empty = self.probe()

        spread = abs(mounted - empty)
        print(f"\n  mounted: {mounted}    empty: {empty}    spread: {spread}")

        #  the Uno's ADC is 0..1023; a sensor that barely moves between the two states cannot
        #  drive any threshold reliably, so say so rather than recommending a coin flip
        if spread < 50:
            print("\n  The two readings are too close to tell apart. The threshold is not the "
                  "problem:\n  check the sensor's wiring, its supply, and its distance to the "
                  "tool. A working\n  probe should swing by hundreds of counts.")
            return False

        thresh = (mounted + empty) // 2
        reads_high = mounted > empty
        print(f"\n  Put these in firmware/main.cpp and reflash:\n"
              f"      const int thresh = {thresh};\n"
              f"      const bool toolReadsHigh = {'true' if reads_high else 'false'};\n"
              f"\n  cd firmware && ./build_flash.sh upload")
        return True

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
    print(f"Connected to {tc.port}. "
          f"Commands: hold, release, status, motor, probe, calibrate, quit")
    actions = {'hold': tc.hold, 'h': tc.hold,
               'release': tc.release, 'r': tc.release,
               'status': tc.status, 's': tc.status,
               'motor': tc.motor, 'm': tc.motor,
               'probe': tc.probe, 'p': tc.probe,
               'calibrate': tc.calibrate}
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
            print(f"  unknown: {word!r} -- try hold, release, status, motor, probe, "
                  f"calibrate, quit")
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
                    choices=['hold', 'release', 'status', 'motor', 'probe', 'calibrate',
                             'monitor'],
                    help='omit for an interactive prompt')
    ap.add_argument('--port', help='serial device (default: autodetect)')
    ap.add_argument('--baud', type=int, default=9600, help='must match Serial.begin() (default 9600)')
    ap.add_argument('--timeout', type=float, default=5.0, help='seconds to wait for a reply')
    ap.add_argument('--settle', type=float, default=3.0,
                    help='seconds to wait for the board to boot (it prints a banner when ready)')
    ap.add_argument('-v', '--verbose', action='store_true', help='show the raw bytes and lines')
    ap.add_argument('--latch', action='store_true',
                    help='leave the board running when this exits, so the motor STAYS on. '
                         'Without it, closing the port resets the board and switches the '
                         'motor off -- which doubles as a dead-man switch.')
    args = ap.parse_args()

    try:
        tc = ToolChanger(args.port, args.baud, args.timeout, args.settle, args.verbose,
                         args.latch)
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
