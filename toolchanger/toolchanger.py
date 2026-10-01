#!/usr/bin/env python3
"""Drive the Arduino toolchanger (main.cpp) over serial from the terminal.

    ./toolchanger.py hold        # lock the ball bearings onto the tool   (sends '1')
    ./toolchanger.py release     # unlock, let the tool go                (sends '0')
    ./toolchanger.py status      # ask whether a tool is actually there   (sends 's')
    ./toolchanger.py motor       # toggle the motor relay K1 on/off       (sends 'm')
    ./toolchanger.py probe       # print the raw proximity reading        (sends 'r')
    ./toolchanger.py bypass      # work the servo WITHOUT the sensor      (sends 'b')
    ./toolchanger.py calibrate   # measure thresh -- READ ONLY, servo never moves
    ./toolchanger.py monitor     # just watch whatever the board prints
    ./toolchanger.py list        # which boards are plugged in, and what to call them
    ./toolchanger.py             # interactive prompt (hold/release/status/motor/quit)

Every command has a one-letter shorthand -- h r s m p c b l -- and any unambiguous prefix works
too, so `./toolchanger.py m` is `motor` and `mon` is `monitor`. (`m` is pinned to motor rather
than left to the prefix rule, which could not choose between the two.)

TWO COUPLERS, ONE SCRIPT. Boards running this sketch are identical over the wire, so the only
thing telling them apart is the USB serial. Run `list`, put the serials in COUPLERS at the top
of this file against names you will remember, and then:

    ./toolchanger.py hold --port tool     # the coupler on the end effector
    ./toolchanger.py hold --port cleat    # the one on the cleat the ORU mounts to

With one board plugged in --port can be omitted. With two it CANNOT: rather than picking the
best-scoring port and hoping, the script refuses and lists them, because guessing means working
the wrong mechanism and getting a perfectly normal confirmation back from it.

Options: --port NAME|SERIAL|PATH   --baud 9600   --timeout 5   --verbose

The wire protocol is exactly what main.cpp implements -- one ASCII byte per command:

    '0'..'9'  changeStatus(n).  0 = unlocked (servo 15 deg), >0 = locked (servo 50 deg).
              The angles live in main.cpp as noLockAngle/lockAngle -- if you retune them
              there, these comments are the thing that goes stale.
              Board may print "changed status from X to Y", then the confirmation line.
    's'       report status without changing it, with no retry.
    'r'       print the raw averaged sensor reading, for calibrating thresh. Reads only.
    'b'       toggle the sensor bypass. Bypassed, hold and release move the servo and confirm
              without consulting the sensor, and the watchdog stops alarming. Cleared by any
              reset, so it never outlives the connection that set it.
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
import os
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
BYPASS = 'b'
MOTOR = 'm'

BANNER = 'toolchanger ready'   #  printed by setup(), i.e. once per board reset
CONFIRM_SUFFIX = 'confirmed!'
EMERGENCY = 'emergency stop'   #  the board appends " raw=N thresh=M"; match on the prefix


class ToolChangerError(RuntimeError):
    pass


#  =====   command names   =====
COMMANDS = ('hold', 'release', 'status', 'motor', 'probe', 'calibrate', 'bypass', 'monitor',
            'list')

#  One letter per command, chosen to match the WIRE BYTE where there is one ('s' status,
#  'r' raw/probe... note 'r' is release here because that is what a human reaching for r means,
#  and the probe gets 'p'). These are explicit rather than derived because two of them would be
#  ambiguous as prefixes: 'm' would not choose between motor and monitor, and 'r' would not
#  choose between release and... nothing today, but it would the moment a 'reset' is added.
SHORTHANDS = {'h': 'hold', 'r': 'release', 's': 'status', 'm': 'motor',
              'p': 'probe', 'c': 'calibrate', 'b': 'bypass', 'l': 'list'}


def resolve_command(word):
    """A command name from whatever the user typed: exact, shorthand, or unambiguous prefix.

    SHORTHANDS WIN OVER PREFIXES, which is the whole reason they are a table. `m` is ambiguous
    between `motor` and `monitor` by prefix, and `motor` is the one anybody typing a single
    letter in a hurry means -- so it is pinned, and `monitor` is reached by `mon`.

    Shared by the CLI and the interactive prompt so the two cannot drift: a shorthand that works
    at one has always worked at the other. Pure, so the table is testable."""
    word = (word or '').strip().lower()
    if not word:
        raise ToolChangerError('no command given')
    if word in COMMANDS:
        return word
    if word in SHORTHANDS:
        return SHORTHANDS[word]
    hits = [c for c in COMMANDS if c.startswith(word)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise ToolChangerError(
            f'{word!r} is ambiguous -- it could be {" or ".join(sorted(hits))}. '
            f'Type more of it, or use one of: '
            + ', '.join(f'{k}={v}' for k, v in sorted(SHORTHANDS.items())))
    raise ToolChangerError(
        f'unknown command {word!r}. Commands: {", ".join(COMMANDS)}. '
        + 'Shorthands: ' + ', '.join(f'{k}={v}' for k, v in sorted(SHORTHANDS.items())))


#  =====   which board   =====
#  WHICH ARDUINO IS WHICH, from configs/couplers.yaml. Two boards running this sketch answer
#  identically over the wire, so the only thing telling them apart is the USB serial; that file
#  maps serials to names and everything here refers to the name.
#
#      ./toolchanger.py hold --port end_effector
#      ./toolchanger.py hold --port cleat
#
#  `./toolchanger.py list` prints the serials of whatever is plugged in, and a block to paste.
#  Point somewhere else with $TOOLCHANGER_COUPLERS. The mapping is OPTIONAL -- without it,
#  --port still takes a serial or a device path directly.
COUPLERS_YAML = os.environ.get(
    'TOOLCHANGER_COUPLERS',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'couplers.yaml'))


def load_couplers(path=None):
    """{name: serial} from configs/couplers.yaml, or {} if there is no usable mapping.

    NEVER FATAL. This driver's one hard dependency is pyserial, and it has to keep working on a
    bench machine with nothing else installed -- so a missing file, a missing PyYAML or a broken
    mapping costs the NAMES and nothing more: `--port <serial>` and `--port /dev/ttyACM0` go on
    working. What it must not do is half-load and leave some names silently absent, so anything
    it cannot read at all is reported rather than swallowed.

    Serials are forced to str: they look numeric and YAML would hand back an int, which loses a
    leading zero and then matches nothing."""
    path = path or COUPLERS_YAML
    if not os.path.isfile(path):
        return {}
    try:
        import yaml
    except ImportError:
        print(f'  (PyYAML is not installed, so {path} was not read -- coupler NAMES are '
              f'unavailable. --port still takes a serial or a device path.)', file=sys.stderr)
        return {}
    try:
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
        return {str(k): str(v).strip() for k, v in (doc.get('couplers') or {}).items()}
    except Exception as exc:                    # noqa: BLE001 -- a bad mapping is not fatal
        print(f'  (could not read {path}: {exc} -- coupler names are unavailable.)',
              file=sys.stderr)
        return {}


COUPLERS = load_couplers()

ARDUINO_HINTS = ('arduino', 'ch340', 'ftdi', 'wch', 'usb serial')


def _by_id_paths():
    """{real device path: /dev/serial/by-id/... } -- the stable name for each port.

    /dev/ttyACM0 is an enumeration ORDER, not an identity: unplug both boards and plug them back
    in the other way round and the numbers swap. The by-id path carries the USB serial, so it
    names the same board every time, which is what a two-coupler cell needs."""
    out = {}
    try:
        for name in os.listdir('/dev/serial/by-id'):
            link = os.path.join('/dev/serial/by-id', name)
            out[os.path.realpath(link)] = link
    except OSError:
        pass                                    # not Linux, or no by-id: fall back to the device
    return out


def candidates():
    """Every port that plausibly is one of these boards, richest identity first."""
    by_id = _by_id_paths()
    found = []
    for p in list_ports.comports():
        text = f"{p.manufacturer or ''} {p.product or ''} {p.description or ''}".lower()
        if not (any(k in text for k in ARDUINO_HINTS) or p.serial_number):
            continue                            # /dev/ttyS* and other noise
        found.append({'device': p.device,
                      'stable': by_id.get(os.path.realpath(p.device), p.device),
                      'serial': p.serial_number or '',
                      'what': (p.product or p.description or '').strip()})
    return sorted(found, key=lambda c: c['device'])


def describe_ports():
    """The `list` command: what is plugged in, and what to call it."""
    found = candidates()
    if not found:
        print('No boards found. Check `ls /dev/ttyACM* /dev/ttyUSB*` and that they are plugged '
              'in.')
        return False
    named = {v: k for k, v in COUPLERS.items()}
    print(f'{len(found)} board(s):\n')
    for c in found:
        alias = next((n for s, n in named.items() if s and s in c['serial']), None)
        print(f"  {c['device']}   serial {c['serial'] or '(none)'}"
              f"{'   --port ' + alias if alias else ''}")
        print(f"      {c['stable']}")
        if c['what']:
            print(f"      {c['what']}")
    if not COUPLERS:
        print(f'\nNothing is named yet. Put these in {os.path.abspath(COUPLERS_YAML)}:\n')
        print('couplers:')
        for n, c in zip(('end_effector', 'cleat'), found):
            print(f"  {n}: '{c['serial']}'")
    else:
        print(f'\nNames come from {os.path.abspath(COUPLERS_YAML)}: '
              + ', '.join(sorted(COUPLERS)) + '.')
    return True


def find_port(match=None):
    """The port for `match`: a coupler name, a serial (or part of one), or a device path.

    REFUSES TO GUESS BETWEEN TWO BOARDS. The old version took the best-scoring port and ran with
    it, which is fine with one Arduino and actively dangerous with two: the boards are identical
    over the wire, so picking the wrong one means commanding the wrong coupler and getting a
    perfectly normal-looking confirmation back from it. With several plugged in and nothing to
    choose by, this raises and lists them instead."""
    if match is not None:
        if str(match) in COUPLERS and not str(COUPLERS[str(match)]).strip():
            raise ToolChangerError(
                f'couplers.yaml has {match!r} but no serial against it. An empty value '
                f'would match every board, so it is refused rather than guessed. Run '
                f'`./toolchanger.py list` and paste the serial in.')
        # THE ALIAS IS RESOLVED FIRST, so a COUPLERS entry may be a device path as legitimately
        # as a serial. Checking the raw word for existence before looking it up would silently
        # skip that -- a name is never a path, so the lookup has to come first.
        match = COUPLERS.get(str(match), str(match))
        if os.path.exists(str(match)):
            return str(match)                   # an explicit path wins, however odd it looks

    found = candidates()
    if not found:
        raise ToolChangerError(
            'No boards found. Plug the Arduino in and check `ls /dev/ttyACM* /dev/ttyUSB*`.')

    if match:
        needle = str(match).lower()
        hits = [c for c in found
                if needle in c['serial'].lower() or needle in c['stable'].lower()
                or needle in c['device'].lower() or needle in c['what'].lower()]
        if not hits:
            known = ', '.join(sorted(COUPLERS)) or '(none named yet)'
            raise ToolChangerError(
                f'Nothing matches {match!r}. Named couplers: {known}. Run '
                f'`./toolchanger.py list` to see what is plugged in, and add {match!r} to '
                f'{os.path.abspath(COUPLERS_YAML)} with that board\'s serial. (The coupler '
                f'apps ask for it by name, so the name has to be mapped there once.)')
        if len(hits) > 1:
            raise ToolChangerError(
                f'{match!r} matches {len(hits)} boards: '
                + ', '.join(f"{h['device']} (serial {h['serial']})" for h in hits)
                + '. Use a longer serial, or a name from couplers.yaml.')
        return hits[0]['stable']

    if len(found) > 1:
        raise ToolChangerError(
            f'{len(found)} boards are plugged in and none was named, so there is nothing to '
            'choose by -- they run the same sketch and answer identically, so guessing would '
            'mean working the wrong coupler. Pass --port with a name or a serial:\n'
            + '\n'.join(f"    --port {c['serial']}   ({c['device']})" for c in found)
            + '\n  `./toolchanger.py list` also prints a couplers.yaml block to '
              'paste in.')
    return found[0]['stable']


class ToolChanger:
    """Blocking control of the toolchanger. `hold()` locks, `release()` unlocks."""

    def __init__(self, port=None, baud=9600, timeout=5.0, settle=3.0, verbose=False,
                 latch=False, name=None):
        # `name` prefixes everything this instance prints. With one board it is noise; with two
        # open at once -- the end-effector coupler and the one on the cleat -- a transcript
        # without it is a list of confirmations with no way to tell which mechanism moved.
        self.name = name or (port if isinstance(port, str) and port in COUPLERS else None)
        self.tag = f'[{self.name}] ' if self.name else ''
        self.port = find_port(port)
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
            print(f"{self.tag}{label}: EMERGENCY STOP -- the sensor disagrees with the "
                  f"commanded state.")
            print(f"   board said: {final}")
            print(f"   {self._explain_emergency(final)}")
            return False
        print(f"{self.tag}{label}: {final}")
        return True

    def hold(self):
        """Lock the bearings onto the tool (servo -> lockAngle, 50 deg)."""
        return self._status_cmd(LOCKED, 'hold')

    def release(self):
        """Unlock and let the tool go (servo -> noLockAngle, 15 deg)."""
        return self._status_cmd(UNLOCKED, 'release')

    def status(self):
        """Ask the board whether the sensor agrees with its current status."""
        return self._status_cmd(QUERY, 'status')

    def motor(self):
        """Toggle relay K1. Returns True if the motor ended up ON."""
        final, _ = self._exchange(MOTOR, lambda ln: ln.startswith('Motor'))
        print(f"{self.tag}motor: {final}")
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

    def bypass(self):
        """Toggle the board's sensor bypass. Returns True if the bypass ended up ON.

        For working the mechanism while the probe is untrustworthy. A disconnected or shorted
        sensor reads a hard rail and then agrees with every command, confirming grips that are
        not happening -- bypassing at least stops the board claiming to have checked.

        The board clears this on reset, and opening the port resets the board, so it only lasts
        as long as this connection."""
        final, _ = self._exchange(BYPASS, lambda ln: ln.startswith('sensor bypass'))
        on = 'ON' in final
        print(f"{self.tag}bypass: {final}")
        if on:
            print("   grip is NOT verified while this is on. It clears when you disconnect.")
        return on

    def probe(self):
        """Print the raw averaged sensor reading. Reads only -- the servo does not move."""
        final, _ = self._exchange(RAW, lambda ln: ln.startswith('raw '))
        print(f"{self.tag}probe: {final}")
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
    """The interactive prompt. Resolves command words the SAME way the CLI does -- one table,
    so a shorthand that works at the shell always works here too."""
    print(f"Connected to {tc.name or tc.port}.")
    print('  Commands: ' + ', '.join(COMMANDS) + ', quit')
    print('  Shorthands: ' + ', '.join(f'{k}={v}' for k, v in sorted(SHORTHANDS.items()))
          + '  (any unambiguous prefix works too)')
    while True:
        try:
            word = input(f'{tc.name or "toolchanger"}> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if word in ('quit', 'exit', 'q'):
            return
        if not word:
            continue
        try:
            name = resolve_command(word)
            if name == 'list':
                describe_ports()
            elif name == 'monitor':
                tc.monitor()
            else:
                getattr(tc, name)()
        except ToolChangerError as exc:
            print(f"  {exc}")


def main():
    ap = argparse.ArgumentParser(
        description='Drive the Arduino toolchanger over serial.',
        epilog="hold = clamp the ball bearings onto the tool, release = let it go.")
    ap.add_argument('command', nargs='?', metavar='COMMAND',
                    help='one of: ' + ', '.join(COMMANDS) + '. Shorthands: '
                         + ', '.join(f'{k}={v}' for k, v in sorted(SHORTHANDS.items()))
                         + '; any unambiguous prefix also works. Omit for an interactive '
                           'prompt.')
    ap.add_argument('--port', metavar='NAME|SERIAL|PATH',
                    help='WHICH COUPLER. A name from COUPLERS at the top of this file, a USB '
                         'serial (or enough of one to be unambiguous), or a device path. With '
                         'one board plugged in this can be omitted; with two it cannot, because '
                         'they are identical over the wire and guessing would work the wrong '
                         'mechanism. `list` shows what is connected.')
    ap.add_argument('--baud', type=int, default=9600,
                    help='must match Serial.begin() (default 9600)')
    ap.add_argument('--timeout', type=float, default=5.0, help='seconds to wait for a reply')
    ap.add_argument('--settle', type=float, default=3.0,
                    help='seconds to wait for the board to boot (it prints a banner when '
                         'ready)')
    ap.add_argument('-v', '--verbose', action='store_true', help='show the raw bytes and lines')
    ap.add_argument('--latch', action='store_true',
                    help='leave the board running when this exits, so the motor STAYS on. '
                         'Without it, closing the port resets the board and switches the '
                         'motor off -- which doubles as a dead-man switch.')
    args = ap.parse_args()

    # `list` answers from the host alone -- no port, no board, nothing opened. Resolved before
    # the connection so it still works when the thing you are trying to diagnose is which board
    # is which.
    try:
        command = resolve_command(args.command) if args.command else None
    except ToolChangerError as exc:
        sys.exit(str(exc))
    if command == 'list':
        sys.exit(0 if describe_ports() else 1)

    try:
        tc = ToolChanger(args.port, args.baud, args.timeout, args.settle, args.verbose,
                         args.latch, name=args.port if args.port in COUPLERS else None)
    except serial.SerialException as exc:
        sys.exit(f"Cannot open the port: {exc}\n"
                 f"Check `ls /dev/ttyACM* /dev/ttyUSB*`, that you are in the dialout group, "
                 f"and that the Arduino IDE serial monitor is closed.")
    except ToolChangerError as exc:
        sys.exit(str(exc))

    try:
        if command is None:
            repl(tc)
        elif command == 'monitor':
            tc.monitor()
        else:
            ok = getattr(tc, command)()
            sys.exit(0 if ok else 2)
    except KeyboardInterrupt:
        print()
    except ToolChangerError as exc:
        sys.exit(str(exc))
    finally:
        tc.close()


if __name__ == '__main__':
    main()
