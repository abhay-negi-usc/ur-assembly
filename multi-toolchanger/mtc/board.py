"""The serial connection to the toolchanger board, and finding which board is which.

Board knows nothing about any device: it opens the port, waits out the reset, sends command
strings and collects reply lines. Each module in mtc/modules/ speaks its own protocol through it.
"""

import os
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is missing. Install it with:  pip install pyserial")

from .base import ToolChangerError

BANNER = 'toolchanger ready'   #  printed by setup(), i.e. once per board reset


#  =====   which board   =====
#  WHICH ARDUINO IS WHICH, from configs/couplers.yaml. Two boards running this sketch answer
#  identically over the wire, so the only thing telling them apart is the USB serial; that file
#  maps serials to names and everything here refers to the name.
#
#      ./multitoolchanger.py hold --port end_effector
#      ./multitoolchanger.py hold --port cleat
#
#  `./multitoolchanger.py list` prints the serials of whatever is plugged in, and a block to
#  paste.
#  Point somewhere else with $TOOLCHANGER_COUPLERS. The mapping is OPTIONAL -- without it,
#  --port still takes a serial or a device path directly.
COUPLERS_YAML = os.environ.get(
    'TOOLCHANGER_COUPLERS',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'configs',
                 'couplers.yaml'))


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
                f'`./multitoolchanger.py list` and paste the serial in.')
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
                f'`./multitoolchanger.py list` to see what is plugged in, and add {match!r} to '
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
            + '\n  `./multitoolchanger.py list` also prints a couplers.yaml block to '
              'paste in.')
    return found[0]['stable']


class Board:
    """One open connection to a toolchanger board. Blocking; one command at a time."""

    def __init__(self, port=None, baud=9600, timeout=5.0, settle=3.0, verbose=False,
                 latch=False, name=None):
        # `name` prefixes everything this connection prints. With one board it is noise; with
        # several open at once a transcript without it cannot say which mechanism moved.
        self.name = name or (port if isinstance(port, str) and port in COUPLERS else None)
        self.tag = f'[{self.name}] ' if self.name else ''
        self.port = find_port(port)
        self.baud = baud
        self.timeout = timeout
        self.verbose = verbose
        # Called on every pass of a wait for the board. The prompt sets it to check for q, and
        # it raises to abandon the wait -- see Session.check_interrupt().
        self.poll_hook = None
        # read timeout is per-readline, kept short so poll_hook runs often; the overall budget
        # is enforced in collect()
        self.ser = serial.Serial(self.port, baud, timeout=0.1)
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

        Seeing the banner also means the board has just run setup(), so every device is in its
        safe state, whatever the previous session left behind."""
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

        Closing the port normally hangs up DTR, which resets the Arduino -- setup() then puts
        every device in its safe state (relay off, screwdrive stopped) the instant this process
        exits. Clearing HUPCL leaves the board running, so a latched relay stays latched and a
        `drive` keeps driving.

        Note this gives up a dead-man switch: with the reset in place, a crashed or killed
        script always leaves the motors off. Latched, they keep running until something
        turns them off or the board loses power."""
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
    def send(self, byte):
        # `byte` is a whole command string: one byte, or a line such as 'd40\n'
        if self.verbose:
            print(f"  -> {byte!r}", file=sys.stderr)
        self.ser.write(byte.encode('ascii'))
        self.ser.flush()

    def exchange(self, byte, terminal):
        """Send one command byte, collect lines until `terminal(line)` is true.

        Returns (final_line, all_lines). Informational chatter such as
        "changed status from 0 to 1" arrives first and is kept in all_lines."""
        self.send(byte)
        return self.collect(byte, terminal, self.timeout)

    def read_line(self):
        """One line from the board, stripped, or None if nothing arrived within ~0.1 s."""
        raw = self.ser.readline()
        line = raw.decode('ascii', errors='replace').strip() if raw else ''
        if line and self.verbose:
            print(f"  <- {line}", file=sys.stderr)
        return line or None

    def collect(self, byte, terminal, timeout):
        """Collect lines until `terminal(line)` is true, for up to `timeout` seconds."""
        lines, deadline = [], time.time() + timeout
        while time.time() < deadline:
            if self.poll_hook is not None:
                self.poll_hook()
            line = self.read_line()
            if not line:
                continue
            lines.append(line)
            if terminal(line):
                return line, lines
        raise ToolChangerError(
            f"No reply to {byte!r} within {timeout:g}s on {self.port}. "
            f"Check the baud rate matches Serial.begin(9600) in firmware/main.cpp, that the "
            f"sketch is actually flashed, and that no serial monitor is holding the port."
            + (f" Got partial output: {lines}" if lines else ""))

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
