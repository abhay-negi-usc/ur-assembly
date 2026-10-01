"""coupler -- the toolchanger lock: a servo that clamps ball bearings onto the tool, and a
proximity sensor that checks a tool is really there. Firmware: firmware/coupler.cpp.

Wire protocol, one byte each:

    '0'..'9'  changeStatus(n).  0 = unlocked (servo 15 deg), >0 = locked (servo 50 deg).
              The angles live in coupler.cpp as noLockAngle/lockAngle -- if you retune them
              there, these comments are the thing that goes stale.
              Board may print "changed status from X to Y", then the confirmation line.
    's'       report status without changing it, with no retry.
    'r'       print the raw averaged sensor reading, for calibrating thresh. Reads only.
    'b'       toggle the sensor bypass. Cleared by any reset.

The board answers "<n> confirmed!" (the sensor agrees with the commanded state) or
"emergency stop raw=N thresh=M" (it does not; raw near thresh means the threshold is mistuned
rather than the tool missing -- run `calibrate`). A hold/release round trip takes the better
part of a second: changeServo() blocks 500 ms and every sensor read averages 10 samples.
"""

from ..base import ToolChangerError
from ..registry import Module

LOCKED = '1'
UNLOCKED = '0'
QUERY = 's'
RAW = 'r'
BYPASS = 'b'

CONFIRM_SUFFIX = 'confirmed!'
EMERGENCY = 'emergency stop'   #  the board appends " raw=N thresh=M"; match on the prefix


class Coupler:
    """Blocking control of the coupler. `hold()` locks, `release()` unlocks."""

    #  a reading this far from the threshold is a clear verdict; anything nearer and the
    #  threshold itself is the thing in doubt, not the tool
    DECISIVE_MARGIN = 100

    def __init__(self, board, settings):
        self.board = board
        self.tag = board.tag

    def _status_cmd(self, byte, label):
        def terminal(line):
            return line.endswith(CONFIRM_SUFFIX) or line.startswith(EMERGENCY)

        final, lines = self.board.exchange(byte, terminal)
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

    @classmethod
    def _explain_emergency(cls, line):
        """Say whether an emergency stop means "no tool" or "threshold mistuned"."""
        try:
            fields = dict(part.split('=', 1) for part in line.split() if '=' in part)
            raw, thresh = int(fields['raw']), int(fields['thresh'])
        except (ValueError, KeyError):
            return 'Could not parse the reading. Run `probe` to see it.'

        if abs(raw - thresh) < cls.DECISIVE_MARGIN:
            return (f"raw {raw} is within {cls.DECISIVE_MARGIN} of thresh {thresh}, so the "
                    f"THRESHOLD is in doubt, not the tool. Run `calibrate`.")
        return (f"raw {raw} is a clear {abs(raw - thresh)} from thresh {thresh}, so the sensor "
                f"is confident: there is genuinely no tool to grip.")

    def check_line(self, line):
        """An emergency stop the watchdog sends unasked fails whatever was waiting."""
        if line.startswith(EMERGENCY):
            raise ToolChangerError(f'coupler reported: {line}')

    def bypass(self):
        """Toggle the board's sensor bypass. Returns True if the bypass ended up ON.

        For working the mechanism while the probe is untrustworthy. A disconnected or shorted
        sensor reads a hard rail and then agrees with every command, confirming grips that are
        not happening -- bypassing at least stops the board claiming to have checked.

        The board clears this on reset, and opening the port resets the board, so it only lasts
        as long as this connection."""
        final, _ = self.board.exchange(BYPASS, lambda ln: ln.startswith('sensor bypass'))
        on = 'ON' in final
        print(f"{self.tag}bypass: {final}")
        if on:
            print("   grip is NOT verified while this is on. It clears when you disconnect.")
        return on

    def probe(self):
        """Print the raw averaged sensor reading. Reads only -- the servo does not move."""
        final, _ = self.board.exchange(RAW, lambda ln: ln.startswith('raw '))
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
        print(f"\n  Put these in firmware/coupler.cpp and reflash:\n"
              f"      const int thresh = {thresh};\n"
              f"      const bool toolReadsHigh = {'true' if reads_high else 'false'};\n"
              f"\n  cd firmware && ./build_flash.sh upload")
        return True


MODULE = Module('coupler', 'servo lock + proximity sensor (firmware/coupler.cpp)',
                device=Coupler)


@MODULE.command()
def cmd_hold(s):
    """Lock the ball bearings onto the tool.

    Moves the servo to lockAngle, then waits up to 1.5 s for the proximity sensor to see the
    tool. If it never does, the board reports an emergency stop and this fails -- unless the
    sensor is bypassed."""
    return s.dev('coupler').hold()


@MODULE.command()
def cmd_release(s):
    """Unlock the ball bearings and let the tool go.

    Moves the servo to noLockAngle. Always succeeds: the tool still sitting in the changer
    afterwards is normal, and is reported rather than treated as a failure."""
    return s.dev('coupler').release()


@MODULE.command()
def cmd_status(s):
    """Report whether the sensor agrees with the lock state. Changes nothing.

    Fails only if the coupler is locked and the sensor sees no tool."""
    return s.dev('coupler').status()


@MODULE.command()
def cmd_probe(s):
    """Print the raw proximity sensor reading. Moves nothing."""
    s.dev('coupler').probe()


@MODULE.command()
def cmd_bypass(s):
    """Toggle the sensor bypass: hold and release stop consulting the sensor.

    For working the mechanism while the probe is untrustworthy. Grips are NOT verified while
    it is on. Any board reset clears it -- including every disconnect."""
    s.dev('coupler').bypass()


@MODULE.command(in_sequence=False)
def cmd_calibrate(s):
    """Measure the sensor with and without a tool, and recommend a threshold.

    Prompts you to mount and remove a tool, reads the sensor each time, and prints the
    `thresh` and `toolReadsHigh` to put in firmware/coupler.cpp. The servo does not move."""
    return s.dev('coupler').calibrate()
