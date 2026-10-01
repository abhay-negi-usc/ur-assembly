"""screwdrive -- the coupling lead screw's 12 V DC motor, on a Cytron MD10C R3.
Firmware: firmware/screwdrive.cpp.

Wire protocol -- a letter, numbers, a newline:

    'd<pct>\\n'          speed, signed percent -100..100, 0 stops -> "screwdrive <pct>%"
    't<pct>,<ms>\\n'     the same for ms -> "screwdrive <pct>% for <ms> ms", then
                        "screwdrive run done" when the BOARD stops it
    'p<duty>,<ms>\\n'    raw duty -255..255 for ms -> "screwdrive pwm <duty> for <ms> ms", then
                        "screwdrive run done". The open-loop rpm command converts to this.
    'a<p1>,<p2>,<ms>\\n' ramp from p1% to p2% over ms, then HOLD p2% ->
                        "screwdrive ramp <p1>% to <p2>% over <ms> ms", then "screwdrive ramp done"

A malformed line stops the motor and says so first. Any new command replaces a run or ramp in
progress. The board times runs and ramps itself, so a host that hangs cannot leave the motor
turning on a timer that never ends.
"""

import re

from ..base import ToolChangerError, parse_int, parse_ms
from ..registry import Module, Param, positive_number

DRIVE = 'd'
TIMED = 't'
PWM = 'p'
RAMP = 'a'
RUN_DONE = 'screwdrive run done'
RAMP_DONE = 'screwdrive ramp done'
STOPPED = r'screwdrive -?\d+%'      # the reply to `d`, and to a run the board refused

DRIVE_RANGE = 100                   # percent
DUTY_RANGE = 255                    # analogWrite()


def duty_for_rpm(rpm, max_rpm):
    """Open-loop duty for `rpm`: linear from 0 to max_rpm at full duty, rounded."""
    return round(rpm * DUTY_RANGE / max_rpm)


class Screwdrive:
    """Blocking control of the screwdrive motor. Returned speeds are what the BOARD reports."""

    def __init__(self, board, settings):
        self.board = board
        self.tag = board.tag
        self.max_rpm = settings.max_rpm

    def drive(self, pct):
        """Run at `pct` percent (-100..100, negative reverses, 0 stops) until told otherwise.

        The motor stops when the port closes (the board resets) unless the board was opened
        with latch=True."""
        pct = parse_int(pct, DRIVE_RANGE, 'drive', 'percent')
        final, lines = self.board.exchange(f'{DRIVE}{pct}\n', lambda ln: re.fullmatch(STOPPED, ln))
        for note in lines[:-1]:
            print(f"   {note}")
        got = int(final[len('screwdrive '):-1])
        print(f"{self.tag}drive: {final}")
        if got != pct:
            raise ToolChangerError(f'asked for {pct}% but the board reports {got}%')
        return got

    def stop(self):
        """Stop the motor. Returns 0, the speed the board reports."""
        return self.drive(0)

    def safe_stop(self):
        """What q and a failed sequence call: stop the motor."""
        self.stop()

    def _timed(self, cmd, started, label, ms, wait, done=RUN_DONE):
        """Send a timed line, check the board started it, optionally wait for it to end.

        Ctrl-C while waiting stops the motor at once; q does too, via the session."""
        final, lines = self.board.exchange(
            cmd, lambda ln: re.fullmatch(started, ln) or re.fullmatch(STOPPED, ln))
        for note in lines[:-1]:
            print(f"   {note}")
        match = re.fullmatch(started, final)
        if not match:
            raise ToolChangerError(f'{label}: the board refused it and stopped: {lines}')
        print(f"{self.tag}{label}: {final}")
        if not wait:
            return match
        try:
            self.board.collect(cmd, lambda ln: ln == done, ms / 1000 + self.board.timeout)
        except KeyboardInterrupt:
            print()
            self.stop()
            raise
        print(f"{self.tag}{label}: done")
        return match

    def run(self, pct, seconds, wait=True):
        """Run at `pct` percent for `seconds`, then stop.

        With wait=True (the default) this blocks until the board reports the run is over. With
        wait=False it returns as soon as the motor starts -- but closing the port still resets
        the board and stops the motor early, unless it was opened with latch=True.
        Returns the percent the board reports."""
        pct = parse_int(pct, DRIVE_RANGE, 'run', 'percent')
        ms = parse_ms(seconds)
        match = self._timed(f'{TIMED}{pct},{ms}\n', r'screwdrive (-?\d+)% for (\d+) ms',
                            'run', ms, wait)
        return int(match.group(1))

    def pwm(self, duty, seconds, wait=True, label='pwm'):
        """Run at raw PWM `duty` (-255..255) for `seconds`, then stop. Blocks like run().
        Returns the duty the board reports."""
        duty = parse_int(duty, DUTY_RANGE, label, 'duty')
        ms = parse_ms(seconds)
        match = self._timed(f'{PWM}{duty},{ms}\n', r'screwdrive pwm (-?\d+) for (\d+) ms',
                            label, ms, wait)
        return int(match.group(1))

    def rpm(self, rpm, seconds, wait=True):
        """Run at `rpm` OPEN LOOP for `seconds`, then stop.

        Sends duty = rpm / max_rpm * 255, max_rpm being the no-load speed from the config.
        Nothing measures the actual speed, so under load the motor turns slower than asked.
        Negative reverses. Blocks like run(). Returns the rpm asked for."""
        rpm = parse_int(rpm, int(self.max_rpm), 'rpm', 'rpm')
        duty = duty_for_rpm(rpm, self.max_rpm)
        print(f"{self.tag}rpm: {rpm} rpm open loop = pwm {duty} (max_rpm {self.max_rpm:g})")
        self.pwm(duty, seconds, wait, label='rpm')
        return rpm

    def ramp(self, start, end, seconds, wait=True):
        """Ramp linearly from `start`% to `end`% over `seconds`, then HOLD `end`%.

        The board does the ramp, stepping the duty as time passes, and keeps running at `end`
        afterwards like drive() -- ramp to 0 to finish stopped. Crossing zero passes through a
        stop rather than braking. Blocks until the ramp is over with wait=True.
        Returns the end percent."""
        start = parse_int(start, DRIVE_RANGE, 'ramp', 'percent')
        end = parse_int(end, DRIVE_RANGE, 'ramp', 'percent')
        ms = parse_ms(seconds)
        match = self._timed(f'{RAMP}{start},{end},{ms}\n',
                            r'screwdrive ramp (-?\d+)% to (-?\d+)% over (\d+) ms',
                            'ramp', ms, wait, done=RAMP_DONE)
        return int(match.group(2))


MODULE = Module('screwdrive', 'coupling lead screw: 12 V DC motor on a Cytron MD10C R3 '
                              '(firmware/screwdrive.cpp)', device=Screwdrive)

#  no-load rpm at full duty (12 V, 100% PWM); the open-loop `rpm` command scales from it
MODULE.setting('max_rpm', 500, check=positive_number)

MODULE.kind('percent',
            lambda t, cfg, n: parse_int(t, DRIVE_RANGE, n, 'percent'),
            lambda cfg: f'whole percent of full PWM, -{DRIVE_RANGE}..{DRIVE_RANGE}; '
                        f'negative reverses, 0 stops')
MODULE.kind('duty',
            lambda t, cfg, n: parse_int(t, DUTY_RANGE, n, 'duty'),
            lambda cfg: f'whole PWM duty, -{DUTY_RANGE}..{DUTY_RANGE}; negative reverses')
MODULE.kind('rpm',
            lambda t, cfg, n: parse_int(t, int(cfg.settings['screwdrive'].max_rpm), n, 'rpm'),
            lambda cfg: f'whole rpm, -{int(cfg.settings["screwdrive"].max_rpm)}..'
                        f'{int(cfg.settings["screwdrive"].max_rpm)} (max_rpm in '
                        f'config/screwdrive.yaml); negative reverses')


@MODULE.command(Param('PCT', 'percent'), runs_on=lambda args: args[0] != 0)
def cmd_drive(s, pct):
    """Run the screwdrive at PCT percent until told otherwise.

    The motor keeps running after the command returns: `stop` it, or let a disconnect (which
    resets the board) stop it. Reversing while running brakes for 200 ms first."""
    s.dev('screwdrive').drive(pct)


@MODULE.command()
def cmd_stop(s):
    """Stop the screwdrive."""
    s.dev('screwdrive').stop()


@MODULE.command(Param('PCT', 'percent'), Param('SECONDS', 'seconds'), timed=True)
def cmd_run(s, pct, seconds):
    """Run the screwdrive at PCT percent for SECONDS, then stop.

    The board times the run and stops the motor itself, so a host that hangs mid-run cannot
    leave it turning. Press q to stop early."""
    s.dev('screwdrive').run(pct, seconds)


@MODULE.command(Param('START', 'percent'), Param('END', 'percent'), Param('SECONDS', 'seconds'),
                timed=True, runs_on=lambda args: args[1] != 0)
def cmd_ramp(s, start, end, seconds):
    """Ramp the screwdrive from START to END percent over SECONDS, then hold END.

    Starts at START at once and changes speed linearly, stepped by the board. Afterwards the
    motor KEEPS RUNNING at END, like drive -- ramp down to 0 to finish stopped, or `stop`.
    Crossing zero (e.g. 50 to -50) passes through a stop instead of braking. Press q to stop."""
    s.dev('screwdrive').ramp(start, end, seconds)


@MODULE.command(Param('DUTY', 'duty'), Param('SECONDS', 'seconds'), timed=True)
def cmd_pwm(s, duty, seconds):
    """Run the screwdrive at raw PWM duty DUTY for SECONDS, then stop.

    DUTY is what analogWrite() gets: 255 is full speed. Timed by the board, like `run`.
    Press q to stop early."""
    s.dev('screwdrive').pwm(duty, seconds)


@MODULE.command(Param('RPM', 'rpm'), Param('SECONDS', 'seconds'), timed=True)
def cmd_rpm(s, rpm, seconds):
    """Run the screwdrive at RPM, open loop, for SECONDS, then stop.

    Sets duty = RPM / max_rpm * 255, where max_rpm is the no-load speed from
    config/screwdrive.yaml. Nothing measures the real speed: under load the motor turns slower
    than asked, and at low rpm it may not start at all. Press q to stop early."""
    s.dev('screwdrive').rpm(rpm, seconds)
