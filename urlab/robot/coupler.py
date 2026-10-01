"""Coupler -- the toolchanger locking mechanism, as the apps see it.

Wraps toolchanger/toolchanger.py's ToolChanger behind hold / release / verify, and falls back to
an operator prompt when the board is not there.

WHAT THE BOARD ACTUALLY PROMISES. hold() and release() each command the servo AND then ask the
proximity sensor whether the result agrees with what was asked. The board answers either
"<n> confirmed!" or "emergency stop raw=N thresh=M", and the driver turns those into True and
FALSE RESPECTIVELY. That return value is the whole safety interlock: "asked to hold but nothing
is gripped" and "asked to release but something is still there" are exactly the two states that
must not be mistaken for success, because the next thing an app does is lift.

    A False from hold() means the coupler is holding NOTHING. Anything that treats it as a
    success then sets a payload for an object that is not there, lifts air, and carries a
    3 kg gravity-compensation bias into every force reading after it.

THE FALLBACK IS DELIBERATE, not a stub. A calibration or a dry run is worth doing on a bench
with the board unplugged. What it costs is precisely the check above, so `verified` says whether
a real sensor agreed, and callers about to lift something heavy are expected to look at it.

BYPASS. The board has a sensor-bypass mode in which it confirms without consulting the probe.
It is OFF unless `toolchanger.bypass` asks for it: a reset clears the bypass and opening the
serial port resets the board, so every session starts with the interlock live and has to opt out
deliberately.

WHAT BYPASSING ACTUALLY BUYS, AND WHAT IT COSTS. It is for working the mechanism when the PROBE
is the thing that is broken -- a disconnected or shorted sensor reads a hard rail and then agrees
with every command, confirming grips that are not happening. Bypassing at least stops the board
claiming to have checked. What it costs is the entire interlock: hold() confirms whatever it is
told, so a refused grip stops being detectable and nothing downstream can tell a real hold from
an empty one.

SO `verified` IS FALSE FOR THE WHOLE SESSION when bypassed, and verify() answers None rather
than the probe. The firmware does NOT bypass its `status` query -- only changeStatus -- so a
bypassed board would happily go on giving sensor answers to a re-check, which is precisely the
untrustworthy reading the bypass exists to stop believing. Reporting "nobody checked" is the
honest translation, and it puts callers on the same warning path they take with no board at all.

TWO THINGS THE BOARD DOES AT BOOT THAT A CALLER MUST UNDO BEFORE MATING. Both were found the
hard way, by a pick that reported a successful grip on an object it had never clamped.

  1. THE MOTOR RELAY COMES UP OFF. setup() runs digitalWrite(relayK1, LOW), and opening the
     serial port resets the board -- so every connection starts with the actuation unpowered.
     The servo will swing on command and nothing will happen. set_motor(True) fixes it, and it
     is a TOGGLE on the wire, so it has to be driven to a state rather than pulsed.

  2. THE BOARD ADOPTS THE SENSOR'S OPINION AS ITS STATE. setup() ends with

         status = checkTool() ? 1 : 0;
         changeServo(status > 0);

     so a board that boots with anything in front of the probe comes up believing it is ALREADY
     LOCKED. changeStatus() then only moves the servo when the new status DIFFERS from the old:

         if (newStatus != status) { ... changeServo(...) }

     A hold() in that state is a no-op on the mechanism -- the board re-reads the sensor, still
     sees the object, and answers "1 confirmed!". Nothing clamped, and a confirmation saying it
     did. Forcing a release first makes the status differ, so the following hold is a real
     transition. prepare_to_mate() does both of these.

CLOSING THE PORT DROPS WHATEVER IS HELD, unless `toolchanger.latch` says otherwise. This is the
sharpest edge in the whole driver and it is worth spelling out. Closing the serial port hangs up
DTR, which RESETS the board, and setup() ends with

    status = checkTool() ? 1 : 0;
    changeServo(status > 0);

-- so the board re-decides whether to clamp from a single sensor reading taken microseconds after
power-on. If the probe says "absent" at that instant, the servo swings to noLockAngle and the
coupler OPENS. Anything it was holding falls. That is most likely in exactly the situation the
sensor bypass exists for: a probe untrustworthy enough to be ignored during the run is a probe
that can be wrong at reset too, and the reset clears the bypass before the reading is taken.

`latch: true` clears HUPCL so the port can close without dropping DTR: no reset, no re-decision,
the servo stays where it is and the relay stays powered. The cost is the dead-man switch -- a
crashed or killed process no longer switches the motor off by dying. That is the right trade
whenever the run holds a part that would be damaged or dangerous to drop, and the wrong one for
a run that is only ever empty-handed.

The tool-side GEOMETRY of the same mechanism is configs/frames.yaml's `coupler_mate` frame; this
module is only the actuation.
"""

from .. import log as urlog
from ..apps._common import ask, prompts_off

log = urlog.get('coupler')


class Coupler:
    """hold / release / verify, through the board when it is there and the operator when it is
    not.

    `verified` is True only when the BOARD's proximity sensor agreed with the last command. It
    is False after any operator-prompted action, because nothing checked."""

    def __init__(self, cfg, section='toolchanger', label=None):
        # WHICH BOARD. This cell has two couplers running the same firmware, so a run that drives
        # both reads two identical streams of "1 confirmed!" -- and the one thing a reply never
        # says is which mechanism moved. `section` picks the config block and `label` names it in
        # every line this instance logs, so a refused hold is attributable to a board rather than
        # to "the coupler".
        block = cfg.section(section)
        self.section = section
        self.label = label or ('end effector' if section == 'toolchanger' else section)
        self.log = urlog.get('coupler' if label is None else f'coupler[{label}]')
        self.cfg = cfg
        self.device = None
        self.verified = False
        self.skip_prompts = prompts_off(cfg)
        self.use_motor = bool(block.get('motor', True))
        # OPT-IN, and never assumed to have worked: `bypassed` is set only once the board has
        # said so. Everything that reports confidence keys off it.
        self.bypass_requested = bool(block.get('bypass', False))
        self.bypassed = False
        # LATCH: keep the board alive when the port closes. See the class docstring.
        self.latched = bool(block.get('latch', False))
        # What we believe the relay is doing. Seeded from the boot banner, because seeing it
        # means setup() has just run and left the relay LOW; without the banner we know
        # nothing and have to find out by toggling.
        self._motor_on = None
        if not bool(block.get('enabled', True)) or cfg.get_path('robot.dry_run'):
            self.log.info('Toolchanger driver disabled for the %s coupler -- the operator '
                          'locks and releases it by hand.', self.label)
            return
        try:
            from toolchanger.toolchanger import ToolChanger
            # settle: opening the port RESETS the board, and setup() has to clear the
            # bootloader, average the sensor and swing the servo before it prints its banner.
            # The driver floors this at 2.5 s; anything less just means it waits anyway.
            self.device = ToolChanger(port=block.get('port'),
                                      baud=int(block.get('baud', 9600)),
                                      timeout=float(block.get('timeout_s', 5.0)),
                                      settle=float(block.get('settle_s', 3.0)),
                                      latch=self.latched, name=label)
            if self.latched:
                self.log.info('The %s coupler is LATCHED: its board is left running when this '
                              'process exits, so whatever it holds stays held. Nothing will '
                              'switch the relay off on its own -- release it deliberately, or '
                              'power the board down.', self.label)
            self._motor_on = False if getattr(self.device, 'booted', False) else None
            if not getattr(self.device, 'booted', True):
                self.log.warning('The %s coupler\'s board never printed its boot banner. On '
                                 'older firmware that has no banner the reply stream can be OFF '
                                 'BY ONE, which would make a confirmation belong to the previous '
                                 'command. Treat confirmations from this session with suspicion '
                                 'and reflash if you can.', self.label)
            self.log.info('The %s coupler is connected on %s.', self.label, self.device.port)
            if self.bypass_requested:
                self._enable_bypass()
        except Exception as exc:                     # noqa: BLE001 -- optional hardware
            self.log.warning('The %s coupler\'s driver is unavailable (%s) -- the operator '
                             'locks and releases it by hand, and nothing can confirm the tool '
                             'is seated.', self.label, exc)

    @property
    def manual(self):
        return self.device is None

    def _enable_bypass(self):
        """Turn the board's sensor bypass ON, and confirm it went on.

        The wire command TOGGLES, like the motor relay. Unlike the relay the starting state IS
        known -- a reset clears the bypass and opening the port reset the board -- so one toggle
        should do it; the answer is checked anyway, because a bypass that silently failed to
        engage would leave the run believing the probe is being ignored while it is not."""
        try:
            on = bool(self.device.bypass())
            if not on:                          # already on somehow: put it back, then on again
                on = bool(self.device.bypass())
        except Exception as exc:                # noqa: BLE001
            self.log.error('Could not enable the sensor bypass: %s -- the probe is still in '
                           'charge.', exc)
            return False
        self.bypassed = on
        if not on:
            self.log.error('The board would not enable the sensor bypass. The probe is still '
                           'being consulted, so an untrustworthy one will still refuse grips.')
            return False
        self.log.warning('SENSOR BYPASS ON for the %s coupler this session. The board will '
                         'confirm every hold and release WITHOUT consulting the proximity '
                         'probe, so nothing from here on can tell a real grip from an empty '
                         'one. Whatever is being lifted is being lifted on trust. It clears '
                         'when this connection closes.', self.label)
        return True

    def hold(self):
        """Lock onto the object. False when the board's sensor says nothing is gripped."""
        return self._command('hold', f'LOCK the {self.label} coupler onto the object, then '
                                     'press Enter (q to abort): ')

    def release(self):
        """Unlock. False when the board's sensor says something is still held."""
        return self._command('release', f'RELEASE the {self.label} coupler, then press Enter '
                                        '(q to abort): ')

    def verify(self):
        """Re-ask the board whether the sensor still agrees with its current state.

        An INDEPENDENT second look, worth taking between locking and lifting: hold() confirms
        once, at the instant the servo finished, and a part that was merely resting where the
        probe could see it can settle out again in the moment after. Returns None when there is
        no board, which is not the same answer as False and must not be collapsed into it."""
        if self.device is None:
            return None
        if self.bypassed:
            # The FIRMWARE would still answer this one from the probe -- only changeStatus is
            # bypassed -- but that is the reading the bypass exists to stop believing. None is
            # "nobody checked", which is the truth and puts callers on the same warning path
            # they take with no board at all.
            return None
        try:
            ok = bool(self.device.status())
        except Exception as exc:                     # noqa: BLE001
            self.log.error('The %s coupler status check failed: %s', self.label, exc)
            return False
        self.verified = ok
        return ok

    # ---------------------------------------------------------------- the motor relay
    def set_motor(self, on):
        """Drive relay K1 to a KNOWN state. True when it ended up there.

        The wire command TOGGLES, so a blind call is a coin flip on the result -- which is why
        the relay "sometimes defaults to on". The board does report the state it ended in, and
        there is no way to ask without changing it, so the honest method is: toggle, look at the
        answer, and toggle back if it went the wrong way. At most two round trips, and none at
        all when the state is already known."""
        if not self.use_motor or self.device is None:
            return True
        if self._motor_on is on:
            return True
        try:
            ended = bool(self.device.motor())
            if ended is not on:
                ended = bool(self.device.motor())
        except Exception as exc:                 # noqa: BLE001
            self._motor_on = None
            self.log.error('Could not switch the motor relay: %s', exc)
            return False
        self._motor_on = ended
        if ended is not on:
            self.log.error('The motor relay would not go %s -- it reports %s after two attempts.',
                      'ON' if on else 'OFF', 'ON' if ended else 'OFF')
            return False
        self.log.info('Motor relay %s.', 'ON' if ended else 'OFF')
        return True

    def prepare_to_mate(self):
        """Power the actuation and force the jaws OPEN -- the only state a mate can start from.

        The release is not belt-and-braces. A board that booted with something in front of the
        probe believes it is already locked, and in that state a hold() moves nothing while
        still answering "confirmed!" (see the module docstring). Commanding a release first
        makes the status differ, so the hold that follows is a real servo transition."""
        if not self.set_motor(True):
            return False
        if self.device is None:
            return True
        self.log.info('Opening the %s coupler before the mate (a board that booted with '
                      'something in front of the probe comes up latched, and would no-op the '
                      'hold).', self.label)
        return self.release()

    def _command(self, what, prompt):
        if self.device is not None:
            try:
                ok = bool(getattr(self.device, what)())
            except Exception as exc:                 # noqa: BLE001 -- surfaced, never swallowed
                self.verified = False
                self.log.error('The %s coupler could not %s: %s', self.label, what, exc)
                return False
            self.verified = ok and not self.bypassed
            if self.bypassed:
                self.log.warning('%s coupler %s: the board confirmed, but the sensor is '
                                 'BYPASSED -- nothing checked whether anything is actually '
                                 'held.', self.label, what)
            if not ok:
                # The board printed the raw reading and its own diagnosis already; what it
                # cannot know is that this stops a pick.
                self.log.error('The %s coupler was REFUSED a %s by its board: the proximity '
                               'sensor disagrees with the commanded state. That mechanism is '
                               'not holding what it was asked to hold, so nothing further '
                               'should move.', self.label, what)
            return ok
        self.verified = False
        if self.skip_prompts:
            self.log.warning('%s coupler %s: no driver and no prompts -- assuming the operator '
                             'did it. NOTHING has verified that the tool is seated.',
                             self.label, what)
            return True
        return ask(prompt)

    def close(self):
        if self.device is not None:
            self.device.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
