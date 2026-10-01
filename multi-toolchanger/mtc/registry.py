"""How a module declares itself: its commands, argument kinds, settings and device.

Every module file in mtc/modules/ makes one `MODULE = Module(...)` and registers onto it:

    MODULE = Module('screwdrive', 'what it is', device=Screwdrive)
    MODULE.setting('max_rpm', 500, check=positive_number)

    @MODULE.command(Param('PCT', 'percent'), timed=True)
    def cmd_run(s, pct):
        \"\"\"One-line summary, shown by `help`.

        Longer explanation, shown by `help run`.\"\"\"
        s.dev('screwdrive').run(pct)

THE DOCSTRING IS THE HELP TEXT, so help cannot drift from the code, and a command without one
will not load. The config file lists which modules to load; only those modules' commands exist.
"""

import inspect
import os
import types

from .base import MAX_RUN_S, ToolChangerError, parse_ms


class Kind:
    """An argument type: how to check and convert it, describe it, and Tab-complete it."""

    def __init__(self, parse, describe, complete=None):
        self.parse = parse            # (text, cfg, name) -> value, or raise ToolChangerError
        self.describe = describe      # (cfg) -> str, for `help <command>`
        self.complete = complete      # (cfg) -> [str], or None for "nothing to offer"


class Param:
    def __init__(self, name, kind, optional=False):
        self.name, self.kind, self.optional = name, kind, optional

    def __str__(self):
        return f'[{self.name}]' if self.optional else self.name


class Command:
    def __init__(self, func, module, params, board, in_sequence, timed, runs_on):
        self.name = func.__name__[len('cmd_'):]
        self.func, self.module, self.params = func, module, params
        self.board = board              # needs an open connection
        self.in_sequence = in_sequence  # may be a sequence step
        self.timed = timed              # runs for a while: q stops it
        self.runs_on = runs_on          # (args, cfg) -> True if it leaves something running

    @property
    def usage(self):
        return ' '.join([self.name] + [str(p) for p in self.params])

    @property
    def summary(self):
        return (inspect.getdoc(self.func) or '').split('\n', 1)[0]

    @property
    def details(self):
        doc = (inspect.getdoc(self.func) or '').split('\n', 1)
        return doc[1].strip() if len(doc) > 1 else ''


class Module:
    """One device (or the shell itself, for `general`): commands, kinds, settings, device.

    `device` is a class built once per connection as device(board, settings). It may define
    `safe_stop()` -- called when q is pressed or a sequence fails, to stop anything moving --
    `let_go()` -- called on Ctrl-C and whenever the connection closes, to stop HOLDING anything
    (the t74 releases its motor) -- and `check_line(line)`, which raises on a message the board
    sends unasked that should fail a wait (the coupler watchdog's emergency stop)."""

    def __init__(self, name, description, device=None, always=False):
        self.name = name
        self.description = description
        self.device = device
        self.always = always        # loaded whatever the config says (only `general`)
        self.commands = {}
        self.kinds = {}
        self._settings = {}         # name -> (default, check)
        self.pins = {}              # role -> (default pin, what it needs) -- see mtc/pins.py
        self.claims = []            # hardware it takes over whole, e.g. 'timer1' 

    def command(self, *params, board=True, in_sequence=True, timed=False, runs_on=None):
        """Register the decorated `cmd_<name>(session, *args)` as the command <name>.

        The function returns False to report failure (a sequence then aborts); anything else
        is success. Arguments arrive already checked and converted by their Kind.

        board=False      runs without a connection
        in_sequence=False  not allowed as a sequence step
        timed=True       runs for a while; q stops it
        runs_on=f        f(args, cfg) is True when the command leaves something running after it
                         returns -- the one-shot CLI then holds the connection open until q,
                         since closing it resets the board"""
        def register(func):
            assert func.__name__.startswith('cmd_'), func.__name__
            assert inspect.getdoc(func), f'{func.__name__} needs a docstring -- it is the help'
            cmd = Command(func, self.name, params, board, in_sequence, timed, runs_on)
            assert cmd.name not in self.commands, cmd.name
            self.commands[cmd.name] = cmd
            return func
        return register

    def kind(self, name, parse, describe, complete=None):
        """Register an argument kind this module's commands use, e.g. 'rpm'."""
        self.kinds[name] = Kind(parse, describe, complete)

    def pin(self, role, default, needs='digital'):
        """Declare a pin the firmware module uses. The config's `pins:` block may move it;
        `needs` is what the pin must be able to do -- 'digital', 'pwm', 'analog' (an ADC input),
        'interrupt' (INT0/INT1) or 'pcint' (a pin-change interrupt). build_flash.sh checks every
        loaded module's pins against the board and each other, and compiles them in as
        PIN_<MODULE>_<ROLE>."""
        self.pins[role] = (default, needs)

    def claim(self, resource):
        """Declare hardware this module takes over entirely, e.g. 'timer1' (the Servo library,
        or a control-loop interrupt). Two modules claiming one resource cannot share a board, and
        PWM on a claimed timer's pins is refused."""
        self.claims.append(resource)

    def configure_pins(self, raw, where):
        """{role: arduino pin number} from the config's `pins:` block, defaults filled in.
        Fatal on an unknown role or a pin name the board does not have."""
        from .pins import parse_pin       # pins imports this module
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ToolChangerError(f'{where}: pins must be a mapping of role -> pin')
        unknown = sorted(set(raw) - set(self.pins))
        if unknown:
            raise ToolChangerError(f'{where}: unknown pin(s) {", ".join(map(str, unknown))} for '
                                   f'{self.name} (it has: {", ".join(self.pins) or "none"})')
        return {role: parse_pin(raw.get(role, default), f'{where}: {self.name} {role}')
                for role, (default, _) in self.pins.items()}

    def setting(self, name, default, check=None):
        """Declare a setting read from the module's config file. `check(value)` returns an
        error message, or None if the value is fine."""
        self._settings[name] = (default, check)

    def configure(self, raw, where):
        """Settings from the module's config `settings:` block, defaults filled in.

        Fatal on an unknown key or a bad value: a typo'd setting silently falling back to its
        default is exactly the kind of thing that is found out the hard way."""
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ToolChangerError(f'{where}: settings must be a mapping')
        unknown = sorted(set(raw) - set(self._settings))
        if unknown:
            known = ', '.join(sorted(self._settings)) or 'none'
            raise ToolChangerError(f'{where}: unknown setting(s) {", ".join(unknown)} for '
                                   f'{self.name} (it has: {known})')
        values = {}
        for name, (default, check) in self._settings.items():
            value = raw.get(name, default)
            problem = check(value) if check else None
            if problem:
                raise ToolChangerError(f'{where}: {self.name} {name} {problem}, got {value!r}')
            values[name] = value
        #  where the module's config lives, for modules that keep measured state beside it
        values['config_dir'] = os.path.dirname(os.path.abspath(where))
        return types.SimpleNamespace(**values)


def positive_number(value):
    """A setting check: a number >= 1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 1:
        return 'must be a number >= 1'
    return None


#  =====   kinds every config has   =====
def _parse_sequence(text, cfg, name):
    """A runnable sequence. A broken one is refused here, before the port is even opened."""
    seq = cfg.sequences.get(text)
    if seq is None:
        known = ', '.join(sorted(cfg.sequences)) or 'none are configured'
        raise ToolChangerError(f'no sequence called {text!r}. Sequences: {known}')
    if seq.errors:
        raise ToolChangerError(f'sequence {text} cannot run ({seq.path}):\n'
                               + '\n'.join(f'    {e}' for e in seq.errors))
    return text


def _parse_topic(text, cfg, name):
    if text not in cfg.commands and text not in cfg.modules:
        raise ToolChangerError(f'nothing called {text!r} -- try `help` for the full list')
    return text


CORE_KINDS = {
    'seconds': Kind(lambda t, cfg, n: parse_ms(t, n) / 1000,
                    lambda cfg: f'seconds, 0.001..{MAX_RUN_S}, decimals allowed'),
    'sequence': Kind(_parse_sequence,
                     lambda cfg: 'a sequence name from the config',
                     lambda cfg: list(cfg.sequences)),
    'topic': Kind(_parse_topic,
                  lambda cfg: 'a command or module name',
                  lambda cfg: list(cfg.commands) + list(cfg.modules)),
}
