"""general -- the shell's own commands: help, sequences, waiting, the connection.

Always loaded, and not listed in the config: a toolchanger without `help`, `sequence` or `quit`
is not a smaller configuration, it is a broken one. It lives in its own file anyway so that
EVERY command is defined the same way, in a module file, with no special case in the core.
"""

from ..base import ToolChangerError
from ..board import describe_ports
from ..registry import Module, Param
from ..session import render_help, run_sequence

MODULE = Module('general', 'the shell itself; always loaded', always=True)


class QuitPrompt(Exception):
    """Raised by `quit` to leave the interactive prompt."""


def _parse_help_sub(text, cfg, name):
    if text != 'sequence':
        raise ToolChangerError(f'`help MODULE {text}`: the only thing after a module is '
                               f'`sequence`')
    return text


MODULE.kind('help_sub', _parse_help_sub, lambda cfg: '`sequence`: that module\'s sequences',
            lambda cfg: ['sequence'])


@MODULE.command(Param('TOPIC', 'topic', optional=True),
                Param('sequence', 'help_sub', optional=True), board=False, in_sequence=False)
def cmd_help(s, topic=None, sub=None):
    """List the general commands and the modules, or describe one module, command or sequences.

    `help` lists the general commands and the modules loaded. `help MODULE` lists a module's
    commands; `help COMMAND` shows one in full. `help sequence` lists every sequence, grouped
    by the file it comes from; `help MODULE sequence` just that module's."""
    print(render_help(s.cfg, topic, sub))


@MODULE.command(board=False, in_sequence=False)
def cmd_list(s):
    """Show which boards are plugged in, and their names from couplers.yaml."""
    return describe_ports()


@MODULE.command(Param('NAME', 'sequence'), timed=True)
def cmd_sequence(s, name):
    """Run a named sequence of commands from the config, start to finish.

    A module's sequences live in config/<module>.yaml and are named <module>_<something>;
    sequences spanning modules live in config/multitoolchanger.yaml. A step may itself be
    `sequence OTHER`, which runs OTHER in full: a module's sequence may nest that module's
    sequences, the main config's may nest any, and none may end up running itself.

    Every step -- nested ones too -- is checked before anything runs. A step that fails -- a
    hold that grips nothing, an emergency stop -- aborts the rest, nested or not, and stops
    anything moving. Press q at any point to stop: moving devices stop and the prompt carries
    on as normal."""
    run_sequence(s, name)


@MODULE.command(Param('SECONDS', 'seconds'), board=False, timed=True)
def cmd_wait(s, seconds):
    """Pause for SECONDS. Mostly for sequences; press q to cut it short.

    The board's messages are printed while waiting, and a device can fail the wait on one --
    the coupler does on an emergency stop from its watchdog, which stops a sequence."""
    s.wait(seconds)


@MODULE.command(in_sequence=False)
def cmd_monitor(s):
    """Print whatever the board sends until Ctrl-C."""
    s.tc.board.monitor()


@MODULE.command(board=False, in_sequence=False)
def cmd_quit(s):
    """Leave the interactive prompt (Ctrl-D also works)."""
    raise QuitPrompt
