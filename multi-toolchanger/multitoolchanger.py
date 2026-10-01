#!/usr/bin/env python3
"""Drive the multi-device Arduino toolchanger (firmware/) over serial from the terminal.

    ./multitoolchanger.py --port dc_motor             # interactive prompt
    ./multitoolchanger.py help                        # every command, grouped by module
    ./multitoolchanger.py help ramp                   # one command in full
    ./multitoolchanger.py ramp 0 60 3 --port dc_motor # any command, once, then exit
    ./multitoolchanger.py sequence screwdrive_attach --port dc_motor

WHAT EXISTS DEPENDS ON THE BOARD. The firmware is built with only some modules
(firmware/build_flash.sh) and announces which when the port opens; the script loads exactly
those -- each a file in mtc/modules/ with its device protocol and commands, plus an optional
config/<module>.yaml with its settings and sequences. Commands of modules that are not loaded do
not exist -- not in help, not in completion. With --no-detect the `modules:` list in
config/multitoolchanger.yaml is used instead. Point at a different deployment's config with
--config or $MULTITOOLCHANGER_CONFIG.

At the prompt: Tab completes, up/down walk the history, and q stops a timed command or a
sequence -- anything moving stops and the prompt carries on as normal.

Opening the port pulls DTR and RESETS the Arduino, so every device starts in its safe state
and we wait for the boot banner before sending (--settle). Closing it resets the board again,
which stops everything -- a dead-man switch -- unless --latch.

From Python:

    import sys; sys.path.insert(0, 'multi-toolchanger')
    from mtc import ToolChanger
    with ToolChanger('dc_motor') as tc:
        tc.screwdrive.run(40, 2.5)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mtc.base import MAX_RUN_S, Interrupted, ToolChangerError  # noqa: E402
from mtc.board import COUPLERS, serial  # noqa: E402
from mtc.config import CONFIG_YAML, available_modules, load_config  # noqa: E402
from mtc.modules.general import QuitPrompt  # noqa: E402
from mtc.session import Session, ToolChanger, complete_line, lookup, parse_args  # noqa: E402

#  the prompt's command history, kept across sessions; up/down arrows walk through it
HISTORY_FILE = os.path.expanduser('~/.multitoolchanger_history')
HISTORY_LENGTH = 500


def _enable_readline(cfg):
    """Arrow-key history, line editing and Tab completion for input(), via readline.

    Never fatal: without readline (not every Python has it) or a writable home directory the
    prompt just has no history or completion."""
    try:
        import atexit
        import readline
    except ImportError:
        return
    try:
        readline.read_history_file(HISTORY_FILE)
    except OSError:
        pass                                    # first run: no history yet
    readline.set_history_length(HISTORY_LENGTH)

    def save():
        try:
            readline.write_history_file(HISTORY_FILE)
        except OSError:
            pass
    atexit.register(save)

    def completer(text, state):
        upto = readline.get_line_buffer()[:readline.get_endidx()]
        options = complete_line(upto, cfg)
        return options[state] if state < len(options) else None

    readline.set_completer(completer)
    readline.set_completer_delims(' \t\n')
    # macOS and some Python builds link libedit instead of GNU readline, which binds differently
    if 'libedit' in (readline.__doc__ or ''):
        readline.parse_and_bind('bind ^I rl_complete')
    else:
        readline.parse_and_bind('tab: complete')


def prompt_name(session):
    """What the prompt shows, first that applies: the couplers.yaml name it was opened by
    (--port cleat), the name it was flashed with (build_flash.sh --name), or `mtc` with its
    modules -- `mtc(coupler,screwdrive)` -- so two unnamed boards still differ."""
    if session.tc.board.name:
        return session.tc.board.name
    if session.tc.board_name:
        return session.tc.board_name
    return f"mtc({','.join(m for m in session.cfg.modules if m != 'general')})"


def repl(session):
    """The interactive prompt. Runs commands through the same Session as the CLI."""
    _enable_readline(session.cfg)
    board = session.tc.board
    where = board.name or board.port
    if session.tc.board_name and session.tc.board_name != board.name:
        where = f'{session.tc.board_name} on {where}'
    print(f"Connected to {where} -- "
          f"{', '.join(m for m in session.cfg.modules if m != 'general') or 'no modules'} "
          f"({'detected on the board' if session.tc.source == 'detected' else 'from the config'})."
          f" `help` lists commands; Tab completes; `quit` or Ctrl-D leaves.")
    while True:
        try:
            line = input(f'{prompt_name(session)}> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        tokens = line.split()
        tokens[0] = tokens[0].lower()
        try:
            session.execute(tokens)
        except QuitPrompt:
            return
        except ToolChangerError as exc:          # includes Interrupted: report, carry on
            print(f"  {exc}")
        except KeyboardInterrupt:
            print('\n  Ctrl-C -- stopping and letting go')
            session.safe_stop()
            session.tc.let_go()                  # unlike q: nothing is left holding


def main():
    ap = argparse.ArgumentParser(
        description='Drive the multi-device Arduino toolchanger over serial.',
        epilog='Run `help` (as a command) for every command this configuration loads.')
    ap.add_argument('command', nargs='?', metavar='COMMAND',
                    help='a command to run once, e.g. hold, ramp, sequence. Omit for an '
                         'interactive prompt. `help` lists them all.')
    ap.add_argument('values', nargs='*', metavar='ARG',
                    help="the command's arguments, e.g. `ramp 0 60 3`")
    ap.add_argument('--port', metavar='NAME|SERIAL|PATH',
                    help='WHICH BOARD. A name from configs/couplers.yaml, a USB serial (or '
                         'enough of one to be unambiguous), or a device path. With one board '
                         'plugged in this can be omitted; with several it cannot, because they '
                         'are identical over the wire. `list` shows what is connected.')
    ap.add_argument('--no-detect', action='store_true',
                    help="load the config's `modules:` list instead of the modules the board's "
                         'firmware reports (it warns if they differ)')
    ap.add_argument('--config', metavar='PATH',
                    help=f'the main config (default {os.path.relpath(CONFIG_YAML)}, or '
                         f'$MULTITOOLCHANGER_CONFIG); module configs sit beside it')
    ap.add_argument('--baud', type=int, default=115200,
                    help='must match Serial.begin() in firmware/main.cpp (default 115200)')
    ap.add_argument('--timeout', type=float, default=5.0, help='seconds to wait for a reply')
    ap.add_argument('--settle', type=float, default=3.0,
                    help='seconds to wait for the board to boot (it prints a banner when '
                         'ready)')
    ap.add_argument('-v', '--verbose', action='store_true', help='show the raw bytes and lines')
    ap.add_argument('--latch', action='store_true',
                    help='leave the board running when this exits, so motors STAY on. Without '
                         'it, closing the port resets the board and stops everything -- which '
                         'doubles as a dead-man switch, and `drive` then runs until q or '
                         'Ctrl-C.')
    args = ap.parse_args()

    detect = not args.no_detect
    try:
        cfg = load_config(args.config)
        session = Session(cfg)
        # Checked before the port is opened: opening it resets the board and moves the servo,
        # so a typo should cost nothing. With detect on, the board decides the modules, so the
        # check is against every module this script knows; the real check follows connecting.
        tokens = [args.command.lower()] + args.values if args.command else None
        cmd = None
        if tokens:
            known = (load_config(args.config, modules=[m for m in available_modules()
                                                       if m != 'general'])
                     if detect else cfg)
            cmd = lookup(tokens[0], known)
            values = parse_args(cmd, tokens[1:], known)
            if not cmd.board:
                # With detect on, the board decides the modules, so `help` without a board
                # describes every module this script has.
                ok = (Session(known) if detect else session).execute(tokens)
                if detect and cmd.name == 'help' and not values:
                    print('(Without a board: every module this script knows. Connected, the '
                          'board decides.)')
                sys.exit(0 if ok else 2)
    except QuitPrompt:
        sys.exit(0)
    except ToolChangerError as exc:
        sys.exit(str(exc))

    try:
        session.tc = ToolChanger(args.port, None if detect else cfg, detect=detect,
                                 config_path=args.config, baud=args.baud, timeout=args.timeout,
                                 settle=args.settle, verbose=args.verbose, latch=args.latch,
                                 name=args.port if args.port in COUPLERS else None)
    except serial.SerialException as exc:
        sys.exit(f"Cannot open the port: {exc}\n"
                 f"Check `ls /dev/ttyACM* /dev/ttyUSB*`, that you are in the dialout group, "
                 f"and that the Arduino IDE serial monitor is closed.")
    except ToolChangerError as exc:
        sys.exit(str(exc))
    session.cfg = cfg = session.tc.config
    if session.tc.source == 'detected':
        print(f"detected modules: {', '.join(m for m in cfg.modules if m != 'general') or 'none'}")
    if cmd is not None and cmd.name not in cfg.commands:
        session.tc.close()
        sys.exit(f"{cmd.name} is a {cmd.module} command, and this board's firmware has no "
                 f"{cmd.module} (it has: {', '.join(session.tc.detected) or 'nothing'})")
    if cmd is not None:
        try:
            values = parse_args(cmd, tokens[1:], cfg)   # again: settings may differ by board
        except ToolChangerError as exc:
            session.tc.close()
            sys.exit(str(exc))

    try:
        if cmd is None:
            repl(session)
            sys.exit(0)
        ok = session.execute(tokens)
        if ok and cmd.runs_on and cmd.runs_on(values, cfg) and not args.latch:
            # closing the port resets the board and stops it, so without --latch whatever the
            # command left running lasts exactly as long as this process: hold it open
            print('   running until q or Ctrl-C (or use --latch to leave it running)')
            with session.watching():
                while True:
                    session.wait(MAX_RUN_S)
        sys.exit(0 if ok else 2)
    except Interrupted as exc:
        print(f"  {exc}")
        sys.exit(0 if cmd.runs_on else 3)
    except KeyboardInterrupt:
        print()
        session.safe_stop()
        sys.exit(130)
    except ToolChangerError as exc:
        sys.exit(str(exc))
    finally:
        session.tc.close()


if __name__ == '__main__':
    main()
