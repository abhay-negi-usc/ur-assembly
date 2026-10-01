"""Running commands: the connection with its devices, q to stop, sequences, help, completion.

The prompt, the one-shot CLI and sequences all go through Session.execute(), so they cannot
disagree about what a command means.
"""

import contextlib
import difflib
import os
import select
import sys
import time

from .base import Interrupted, ToolChangerError
from .board import Board
from .config import MAIN, load_config


class ToolChanger:
    """A board connection plus a device object for every module the config loads.

        from mtc import ToolChanger
        with ToolChanger('dc_motor') as tc:
            tc.screwdrive.run(40, 2.5)
            tc.coupler.hold()

    Board options (baud, timeout, settle, verbose, latch, name) pass straight to Board."""

    def __init__(self, port=None, config=None, **board_options):
        self.config = config or load_config()
        self.board = Board(port, **board_options)
        self.devices = {name: module.device(self.board, self.config.settings[name])
                        for name, module in self.config.modules.items() if module.device}

    def __getattr__(self, name):
        devices = self.__dict__.get('devices', {})
        if name in devices:
            return devices[name]
        raise AttributeError(f'{name!r} -- not a loaded module or a ToolChanger attribute '
                             f'(loaded: {", ".join(devices) or "none"})')

    def close(self):
        self.board.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


#  =====   q to stop   =====
class KeyWatcher:
    """Notices a key press (q) without waiting for Enter, for as long as it is entered.

    Puts the terminal in cbreak mode -- keys arrive one at a time and are not echoed -- and
    restores it however the block exits, dropping anything typed meanwhile so a stray q does
    not land at the next prompt. Ctrl-C still works as usual. When stdin is not a terminal
    (piped input, tests) it watches nothing."""

    def __init__(self, keys=b'qQ'):
        self.keys = keys
        self.fd = None
        self._saved = None

    @property
    def active(self):
        return self._saved is not None

    def __enter__(self):
        try:
            fd = sys.stdin.fileno()
            if os.isatty(fd):
                import termios
                import tty
                self._saved = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                self.fd = fd
        except (OSError, ValueError, AttributeError):
            self._saved = None
        return self

    def pressed(self):
        """True if q has been pressed since the last call. Never blocks."""
        if not self.active:
            return False
        hit = False
        while select.select([self.fd], [], [], 0)[0]:
            data = os.read(self.fd, 64)
            if not data:
                break
            hit = hit or any(k in data for k in self.keys)
        return hit

    def __exit__(self, *exc):
        if self.active:
            import termios
            termios.tcflush(self.fd, termios.TCIFLUSH)
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            self._saved = None
        return False


#  =====   commands   =====
def lookup(word, cfg):
    """The Command called exactly `word`. Close misses are suggested, never guessed."""
    cmd = cfg.commands.get(word)
    if cmd is None:
        near = difflib.get_close_matches(word, list(cfg.commands), n=3)
        raise ToolChangerError(f'unknown command {word!r}'
                               + (f' -- did you mean {" or ".join(near)}?' if near else '')
                               + ' `help` lists them all.')
    return cmd


def parse_args(cmd, values, cfg):
    """Check and convert a command's arguments. Raises ToolChangerError with the usage."""
    required = sum(not p.optional for p in cmd.params)
    if not required <= len(values) <= len(cmd.params):
        raise ToolChangerError(f'usage: {cmd.usage}')
    return [cfg.kinds[p.kind].parse(v, cfg, p.name) for p, v in zip(cmd.params, values)]


class Session:
    def __init__(self, cfg, tc=None, watcher=KeyWatcher):
        self.cfg = cfg
        self.tc = tc
        self.watcher_factory = watcher
        self._watcher = None

    def dev(self, module):
        """The connected device for `module`."""
        if self.tc is None:
            raise ToolChangerError(f'{module} needs a board, and none is connected')
        return self.tc.devices[module]

    def execute(self, tokens):
        """Run one command from its words. Returns False if it reported failure."""
        cmd = lookup(tokens[0], self.cfg)
        args = parse_args(cmd, tokens[1:], self.cfg)
        if cmd.board and self.tc is None:
            raise ToolChangerError(f'{cmd.name} needs a board, and none is connected')
        if not cmd.timed:
            return cmd.func(self, *args) is not False
        with self.watching():
            return cmd.func(self, *args) is not False

    @contextlib.contextmanager
    def watching(self):
        """Watch for q while the block runs. Nested blocks share the outermost watcher, so a
        sequence's steps are covered by the one the sequence opened."""
        if self._watcher is not None:
            yield
            return
        with self.watcher_factory() as watcher:
            if watcher.active:
                print('   (press q to stop)')
            self._watcher = watcher
            if self.tc:
                self.tc.board.poll_hook = self.check_interrupt
            try:
                yield
            finally:
                self._watcher = None
                if self.tc:
                    self.tc.board.poll_hook = None

    def check_interrupt(self):
        """Called while waiting: if q was pressed, stop everything and raise Interrupted."""
        if self._watcher is None or not self._watcher.pressed():
            return
        print('\n  q -- stopping')
        stopped = self.safe_stop()
        raise Interrupted('interrupted' + (f' -- {", ".join(stopped)} stopped' if stopped else ''))

    def safe_stop(self):
        """Stop every device that can be stopped, best effort. Returns the ones stopped."""
        if self.tc is None:
            return []
        stopped = []
        hook, self.tc.board.poll_hook = self.tc.board.poll_hook, None   # must not recurse
        try:
            for name, dev in self.tc.devices.items():
                if hasattr(dev, 'safe_stop'):
                    try:
                        dev.safe_stop()
                        stopped.append(name)
                    except ToolChangerError as exc:
                        print(f'  WARNING: could not stop {name}: {exc}')
        finally:
            self.tc.board.poll_hook = hook
        return stopped

    def wait(self, seconds):
        """Sleep, printing what the board says. q interrupts; a device's check_line() can fail
        it (the coupler does, on an emergency stop from its watchdog)."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            self.check_interrupt()
            if self.tc is None:
                time.sleep(min(0.05, max(0.0, deadline - time.time())))
                continue
            line = self.tc.board.read_line()
            if line:
                print(f'   {line}')
                for dev in self.tc.devices.values():
                    if hasattr(dev, 'check_line'):
                        dev.check_line(line)


#  =====   sequences   =====
def validate_steps(seq, cfg):
    """Every problem with a sequence's steps, as strings. Empty means it can run.

    A module's own sequence may only use that module's commands (and general ones such as
    wait); anything spanning modules belongs in the main config."""
    errors = []
    for i, step in enumerate(seq.steps, 1):
        words = step.split()
        try:
            if not words:
                raise ToolChangerError('empty step')
            cmd = lookup(words[0], cfg)
            if not cmd.in_sequence:
                raise ToolChangerError(f'{cmd.name} cannot be used in a sequence')
            if seq.source != MAIN and cmd.module not in (seq.source, 'general'):
                raise ToolChangerError(f'{cmd.name} is a {cmd.module} command; a sequence using '
                                       f'more than one module belongs in the main config')
            parse_args(cmd, words[1:], cfg)
        except ToolChangerError as exc:
            errors.append(f'step {i} ({step!r}): {exc}')
    return errors


def run_sequence(s, name):
    """Run every step of sequence `name` in order; stop everything if any of it fails."""
    seq = s.cfg.sequences[name]
    if seq.errors:
        raise ToolChangerError(f'sequence {name} cannot run ({seq.path}):\n'
                               + '\n'.join(f'    {e}' for e in seq.errors))
    print(f'sequence {name}: {len(seq.steps)} steps'
          + (f' -- {seq.description}' if seq.description else ''))
    for i, step in enumerate(seq.steps, 1):
        where = f'step {i}/{len(seq.steps)} ({step})'
        print(f'[{name} {i}/{len(seq.steps)}] {step}')
        try:
            ok, why = s.execute(step.split()), 'it reported a failure'
        except Interrupted as exc:
            raise Interrupted(f'sequence {name}: {exc} (at {where})')
        except ToolChangerError as exc:
            ok, why = False, str(exc)
        if not ok:
            stopped = s.safe_stop()
            raise ToolChangerError(f'sequence {name} aborted at {where}: {why}.'
                                   + (f' Stopped: {", ".join(stopped)}.' if stopped else ''))
    print(f'sequence {name}: done')


#  =====   help   =====
def render_help(cfg, topic=None):
    """The text `help` prints: everything, one module, or one command."""
    if topic in cfg.commands:
        cmd = cfg.commands[topic]
        out = [f'{cmd.usage}    ({cmd.module})', f'  {cmd.summary}']
        if cmd.details:
            out += [''] + [f'  {ln}' for ln in cmd.details.splitlines()]
        if cmd.params:
            out += ['', '  arguments:']
            out += [f'    {str(p):<10} {cfg.kinds[p.kind].describe(cfg)}' for p in cmd.params]
        notes = []
        if cmd.timed:
            notes.append('press q to stop it')
        if not cmd.in_sequence:
            notes.append('not allowed in sequences')
        if not cmd.board:
            notes.append('works without a board')
        if notes:
            out += ['', '  ' + '; '.join(notes) + '.']
        return '\n'.join(out)

    modules = [topic] if topic in cfg.modules else list(cfg.modules)
    width = max(len(name) for name in [c.usage for c in cfg.commands.values()]
                + list(cfg.sequences)) + 3
    out = []
    for name in modules:
        out.append(f'{name} -- {cfg.modules[name].description}')
        out += [f'  {c.usage:<{width}}{c.summary}'
                for c in cfg.commands.values() if c.module == name]
        out.append('')
    if topic is None or topic in cfg.modules:
        for source in modules + ([MAIN] if topic is None else []):
            seqs = [q for q in cfg.sequences.values() if q.source == source]
            if not seqs:
                continue
            out.append(f'sequences from {os.path.relpath(seqs[0].path)} -- '
                       f'run with `sequence NAME`')
            for q in seqs:
                state = (f'[CANNOT RUN: {len(q.errors)} problem(s) -- `sequence {q.name}` '
                         f'says what] ' if q.errors else '')
                out.append(f'  {q.name:<{width}}{state}{q.description}')
            out.append('')
    if topic is None:
        out.append('`help COMMAND` or `help MODULE` for details. Tab completes; q stops a '
                   'timed command or a sequence.')
    return '\n'.join(out).rstrip()


#  =====   Tab completion   =====
def complete_line(line, cfg):
    """Candidates for the word being typed at the end of `line`, each with a trailing space.

    The first word completes to a command; later words complete from their argument's Kind --
    sequence names after `sequence`, commands and modules after `help`, nothing for numbers."""
    words = line.split()
    if not line or line[-1].isspace():
        words.append('')
    text, before = words[-1], words[:-1]
    if not before:
        pool = list(cfg.commands)
    else:
        cmd = cfg.commands.get(before[0].lower())
        index = len(before) - 1
        if cmd is None or index >= len(cmd.params):
            return []
        kind = cfg.kinds[cmd.params[index].kind]
        pool = kind.complete(cfg) if kind.complete else []
    return sorted(w + ' ' for w in pool if w.startswith(text))
