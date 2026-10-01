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
from .config import MAIN, PROTOCOL, available_modules, load_config


def _warn(message):
    print(f'  warning: {message}', file=sys.stderr)


class ToolChanger:
    """A board connection plus a device object for every module loaded.

        from mtc import ToolChanger
        with ToolChanger('dc_motor') as tc:
            tc.screwdrive.run(40, 2.5)
            tc.coupler.hold()

    WHICH MODULES: by default (detect=True) the ones the board's firmware says it was built
    with; their settings and sequences still come from the config directory. detect=False uses
    the config's `modules:` list, and warns if the board disagrees. Firmware too old to say
    falls back to the config list either way.

    `config` is a loaded Config, or None to load `config_path` (default: the usual one). Board
    options (baud, timeout, settle, verbose, latch, name) pass straight to Board.

    `board_name` is the name the board was flashed with (build_flash.sh --name), or None."""

    def __init__(self, port=None, config=None, detect=True, config_path=None, **board_options):
        self.board = Board(port, **board_options)
        try:
            #  the board's modules (or None), protocol, and the name it was flashed with
            self.detected, proto, self.board_name = self.board.identify()
            if proto is not None and proto != PROTOCOL:
                raise ToolChangerError(
                    f'the board speaks protocol {proto} and this script speaks {PROTOCOL} -- '
                    f'reflash it: cd firmware && ./build_flash.sh upload')
            path = config_path or (config.path if config else None)
            if detect and self.detected is not None:
                known = set(available_modules())
                unknown = [m for m in self.detected if m not in known]
                if unknown:
                    _warn(f'the board has {", ".join(unknown)}, which this script has no module '
                          f'for (mtc/modules/) -- skipped')
                config = load_config(path, modules=[m for m in self.detected if m in known])
                self.source = 'detected'
            else:
                if detect and not self.board.booted:
                    _warn(f'no boot banner at {self.board.baud} baud -- the board may run firmware '
                          f'from before the switch to 115200 (protocol 3). Using the config list. '
                          f'Reflash: cd firmware && ./build_flash.sh upload')
                elif detect:
                    _warn('the board does not say which modules it has (firmware from before '
                          'protocol 2?) -- using the config list. Reflash to fix.')
                config = config or load_config(path)
                self.source = 'config'
                if self.detected is not None:
                    self._compare(config)
            self.config = config
            self.devices = {name: module.device(self.board, config.settings[name])
                            for name, module in config.modules.items() if module.device}
        except BaseException:
            self.board.close()
            raise

    def _compare(self, config):
        """Warn when the config's modules and the board's firmware disagree (--no-detect)."""
        wanted = [m for m in config.modules if m != 'general']
        missing = [m for m in wanted if m not in self.detected]
        extra = [m for m in self.detected if m not in wanted]
        if missing:
            _warn(f'the config loads {", ".join(missing)} but the board\'s firmware does not have '
                  f'{"it" if len(missing) == 1 else "them"} -- those commands will time out. '
                  f'Reflash (cd firmware && ./build_flash.sh upload) or drop --no-detect.')
        if extra:
            _warn(f'the board also has {", ".join(extra)}, which the config does not load.')

    def __getattr__(self, name):
        devices = self.__dict__.get('devices', {})
        if name in devices:
            return devices[name]
        raise AttributeError(f'{name!r} -- not a loaded module or a ToolChanger attribute '
                             f'(loaded: {", ".join(devices) or "none"})')

    def let_go(self):
        """Ctrl-C or leaving: every device stops holding anything (the t74 releases its motor).
        Best effort -- a device that does not answer is warned about, not fatal, so the rest
        still let go and the port still closes. Returns the devices that let go."""
        done = []
        hook, self.board.poll_hook = self.board.poll_hook, None   # q must not interrupt it
        try:
            for name, dev in self.__dict__.get('devices', {}).items():
                if hasattr(dev, 'let_go'):
                    try:
                        dev.let_go()
                        done.append(name)
                    except Exception as exc:             # serial errors too: closing anyway
                        print(f'  WARNING: {name} did not let go: {exc}')
        finally:
            self.board.poll_hook = hook
        return done

    def close(self):
        """Let go of everything, then close the port. Even with --latch (which leaves the
        board running), nothing is left holding."""
        if self.board.ser.is_open:
            self.let_go()
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
        self._running = []      # [name, step, steps] of each sequence running, outermost first

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
    """Every problem with a sequence's own steps, as strings. Empty means they are all valid
    (whether the sequences it nests can run is check_nesting's job, once all are validated).

    A module's own sequence may only use that module's commands (and general ones such as
    wait), and only nest that module's sequences; anything spanning modules belongs in the
    main config, which may nest any sequence."""
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
            if cmd.name == 'sequence':
                _check_nested(seq, words[1:], cfg)
            else:
                parse_args(cmd, words[1:], cfg)
        except ToolChangerError as exc:
            errors.append(f'step {i} ({step!r}): {exc}')
    return errors


def _check_nested(seq, args, cfg):
    """A `sequence NAME` step: NAME exists, and is in reach of `seq`. Not whether NAME can run
    -- that depends on sequences not validated yet, and check_nesting settles it after."""
    if len(args) != 1:
        raise ToolChangerError('usage: sequence NAME')
    inner = cfg.sequences.get(args[0])
    if inner is None:
        raise ToolChangerError(f'no sequence called {args[0]!r}')
    if seq.source != MAIN and inner.source != seq.source:
        raise ToolChangerError(f'{inner.name} is a {inner.source} sequence; a sequence using more '
                               f'than one module belongs in the main config')


def nested_names(seq):
    """The sequences `seq` runs as steps, in order: [(step number, name)]."""
    return [(i, words[1]) for i, words in enumerate((s.split() for s in seq.steps), 1)
            if len(words) == 2 and words[0] == 'sequence']


def check_nesting(cfg):
    """After every sequence's own steps are validated: refuse loops (a sequence that ends up
    running itself would never finish), and mark a sequence that nests one that cannot run as
    unable to run too -- so `help` says so, and nothing starts only to fail halfway."""
    calls = {name: [n for _, n in nested_names(q) if n in cfg.sequences]
             for name, q in cfg.sequences.items() if not q.errors}
    looped = set()
    state = {}                                  # name -> 'open' while on the path, then 'done'

    def visit(name, path):
        state[name] = 'open'
        for nxt in calls.get(name, []):
            if state.get(nxt) == 'open':
                loop = path[path.index(nxt):] + [nxt]
                for member in loop[:-1]:
                    if member not in looped:
                        looped.add(member)
                        cfg.sequences[member].errors.append(
                            f'it runs itself: {" -> ".join(loop)}')
            elif nxt not in state:
                visit(nxt, path + [nxt])
        state[name] = 'done'

    for name in calls:
        if name not in state:
            visit(name, [name])

    changed = True
    while changed:                              # broken-ness climbs to whatever nests it
        changed = False
        for name, q in cfg.sequences.items():
            if q.errors:
                continue
            for i, inner in nested_names(q):
                if cfg.sequences[inner].errors:
                    q.errors.append(f'step {i} (sequence {inner}): {inner} cannot run '
                                    f'(`sequence {inner}` says why)')
                    changed = True
                    break


#  A sequence's failure or q, already given its place in the nesting and acted on (devices
#  stopped), so the sequences around it pass it up unchanged.
class SequenceAborted(ToolChangerError):
    pass


class SequenceInterrupted(Interrupted):
    pass


def run_sequence(s, name):
    """Run every step of sequence `name` in order; stop everything if any of it fails.

    A step may be `sequence OTHER`: it runs here, in full, as one step. The progress lines show
    the path -- [outer 2/3 > inner 1/2] -- and a failure or q deep inside stops the devices
    once and ends every sequence around it, saying where it happened."""
    seq = s.cfg.sequences[name]
    if seq.errors:
        raise ToolChangerError(f'sequence {name} cannot run ({seq.path}):\n'
                               + '\n'.join(f'    {e}' for e in seq.errors))
    if any(level[0] == name for level in s._running):     # check_nesting refuses loops
        raise ToolChangerError(f'sequence {name} is already running (it nests itself)')
    level = [name, 0, len(seq.steps)]
    s._running.append(level)
    indent = '  ' * (len(s._running) - 1)
    try:
        print(f'{indent}sequence {name}: {len(seq.steps)} steps'
              + (f' -- {seq.description}' if seq.description else ''))
        for i, step in enumerate(seq.steps, 1):
            level[1] = i
            path = ' > '.join(f'{n} {k}/{m}' for n, k, m in s._running)
            outer = s._running[0][0]
            print(f'{indent}[{path}] {step}')
            try:
                ok, why = s.execute(step.split()), 'it reported a failure'
            except (SequenceAborted, SequenceInterrupted):
                raise                       # a nested sequence's: reported and stopped already
            except Interrupted as exc:
                raise SequenceInterrupted(f'sequence {outer}: {exc} (at {path}: {step})')
            except ToolChangerError as exc:
                ok, why = False, str(exc)
            if not ok:
                stopped = s.safe_stop()
                raise SequenceAborted(f'sequence {outer} aborted at {path} ({step}): {why}.'
                                      + (f' Stopped: {", ".join(stopped)}.' if stopped else ''))
        print(f'{indent}sequence {name}: done')
    finally:
        s._running.pop()


#  =====   help   =====
def _listing(rows):
    """Two aligned columns."""
    width = max((len(a) for a, _ in rows), default=0) + 3
    return [f'  {a:<{width}}{b}' for a, b in rows]


def _plural(n, word):
    return f'{n} {word}{"" if n == 1 else "s"}'


def _sequences_help(cfg, modules=None):
    """The sequences, grouped by the file they come from; only `modules`' if given."""
    sources = modules if modules is not None else list(cfg.modules) + [MAIN]
    out = []
    for source in sources:
        seqs = [q for q in cfg.sequences.values() if q.source == source]
        if not seqs:
            continue
        out.append(f'sequences from {os.path.relpath(seqs[0].path)}')
        out += _listing([(q.name, (f'[CANNOT RUN: {_plural(len(q.errors), "problem")} -- '
                                   f'`sequence {q.name}` says what] ' if q.errors else '')
                          + q.description) for q in seqs])
        out.append('')
    if not out:
        which = f'{" or ".join(modules)} has' if modules else 'this configuration has'
        return f'{which} no sequences.'
    out.append('Run one with `sequence NAME`; every step is checked first, and q stops it.')
    return '\n'.join(out)


def _command_help(cfg, cmd):
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


def _module_help(cfg, name):
    cmds = [c for c in cfg.commands.values() if c.module == name]
    out = [f'{name} -- {cfg.modules[name].description}']
    out += _listing([(c.usage, c.summary) for c in cmds])
    n = sum(q.source == name for q in cfg.sequences.values())
    if n:
        out += ['', f'{_plural(n, "sequence")}: `help {name} sequence`.']
    return '\n'.join(out)


def render_help(cfg, topic=None, sub=None):
    """The text `help` prints.

        help                    the general commands, and the modules loaded
        help MODULE             that module's commands
        help COMMAND            one command in full
        help sequence           every sequence, grouped by the file it comes from
        help MODULE sequence    that module's sequences"""
    if sub is not None:
        if topic not in cfg.modules:
            raise ToolChangerError(f'`help {topic} sequence` needs a module name: '
                                   f'{", ".join(cfg.modules)}')
        return _sequences_help(cfg, [topic])
    if topic == 'sequence':
        return _sequences_help(cfg)
    if topic in cfg.commands:
        return _command_help(cfg, cfg.commands[topic])
    if topic in cfg.modules:
        return _module_help(cfg, topic)

    general = [c for c in cfg.commands.values() if c.module == 'general']
    out = ['general commands']
    out += _listing([(c.usage, c.summary) for c in general])
    rows = []
    for name, module in cfg.modules.items():
        if name == 'general':
            continue
        n_cmd = sum(c.module == name for c in cfg.commands.values())
        n_seq = sum(q.source == name for q in cfg.sequences.values())
        counts = _plural(n_cmd, 'command') + (f', {_plural(n_seq, "sequence")}' if n_seq else '')
        rows.append((name, counts, module.description))
    out += ['', 'modules']
    if rows:
        width = max(len(counts) for _, counts, _ in rows) + 3
        out += _listing([(name, f'{counts:<{width}}{desc}') for name, counts, desc in rows])
    else:
        out.append('  (none loaded)')
    out += ['',
            '`help MODULE` lists its commands; `help COMMAND` shows one in full.',
            '`help sequence` lists every sequence; `help MODULE sequence` one module\'s.',
            'Tab completes; q stops a timed command or a sequence.']
    return '\n'.join(out)


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
