"""Screwdriver -- the multi-toolchanger's screwdrive module, as the apps see it.

Runs NAMED SEQUENCES from multi-toolchanger/config/screwdrive.yaml (or any sequence the board's
config defines), through the same Session the multitoolchanger.py prompt uses -- so a sequence
means exactly what `sequence NAME` means there: every step checked before anything runs, a
failed step stops the motor and aborts the rest, and q stops it.

    sd = Screwdriver(cfg)                 # cfg's `fastening:` block
    sd.run('screwdrive_predrive')         # True when the sequence finished

STOPPING FROM ANOTHER THREAD. stop() only sets a flag; the running sequence notices it at its
next poll of the board, stops the motor itself and returns False. Only the thread running the
sequence ever talks to the serial port, so a stop requested by the servo loop (a force-guard
trip while fastening) cannot interleave bytes with it.

SEQUENCE NAMES ARE CHECKED UP FRONT, against the multi-toolchanger config on disk, so a typo
stops the run at startup rather than with a part half assembled. A dry run checks them too and
then only logs what it would run.
"""

import os
import sys
import threading

from .. import log as urlog

log = urlog.get('screwdriver')

MTC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__)))), 'multi-toolchanger')


def _mtc():
    """The multi-toolchanger package. It lives in a directory with a hyphen in its name, so it
    is put on the path rather than imported as part of this one."""
    if MTC_DIR not in sys.path:
        sys.path.insert(0, MTC_DIR)
    import mtc
    from mtc import session
    return mtc, session


class _StopOrKey:
    """Session watcher: q on the keyboard (the usual KeyWatcher) OR a stop() from this process."""

    def __init__(self, event, keys):
        self.event = event
        self.keys = keys

    def __enter__(self):
        self.keys.__enter__()
        return self

    def __exit__(self, *exc):
        return self.keys.__exit__(*exc)

    @property
    def active(self):
        return self.keys.active

    def pressed(self):
        return self.event.is_set() or self.keys.pressed()


def check_sequences(names, config_path=None, config=None):
    """Refuse a sequence name that does not exist or cannot run. Raises ValueError."""
    mtc, _session = _mtc()
    config = config or mtc.load_config(config_path)
    for name in names:
        seq = config.sequences.get(name)
        if seq is None:
            known = ', '.join(sorted(n for n in config.sequences if n.startswith('screwdrive_')))
            raise ValueError(f'fastening: no sequence {name!r} in {config.path} or its module '
                             f'files. Screwdrive sequences: {known or "(none)"}')
        if seq.errors:
            raise ValueError(f'fastening: sequence {name!r} cannot run ({seq.path}): '
                             + '; '.join(seq.errors))


class Screwdriver:
    """Named screwdrive sequences on the multi-toolchanger board, from a config block."""

    def __init__(self, cfg, section='fastening', sequences=()):
        block = cfg.section(section)
        self.port = block.get('port', 'dc_motor')
        self.config_path = block.get('config') or None
        self.dry_run = bool(cfg.get_path('robot.dry_run'))
        self.tc = self.session = None
        self._stop = threading.Event()
        check_sequences(sequences, self.config_path)
        if self.dry_run:
            log.info('DRY RUN: no screwdriver connection; sequences are logged, not run.')
            return
        mtc, session = _mtc()
        try:
            self.tc = mtc.ToolChanger(port=self.port, config_path=self.config_path,
                                      baud=int(block.get('baud', 115200)),
                                      timeout=float(block.get('timeout_s', 5.0)),
                                      settle=float(block.get('settle_s', 3.0)), name='screwdriver')
        except (mtc.ToolChangerError, OSError) as exc:
            raise ValueError(f'fastening: could not connect to the screwdriver board '
                             f'{self.port!r}: {exc}') from exc
        if 'screwdrive' not in self.tc.devices:
            have = ', '.join(self.tc.devices) or 'nothing'
            self.tc.close()
            raise ValueError(f'fastening: the board on {self.port!r} has no screwdrive module '
                             f'(it has {have}) -- wrong port?')
        # The board's own config (its detected modules) is what runs, so check against it too.
        check_sequences(sequences, config=self.tc.config)
        self.session = mtc.Session(self.tc.config, self.tc,
                                   watcher=lambda: _StopOrKey(self._stop, session.KeyWatcher()))
        log.info('Screwdriver connected on %r.', self.port)

    def run(self, name):
        """Run sequence `name` to the end. True if it finished; False if a step failed, q was
        pressed or stop() was called -- in every one of those the motor has been stopped."""
        if self.dry_run:
            log.info('DRY RUN: would run screwdriver sequence %r.', name)
            return True
        mtc, _session = _mtc()
        self._stop.clear()
        log.info('Screwdriver: running %r.', name)
        try:
            ok = bool(self.session.execute(['sequence', name]))
        except (mtc.ToolChangerError, mtc.Interrupted) as exc:
            log.error('Screwdriver sequence %r did not finish: %s', name, exc)
            return False
        if ok:
            log.info('Screwdriver: %r done.', name)
        return ok

    def stop(self):
        """Ask the running sequence to stop. Safe from any thread; see the module docstring."""
        self._stop.set()

    def close(self):
        if self.tc is not None:
            self.tc.close()
            self.tc = None
