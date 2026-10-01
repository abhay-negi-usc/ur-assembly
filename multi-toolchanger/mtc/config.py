"""Loading the config: which modules this toolchanger has, their settings and sequences.

    config/multitoolchanger.yaml   which modules to load, and sequences that span modules
    config/<module>.yaml           that module's settings and its own sequences (optional)

The module files sit next to the main one, so a different deployment is a different directory:
--config path/to/other/multitoolchanger.yaml, or $MULTITOOLCHANGER_CONFIG.
"""

import importlib
import os
import pkgutil
import re

from . import modules as modules_pkg
from .base import ToolChangerError
from .registry import CORE_KINDS

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_YAML = os.environ.get(
    'MULTITOOLCHANGER_CONFIG', os.path.join(HERE, '..', 'config', 'multitoolchanger.yaml'))

#  The wire protocol's version; must match PROTOCOL in firmware/main.cpp. The host refuses a
#  board that reports a different one rather than send it commands it would misread.
PROTOCOL = 5

SEQUENCE_NAME = re.compile(r'[A-Za-z0-9_.-]+')
MAIN = 'main'   # the `source` of a sequence from the main file


class Sequence:
    def __init__(self, name, description, steps, source, path, errors=None):
        self.name, self.description, self.steps = name, description, steps
        self.source = source        # the module it came from, or MAIN
        self.path = path            # the file it came from
        self.errors = errors or []  # non-empty: it is listed but refuses to run


class Config:
    """Everything loaded: modules (general first, then in config order), their settings, the
    commands and kinds they define, and every sequence by name."""

    def __init__(self, path):
        self.path = path
        self.modules = {}       # name -> Module
        self.settings = {}      # module name -> SimpleNamespace
        self.pins = {}          # module name -> {role: arduino pin}; used by build_flash.sh
        self.commands = {}      # command name -> Command
        self.kinds = dict(CORE_KINDS)
        self.sequences = {}     # name -> Sequence


def available_modules():
    return sorted(m.name for m in pkgutil.iter_modules(modules_pkg.__path__))


def _read_yaml(path, required):
    if not os.path.isfile(path):
        if required:
            raise ToolChangerError(f'no config at {path}. Pass --config, or set '
                                   f'$MULTITOOLCHANGER_CONFIG.')
        return {}
    try:
        import yaml
    except ImportError:
        raise ToolChangerError(f'{path} needs PyYAML to read it: pip install pyyaml')
    try:
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ToolChangerError(f'could not read {path}: {exc}')
    if not isinstance(doc, dict):
        raise ToolChangerError(f'{path}: expected a mapping at the top level')
    return doc


def _import_module(name):
    try:
        mod = importlib.import_module(f'{modules_pkg.__name__}.{name}')
    except ModuleNotFoundError as exc:
        if exc.name != f'{modules_pkg.__name__}.{name}':
            raise                                   # the module itself imports something missing
        raise ToolChangerError(f'no module called {name!r}. Available: '
                               + ', '.join(m for m in available_modules() if m != 'general'))
    return mod.MODULE


def _read_sequences(raw, source, path):
    """{name: Sequence} from a `sequences:` block. Structure problems mark the sequence, they
    do not stop the load -- one bad sequence must not take the others down with it."""
    out = {}
    if raw is None:
        return out
    if not isinstance(raw, dict):
        raise ToolChangerError(f'{path}: sequences must be a mapping of name -> steps')
    for name, body in raw.items():
        name = str(name)
        if isinstance(body, list):
            body = {'steps': body}
        if not isinstance(body, dict):
            body = {'steps': None}
        steps = body.get('steps')
        seq = Sequence(name, str(body.get('description') or '').strip(),
                       [str(s).strip() for s in steps] if isinstance(steps, list) else [],
                       source, path)
        if not SEQUENCE_NAME.fullmatch(name):
            seq.errors.append('the name may only use letters, digits, _ . and -')
        if source != MAIN and not name.startswith(f'{source}_'):
            seq.errors.append(f'sequences in {os.path.basename(path)} must be named '
                              f'{source}_<something>, so it is clear where they come from')
        if not isinstance(steps, list):
            seq.errors.append('needs a list of steps, or a mapping with `steps:`')
        elif not steps:
            seq.errors.append('has no steps')
        out[name] = seq
    return out


def load_config(path=None, modules=None):
    """Load the main config, the modules it names, and each module's own config.

    `modules`, when given, REPLACES the main file's `modules:` list -- that is how the modules
    the board reports are loaded (autodetect). Module configs are still read from beside the
    main file, so settings and sequences come from the same place either way.

    FATAL: a missing or unreadable main file, an unknown module, two modules defining the same
    command, a bad setting. Falling back to defaults would quietly run with the wrong modules or
    the wrong numbers, which is worse than refusing to start.

    NOT FATAL: a bad sequence. It is kept and marked with what is wrong, `help` shows it as
    unable to run, and everything else carries on."""
    from .session import check_nesting, validate_steps   # session imports this module

    path = os.path.abspath(path or CONFIG_YAML)
    doc = _read_yaml(path, required=True)
    unknown = sorted(set(doc) - {'modules', 'sequences'})
    if unknown:
        raise ToolChangerError(f'{path}: unknown key(s) {", ".join(unknown)} '
                               f'(expected modules, sequences)')
    names = list(modules) if modules is not None else (doc.get('modules') or [])
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ToolChangerError(f'{path}: modules must be a list of module names')
    if 'general' in names:
        raise ToolChangerError(f'{path}: general is always loaded -- leave it out of modules')
    if len(set(names)) != len(names):
        raise ToolChangerError(f'{path}: a module is listed twice')

    cfg = Config(path)
    sequences = []
    for name in ['general'] + names:
        module = _import_module(name)
        mod_path = os.path.join(os.path.dirname(path), f'{name}.yaml')
        mod_doc = _read_yaml(mod_path, required=False)
        unknown = sorted(set(mod_doc) - {'settings', 'pins', 'sequences'})
        if unknown:
            raise ToolChangerError(f'{mod_path}: unknown key(s) {", ".join(unknown)} '
                                   f'(expected settings, pins, sequences)')
        cfg.modules[name] = module
        cfg.settings[name] = module.configure(mod_doc.get('settings'), mod_path)
        cfg.pins[name] = module.configure_pins(mod_doc.get('pins'), mod_path)
        for cmd_name, cmd in module.commands.items():
            if cmd_name in cfg.commands:
                raise ToolChangerError(f'{name} and {cfg.commands[cmd_name].module} both define '
                                       f'the command {cmd_name!r}')
            cfg.commands[cmd_name] = cmd
        for kind_name, kind in module.kinds.items():
            if kind_name in cfg.kinds:
                raise ToolChangerError(f'{name} redefines the argument kind {kind_name!r}')
            cfg.kinds[kind_name] = kind
        sequences.append(_read_sequences(mod_doc.get('sequences'), name, mod_path))
    sequences.append(_read_sequences(doc.get('sequences'), MAIN, path))

    for group in sequences:
        for name, seq in group.items():
            if name in cfg.sequences:
                seq.errors.append(f'the name is also used in {cfg.sequences[name].path}')
                name = f'{name} ({seq.source})'     # keep both visible in `help`
            cfg.sequences[name] = seq
    for seq in cfg.sequences.values():
        if not seq.errors:
            seq.errors += validate_steps(seq, cfg)
    check_nesting(cfg)
    return cfg
