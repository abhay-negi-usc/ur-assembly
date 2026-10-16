"""ur5cell -- the UR5 (CB3.1) cell: its apps, configs and tests, on top of the shared `urlab` core.

The core is the same library the UR10e cell runs; this package holds only what is THIS cell's --
its machine files (configs/ur5-*.yaml) and its task apps. See ur5cell/README.md.
"""

import os

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')


def config_path(name):
    """ur5cell/configs/<name>.yaml as an absolute path -- the `default_config` a UR5 app hands
    run_app, since urlab's bare-name lookup only searches the UR10e's configs/."""
    return os.path.join(CONFIG_DIR, name if name.endswith('.yaml') else name + '.yaml')


def use_ur5_configs(default_config=None, argv=None):
    """Make THIS process resolve configs in ur5cell/configs -- the UR5 entry points call it first.

    Sets $URLAB_CONFIG_DIR to ur5cell/configs, so a bare `--config <name>` can only ever name a UR5
    config: a UR10e name (cartesian, bnc_assembly, ...) is an error, never the UR10e's file. And when
    no --config was given, inserts `--config <default_config>`, since urlab's own apps default to
    UR10e configs.

    A FUNCTION, not an import side effect: merely importing ur5cell (its tests do) must not
    redirect configs for everything else in the same process. `argv` defaults to sys.argv and is
    edited in place."""
    import sys

    from urlab.config import CONFIG_DIR_ENV
    prev = os.environ.get(CONFIG_DIR_ENV, '').strip()
    if prev and os.path.realpath(os.path.expanduser(prev)) != os.path.realpath(CONFIG_DIR):
        print(f'ur5cell: {CONFIG_DIR_ENV} was {prev!r}; using {CONFIG_DIR} for this UR5 run.',
              file=sys.stderr)
    os.environ[CONFIG_DIR_ENV] = CONFIG_DIR
    argv = sys.argv if argv is None else argv
    given = any(a == '--config' or a.startswith('--config=') for a in argv[1:])
    if default_config and not given:
        argv[1:1] = ['--config', default_config]
    return argv


def run_urlab_app(module, default_config):
    """Run one of urlab's generic apps ON THE UR5 (see use_ur5_configs) -- what the
    ur5cell/apps/<name>.py wrappers are."""
    import importlib
    use_ur5_configs(default_config)
    return importlib.import_module(module).main()
