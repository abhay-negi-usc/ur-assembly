"""Which pins each module uses, checked against the board and each other at build time.

Each module declares its pins (registry.Module.pin) with what they must be able to do, and any
hardware it takes over whole (Module.claim). The config's `pins:` block moves them. Before
compiling, build_flash.sh runs

    python3 -m mtc.pins [--config PATH] [--modules a,b]

which loads the selected modules, refuses any combination that cannot work on the board, and
prints the module list and the -DPIN_<MODULE>_<ROLE>=<n> flags to compile them in with.

THE BOARD is an Arduino Uno (ATmega328P). What its pins can do:

    D0 D1         the serial link to the PC -- never available
    PWM           D5 D6 (Timer0), D9 D10 (Timer1), D3 D11 (Timer2)
    interrupt     D2 (INT0), D3 (INT1)
    analog        A0..A5 (pins 14..19, which also work as digital pins)
    pin change    every pin; one interrupt vector per port: D0-D7, D8-D13, A0-A5

Timer0 also runs millis()/delay(), so no module may claim it.
"""

import argparse
import os
import sys

from .base import ToolChangerError

ANALOG_BASE = 14                                   # A0 is Arduino pin 14 on an Uno
RESERVED = {0: 'D0 is the serial link to the PC', 1: 'D1 is the serial link to the PC'}
PWM_TIMER = {3: 'timer2', 5: 'timer0', 6: 'timer0', 9: 'timer1', 10: 'timer1', 11: 'timer2'}
INTERRUPT = {2: 'INT0', 3: 'INT1'}
ANALOG = set(range(ANALOG_BASE, ANALOG_BASE + 6))
LAST_PIN = ANALOG_BASE + 5
CORE_CLAIMS = {'timer0': 'the Arduino core (millis and delay)'}
NEEDS = ('digital', 'pwm', 'analog', 'interrupt', 'pcint')


def parse_pin(value, where):
    """An Arduino pin number from 6, '6', 'D6' or 'A3'."""
    text = str(value).strip().upper()
    try:
        if text.startswith('A'):
            n = ANALOG_BASE + int(text[1:])
            ok = n in ANALOG
        else:
            n = int(text[1:] if text.startswith('D') else text)
            ok = 0 <= n <= LAST_PIN
    except ValueError:
        ok = False
    if isinstance(value, bool) or not ok:
        raise ToolChangerError(f'{where}: {value!r} is not an Uno pin (D2..D13, A0..A5)')
    return n


def pin_name(n):
    return f'A{n - ANALOG_BASE}' if n >= ANALOG_BASE else f'D{n}'


def pcint_port(n):
    """The pin-change interrupt vector a pin shares with the rest of its port."""
    return 'pcint D0-D7' if n <= 7 else ('pcint D8-D13' if n <= 13 else 'pcint A0-A5')


def check(cfg, modules=None):
    """Every reason the loaded modules' pins cannot work together on the board, as strings."""
    names = [m for m in (modules or cfg.modules) if cfg.modules[m].pins or cfg.modules[m].claims]
    problems = []
    users = {}                      # pin -> [module.role]
    claims = dict(CORE_CLAIMS)      # resource -> who has it

    def take(resource, who):
        if resource in claims:
            problems.append(f'{who} needs {resource}, which {claims[resource]} already has')
        else:
            claims[resource] = who

    for m in names:
        module = cfg.modules[m]
        for resource in module.claims:
            take(resource, m)
        for role, (_, needs) in module.pins.items():
            pin = cfg.pins[m][role]
            who = f'{m}.{role}'
            users.setdefault(pin, []).append(who)
            if pin in RESERVED:
                problems.append(f'{who} is on {pin_name(pin)}: {RESERVED[pin]}')
            if needs == 'pwm' and pin not in PWM_TIMER:
                problems.append(f'{who} needs PWM, and {pin_name(pin)} has none '
                                f'(PWM pins: {", ".join(pin_name(p) for p in sorted(PWM_TIMER))})')
            elif needs == 'analog' and pin not in ANALOG:
                problems.append(f'{who} needs an analog input, and {pin_name(pin)} is not one '
                                f'(A0..A5)')
            elif needs == 'interrupt' and pin not in INTERRUPT:
                problems.append(f'{who} needs a hardware interrupt, and {pin_name(pin)} has none '
                                f'(only D2 and D3)')
            elif needs == 'pcint':
                take(pcint_port(pin), who)

    #  PWM runs on a timer, so it fails on a timer another module has taken over
    for m in names:
        for role, (_, needs) in cfg.modules[m].pins.items():
            pin = cfg.pins[m][role]
            timer = PWM_TIMER.get(pin)
            if needs == 'pwm' and timer and timer != 'timer0' and timer in claims:
                problems.append(f'{m}.{role} needs PWM on {pin_name(pin)}, which runs on '
                                f'{timer} -- and {claims[timer]} has taken {timer} over')

    for pin, who in sorted(users.items()):
        if len(who) > 1:
            problems.append(f'{pin_name(pin)} is wanted by {" and ".join(who)}')
    return problems


def defines(cfg, modules=None):
    """The -D flags that compile each module's pins in."""
    flags = []
    for m in modules or cfg.modules:
        for role, pin in cfg.pins.get(m, {}).items():
            flags.append(f'-DPIN_{m.upper()}_{role.upper()}={pin}')
    return flags


def describe(cfg, modules=None):
    return '  '.join(f'{m}.{role} {pin_name(pin)}' for m in (modules or cfg.modules)
                     for role, pin in cfg.pins.get(m, {}).items())


def main(argv=None):
    """For build_flash.sh: the modules to build and their -D flags, or the problems."""
    from .config import CONFIG_YAML, _read_yaml, available_modules, load_config

    ap = argparse.ArgumentParser(description='Check module pins and print build flags.')
    ap.add_argument('--config', default=os.environ.get('MULTITOOLCHANGER_CONFIG', CONFIG_YAML))
    ap.add_argument('--modules', help='comma-separated; default: the config\'s modules: list')
    ap.add_argument('--firmware', help='firmware directory, to check each module has a .cpp')
    args = ap.parse_args(argv)
    try:
        if args.modules is not None:
            names = [m for m in args.modules.split(',') if m]
            where = '--modules'
        else:
            names = _read_yaml(os.path.abspath(args.config), required=True).get('modules') or []
            where = os.path.abspath(args.config)
        if 'general' in names:
            raise ToolChangerError('general is host-only -- it has no firmware; leave it out')
        if not names:
            raise ToolChangerError(f'no modules selected ({where})')
        if args.firmware:
            missing = [m for m in names
                       if not os.path.isfile(os.path.join(args.firmware, f'{m}.cpp'))]
            if missing:
                have = [m for m in available_modules()
                        if os.path.isfile(os.path.join(args.firmware, f'{m}.cpp'))]
                raise ToolChangerError(f'no firmware for {", ".join(missing)}. '
                                       f'Available: {", ".join(have)}')
        cfg = load_config(args.config, modules=names)
        problems = check(cfg, names)
        if problems:
            raise ToolChangerError(f'these modules cannot share one board ({where}):\n'
                                   + '\n'.join(f'    {p}' for p in problems)
                                   + '\n  Move pins in the modules\' `pins:` blocks in '
                                   + os.path.dirname(cfg.path) + ', or use another board.')
    except ToolChangerError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f'==> pins: {describe(cfg, names)}', file=sys.stderr)
    print(' '.join(names))
    print(' '.join(defines(cfg, names)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
