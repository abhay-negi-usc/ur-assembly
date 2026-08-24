"""Interactive gripper position control -- type a target, the gripper moves there, and the
MEASURED position is printed back.

    python -m urlab.apps.gripper_control
    python -m urlab.apps.gripper_control --config cable_pick_assemble
    python -m urlab.apps.gripper_control --port /dev/ttyUSB0

Positions are COUNTS (0 = fully open .. 255 = fully closed), the gripper's native Modbus units.
At the prompt type a number to command it; 'o'/'open' and 'close' also work, 'c'/'clear' clears
a latched fault, 'q' or Ctrl-C quits.  The measured position after a move can DIFFER from the
command when the fingers stall on an object early -- that gap is exactly what the grasp check
reads (see skills/pick.py GraspCheck).

GRIP FORCE is settable too, live: 'f 80' (or 'force 80') sets the force register, 'f' alone
reports it.  It is also 0-255 counts and rides along in the SAME Modbus word as the speed, so a
new value takes effect on the NEXT move -- it does not re-squeeze whatever is already held.
Lower force is what you want when tuning a grasp on something crushable; the grasp check reads
POSITION, and a soft object seats deeper as the force goes up, which shifts those count bands.

ONE LETTER MOVED: 'c' used to be the close shorthand and is now CLEAR, so that 'f' can belong
to force alone.  CLOSE MUST BE SPELLED OUT.  The two are not interchangeable by accident --
clearing re-activates the gripper, which re-homes the fingers and RELEASES anything held.
"""

import argparse
import sys

from .. import config as urconfig
from .. import log as urlog
from ..robot.gripper import GripperError, Robotiq2F85

log = urlog.get('gripper-control')


# Robotiq 2F-85 datasheet grip-force range, spanning force counts 0..255.
FORCE_MIN_N, FORCE_MAX_N = 20.0, 235.0


def force_newtons(counts):
    """Indicative grip force in N for a force-register value, linear across the 2F-85's
    datasheet range (0 counts ~= 20 N, 255 ~= 235 N). APPROXIMATE -- the register sets a motor
    current limit, not a calibrated force, and the real value varies with the grip position."""
    return FORCE_MIN_N + (max(0, min(255, counts)) / 255.0) * (FORCE_MAX_N - FORCE_MIN_N)


def parse_command(raw, open_counts, closed_counts):
    """One prompt line -> ('quit' | 'clear' | 'move' | 'force' | 'help', counts or None).

    Pure, so the REPL's grammar is testable without hardware. Numeric values are clamped to the
    gripper's 0-255 count range, which both the position and the force register use. 'f'/'force'
    with no argument means REPORT, and carries None.

    'f' means FORCE and 'c' means CLEAR -- each letter has exactly one meaning. 'c' used to be
    close; CLOSE now has no shorthand, deliberately, because a mistyped close that clears
    instead re-homes the fingers and drops whatever is held."""
    raw = raw.strip().lower()
    head, _, rest = raw.partition(' ')
    rest = rest.strip()

    if head in ('f', 'force', 'grip'):
        if not rest:
            return 'force', None
        try:
            return 'force', max(0, min(255, int(round(float(rest)))))
        except ValueError:
            return 'help', None
    if raw in ('q', 'quit', 'exit'):
        return 'quit', None
    if raw in ('c', 'clear', 'fault'):
        return 'clear', None
    if raw in ('o', 'open'):
        return 'move', open_counts
    if raw == 'close':
        return 'move', closed_counts
    try:
        return 'move', max(0, min(255, int(round(float(raw)))))
    except ValueError:
        return 'help', None


def _report_fault(gripper):
    """Print a statement if a fault is latched (it blocks further motion until cleared)."""
    fault = gripper.fault()
    if fault:
        print(f'  >> Gripper FAULT 0x{fault:02X}: motion commands will be IGNORED until '
              f'cleared. Type "c"/"clear" to clear it (re-activates + RE-HOMES the fingers).')
    return fault


def _clear_fault(gripper):
    if gripper.clear_fault():
        print(f'  fault cleared; position now {gripper.position()} counts.')
    else:
        print('  fault clearing FAILED -- check power, e-stop, and air supply, then retry '
              '"clear".')


def _describe_force(gripper):
    return f'{gripper.force} counts (~{force_newtons(gripper.force):.0f} N)'


def _set_force(gripper, counts):
    """Report the force register, or retarget it. None = report only."""
    if counts is None:
        print(f'  force is {_describe_force(gripper)}.')
        return
    previous = gripper.force
    gripper.force = counts                    # rides in the same word as speed on the next write
    print(f'  force {previous} -> {_describe_force(gripper)}; takes effect on the NEXT move '
          f'(anything already held keeps the force it was grasped with).')


def _move(gripper, target):
    ok = gripper.go_to(target, f'-> {target}')
    measured = gripper.position()
    if ok:
        note = '' if measured >= target - 1 else '  (stalled short -- holding an object?)'
        print(f'  commanded {target} -> measured {measured} counts{note}  [force '
              f'{_describe_force(gripper)}]')
    else:
        print(f'  move to {target} did not complete (measured {measured} counts).')
    _report_fault(gripper)


def repl(gripper):
    print(f'Gripper ready on {gripper.port}. Current position: {gripper.position()} counts, '
          f'force {_describe_force(gripper)}.')
    print('Enter a position 0-255 (0=open, 255=closed), "o"/"open", "close" (spelled out), '
          '"f N" 0-255 to set grip force, "f" to show it, "c"/"clear" to clear a fault '
          '(RE-HOMES the fingers, releasing anything held), "q" to quit.')
    _report_fault(gripper)
    while True:
        try:
            raw = input('position> ')
        except EOFError:
            return
        if not raw.strip():
            continue
        cmd, target = parse_command(raw, gripper.open_counts, gripper.closed_counts)
        if cmd == 'quit':
            return
        if cmd == 'clear':
            _clear_fault(gripper)
        elif cmd == 'force':
            _set_force(gripper, target)
        elif cmd == 'move':
            _move(gripper, target)
        else:
            print('  enter a number 0-255, "o"/"open", "close", "f N" (force), "c"/"clear", '
                  'or "q".')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', default='gripper',
                    help='config name in configs/ (for the gripper.* block); default: gripper')
    ap.add_argument('--port', default=None, help='override gripper.port (e.g. /dev/ttyUSB0)')
    ap.add_argument('--force', type=int, default=None,
                    help='override gripper.force_counts (0-255) for this session; also settable '
                         'live at the prompt with "f N"')
    ap.add_argument('--dry-run', action='store_true',
                    help='do not open the gripper; echo commands and simulate the position')
    args = ap.parse_args()

    cfg = urconfig.load(args.config)
    if args.port:
        cfg.set_path('gripper.port', args.port)
    if args.force is not None:
        # Set BEFORE constructing, so the activation stroke uses it too.
        cfg.set_path('gripper.force_counts', max(0, min(255, args.force)))
    if args.dry_run:
        cfg.set_path('robot.dry_run', True)

    try:
        gripper = Robotiq2F85(cfg)             # opens the port + activates the gripper
    except GripperError as exc:
        print(f'Could not open the gripper: {exc}', file=sys.stderr)
        return 1

    try:
        repl(gripper)
    except KeyboardInterrupt:
        pass
    finally:
        gripper.disconnect()
        print('\nDisconnected.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
