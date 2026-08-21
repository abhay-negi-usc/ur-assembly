"""Interactive gripper position control -- type a target, the gripper moves there, and the
MEASURED position is printed back.

    python -m urlab.apps.gripper_control
    python -m urlab.apps.gripper_control --config cable_pick_assemble
    python -m urlab.apps.gripper_control --port /dev/ttyUSB0

Positions are COUNTS (0 = fully open .. 255 = fully closed), the gripper's native Modbus units.
At the prompt type a number to command it; 'o'/'open' and 'c'/'close' also work, 'clear' clears
a latched fault, 'q' or Ctrl-C quits.  The measured position after a move can DIFFER from the
command when the fingers stall on an object early -- that gap is exactly what the grasp check
reads (see skills/pick.py GraspCheck).
"""

import argparse
import sys

from .. import config as urconfig
from .. import log as urlog
from ..robot.gripper import GripperError, Robotiq2F85

log = urlog.get('gripper-control')


def parse_command(raw, open_counts, closed_counts):
    """One prompt line -> ('quit' | 'clear' | 'move' | 'help', target counts or None).

    Pure, so the REPL's grammar is testable without hardware. Numeric targets are clamped to
    the gripper's 0-255 count range."""
    raw = raw.strip().lower()
    if raw in ('q', 'quit', 'exit'):
        return 'quit', None
    if raw in ('clear', 'f', 'fault'):
        return 'clear', None
    if raw in ('o', 'open'):
        return 'move', open_counts
    if raw in ('c', 'close'):
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
              f'cleared. Type "clear" to clear it (re-activates + RE-HOMES the fingers).')
    return fault


def _clear_fault(gripper):
    if gripper.clear_fault():
        print(f'  fault cleared; position now {gripper.position()} counts.')
    else:
        print('  fault clearing FAILED -- check power, e-stop, and air supply, then retry '
              '"clear".')


def _move(gripper, target):
    ok = gripper.go_to(target, f'-> {target}')
    measured = gripper.position()
    if ok:
        note = '' if measured >= target - 1 else '  (stalled short -- holding an object?)'
        print(f'  commanded {target} -> measured {measured} counts{note}')
    else:
        print(f'  move to {target} did not complete (measured {measured} counts).')
    _report_fault(gripper)


def repl(gripper):
    print(f'Gripper ready on {gripper.port}. Current position: {gripper.position()} counts.')
    print('Enter a position 0-255 (0=open, 255=closed), "o"/"c" open/close, "clear" to clear '
          'a fault, "q" to quit.')
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
        elif cmd == 'move':
            _move(gripper, target)
        else:
            print('  enter a number 0-255, "o"/"c", "clear", or "q".')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', default='gripper',
                    help='config name in configs/ (for the gripper.* block); default: gripper')
    ap.add_argument('--port', default=None, help='override gripper.port (e.g. /dev/ttyUSB0)')
    ap.add_argument('--dry-run', action='store_true',
                    help='do not open the gripper; echo commands and simulate the position')
    args = ap.parse_args()

    cfg = urconfig.load(args.config)
    if args.port:
        cfg.set_path('gripper.port', args.port)
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
