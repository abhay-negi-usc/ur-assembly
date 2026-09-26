"""COUPLER ACTUATE -- lock the toolchanger, wait, unlock. The bench check before a pick.

    python -m urlab.apps.coupler_actuate

Confirm, LOCK, confirm, RELEASE. Nothing else moves and the arm is never contacted.

WHY THIS EXISTS WHEN toolchanger/toolchanger.py ALREADY HAS A REPL. That one tests the DRIVER,
on the driver's own defaults. This one tests THE PATH THE DEMO TAKES: the `toolchanger:` block in
configs/, urlab.robot.coupler.Coupler on top of it, and the interlock as apps/coupler_pick_place
reads it. The failures that only show up here are the boring ones that stop a demo dead -- a
port named in the config that does not exist, a settle_s too short for the board to finish
booting, a baud mismatch, a wrapper that mishandles a refusal. Running this first turns "the
pick did not work" into "the coupler is fine, look elsewhere".

IT REFUSES TO RUN WITHOUT THE BOARD. Coupler falls back to asking the operator to work the
mechanism by hand, which is right for a calibration and useless here -- a coupler test that
reports success because a human pressed Enter has tested a human. `require_driver: false` if you
really want the manual path.

NO ROBOT IS CONNECTED. The arm is not needed to work the coupler, and requiring it would mean
this could not be run at a bench, or with the controller off, which are exactly the times you
want it.

THE SENSOR IS THE POINT, not the servo. The board confirms every lock against its proximity
probe and answers "confirmed!" or "emergency stop"; this prints which, per step, so a servo that
swings while the probe sees nothing is visible rather than silently passing. Put a tool in the
coupler before running it, or the lock SHOULD be refused -- that refusal is a passing test of the
interlock, and the script says so.
"""

import logging
import sys

from .. import config as urconfig
from .. import log as urlog
from ..apps._common import ask, prompts_off
from ..robot.coupler import Coupler

log = urlog.get('coupler-actuate')


def _report(what, ok, verified):
    """One line per actuation, saying BOTH whether it was accepted and whether anything checked.

    Those are different facts and collapsing them is how a manual run gets mistaken for a
    verified one."""
    if ok and verified:
        log.info('  %s OK -- and the board\'s proximity sensor AGREED.', what.upper())
    elif ok:
        log.warning('  %s reported OK, but NOTHING VERIFIED IT (no board, or no sensor).', what)
    else:
        log.error('  %s REFUSED. The sensor disagrees with the commanded state: asked to %s and '
                  'the probe says otherwise. The board printed the raw reading above -- if it '
                  'sits near the threshold, the THRESHOLD is what is wrong, not the tool '
                  '(`toolchanger/toolchanger.py calibrate`).', what.upper(), what)


def cycle(coupler, index, total, confirm):
    """One lock/unlock, gated by the operator on both sides. False on any refusal."""
    label = '' if total == 1 else f' ({index} of {total})'
    if not confirm(f'Ready to LOCK the coupler{label}. Stand clear, then press Enter '
                   '(q to stop): '):
        return False
    ok = coupler.hold()
    _report('hold', ok, coupler.verified)
    if not ok:
        return False

    if not confirm(f'LOCKED{label}. Check the mechanism, then press Enter to RELEASE '
                   '(q to stop, leaving it locked): '):
        # Deliberately NOT auto-releasing: the operator may be holding the tool, and a coupler
        # that lets go because someone typed 'q' is a dropped tool.
        log.warning('Stopped with the coupler STILL LOCKED, as asked. Release it with '
                    '`toolchanger/toolchanger.py release` when ready.')
        return False
    ok = coupler.release()
    _report('release', ok, coupler.verified)
    return ok


def build_and_run(cfg, coupler):
    """The whole run: report where the board stands, then cycle. True if every step passed."""
    if coupler.manual and bool(cfg.get('require_driver', True)):
        log.error('No toolchanger driver -- there is nothing here to test. A coupler check that '
                  'passes because a human pressed Enter has tested the human. Fix the port '
                  '(toolchanger.port, currently %r) or set require_driver: false to use the '
                  'manual path deliberately.', cfg.get_path('toolchanger.port'))
        return False

    # Where the board thinks it stands BEFORE anything is commanded. A disagreement here means
    # the previous run left it locked, or the probe is seeing something that is not there.
    initial = coupler.verify()
    if initial is None:
        log.warning('Manual mode: nothing will be verified in this run.')
    elif initial:
        log.info('Board check: the sensor AGREES with the coupler\'s current state.')
    else:
        log.warning('Board check: the sensor DISAGREES with the coupler\'s current state before '
                    'anything has been commanded. Either the last run left it locked, or the '
                    'probe is mis-tuned. `toolchanger/toolchanger.py probe` shows the reading.')

    # Power the actuation and open the jaws before the first lock. Without this the servo
    # swings against a dead relay, and a board that booted with something in front of the probe
    # answers the hold by re-reading the sensor and confirming WITHOUT MOVING -- the two ways a
    # coupler reports a grip it never made.
    if not coupler.prepare_to_mate():
        log.error('Could not power and open the coupler -- there is nothing to test past here.')
        return False

    confirm = (lambda _p: True) if prompts_off(cfg) else ask
    total = max(1, int(cfg.get('cycles', 1)))
    ok = True
    for i in range(1, total + 1):
        if not cycle(coupler, i, total, confirm):
            ok = False
            break
    else:
        log.info('%d cycle%s completed.', total, '' if total == 1 else 's')
    # Only ever on the way out, and only once nothing is held: cutting the actuation under a
    # held tool is how one gets dropped. A run that stopped mid-cycle deliberately leaves the
    # relay alone and says so.
    if ok:
        coupler.set_motor(False)
    else:
        log.warning('Leaving the motor relay AS IT IS -- the coupler may still be holding '
                    'something, and cutting the actuation under it would drop it.')
    return ok


def main():
    # NOT run_app: that builds a Robot, and the arm is not needed to work the coupler. Requiring
    # it would mean this could not be run at a bench or with the controller off.
    parser = urconfig.arg_parser(__doc__.splitlines()[0], 'coupler_actuate')
    args = parser.parse_args()
    cfg = urconfig.from_args(args)
    urlog.setup(logging.DEBUG if cfg.get('debug') else logging.INFO)
    log.info('Config: %s', cfg.get('_config_path'))

    ok = False
    coupler = None
    try:
        coupler = Coupler(cfg)
        ok = bool(build_and_run(cfg, coupler))
    except KeyboardInterrupt:
        log.warning('Interrupted. The coupler is left exactly as it was -- check it before '
                    'the next run.')
    except Exception:                      # noqa: BLE001 -- log the traceback, still tear down
        log.exception('Unhandled error:')
    finally:
        if coupler is not None:
            coupler.close()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
