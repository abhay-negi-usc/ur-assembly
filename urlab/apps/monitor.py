"""Live end-effector + attached-frame monitor -- watch tool0 and every tool0-attached frame while
you MANUALLY DRIVE the arm.

READ-ONLY by default (RTDE Receive only): it uploads no control script and needs no Remote Control,
so it coexists with PENDANT FREEDRIVE. Put the robot in Freedrive on the pendant, run this, and push
the arm around by hand -- the poses update live. (This is the opposite of the demos, which need
Remote Control because they DRIVE the arm.)

    python -m urlab.apps.monitor
    python -m urlab.apps.monitor --config cable_pick_place --rate 20
    python -m urlab.apps.monitor --freedrive          # SOFTWARE freedrive via teachMode (needs
                                                       # Remote Control) -- drive by hand without the
                                                       # pendant's Freedrive button
    python -m urlab.apps.monitor --wrench --gripper    # also show the TCP wrench + gripper counts
    python -m urlab.apps.monitor --csv poses.csv       # append every sample to a CSV

Every frame is in base_link, built from the SAME config sections the demos use, so what you read
here is exactly what a demo would command:
    tool0            -- the flange (the all-zeros pendant TCP; getActualTCPPose is this pose)
    fingertip        -- fingertip_grasp    (tool0 -> fingertip)
    camera           -- hand_eye           (tool0 -> camera)
    grasp            -- grasp_tcp_offset   (tool0 -> grasp TCP)
    connector_holder -- connector_holder   (tool0 -> connector_holder; only if configured)
Poses print as xyz (mm) + rpy (deg, EXTRINSIC XYZ). This is the tool to MEASURE the config values
the README's hardware checklist leaves open (assembly.target, a grasp pose, etc.): jog to the spot,
read the frame off here, paste it in.

NOTE: it assumes the pendant TCP is the all-zeros tool0 (the urlab convention). If a non-zero TCP is
set on the pendant, getActualTCPPose reports THAT, and every frame here is offset by it. The wrench
(--wrench) is uncompensated (no payload is set in read-only mode), so it reads the tool's own weight.
"""

import argparse
import sys
import time

import numpy as np

from .. import config as urconfig
from .. import log as urlog
from ..transforms import from_cfg, matrix_to_xyzrpy, rtde_to_matrix

log = urlog.get('monitor')


def _frames(cfg):
    """{name: T_tool0_frame} for every tool0-attached frame, from the same config sections the
    Robot facade uses (robot/robot.py)."""
    frames = {
        'tool0': np.eye(4),
        'fingertip': from_cfg(cfg.section('fingertip_grasp')),
        'camera': from_cfg(cfg.section('hand_eye')),
        'grasp': from_cfg(cfg.section('grasp_tcp_offset')),
    }
    if cfg.get('connector_holder'):                    # tool0 -> connector_holder (-> connector)
        T_holder = from_cfg(cfg.section('connector_holder'))
        frames['connector_holder'] = T_holder
        frames['connector'] = T_holder @ from_cfg(cfg.section('connector_in_holder'))
    return frames


def _fmt(T):
    xyz, rpy = matrix_to_xyzrpy(T)
    d = np.degrees(rpy)
    return (f'xyz=[{xyz[0] * 1000:+8.2f}, {xyz[1] * 1000:+8.2f}, {xyz[2] * 1000:+8.2f}] mm  '
            f'rpy=[{d[0]:+7.2f}, {d[1]:+7.2f}, {d[2]:+7.2f}] deg')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', default='cable_pick_assemble',
                    help='config name in configs/ or a path (for robot.ip + the frame offsets)')
    ap.add_argument('--robot-ip', default=None, help='override robot.ip')
    ap.add_argument('--rate', type=float, default=10.0, help='refresh rate in Hz (default 10)')
    ap.add_argument('--freedrive', action='store_true',
                    help='enable SOFTWARE freedrive via teachMode (needs Remote Control). Default is '
                         'READ-ONLY -- drive with the pendant Freedrive button instead.')
    ap.add_argument('--wrench', action='store_true',
                    help='also show the TCP wrench (uncompensated: reads the tool weight)')
    ap.add_argument('--gripper', action='store_true',
                    help='also show gripper position in counts (connects the Modbus port; activates '
                         'the gripper only if it is not already activated)')
    ap.add_argument('--csv', default=None, help='append every sample to this CSV')
    ap.add_argument('--once', action='store_true', help='print one sample and exit')
    args = ap.parse_args()

    cfg = urconfig.load(args.config)
    ip = args.robot_ip or cfg.get_path('robot.ip')
    if not ip:
        print('No robot.ip in the config and no --robot-ip given.', file=sys.stderr)
        return 1

    try:
        from rtde_receive import RTDEReceiveInterface
    except ImportError:
        print('ur_rtde is not installed (pip install ur_rtde).', file=sys.stderr)
        return 1

    frames = _frames(cfg)
    base = cfg.get('base_frame', 'base_link')
    cam_name = cfg.get('camera_frame', 'camera')

    print(f'Connecting (read-only) to {ip} ...')
    rtde_r = RTDEReceiveInterface(ip)
    rtde_c = None
    if args.freedrive:
        from rtde_control import RTDEControlInterface
        rtde_c = RTDEControlInterface(ip)          # uploads a control script -- needs Remote Control
        rtde_c.teachMode()
        print('SOFTWARE freedrive ON (teachMode). Push the arm by hand.  Ctrl-C to stop.')
    else:
        print('READ-ONLY. Put the robot in FREEDRIVE on the pendant and drive it by hand. '
              ' Ctrl-C to stop.')

    gripper = None
    if args.gripper:
        try:
            from ..robot.gripper import Robotiq2F85
            gripper = Robotiq2F85(cfg)
        except Exception as exc:                   # noqa: BLE001 -- gripper is optional
            print(f'  (gripper unavailable: {exc})')
            gripper = None

    writer, csvf = None, None
    if args.csv:
        import csv as _csv
        csvf = open(args.csv, 'w', newline='')
        writer = _csv.writer(csvf)
        hdr = ['t'] + [f'q{i}_deg' for i in range(6)]
        for fn in frames:
            hdr += [f'{fn}_{s}' for s in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')]
        if args.wrench:
            hdr += ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
        if gripper is not None:
            hdr += ['gripper_counts']
        writer.writerow(hdr)

    period = 1.0 / max(1e-3, args.rate)
    n_lines = 0
    t0 = time.monotonic()
    try:
        while True:
            q = np.degrees(np.asarray(rtde_r.getActualQ(), dtype=float))
            T_base_tool0 = rtde_to_matrix(rtde_r.getActualTCPPose())

            lines = ['joints (deg): [' + ', '.join(f'{v:+7.2f}' for v in q) + ']']
            for name, T_t0 in frames.items():
                label = cam_name if name == 'camera' else name
                lines.append(f'  {base} <- {label:16s}: {_fmt(T_base_tool0 @ T_t0)}')

            ft = None
            if args.wrench:
                ft = np.asarray(rtde_r.getActualTCPForce(), dtype=float)
                lines.append(f'  wrench: F=[{ft[0]:+6.1f}, {ft[1]:+6.1f}, {ft[2]:+6.1f}] N   '
                             f'T=[{ft[3]:+5.2f}, {ft[4]:+5.2f}, {ft[5]:+5.2f}] Nm')
            gpos = None
            if gripper is not None:
                gpos = gripper.position()
                lines.append(f'  gripper: {gpos} counts')

            if n_lines:                            # redraw in place (move the cursor back up)
                sys.stdout.write(f'\x1b[{n_lines}A')
            sys.stdout.write('\r' + '\n'.join(ln.ljust(96) for ln in lines) + '\n')
            sys.stdout.flush()
            n_lines = len(lines)

            if writer is not None:
                row = [f'{time.monotonic() - t0:.3f}'] + [f'{v:.3f}' for v in q]
                for _, T_t0 in frames.items():
                    xyz, rpy = matrix_to_xyzrpy(T_base_tool0 @ T_t0)
                    row += [f'{xyz[0]:.5f}', f'{xyz[1]:.5f}', f'{xyz[2]:.5f}',
                            f'{rpy[0]:.5f}', f'{rpy[1]:.5f}', f'{rpy[2]:.5f}']
                if ft is not None:
                    row += [f'{v:.4f}' for v in ft]
                if gpos is not None:
                    row += [str(gpos)]
                writer.writerow(row)
                csvf.flush()

            if args.once:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        if rtde_c is not None:
            rtde_c.endTeachMode()
            rtde_c.disconnect()
        rtde_r.disconnect()
        if csvf is not None:
            csvf.close()
        print('\nStopped.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
