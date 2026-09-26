"""ASSEMBLY CALIBRATION -- teach where an object has to end up to be assembled, while holding it.

    python -m urlab.apps.coupler_assembly_calibration --set object_name=ORU_v7 \\
                                                      --set assembly_name=rack_slot_1

Writes one `assemblies:` entry into configs/objects.yaml, under the object: the base_link pose of
the object's mating frame at the assembled position. apps/coupler_pick_assemble drives to it.

KINEMATIC, AND ONLY THAT. The pose is read off forward kinematics with the object hand-guided
into its fixture, so it is exactly as good as the cell staying put. Unbolt the fixture, nudge the
bench, re-zero the robot, and every assembly taught here is wrong with nothing to notice it. That
is the trade a taught pose makes; the alternative -- a marker rig on the fixture, found by sight
every run -- is what frames.yaml's marker_rigs exist for. Taught poses are quicker, need no
line of sight, and are right until something moves.

WHAT IS RECORDED IS THE COUPLER FRAME, not the flange. Once the object is locked on, its mating
frame IS `coupler_mate` (see apps/object_calibration), so

    T_base_assembly = T_base_tool0 @ T_tool0_coupler_mate

is the object's own pose at the assembly, in terms that survive a change of tool stack. Recording
tool0 instead would bake the 55 mm coupler into the number and silently invalidate it the first
time the coupler is re-machined.

NO CAMERA. Nothing is detected -- the object is already in hand and the fixture is wherever it
is. That also means this runs with the camera unplugged.

THE PAYLOAD IS SET BEFORE FREEDRIVE, which matters more than it sounds. teachMode hands the arm
to gravity compensation; told the wrong mass it will sag or fight while the operator is trying to
seat a part by hand, and the pose finally recorded is the one they settled for rather than the
one they meant.

Output: data/experiments/assembly_calibration_<stamp>/ -- the yaml entry and a per-approach CSV.
The catalogue is MERGED (this object's other assemblies, and every other object, left alone) and
the previous version is backed up into the run directory first.
"""

import csv as _csv
import os
import shutil
from datetime import datetime

import numpy as np

from .. import behaviors as bt
from .. import log as urlog
from .. import tool_frames
from ..apps._common import ask, experiment_dir, prompts_off
from ..robot.coupler import Coupler
from ..transforms import average_pose, matrix_to_xyzrpy, pose_error, translation_matrix
from ._runner import run_app
from .coupler_pick_place import parse_offset
from .object_calibration import merge_catalogue, yaml_document

log = urlog.get('assembly-calib')

_CSV_HEADER = ['approach', 'x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg',
               'point_dev_mm', 'rot_dev_deg']


def fuse_approaches(poses):
    """(T_base_assembly, residual_mm, residual_deg) over the taught approaches.

    A PLAIN RIGID-POSE MEAN, all six degrees of freedom. Unlike the pick, nothing here is
    reconstructed and nothing is unconstrained: the fixture decides the full pose, so every axis
    of the spread is evidence of how repeatably the part can be seated by hand.

    THE SPREAD IS THE POINT. One approach gives a pose with no error bar at all -- and an
    assembly taught once is a number, not a measurement. Pure, so the arithmetic is testable
    without a robot."""
    if not poses:
        raise ValueError('fuse_approaches needs at least one approach')
    rows = list(poses)
    T, lin, ang = average_pose(rows)
    return T, lin * 1000.0, float(np.degrees(ang))


class _AssemblyCalibration:
    """One assembly: hold the object, hand-guide it home, repeat, fuse, write the catalogue."""

    def __init__(self, cfg, robot, name, assembly, obj, coupler, out_dir):
        self.cfg = cfg
        self.robot = robot
        self.name = name
        self.assembly = assembly
        self.obj = obj
        self.coupler = coupler
        self.out_dir = out_dir
        self.T_tool0_coupler = tool_frames.coupler_mate(cfg)
        self.repeats = max(1, int(cfg.get('approaches', 3)))
        self.retreat = parse_offset(cfg.section('retreat'), 'retreat', 100.0)
        self.poses = []
        self.result = None

    # ---- hold the object ---------------------------------------------------------------------
    def take_object(self):
        """Get the object onto the coupler, and tell the controller it is there.

        THE PAYLOAD GOES IN BEFORE ANY FREEDRIVE. teachMode is gravity compensation; told the
        wrong mass it sags or fights, and a pose seated against a fighting arm is the pose the
        operator gave up at."""
        if not self.coupler.prepare_to_mate():
            log.error('The coupler could not be powered and opened.')
            return False
        if not self._hand_guide('Seat the COUPLER in the object and support it, then press '
                                'Enter to LOCK (q to abort): '):
            return False
        if not self.coupler.hold():
            return False
        if self.coupler.verify() is False:
            log.error('The coupler locked and then the sensor disagreed -- not holding.')
            return False

        held = self.obj.get('held_mass_kg')
        if not held:
            log.warning('%r has no held_mass_kg, so freedrive will be compensating for a tool '
                        'that is lighter than what is actually on it. Expect it to sag.',
                        self.name)
            return True
        from .coupler_pick_place import combined_payload
        cog_mm = self.cfg.get('held_cog_mm')
        cog_m = (np.asarray(cog_mm, dtype=float) / 1000.0 if cog_mm is not None
                 else self.T_tool0_coupler[:3, 3])
        payload = combined_payload(self.cfg.section('robot').get('payload', {}), held, cog_m)
        log.info('Payload -> %.2f kg before freedrive.', payload['mass_kg'])
        self.robot.arm.set_payload(payload)
        return True

    def release_object(self):
        payload = dict(self.cfg.section('robot').get('payload', {}) or {})
        self.robot.arm.set_payload(payload)
        log.info('Payload -> %.2f kg (tool alone). Release the object by hand when ready.',
                 float(payload.get('mass_kg', 0.0)))
        return True

    # ---- teach -------------------------------------------------------------------------------
    def collect(self):
        for k in range(self.repeats):
            log.info('---- APPROACH %d of %d ----', k + 1, self.repeats)
            if not self._hand_guide(
                    'Seat the OBJECT in its assembled position, then press Enter '
                    '(q to abort): '):
                return False
            T = self.robot.arm.tcp_pose() @ self.T_tool0_coupler
            self.poses.append(T)
            xyz, rpy = matrix_to_xyzrpy(T)
            log.info('  approach %d: object at xyz %+8.2f %+8.2f %+8.2f mm  rpy %+7.2f %+7.2f '
                     '%+7.2f deg', k + 1, xyz[0] * 1000.0, xyz[1] * 1000.0, xyz[2] * 1000.0,
                     *np.degrees(rpy))
            if k + 1 < self.repeats and not self._retreat(T):
                return False
        return bool(self.poses)

    def _retreat(self, T_at):
        """Back the object out along the configured axis so the next approach starts clear.

        Straight back out along the assembly axis by default, for the same reason every other
        retreat in this cell is: any other direction levers the part against the fixture it is
        still inside."""
        if self.retreat['distance_m'] <= 0:
            return True
        step = -self.retreat['distance_m'] * self.retreat['axis']
        T_out = (T_at @ translation_matrix(step) if self.retreat['frame'] == 'coupler'
                 else np.array(T_at, dtype=float))
        if self.retreat['frame'] == 'base':
            T_out[:3, 3] = T_at[:3, 3] + step
        return self.robot.arm.move_frame_to(T_out, self.T_tool0_coupler,
                                            'retreat off the assembly')

    def _hand_guide(self, prompt):
        """Software freedrive for the length of ONE prompt, guaranteed off again after.

        The try/finally is not decoration: teachMode leaves the arm compliant, and an abort or a
        Ctrl-C would otherwise return with it still on -- carrying the object."""
        if self.robot.arm.dry_run or prompts_off(self.cfg):
            return True
        self._freedrive(True)
        try:
            return ask(prompt)
        finally:
            self._freedrive(False)

    def _freedrive(self, on):
        rtde_c = getattr(self.robot.arm, 'rtde_c', None)
        if rtde_c is None:
            return True
        try:
            rtde_c.teachMode() if on else rtde_c.endTeachMode()
        except Exception as exc:                  # noqa: BLE001 -- freedrive is a convenience
            log.warning('Could not %s software freedrive (%s) -- use the pendant.',
                        'enter' if on else 'leave', exc)
            return False
        if on:
            log.info('SOFTWARE FREEDRIVE ON -- the arm is compliant and CARRYING THE OBJECT.')
        return True

    # ---- fuse + write ------------------------------------------------------------------------
    def fuse(self):
        self.result = fuse_approaches(self.poses)
        _T, res_mm, res_deg = self.result
        log.info('FUSED %d approach%s: %.2f mm / %.2f deg spread.',
                 len(self.poses), '' if len(self.poses) == 1 else 'es', res_mm, res_deg)
        if len(self.poses) < 2:
            log.warning('ONE APPROACH ONLY -- the residuals are zero by construction and say '
                        'nothing about how repeatably this part can be seated. Set '
                        'approaches: >= 3 before trusting the number.')
        cap_mm = self.cfg.get('max_residual_mm')
        cap_deg = self.cfg.get('max_residual_deg')
        ok = True
        for value, cap, unit, what in ((res_mm, cap_mm, 'mm', 'position'),
                                       (res_deg, cap_deg, 'deg', 'orientation')):
            if cap is not None and value > float(cap):
                log.error('The %s spread is %.2f %s, over the configured %.2f %s.',
                          what, value, unit, float(cap), unit)
                ok = False
        return ok

    def write_outputs(self):
        T, res_mm, res_deg = self.result
        stamp = f'{datetime.now():%Y-%m-%d}'
        entry = dict(self.obj)
        entry['assemblies'] = dict(entry.get('assemblies') or {})
        entry['assemblies'][self.assembly] = {
            'T_base_assembly': T,
            'meta': {'approaches': len(self.poses), 'residual_mm': round(res_mm, 2),
                     'residual_deg': round(res_deg, 2), 'measured': stamp}}

        self._write_csv()
        path = tool_frames.objects_path(self.cfg)
        if not bool(self.cfg.get('write_catalogue', True)):
            log.info('write_catalogue is off -- %s not touched.', path)
            return True
        if os.path.isfile(path):
            shutil.copy2(path, os.path.join(self.out_dir, 'objects.yaml.bak'))
        try:
            merged = merge_catalogue(path, self.name, entry)
        except ValueError as exc:
            log.error('%s -- refusing to rewrite the catalogue over a file that does not load.',
                      exc)
            return False
        with open(path, 'w') as fh:
            fh.write(yaml_document(merged, stamp=stamp))
        xyz, rpy = matrix_to_xyzrpy(T)
        log.info('WROTE assembly %r for %r into %s:\n    xyz_mm:  [%s]\n    rpy_deg: [%s]',
                 self.assembly, self.name, path,
                 ', '.join('%+.2f' % (v * 1000.0) for v in xyz),
                 ', '.join('%+.2f' % np.degrees(v) for v in rpy))
        return True

    def _write_csv(self):
        T_mean = self.result[0]
        with open(os.path.join(self.out_dir, 'approaches.csv'), 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(_CSV_HEADER)
            for i, T in enumerate(self.poses, start=1):
                xyz, rpy = matrix_to_xyzrpy(T)
                lin, ang = pose_error(T_mean, T)
                w.writerow([i]
                           + [round(float(v) * 1000.0, 3) for v in xyz]
                           + [round(float(np.degrees(v)), 3) for v in rpy]
                           + [round(lin * 1000.0, 4), round(float(np.degrees(ang)), 4)])


def build_and_run(cfg, robot, camera, args):
    name, assembly = cfg.get('object_name'), cfg.get('assembly_name')
    path = tool_frames.objects_path(cfg)
    try:
        catalogue = tool_frames.load_objects(cfg)
    except ValueError as exc:
        log.error('%s', exc)
        return False
    if not name or name not in catalogue:
        log.error('object_name must name a catalogued object. In %s: %s.',
                  path, ', '.join(sorted(catalogue)) or '(none)')
        return False
    if not assembly:
        log.error('assembly_name is required -- it is the key this run writes under %r. '
                  'Existing: %s.', name,
                  ', '.join(sorted(catalogue[name]['assemblies'])) or '(none)')
        return False
    obj = catalogue[name]
    if assembly in obj['assemblies']:
        log.info('Re-teaching assembly %r for %r (previous: %s).', assembly, name,
                 obj['assemblies'][assembly]['meta'].get('measured', '?'))

    out_dir = experiment_dir(cfg, 'assembly_calibration')
    log.info('Output: %s', out_dir)
    log.info('Teaching assembly %r for %r over %d approach(es). Nothing is detected -- this is '
             'a KINEMATIC record, good only while the cell stays put.',
             assembly, name, int(cfg.get('approaches', 3)))

    coupler = Coupler(cfg)
    cal = _AssemblyCalibration(cfg, robot, name, assembly, obj, coupler, out_dir)
    root = bt.sequence(
        'assembly-calibration',
        bt.Action('lock the object onto the coupler', cal.take_object),
        bt.Action('teach the assembled position', cal.collect),
        bt.Action('fuse the approaches', cal.fuse),
        bt.Action('write objects.yaml', cal.write_outputs),
        bt.Action('hand the object back', cal.release_object))
    try:
        return bt.run_tree(root, log)
    finally:
        coupler.close()


def main():
    # needs_camera=False: nothing is detected here. It runs with the camera unplugged.
    run_app('Assembly calibration: where a held object sits when it is assembled',
            'coupler_assembly_calibration', build_and_run, with_gripper=False,
            needs_camera=False)


if __name__ == '__main__':
    main()
