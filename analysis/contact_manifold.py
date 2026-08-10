"""Build a CONTACT MANIFOLD csv from uncertain_sampling and/or estimator_eval logs.

The contact manifold pairs "at this relative pose, the part feels this wrench" over many trials,
in the two column groups that are frame-invariant w.r.t. the cell:

    connector_target_*    the connector's pose w.r.t. the TARGET connector -- identity at a perfect
                          mate, so this is the misalignment that produced the contact
    wrench_connector_*    the contact wrench re-expressed in the CONNECTOR frame -- the response to
                          that misalignment, along the part's own axes

Two sources are understood, freely mixed in one build:

  * uncertain_sampling runs (one 34-column CSV per run; docs/uncertain_sampling.md). The manifold
    columns are copied out directly -- the sampler logs the TRUE relative pose (the part is
    fixtured and the belief is never wrong). The cell-specific `tool0_base_*` / `wrench_base_*`
    columns are deliberately NOT carried over.

  * estimator_eval experiment folders (data/experiments/estimator_eval_*/, one
    trial_TTT_attempt_AA_observations.csv per insertion attempt). Those rows are logged in the
    BELIEVED frame, which is wrong by exactly that attempt's injected/remaining belief error --
    known, because the part is fixtured. Each file is REBASED into the true frame using the
    err_before_* columns of the run's own trials.csv (rel_true = rel_logged @ inverse(E); the
    wrench changes frames the same way) -- the same recipe that built the augmented manifold.
    Passing a run folder sweeps its *_observations.csv; trials.csv / summary.csv are read as
    metadata for the rebase, never as samples.

Both sources land in the same 16-column schema (pose incl. quaternion + wrench), so the output is
a drop-in manifold for the estimator (urlab/skills/manifold.py) and the analysis stack. The output
path and filename are YOURS to choose via the required --out. Tip: keep the *_contact_manifold.csv
suffix when the output lands beside its inputs -- directory sweeps skip that suffix, which is what
keeps rebuilds idempotent.

    # every sampling run for the banana connector
    python analysis/contact_manifold.py data/uncertain_assembly_sampling/banana --out data/uncertain_assembly_sampling/banana/banana_connector_contact_manifold.csv

    # sampling runs + two eval campaigns, contact samples only (the estimator's own threshold)
    python analysis/contact_manifold.py data/uncertain_assembly_sampling/banana "data/experiments/estimator_eval_2026*" --min-force 3.0 --out configs/data/banana_manifold_augmented.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation

# The columns carried into the manifold, in output order. Names match the sampler's header exactly
# (urlab/apps/uncertain_sampling.py) so downstream tooling can find them by name. Units are in the
# names: translation mm, rotation deg, forces N / Nm.
_POSE = ('x_mm', 'y_mm', 'z_mm', 'qx', 'qy', 'qz', 'qw', 'yaw_deg', 'pitch_deg', 'roll_deg')
_WRENCH = ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')
POSE_COLS = [f'connector_target_{s}' for s in _POSE]
WRENCH_COLS = [f'wrench_connector_{s}' for s in _WRENCH]
MANIFOLD_COLS = POSE_COLS + WRENCH_COLS

_FORCE_COLS = [f'wrench_connector_{s}' for s in ('fx', 'fy', 'fz')]

MANIFOLD_SUFFIX = '_contact_manifold.csv'

# estimator_eval observation files: the estimator's own 6-vec pose convention (extrinsic-XYZ rpy,
# no quaternion; urlab/skills/manifold.py POSE_COLS) + the same wrench names. err_before_* in the
# run's trials.csv is the belief error E (T_true @ E = T_believed) DURING that attempt -- the
# rebase key.
_DIMS6 = ('x_mm', 'y_mm', 'z_mm', 'roll_deg', 'pitch_deg', 'yaw_deg')
EVAL_POSE_COLS = [f'connector_target_{d}' for d in _DIMS6]
EVAL_OBS_COLS = EVAL_POSE_COLS + WRENCH_COLS
ERR_BEFORE_COLS = [f'err_before_{d}' for d in _DIMS6]
EVAL_META = ('trials.csv', 'summary.csv')
_EVAL_OBS_RE = re.compile(r'^trial_(\d+)_(?:attempt_(\d+)|final_insertion)_observations\.csv$')


def _eval_key(basename):
    """(trial, attempt) of an estimator_eval observation file -- trials.csv's key -- or None."""
    m = _EVAL_OBS_RE.match(basename)
    if not m:
        return None
    return int(m.group(1)), (str(int(m.group(2))) if m.group(2) else 'final_insertion')


def expand_inputs(patterns):
    """File paths from a mix of files, directories and globs. Directories contribute their *.csv
    (a glob that matches directories -- e.g. data/experiments/estimator_eval_* -- does the same).

    Sorted and de-duplicated so a rebuild is reproducible and passing the same file twice (e.g. via
    an overlapping glob) cannot silently double-count it.

    Previously-built manifolds are EXCLUDED. The output typically lands beside its inputs and
    carries the same columns, so a second run over the same directory would happily read its own
    output back in and silently double every sample. Only a directory/glob sweep is filtered --
    naming a manifold file explicitly still works, for deliberately merging one into another.
    An estimator_eval run's trials.csv / summary.csv are likewise dropped from sweeps: trials.csv
    is the rebase's truth table, not observation data."""
    out, self_ingest, meta = [], [], []
    for pat in patterns:
        swept = True
        if os.path.isdir(pat):
            hits = sorted(glob.glob(os.path.join(pat, '*.csv')))
        elif os.path.isfile(pat):
            hits, swept = [pat], False
        else:
            hits = []
            for h in sorted(glob.glob(pat, recursive=True)):
                hits.extend(sorted(glob.glob(os.path.join(h, '*.csv')))
                            if os.path.isdir(h) else [h])
        if swept:
            keep = []
            for h in hits:
                if h.endswith(MANIFOLD_SUFFIX):
                    self_ingest.append(h)
                elif os.path.basename(h) in EVAL_META:
                    meta.append(h)
                else:
                    keep.append(h)
            hits = keep
        if not hits:
            print(f'warning: no CSV matched {pat!r}', file=sys.stderr)
        out.extend(hits)
    for h in self_ingest:
        print(f'  (ignoring existing manifold {os.path.basename(h)})', file=sys.stderr)
    for h in meta:
        print(f'  ({os.path.basename(h)} is eval-run metadata -- used for the rebase, '
              'not ingested)', file=sys.stderr)
    seen, unique = set(), []
    for p in out:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def classify(path):
    """('sampling' | 'eval', None) from the file's header, or (None, reason-to-skip).

    A previously-built manifold named EXPLICITLY (a deliberate merge) classifies as 'sampling' --
    it carries the same columns and copies through unchanged."""
    if os.path.basename(path) in EVAL_META:
        return None, ("an estimator_eval run's metadata, not observation data -- pass the run "
                      'DIRECTORY instead')
    with open(path, newline='') as fh:
        cols = next(csv.reader(fh), None) or []
    if all(c in cols for c in MANIFOLD_COLS):
        return 'sampling', None
    if all(c in cols for c in EVAL_OBS_COLS):
        return 'eval', None
    # Distinguish the one format change we know about. Logs written before translations moved to
    # mm carry the same fields WITHOUT the `_mm` suffix and in METRES; reading them as-is would
    # concatenate metres with millimetres, which is worse than skipping them.
    if 'connector_target_x' in cols:
        return None, ('pre-mm log format (translations in METRES, un-suffixed) -- re-collect, or '
                      'scale x/y/z by 1000 and rename to *_mm first')
    missing = [c for c in MANIFOLD_COLS if c not in cols]
    return None, f'missing {len(missing)} manifold column(s), first: {missing[0]}'


def read_manifold_rows(path, min_force=None):
    """(rows, n_read, note) -- the manifold columns of one sampling log.

    `note` is a human-readable reason when the file is skipped entirely, so the caller can report
    it rather than failing the whole batch."""
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in MANIFOLD_COLS if c not in (reader.fieldnames or [])]
        if missing:
            return [], 0, f'missing {len(missing)} column(s), first: {missing[0]}'
        rows, n_read = [], 0
        for rec in reader:
            n_read += 1
            if min_force is not None:
                try:
                    f = math.sqrt(sum(float(rec[c]) ** 2 for c in _FORCE_COLS))
                except (TypeError, ValueError):
                    continue                       # unparseable row -- drop it rather than crash
                if f < min_force:
                    continue
            rows.append(rec)
        return rows, n_read, None


# ---------------------------------------------------------------------------------------------
# estimator_eval ingestion: believed-frame observation rows -> true-frame manifold rows.
# ---------------------------------------------------------------------------------------------

def _mats_from_vec6(v):
    """(..., 6) [x,y,z (mm), roll,pitch,yaw (deg, extrinsic XYZ)] -> (..., 4, 4) -- the estimator's
    own convention (urlab.skills.manifold.mats_from_vec6, duplicated so analysis/ stays free of
    urlab imports; the smoke test pins the two against each other via the round-trip identity)."""
    v = np.asarray(v, dtype=float)
    shp = v.shape[:-1]
    T = np.zeros(shp + (4, 4))
    T[..., :3, :3] = (Rotation.from_euler('xyz', v.reshape(-1, 6)[:, 3:], degrees=True)
                      .as_matrix().reshape(shp + (3, 3)))
    T[..., :3, 3] = v[..., :3]
    T[..., 3, 3] = 1.0
    return T


def _rebase_to_truth(rows12, err6):
    """estimator_eval rows [pose6 | f | tau] from the BELIEVED frame into the TRUE frame, emitted
    in MANIFOLD_COLS order (N, 16) -- quaternion computed from the rebased rotation.

    A logged pose is rel_logged = rel_true @ E with E the attempt's belief error (err_before_*,
    mm/deg), so rel_true = rel_logged @ inverse(E). The wrench columns live in the believed
    connector frame, which moves the same way: transform_wrench's convention with the lever arm in
    METRES -- the exact inverse of estimator_eval._rebase_rows' accumulation update."""
    E = _mats_from_vec6(np.asarray(err6, dtype=float))
    if not len(rows12):
        return np.zeros((0, len(MANIFOLD_COLS)))
    pose = _mats_from_vec6(rows12[:, :6]) @ np.linalg.inv(E)
    rot = Rotation.from_matrix(pose[:, :3, :3])
    rpy = rot.as_euler('xyz', degrees=True)
    # inverse(inverse(E) in metres) = E in metres: rotate by E's rotation, lever arm E's t / 1000.
    R, p = E[:3, :3], E[:3, 3] / 1000.0
    f = rows12[:, 6:9] @ R.T
    tau = rows12[:, 9:12] @ R.T + np.cross(p, f)
    return np.hstack([pose[:, :3, 3], rot.as_quat(), rpy[:, ::-1], f, tau])


def load_truth_table(run_dir):
    """(trial, attempt) -> that attempt's belief error [err_before_* mm/deg] from an
    estimator_eval run's trials.csv; None when the folder has no trials.csv at all."""
    path = os.path.join(run_dir, 'trials.csv')
    if not os.path.isfile(path):
        return None
    table = {}
    with open(path, newline='') as fh:
        for rec in csv.DictReader(fh):
            try:
                table[(int(rec['trial']), rec['attempt'])] = [float(rec[c])
                                                              for c in ERR_BEFORE_COLS]
            except (KeyError, TypeError, ValueError):
                continue                # malformed/aborted row -- its observations just won't pair
    return table


def read_eval_rows(path, truth, min_force=None):
    """(rows16, n_read, note) -- one estimator_eval observation CSV, rebased into the TRUE frame.

    rows16 is a float array in MANIFOLD_COLS order. `note` explains a whole-file skip: an
    unrecognized filename (no way to pair it with a trials.csv row), a run folder without
    trials.csv, or a missing row (a run aborted mid-attempt saves observations first)."""
    key = _eval_key(os.path.basename(path))
    if key is None:
        return None, 0, ('does not match trial_*_{attempt_*|final_insertion}_observations.csv -- '
                         'cannot pair it with a trials.csv row for the rebase')
    if truth is None:
        return None, 0, 'no trials.csv beside it -- believed-frame rows cannot be rebased'
    err6 = truth.get(key)
    if err6 is None:
        return None, 0, (f'trials.csv has no row for trial {key[0]} attempt {key[1]} '
                         '(run aborted mid-attempt?)')
    rows, n_read = [], 0
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for rec in reader:
            n_read += 1
            try:
                rows.append([float(rec[c]) for c in EVAL_OBS_COLS])
            except (TypeError, ValueError):
                continue                           # unparseable row -- drop it rather than crash
    arr = np.asarray(rows, dtype=float).reshape(-1, 12)
    if min_force is not None and len(arr):
        arr = arr[np.linalg.norm(arr[:, 6:9], axis=1) >= float(min_force)]
    return _rebase_to_truth(arr, err6), n_read, None


def build(paths, out_path, min_force=None, with_source=False):
    """Concatenate every input into one manifold CSV. Returns the number of rows written.

    Sampling logs copy through; estimator_eval observation files are rebased into the true frame
    via their run folder's trials.csv (loaded once per folder)."""
    header = (['source_file', 'trial'] if with_source else []) + MANIFOLD_COLS
    written = skipped = total_read = 0
    truths = {}                                    # eval run dir -> trials.csv table (or None)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for path in paths:
            base = os.path.basename(path)
            kind, note = classify(path)
            out_rows, n_read = [], 0
            if kind == 'sampling':
                recs, n_read, note = read_manifold_rows(path, min_force)
                for rec in recs:
                    lead = [base, rec.get('trial', '')] if with_source else []
                    out_rows.append(lead + [rec[c] for c in MANIFOLD_COLS])
            elif kind == 'eval':
                run_dir = os.path.dirname(os.path.abspath(path))
                if run_dir not in truths:
                    truths[run_dir] = load_truth_table(run_dir)
                arr, n_read, note = read_eval_rows(path, truths[run_dir], min_force)
                if not note:
                    trial_id = '{}/{}'.format(*_eval_key(base))
                    for r in arr:
                        lead = [base, trial_id] if with_source else []
                        out_rows.append(lead + [f'{v:.6f}' for v in r])
            if note:
                print(f'  SKIP {base}: {note}', file=sys.stderr)
                skipped += 1
                continue
            total_read += n_read
            writer.writerows(out_rows)
            written += len(out_rows)
            kept = '' if min_force is None else f'  (kept {len(out_rows)}/{n_read})'
            tag = '  [eval, rebased into the true frame]' if kind == 'eval' else ''
            print(f'  {base}: {len(out_rows)} rows{kept}{tag}')

    if min_force is not None:
        print(f'\nforce filter |f| >= {min_force} N: kept {written} of {total_read} samples')
    if skipped:
        print(f'{skipped} file(s) skipped -- see the SKIP lines above', file=sys.stderr)
    return written


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Columns kept: connector_target_* (pose wrt target) + wrench_connector_* (wrench '
               'in the connector frame). estimator_eval rows are rebased into the TRUE frame via '
               "their run's trials.csv before landing in those columns.")
    ap.add_argument('inputs', nargs='+',
                    help='sampling CSVs, estimator_eval run folders, directories, or globs')
    ap.add_argument('--out', required=True, metavar='PATH',
                    help='output CSV path INCLUDING the filename. Tip: keep the '
                         f'*{MANIFOLD_SUFFIX} suffix when it lands beside its inputs -- directory '
                         'sweeps skip that suffix, keeping rebuilds idempotent')
    ap.add_argument('--min-force', type=float, default=None, metavar='N',
                    help='keep only samples with |f| >= N in the connector frame (default: keep '
                         'all samples, including the free-space approach)')
    ap.add_argument('--with-source', action='store_true',
                    help='also emit source_file and trial columns for provenance '
                         '(eval rows: trial/attempt)')
    args = ap.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        ap.error('no input CSVs found')

    # Reading the file we are about to truncate would lose data AND double-count it.
    out_key = os.path.normcase(os.path.abspath(args.out))
    if any(os.path.normcase(os.path.abspath(p)) == out_key for p in paths):
        ap.error(f'the output {args.out!r} is also an input -- choose a different --out')

    print(f'{len(paths)} input file(s)')
    n = build(paths, args.out, args.min_force, args.with_source)
    if n == 0:
        print('\nNo rows written -- every input was skipped or filtered out.', file=sys.stderr)
        return 1
    print(f'\nWrote {n} samples x {len(MANIFOLD_COLS)} columns -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
