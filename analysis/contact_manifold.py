"""Build a CONTACT MANIFOLD csv from uncertain_sampling and/or estimator_eval logs.

The contact manifold pairs "at this relative pose, the part feels this wrench" over many trials,
in the two column groups that are frame-invariant w.r.t. the cell:

    connector_target_*    the connector's pose w.r.t. the TARGET connector -- identity at a perfect
                          mate, so this is the misalignment that produced the contact
    wrench_connector_*    the contact wrench re-expressed in the CONNECTOR frame -- the response to
                          that misalignment, along the part's own axes

Three sources are understood, freely mixed in one build:

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

  * wiggle_sampling runs (apps/wiggle_sampling.py; configs/data/wiggle_sampling/<geometry>/
    run_<stamp>/samples.csv). The pose and wrench columns are the sampler's, so they copy through
    like a sampling log -- but a wiggle run also logs its TRANSIT, and those rows are not contact
    at any misalignment. `datum` is the free-space return-to-reference pose 60 mm off the mate in
    two axes; `approach` and `retract` run from a 25 mm standoff. By default only the contact
    phases are kept (quiet, wiggle, press, recentre, both tugs); --segments overrides that.
    Point at a geometry directory to sweep every run under it, or at a single run_<stamp>/.
    bursts.csv is per-burst metadata and is skipped, like an eval run's trials.csv.

All three sources land in the same 16-column schema (pose incl. quaternion + wrench), so the
output is a drop-in manifold for the estimator (urlab/skills/manifold.py) and the analysis stack.
The output path and filename are YOURS to choose via the required --out. Tip: keep the
*_contact_manifold.csv suffix when the output lands beside its inputs -- directory sweeps skip
that suffix, which is what keeps rebuilds idempotent.

    # every sampling run for the banana connector
    python analysis/contact_manifold.py data/uncertain_assembly_sampling/banana --out data/uncertain_assembly_sampling/banana/banana_connector_contact_manifold.csv

    # sampling runs + two eval campaigns, contact samples only (the estimator's own threshold)
    python analysis/contact_manifold.py data/uncertain_assembly_sampling/banana "data/experiments/estimator_eval_2026*" --min-force 3.0 --out configs/data/banana_manifold_augmented.csv

    # every wiggle run for one geometry (contact phases only, which is the default)
    python analysis/contact_manifold.py configs/data/wiggle_sampling/bnc --out configs/data/bnc_wiggle_manifold.csv

    # one run, and keep the steady holds only -- no oscillation, no tug
    python analysis/contact_manifold.py configs/data/wiggle_sampling/bnc/run_20260819_023144 --segments quiet,press --out configs/data/bnc_quiet_manifold.csv

    # a wiggle geometry thinned to 200k rows -- kNN cost scales with the row count, and a 2.5M-row map is mostly redundant
    python analysis/contact_manifold.py configs/data/wiggle_sampling/bnc --downsample 200000 --seed 0 --out configs/data/bnc_wiggle_200k_manifold.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
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

# wiggle_sampling: samples.csv carries the sampler's columns PLUS a `segment` label naming which
# phase of the burst each row came from. bursts.csv is the per-burst summary -- metadata here.
WIGGLE_META = ('bursts.csv',)
# The phases where the part is AT a pose and in contact. `approach`/`retract` are transit from a
# 25 mm standoff and `datum` is the free-space reference pose 60 mm away; none of them is contact
# at a misalignment, and all of them stretch the manifold's extent enough to distort the kNN
# distance scaling that the support/OOD reference is computed against.
WIGGLE_CONTACT_SEGMENTS = ('quiet', 'wiggle', 'press', 'recentre', 'tug_axial', 'tug_lateral')
WIGGLE_TRANSIT_SEGMENTS = ('approach', 'retract', 'datum')

# How many rows a file needs before _degenerate_note is willing to call it a dry run. See there.
_DEGENERATE_MIN_ROWS = 200
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
            # ONE LEVEL DEEPER TOO: a wiggle_sampling geometry directory holds run_<stamp>/
            # subdirectories rather than CSVs, so a sweep that only looked at *.csv found nothing
            # there. Harmless for the flat layouts -- they have no subdirectories to match.
            hits = sorted(set(glob.glob(os.path.join(pat, '*.csv'))
                              + glob.glob(os.path.join(pat, '*', '*.csv'))))
        elif os.path.isfile(pat):
            hits, swept = [pat], False
        else:
            hits = []
            for h in sorted(glob.glob(pat, recursive=True)):
                hits.extend(sorted(set(glob.glob(os.path.join(h, '*.csv'))
                                       + glob.glob(os.path.join(h, '*', '*.csv'))))
                            if os.path.isdir(h) else [h])
        if swept:
            keep = []
            for h in hits:
                if h.endswith(MANIFOLD_SUFFIX):
                    self_ingest.append(h)
                elif os.path.basename(h) in EVAL_META + WIGGLE_META:
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
        why = ('used for the rebase, not ingested' if os.path.basename(h) in EVAL_META
               else 'per-burst summary, not observation data')
        print(f'  ({os.path.basename(h)} is run metadata -- {why})', file=sys.stderr)
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
    if os.path.basename(path) in WIGGLE_META:
        return None, ("a wiggle_sampling run's per-burst summary, not observation data -- pass "
                      'the run DIRECTORY (or the geometry directory above it) instead')
    with open(path, newline='') as fh:
        cols = next(csv.reader(fh), None) or []
    if all(c in cols for c in MANIFOLD_COLS):
        # A wiggle run carries the sampler's columns plus a phase label. Distinguished so the
        # transit rows can be dropped and so the report says which kind it was.
        return ('wiggle' if 'segment' in cols else 'sampling'), None
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


def read_manifold_rows(path, min_force=None, segments=None):
    """(rows, n_seen, n_offseg, note) -- the manifold columns of one sampling or wiggle log.

    `segments`, when given, keeps only rows whose `segment` column is in it -- how a wiggle run's
    transit and free-space datum rows are left out. It is ignored for logs with no such column,
    so a plain sampling log reads exactly as before.

    TWO FILTERS, TWO COUNTS. `n_offseg` is what the segment filter removed and `n_seen` is what
    then reached the force filter, so each rejection can be attributed to the filter that actually
    made it. A single combined figure reads as though the force threshold rejected the transit
    rows too, which makes the threshold look far more aggressive than it is.

    `note` is a human-readable reason when the file is skipped entirely, so the caller can report
    it rather than failing the whole batch."""
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in MANIFOLD_COLS if c not in (reader.fieldnames or [])]
        if missing:
            return [], 0, 0, f'missing {len(missing)} column(s), first: {missing[0]}'
        has_seg = 'segment' in (reader.fieldnames or [])
        keep_seg = set(segments) if (segments and has_seg) else None
        rows, n_seen, n_offseg = [], 0, 0
        for rec in reader:
            if keep_seg is not None and rec.get('segment') not in keep_seg:
                n_offseg += 1
                continue
            n_seen += 1
            if min_force is not None:
                try:
                    f = math.sqrt(sum(float(rec[c]) ** 2 for c in _FORCE_COLS))
                except (TypeError, ValueError):
                    continue                       # unparseable row -- drop it rather than crash
                if f < min_force:
                    continue
            rows.append(rec)
        note = _degenerate_note(rows)
        if note:
            return [], n_seen, n_offseg, note
        return rows, n_seen, n_offseg, None


def _degenerate_note(rows):
    """A reason to reject a whole file whose rows cannot be real contact, or None.

    A DRY RUN produces a log that is structurally perfect and physically meaningless: with no
    robot, arm.fk() returns one fixed placeholder pose for every cycle and the F/T reads zeros. The
    file then has the right columns, the right row count and a valid quaternion in every row -- and
    would drop tens of thousands of identical, force-free samples into a manifold, where they are
    not merely useless but actively harmful: they pull the kNN scaling and sit as a dense fake
    cluster the estimator can match against.

    Two signatures, both of which a real log fails immediately: the pose never moves, or the wrench
    is identically zero throughout.

    ONLY APPLIED TO A SUBSTANTIAL FILE. "The pose never moves" is a claim about a distribution, and
    a handful of rows cannot support it -- a short capture, a hand-built fixture or a single held
    station can legitimately repeat one pose, and rejecting those would turn a guard against a
    silent hazard into a silent hazard of its own. A dry run writes thousands of rows, so the floor
    costs nothing on the case this exists to catch."""
    if len(rows) < _DEGENERATE_MIN_ROWS:
        return None
    try:
        xs = {rec['connector_target_x_mm'] for rec in rows}
        zs = {rec['connector_target_z_mm'] for rec in rows}
        if len(xs) == 1 and len(zs) == 1:
            return (f'every one of {len(rows)} rows has the same pose '
                    f'(x={next(iter(xs))}, z={next(iter(zs))}) -- this is a DRY-RUN log, where '
                    'arm.fk() returns a fixed placeholder; it is not contact data')
        if all(float(rec[c]) == 0.0 for rec in rows for c in _FORCE_COLS):
            return (f'the wrench is identically zero in all {len(rows)} rows -- a DRY-RUN log '
                    '(no F/T sensor), not contact data')
    except (KeyError, TypeError, ValueError):
        return None                                # malformed rows are the caller's problem
    return None


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
    """(rows16, n_seen, 0, note) -- one estimator_eval observation CSV, rebased into the TRUE
    frame. The third slot is the segment-drop count, always zero here; it exists so this and
    read_manifold_rows share one return shape.

    rows16 is a float array in MANIFOLD_COLS order. `note` explains a whole-file skip: an
    unrecognized filename (no way to pair it with a trials.csv row), a run folder without
    trials.csv, or a missing row (a run aborted mid-attempt saves observations first)."""
    key = _eval_key(os.path.basename(path))
    if key is None:
        return None, 0, 0, (
            'does not match trial_*_{attempt_*|final_insertion}_observations.csv -- '
            'cannot pair it with a trials.csv row for the rebase')
    if truth is None:
        return None, 0, 0, 'no trials.csv beside it -- believed-frame rows cannot be rebased'
    err6 = truth.get(key)
    if err6 is None:
        return None, 0, 0, (f'trials.csv has no row for trial {key[0]} attempt {key[1]} '
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
    return _rebase_to_truth(arr, err6), n_read, 0, None


def build(paths, out_path, min_force=None, with_source=False, segments=None,
          downsample=None, seed=None):
    """Concatenate every input into one manifold CSV. Returns the number of rows written.

    Sampling logs copy through; estimator_eval observation files are rebased into the true frame
    via their run folder's trials.csv (loaded once per folder).

    `downsample` (a row count) keeps a uniformly random sample of that size from the POOLED
    result -- pooled, not per file; see the Algorithm R comment below for why. It runs LAST,
    after the segment and force filters, so it thins what those chose to keep rather than
    competing with them. `seed` makes the draw reproducible; the caller is expected to supply
    one so the run can be repeated (main() invents and prints one when the operator does not)."""
    header = (['source_file', 'trial'] if with_source else []) + MANIFOLD_COLS
    written = skipped = total_read = total_offseg = n_pooled = 0
    truths = {}                                    # eval run dir -> trials.csv table (or None)
    rng = random.Random(seed)
    reservoir = []                                 # [(row, source basename)] when downsampling

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for path in paths:
            base = os.path.basename(path)
            kind, note = classify(path)
            out_rows, n_read, n_offseg = [], 0, 0
            if kind in ('sampling', 'wiggle'):
                recs, n_read, n_offseg, note = read_manifold_rows(
                    path, min_force, segments if kind == 'wiggle' else None)
                for rec in recs:
                    lead = [base, rec.get('trial', '')] if with_source else []
                    out_rows.append(lead + [rec[c] for c in MANIFOLD_COLS])
            elif kind == 'eval':
                run_dir = os.path.dirname(os.path.abspath(path))
                if run_dir not in truths:
                    truths[run_dir] = load_truth_table(run_dir)
                arr, n_read, _, note = read_eval_rows(path, truths[run_dir], min_force)
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
            total_offseg += n_offseg
            if downsample is None:
                writer.writerows(out_rows)
                written += len(out_rows)
                n_pooled += len(out_rows)
            else:
                for row in out_rows:
                    # ALGORITHM R over the POOLED stream, not per file. Sampling each file to its
                    # own quota would preserve every file's share by construction, which is a
                    # different (and unasked-for) dataset: it silently up-weights a short run
                    # against a long one. Pooling keeps every row equally likely; the imbalance
                    # that leaves is real and is REPORTED below rather than engineered away.
                    if len(reservoir) < downsample:
                        reservoir.append((row, base))
                    else:
                        j = rng.randrange(n_pooled + 1)   # n_pooled = this row's 0-based index
                        if j < downsample:
                            reservoir[j] = (row, base)
                    n_pooled += 1
            bits = ([f'{n_offseg} off-segment'] if n_offseg else [])
            if min_force is not None:
                bits.append(f'{n_read - len(out_rows)} below {min_force} N')
            kept = f'  (dropped {", ".join(bits)})' if bits else ''
            tag = {'eval': '  [eval, rebased into the true frame]',
                   'wiggle': f'  [wiggle run: {os.path.basename(os.path.dirname(path))}]'
                   }.get(kind, '')
            # "pooled" not "rows" when downsampling: these rows are CANDIDATES at this point, and
            # how many of them survive is not known until the whole stream has been seen.
            print(f'  {base}: {len(out_rows)} rows'
                  f'{" pooled" if downsample is not None else ""}{kept}{tag}')

        if downsample is not None:
            writer.writerows(row for row, _ in reservoir)
            written = len(reservoir)

    if total_offseg:
        # Named neutrally: with the DEFAULT set what goes is transit and the free-space datum, but
        # --segments can keep any subset, and then most of what is dropped is contact the caller
        # chose to leave out. Saying "transit" there would misdescribe the caller's own filter.
        dropped_kinds = [s for s in WIGGLE_CONTACT_SEGMENTS + WIGGLE_TRANSIT_SEGMENTS
                         if s not in (segments or ())]
        print(f"\nsegment filter kept [{','.join(segments)}]: dropped {total_offseg} row(s) "
              f"of [{','.join(dropped_kinds)}]")
    if min_force is not None:
        # n_pooled, NOT written: with --downsample those differ, and charging the downsampler's
        # thinning to the force threshold would say the threshold rejects far more than it does.
        scope = ' that reached it' if total_offseg else ''
        print(f'force filter |f| >= {min_force} N: kept {n_pooled} of {total_read} '
              f'samples{scope}')
    if downsample is not None:
        if downsample >= n_pooled:
            print(f'\ndownsample {downsample} >= the {n_pooled} pooled row(s): kept them all, '
                  'nothing was thinned')
        else:
            # The per-source spread is the thing uniform sampling does NOT protect: pooling makes
            # every ROW equally likely, so a source contributes in proportion to its size. Printed
            # so a 2.5M-row wiggle run drowning out a 10k sampling log is visible, not inferred.
            per = {}
            for _row, src in reservoir:
                per[src] = per.get(src, 0) + 1
            print(f'\ndownsample: kept a uniform random {written} of {n_pooled} pooled row(s), '
                  f'seed {seed}')
            n_src = len(paths) - skipped
            if n_src > 1:                          # with one source there is no spread to report
                lo = min(per.items(), key=lambda kv: kv[1])
                hi = max(per.items(), key=lambda kv: kv[1])
                gone = n_src - len(per)
                print(f'  spread over {len(per)} of {n_src} source file(s): {hi[0]} contributes '
                      f'{hi[1]}, {lo[0]} contributes {lo[1]}'
                      + (f'; {gone} drew no rows at all' if gone else ''))
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
                    help='sampling CSVs, estimator_eval run folders, wiggle_sampling geometry or '
                         'run directories, or globs')
    ap.add_argument('--out', required=True, metavar='PATH',
                    help='output CSV path INCLUDING the filename. Tip: keep the '
                         f'*{MANIFOLD_SUFFIX} suffix when it lands beside its inputs -- directory '
                         'sweeps skip that suffix, keeping rebuilds idempotent')
    ap.add_argument('--min-force', type=float, default=None, metavar='N',
                    help='keep only samples with |f| >= N in the connector frame (default: keep '
                         'all samples, including the free-space approach)')
    ap.add_argument('--segments', default=','.join(WIGGLE_CONTACT_SEGMENTS), metavar='LIST',
                    help='wiggle_sampling runs only: comma-separated phases to keep, or "all". '
                         f'Default {",".join(WIGGLE_CONTACT_SEGMENTS)} -- the phases where the '
                         'part is at a pose and in contact. The excluded ones '
                         f'({",".join(WIGGLE_TRANSIT_SEGMENTS)}) are transit and the free-space '
                         'reference pose, 25-60 mm off the mate; they are not contact at a '
                         'misalignment and they distort the kNN distance scaling')
    ap.add_argument('--downsample', type=int, default=None, metavar='N',
                    help='keep a uniformly random N rows of the POOLED result (reservoir '
                         'sampling), applied AFTER --segments and --min-force so it thins what '
                         'they kept. Every row is equally likely, so each source contributes in '
                         'proportion to its size -- the spread is printed. The sample is held in '
                         'memory, so N is bounded by RAM, not by the input size')
    ap.add_argument('--seed', type=int, default=None, metavar='INT',
                    help='RNG seed for --downsample. Omitted = a fresh random seed, which is '
                         'PRINTED either way so any build can be reproduced exactly')
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

    segs = None if args.segments.strip().lower() == 'all' else [
        t.strip() for t in args.segments.split(',') if t.strip()]
    if segs is not None:
        unknown = [t for t in segs
                   if t not in WIGGLE_CONTACT_SEGMENTS + WIGGLE_TRANSIT_SEGMENTS]
        if unknown:
            ap.error(f'unknown segment(s) {unknown}; known: '
                     f'{list(WIGGLE_CONTACT_SEGMENTS + WIGGLE_TRANSIT_SEGMENTS)}, or "all"')

    if args.downsample is not None and args.downsample < 1:
        ap.error(f'--downsample must be >= 1 (got {args.downsample})')
    if args.seed is not None and args.downsample is None:
        print('note: --seed does nothing without --downsample -- nothing else here is random.',
              file=sys.stderr)
    # Invented rather than defaulted to a constant: a fixed default would hand two "independent"
    # downsamples of one input the SAME rows, which quietly breaks anything built on repeated
    # draws (a held-out split, a sampling-variance check). Random and always printed instead.
    seed = args.seed
    if args.downsample is not None and seed is None:
        seed = random.randrange(2 ** 31)

    print(f'{len(paths)} input file(s)')
    n = build(paths, args.out, args.min_force, args.with_source, segs, args.downsample, seed)
    if n == 0:
        print('\nNo rows written -- every input was skipped or filtered out.', file=sys.stderr)
        return 1
    print(f'\nWrote {n} samples x {len(MANIFOLD_COLS)} columns -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
