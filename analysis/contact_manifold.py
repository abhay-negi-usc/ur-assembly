"""Build a CONTACT MANIFOLD csv from a set of uncertain_sampling logs.

Each sampling run writes one timestamped CSV of 34 columns (see docs/uncertain_sampling.md). The
contact manifold is the part of that which describes CONTACT itself, independent of where the robot
happened to be in the cell:

    connector_target_*    the connector's pose w.r.t. the TARGET connector -- identity at a perfect
                          mate, so this is the misalignment that produced the contact
    wrench_connector_*    the contact wrench re-expressed in the CONNECTOR frame -- the response to
                          that misalignment, along the part's own axes

Pairing those two over many trials is the manifold: "at this relative pose, the part feels this
wrench". Both are already frame-invariant w.r.t. the cell, so runs recorded on different days, at
different socket positions, can be concatenated directly. The raw `tool0_base_*` and `wrench_base_*`
columns are cell-specific and are deliberately NOT carried over.

Output is named for the connector type: `<cable>_connector_contact_manifold.csv`, e.g.
`banana_connector_contact_manifold.csv`. The cable is inferred from the `<cable>/` subdirectory the
sampler writes into, or forced with --cable.

    # every run for the banana connector
    python analysis/contact_manifold.py data/uncertain_assembly_sampling/banana

    # explicit files / globs, contact samples only, written somewhere else
    python analysis/contact_manifold.py 'data/**/banana/*.csv' --min-force 0.5 --out-dir analysis
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

# The columns carried into the manifold, in output order. Names match the sampler's header exactly
# so downstream tooling can find them by name.
# Must match the sampler's header exactly (urlab/apps/uncertain_sampling.py). Units are in the
# names: translation mm, rotation deg, forces N / Nm.
_POSE = ('x_mm', 'y_mm', 'z_mm', 'qx', 'qy', 'qz', 'qw', 'yaw_deg', 'pitch_deg', 'roll_deg')
_WRENCH = ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')
POSE_COLS = [f'connector_target_{s}' for s in _POSE]
WRENCH_COLS = [f'wrench_connector_{s}' for s in _WRENCH]
MANIFOLD_COLS = POSE_COLS + WRENCH_COLS

_FORCE_COLS = [f'wrench_connector_{s}' for s in ('fx', 'fy', 'fz')]

MANIFOLD_SUFFIX = '_contact_manifold.csv'


def expand_inputs(patterns):
    """File paths from a mix of files, directories and globs. Directories contribute their *.csv.

    Sorted and de-duplicated so a rebuild is reproducible and passing the same file twice (e.g. via
    an overlapping glob) cannot silently double-count it.

    Previously-built manifolds are EXCLUDED. The default output lands beside its inputs and carries
    the same columns, so a second run over the same directory would happily read its own output back
    in and silently double every sample. Only a directory/glob sweep is filtered -- naming a manifold
    file explicitly still works, for deliberately merging one into another."""
    out, self_ingest = [], []
    for pat in patterns:
        swept = True
        if os.path.isdir(pat):
            hits = sorted(glob.glob(os.path.join(pat, '*.csv')))
        elif os.path.isfile(pat):
            hits, swept = [pat], False
        else:
            hits = sorted(glob.glob(pat, recursive=True))
        if swept:
            keep = [h for h in hits if not h.endswith(MANIFOLD_SUFFIX)]
            self_ingest.extend(h for h in hits if h.endswith(MANIFOLD_SUFFIX))
            hits = keep
        if not hits:
            print(f'warning: no CSV matched {pat!r}', file=sys.stderr)
        out.extend(hits)
    for h in self_ingest:
        print(f'  (ignoring existing manifold {os.path.basename(h)})', file=sys.stderr)
    seen, unique = set(), []
    for p in out:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def infer_cable(paths):
    """The cable name from the `<cable>/` directory the sampler writes into.

    Returns (name, None) or (None, reason). Mixing cables is an ERROR, not something to merge: two
    connector types have different geometry, so their manifolds are different objects."""
    names = {os.path.basename(os.path.dirname(os.path.abspath(p))) for p in paths}
    if not names:
        return None, 'no input files'
    if len(names) > 1:
        return None, ('inputs span multiple cable directories ' + repr(sorted(names))
                      + ' -- pass --cable to force one, but check you meant to merge them')
    name = names.pop()
    if not name or name in ('.', '..'):
        return None, 'could not infer a cable name from the input path'
    return name, None


def read_manifold_rows(path, min_force=None):
    """(rows, n_read, note) -- the manifold columns of one sampling log.

    `note` is a human-readable reason when the file is skipped entirely (missing columns = a log
    from an older format), so the caller can report it rather than failing the whole batch."""
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        missing = [c for c in MANIFOLD_COLS if c not in cols]
        if missing:
            # Distinguish the one format change we know about. Logs written before translations
            # moved to mm carry the same fields WITHOUT the `_mm` suffix and in METRES; reading them
            # as-is would concatenate metres with millimetres, which is worse than skipping them.
            if 'connector_target_x' in cols:
                return [], 0, ('pre-mm log format (translations in METRES, un-suffixed) -- '
                               're-collect, or scale x/y/z by 1000 and rename to *_mm first')
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


def build(paths, out_path, min_force=None, with_source=False):
    """Concatenate the manifold columns of every input into one CSV. Returns the rows written."""
    header = (['source_file', 'trial'] if with_source else []) + MANIFOLD_COLS
    written = skipped = total_read = 0

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for path in paths:
            rows, n_read, note = read_manifold_rows(path, min_force)
            if note:
                print(f'  SKIP {os.path.basename(path)}: {note}', file=sys.stderr)
                skipped += 1
                continue
            total_read += n_read
            src = os.path.basename(path)
            for rec in rows:
                lead = [src, rec.get('trial', '')] if with_source else []
                writer.writerow(lead + [rec[c] for c in MANIFOLD_COLS])
            written += len(rows)
            kept = '' if min_force is None else f'  (kept {len(rows)}/{n_read})'
            print(f'  {os.path.basename(path)}: {len(rows)} rows{kept}')

    if min_force is not None:
        print(f'\nforce filter |f| >= {min_force} N: kept {written} of {total_read} samples')
    if skipped:
        print(f'{skipped} file(s) skipped -- see the SKIP lines above', file=sys.stderr)
    return written


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Columns kept: connector_target_* (pose wrt target) + wrench_connector_* (wrench in '
               'the connector frame).')
    ap.add_argument('inputs', nargs='+',
                    help='sampling CSVs, directories of them, or globs')
    ap.add_argument('--cable', help='connector type for the output name (default: inferred from the '
                                    'containing <cable>/ directory)')
    ap.add_argument('--out-dir', help='where to write (default: the directory of the first input)')
    ap.add_argument('--out', help='full output path, overriding --cable/--out-dir naming')
    ap.add_argument('--min-force', type=float, default=None, metavar='N',
                    help='keep only samples with |f| >= N in the connector frame (default: keep all '
                         'samples, including the free-space approach)')
    ap.add_argument('--with-source', action='store_true',
                    help='also emit source_file and trial columns for provenance')
    args = ap.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        ap.error('no input CSVs found')

    cable = args.cable
    if not cable and not args.out:
        cable, why = infer_cable(paths)
        if not cable:
            ap.error(f'{why}')

    if args.out:
        out_path = args.out
    else:
        out_dir = args.out_dir or os.path.dirname(os.path.abspath(paths[0]))
        out_path = os.path.join(out_dir, f'{cable}_connector_contact_manifold.csv')

    # Reading the file we are about to truncate would lose data AND double-count it.
    out_key = os.path.normcase(os.path.abspath(out_path))
    if any(os.path.normcase(os.path.abspath(p)) == out_key for p in paths):
        ap.error(f'the output {out_path!r} is also an input -- choose a different --out/--out-dir')

    print(f'{len(paths)} input file(s)' + (f'; cable = {cable}' if cable else ''))
    n = build(paths, out_path, args.min_force, args.with_source)
    if n == 0:
        print('\nNo rows written -- every input was skipped or filtered out.', file=sys.stderr)
        return 1
    print(f'\nWrote {n} samples x {len(MANIFOLD_COLS)} columns -> {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
